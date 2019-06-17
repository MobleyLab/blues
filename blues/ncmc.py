"""Provides moves and classes for running the BLUES simulation."""

import abc
import copy
import logging

import mdtraj
import numpy
from openmmtools import alchemy, cache
from openmmtools.mcmc import LangevinDynamicsMove, MCMCMove
from openmmtools.states import CompoundThermodynamicState, ThermodynamicState
from simtk import openmm, unit

from blues import utils
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.systemfactory import generateAlchSystem
import traceback

logger = logging.getLogger(__name__)

class ReportLangevinDynamicsMove(object):
    """Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move class allows the attachment of a reporter for storing the data from running this segment of dynamics. This move assigns a velocity from the Maxwell-Boltzmann distribution and executes a number of Maxwell-Boltzmann steps to propagate dynamics. This is not a *true* Monte Carlo move, in that the generation of the correct distribution is only exact in the limit of infinitely small timestep; in other words, the discretization error is assumed to be negligible. Use HybridMonteCarloMove instead to ensure the exact distribution is generated.

    .. warning::
        No Metropolization is used to ensure the correct phase space
        distribution is sampled. This means that timestep-dependent errors
        will remain uncorrected, and are amplified with larger timesteps.
        Use this move at your own risk!

    Parameters
    ----------
    n_steps : int, optional
        The number of integration timesteps to take each time the
        move is applied (default is 1000).
    timestep : simtk.unit.Quantity, optional
        The timestep to use for Langevin integration
        (time units, default is 1*simtk.unit.femtosecond).
    collision_rate : simtk.unit.Quantity, optional
        The collision rate with fictitious bath particles
        (1/time units, default is 10/simtk.unit.picoseconds).
    reassign_velocities : bool, optional
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move (default is False).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).
    reporters : list
        A list of the storage classes inteded for reporting the simulation data.
        This can be either blues.storage.(NetCDF4Storage/BLUESStateDataStorage).

    Attributes
    ----------
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    timestep : simtk.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : simtk.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    reassign_velocities : bool
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move.
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.
    reporters : list
        A list of the storage classes inteded for reporting the simulation data.
        This can be either blues.storage.(NetCDF4Storage/BLUESStateDataStorage).

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import SamplerState, ThermodynamicState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)

    Create reporters for storing our simulation data.

    >>> from blues.storage import NetCDF4Storage, BLUESStateDataStorage
    nc_storage = NetCDF4Storage('test-md.nc',
                                reportInterval=5,
                                crds=True, vels=True, frcs=True)
    state_storage = BLUESStateDataStorage('test.log',
                                          reportInterval=5,
                                          step=True, time=True,
                                          potentialEnergy=True,
                                          kineticEnergy=True,
                                          totalEnergy=True,
                                          temperature=True,
                                          volume=True,
                                          density=True,
                                          progress=True,
                                          remainingTime=True,
                                          speed=True,
                                          elapsedTime=True,
                                          systemMass=True,
                                          totalSteps=10)

    Create a Langevin move with default parameters

    >>> move = ReportLangevinDynamicsMove()

    or create a Langevin move with specified parameters.

    >>> move = ReportLangevinDynamicsMove(timestep=0.5*unit.femtoseconds,
                                          collision_rate=20.0/unit.picoseconds, n_steps=10,
                                          reporters=[nc_storage, state_storage])

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self,
                 n_steps=1000,
                 timestep=2.0 * unit.femtosecond,
                 collision_rate=1.0 / unit.picoseconds,
                 reassign_velocities=True,
                 context_cache=None,
                 reporters=[]):
        self.n_steps = n_steps
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.reassign_velocities = reassign_velocities
        self.context_cache = context_cache
        self.reporters = list(reporters)
        self.currentStep = 0

    def _get_integrator(self, thermodynamic_state):
        """
        Generates a LangevinIntegrator for the Simulations.
        Parameters
        ----------

        Returns
        -------
        integrator : openmm.LangevinIntegrator
            The LangevinIntegrator object intended for the System.
        """
        integrator = openmm.LangevinIntegrator(thermodynamic_state.temperature,
                            self.collision_rate,
                            self.timestep)
        return integrator

    def _before_integration(self, context, thermodynamic_state):
        """Execute code after Context creation and before integration."""
        context_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=thermodynamic_state.is_periodic)
        self.initial_positions = context_state.getPositions(asNumpy=True)
        self.initial_energy = thermodynamic_state.reduced_potential(context)
        self._usesPBC = thermodynamic_state.is_periodic

    def _after_integration(self, context, thermodynamic_state):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        context_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.final_positions = context_state.getPositions(asNumpy=True)
        self.final_energy = thermodynamic_state.reduced_potential(context)

    def apply(self, thermodynamic_state, sampler_state):
        """Propagate the state through the integrator.

        This updates the SamplerState after the integration.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        """
        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create integrator.
        integrator = self._get_integrator(thermodynamic_state)

        # Create context.
        context, integrator = context_cache.get_context(thermodynamic_state, integrator)
        thermodynamic_state.apply_to_context(context)

        # If we reassign velocities, we can ignore the ones in sampler_state.
        sampler_state.apply_to_context(context, ignore_velocities=self.reassign_velocities)
        if self.reassign_velocities:
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)

        # Subclasses may implement _before_integration().
        self._before_integration(context, thermodynamic_state)

        try:
            nextReport = [None] * len(self.reporters)
            endStep = self.currentStep + self.n_steps
            while self.currentStep < endStep:
                nextSteps = endStep - self.currentStep
                anyReport = False
                for i, reporter in enumerate(self.reporters):
                    nextReport[i] = reporter.describeNextReport(self)
                    if nextReport[i][0] > 0 and nextReport[i][0] <= nextSteps:
                        nextSteps = nextReport[i][0]
                        anyReport = True

                stepsToGo = nextSteps
                while stepsToGo > 10:
                    integrator.step(10)
                    stepsToGo -= 10
                integrator.step(stepsToGo)
                self.currentStep += nextSteps

                if anyReport:
                    reports = []
                    context_state = context.getState(
                        getPositions=True,
                        getVelocities=True,
                        getEnergy=True,
                        enforcePeriodicBox=thermodynamic_state.is_periodic)

                    context_state.currentStep = self.currentStep
                    context_state.system = thermodynamic_state.get_system()

                    for reporter, report in zip(self.reporters, nextReport):
                        reports.append((reporter, report))
                    for reporter, next in reports:
                        reporter.report(context_state, integrator)

        except Exception as e:
            logger.error(e)
            # Catches particle positions becoming nan during integration.
        else:
            # We get also velocities here even if we don't need them because we
            # will recycle this State to update the sampler state object. This
            # way we won't need a second call to Context.getState().
            context_state = context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=thermodynamic_state.is_periodic)

            # Subclasses can read here info from the context to update internal statistics.
            self._after_integration(context, thermodynamic_state)

            # Updated sampler state.
            # This is an optimization around the fact that Collective Variables are not a part of the State,
            # but are a part of the Context. We do this call twice to minimize duplicating information fetched from
            # the State.
            # Update everything but the collective variables from the State object
            sampler_state.update_from_context(
                context_state, ignore_positions=False, ignore_velocities=False, ignore_collective_variables=True)
            # Update only the collective variables from the Context
            #sampler_state.update_from_context(
            #    context, ignore_positions=True, ignore_velocities=True, ignore_collective_variables=False)


class NCMCMove(MCMCMove):
    """A general NCMC move that applies an alchemical integrator.

    This class is intended to be inherited by NCMCMoves that need to alchemically modify and perturb part of the system. The child class has to implement the _propose_positions method. Reporters can be attached to report
    data from the NCMC part of the simulation.

    You can decide to override _before_integration() and _after_integration()
    to execute some code at specific points of the workflow, for example to
    read data from the Context before the it is destroyed.

    Parameters
    ----------
    n_steps : int, optional
        The number of integration timesteps to take each time the
        move is applied (default is 1000).
    timestep : simtk.unit.Quantity, optional
        The timestep to use for Langevin integration
        (time units, default is 1*simtk.unit.femtosecond).
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).
    reporters : list
        A list of the storage classes inteded for reporting the simulation data.
        This can be either blues.storage.(NetCDF4Storage/BLUESStateDataStorage).

    Attributes
    ----------
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    timestep : simtk.unit.Quantity
        The timestep to use for Langevin integration (time units).
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.
    reporters : list
        A list of the storage classes inteded for reporting the simulation data.
        This can be either blues.storage.(NetCDF4Storage/BLUESStateDataStorage).
    """

    def __init__(self,
                 n_steps=1000,
                 timestep=2.0 * unit.femtosecond,
                 atom_subset=None,
                 context_cache=None,
                 nprop=1,
                 propLambda=0.3,
                 reporters=[]):
        self.timestep = timestep
        self.n_steps = n_steps
        self.nprop = nprop
        self.propLambda = propLambda
        self.atom_subset = atom_subset
        self.context_cache = context_cache
        self.reporters = list(reporters)

        self.n_accepted = 0
        self.n_proposed = 0
        self.logp_accept = 0
        self.initial_energy = 0
        self.initial_positions = None
        self.final_energy = 0
        self.final_positions = None
        self.proposed_positions = None
        self.currentStep = 0

    @property
    def statistics(self):
        """Statistics as a dictionary."""
        return dict(
            n_accepted=self.n_accepted,
            n_proposed=self.n_proposed,
            initial_energy=self.initial_energy,
            initial_positions=self.initial_positions,
            final_energy=self.final_energy,
            proposed_positions=self.proposed_positions,
            final_positions=self.final_positions,
            logp_accept=self.logp_accept)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value['n_accepted']
        self.n_proposed = value['n_proposed']
        self.initial_energy = value['initial_energy']
        self.initial_positions = value['initial_positions']
        self.final_energy = value['final_energy']
        self.proposed_positions = value['proposed_positions']
        self.final_positions = value['final_positions']
        self.logp_accept = value['logp_accept']

    def _before_integration(self, context, thermodynamic_state):
        """Execute code after Context creation and before integration."""
        context_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.initial_positions = context_state.getPositions(asNumpy=True)
        self.initial_box_vectors = context_state.getPeriodicBoxVectors()
        self.initial_energy = thermodynamic_state.reduced_potential(context)

    def _after_integration(self, context, thermodynamic_state):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        context_state = context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.final_positions = context_state.getPositions(asNumpy=True)
        self.final_box_vectors = context_state.getPeriodicBoxVectors()
        self.final_energy = thermodynamic_state.reduced_potential(context)
        self.logp_accept = context._integrator.getLogAcceptanceProbability(context)

    def _get_integrator(self, thermodynamic_state):
        return AlchemicalExternalLangevinIntegrator(
            alchemical_functions={
                'lambda_sterics':
                'min(1, (1/0.3)*abs(lambda-0.5))',
                'lambda_electrostatics':
                'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            },
            splitting="H V R O R V H",
            temperature=thermodynamic_state.temperature,
            nsteps_neq=self.n_steps,
            timestep=self.timestep,
            nprop=self.nprop,
            propLambda=self.propLambda)

    def apply(self, thermodynamic_state, sampler_state):
        """Apply a move to the sampler state.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to apply the move.
        sampler_state : openmmtools.states.SamplerState
           The initial sampler state to apply the move to. This is modified.

        """
        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create integrator
        integrator = self._get_integrator(thermodynamic_state)

        # Create context
        context, integrator = context_cache.get_context(thermodynamic_state, integrator)
        #thermodynamic_state.apply_to_context(context)

        # Compute initial energy. We don't need to set velocities to compute the potential.
        # TODO assume sampler_state.potential_energy is the correct potential if not None?
        sampler_state.apply_to_context(context, ignore_velocities=False)

        self._before_integration(context, thermodynamic_state)

        try:
            # #NML: Old Way
            # for step in range(int(self.n_steps)):
            #     alchLambda = integrator.getGlobalVariableByName('lambda')
            #     if alchLambda == 0.5:
            #         positions = context.getState(getPositions=True).getPositions(asNumpy=True)
            #         proposed_positions = self._propose_positions(positions[self.atom_subset])
            #         for index, atomidx in enumerate(self.atom_subset):
            #             positions[atomidx] = proposed_positions[index]
            #         context.setPositions(positions)
            #     if step % self.reporters[0]._reportInterval == 0:
            #         context_state = context.getState(
            #             getPositions=True,
            #             getVelocities=True,
            #             getEnergy=True,
            #             enforcePeriodicBox=thermodynamic_state.is_periodic)
            #         context_state.currentStep = self.currentStep
            #         context_state.system = thermodynamic_state.get_system()
            #         self.reporters[0].report(context_state, integrator)
            #     integrator.step(1)
            #     self.currentStep+=1

            nextReport = [None] * len(self.reporters)
            endStep = self.currentStep + self.n_steps
            while self.currentStep < endStep:
                nextSteps = endStep - self.currentStep
                anyReport = False
                for i, reporter in enumerate(self.reporters):
                    nextReport[i] = reporter.describeNextReport(self)
                    if nextReport[i][0] > 0 and nextReport[i][0] <= nextSteps:
                        nextSteps = nextReport[i][0]
                        anyReport = True

                alchLambda = integrator.getGlobalVariableByName('lambda')
                if alchLambda == 0.5:
                    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                    proposed_positions = self._propose_positions(positions[self.atom_subset])
                    for index, atomidx in enumerate(self.atom_subset):
                        positions[atomidx] = proposed_positions[index]
                    context.setPositions(positions)

                stepsToGo = nextSteps
                while stepsToGo > 10:
                    integrator.step(10)
                    stepsToGo -= 10
                integrator.step(stepsToGo)
                self.currentStep += nextSteps

                if anyReport:
                    context_state = context.getState(
                        getPositions=True,
                        getVelocities=True,
                        getEnergy=True,
                        enforcePeriodicBox=thermodynamic_state.is_periodic)

                    context_state.currentStep = self.currentStep
                    context_state.system = thermodynamic_state.get_system()

                    reports = []
                    for reporter, report in zip(self.reporters, nextReport):
                        reports.append((reporter, report))
                    for reporter, next in reports:
                        reporter.report(context_state, integrator)

        except Exception as e:
            logger.error(e)
            # Catches particle positions becoming nan during integration.
        else:
            context_state = context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=thermodynamic_state.is_periodic)

            self._after_integration(context, thermodynamic_state)
            # Update everything but the collective variables from the State object
            sampler_state.update_from_context(
                context_state, ignore_positions=False, ignore_velocities=False, ignore_collective_variables=True)
            # Update only the collective variables from the Context
            #sampler_state.update_from_context(
            #    context, ignore_positions=True, ignore_velocities=True, ignore_collective_variables=False)

    @abc.abstractmethod
    def _propose_positions(self, positions):
        """Return new proposed positions.

        These method must be implemented in subclasses.

        Parameters
        ----------
        positions : nx3 numpy.ndarray
            The original positions of the subset of atoms that these move
            applied to.

        Returns
        -------
        proposed_positions : nx3 numpy.ndarray
            The new proposed positions.

        """
        pass


class RandomLigandRotationMove(NCMCMove):
    """An NCMC move which proposes random rotations.

    This class will propose a random rotation (as a rigid body) using the center of mass of the selected atoms. This class does not metropolize the proposed moves. Reporters can be attached to record the ncmc simulation data, mostly useful for debugging by storing coordinates of the proposed moves or monitoring the ncmc simulation progression by attaching a state reporter.

    Parameters
    ----------
    n_steps : int, optional
        The number of integration timesteps to take each time the
        move is applied (default is 1000).
    timestep : simtk.unit.Quantity, optional
        The timestep to use for Langevin integration
        (time units, default is 1*simtk.unit.femtosecond).
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).
    reporters : list
        A list of the storage classes inteded for reporting the simulation data.
        This can be either blues.storage.(NetCDF4Storage/BLUESStateDataStorage).

    Attributes
    ----------
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    timestep : simtk.unit.Quantity
        The timestep to use for Langevin integration (time units).
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.
    reporters : list
        A list of the storage classes inteded for reporting the simulation data.
        This can be either blues.storage.(NetCDF4Storage/BLUESStateDataStorage).

    Examples
    --------
    First we need to create the thermodynamic state, alchemical thermodynamic state,  and the sampler state to propagate. Here we create a toy system of a charged ethylene molecule in between two charged particles.

    >>> from simtk import unit
    >>> from openmmtools import testsystems, alchemy
    >>> from openmmtools.states import SamplerState, ThermodynamicState
    >>> from blues.systemfactories import generateAlchSystem
    >>> from blues import utils

    >>> structure_pdb = utils.get_data_filename('blues', 'tests/data/ethylene_structure.pdb')
    >>> structure = parmed.load_file(structure_pdb)
    >>> system_xml = utils.get_data_filename('blues', 'tests/data/ethylene_system.xml')
        with open(system_xml, 'r') as infile:
            xml = infile.read()
            system = openmm.XmlSerializer.deserialize(xml)
    >>> thermodynamic_state = ThermodynamicState(system=system, temperature=200*unit.kelvin)
    >>> sampler_state = SamplerState(positions=structure.positions.in_units_of(unit.nanometers))
    >>> alchemical_atoms = [2, 3, 4, 5, 6, 7]
    >>> alch_system = generateAlchSystem(thermodynamic_state.get_system(), alchemical_atoms)
    >>> alch_state = alchemy.AlchemicalState.from_system(alch_system)
    >>> alch_thermodynamic_state = ThermodynamicState(
            alch_system, thermodynamic_state.temperature)
    >>> alch_thermodynamic_state = CompoundThermodynamicState(
            alch_thermodynamic_state, composable_states=[alch_state])

    Create reporters for storing our ncmc simulation data.

    >>> from blues.storage import NetCDF4Storage, BLUESStateDataStorage
    nc_storage = NetCDF4Storage('test-ncmc.nc',
                                reportInterval=5,
                                crds=True, vels=True, frcs=True,
                                protocolWork=True, alchemicalLambda=True)
    state_storage = BLUESStateDataStorage('test-ncmc.log',
                                          reportInterval=5,
                                          step=True, time=True,
                                          potentialEnergy=True,
                                          kineticEnergy=True,
                                          totalEnergy=True,
                                          temperature=True,
                                          volume=True,
                                          density=True,
                                          progress=True,
                                          remainingTime=True,
                                          speed=True,
                                          elapsedTime=True,
                                          systemMass=True,
                                          totalSteps=10,
                                          protocolWork=True,
                                          alchemicalLambda=True)

    Create a RandomLigandRotationMove move

    >>> rot_move = RandomLigandRotationMove(n_steps=5,
                                            timestep=1*unit.femtoseconds,
                                            atom_subset=alchemical_atoms,
                                            reporters=[nc_storage, state_storage])

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, structure.positions)
    False
    """

    def _before_integration(self, context, thermodynamic_state):
        super(RandomLigandRotationMove, self)._before_integration(context, thermodynamic_state)
        masses, totalmass = utils.getMasses(self.atom_subset, thermodynamic_state.topology)
        self.masses = masses

    def _propose_positions(self, positions):
        """Return new proposed positions.

        These method must be implemented in subclasses.

        Parameters
        ----------
        positions : nx3 numpy.ndarray
            The original positions of the subset of atoms that these move
            applied to.

        Returns
        -------
        proposed_positions : nx3 numpy.ndarray
            The new proposed positions.
        """
        # print('Proposing positions...')
        # Calculate the center of mass

        center_of_mass = utils.getCenterOfMass(positions, self.masses)
        reduced_pos = positions - center_of_mass
        # Define random rotational move on the ligand
        rand_quat = mdtraj.utils.uniform_quaternion(size=None)
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
        # multiply lig coordinates by rot matrix and add back COM translation from origin
        proposed_positions = numpy.dot(reduced_pos, rand_rotation_matrix) * positions.unit + center_of_mass

        return proposed_positions


# =============================================================================
# NCMC+MD (BLUES) SAMPLER
# =============================================================================
class BLUESSampler(object):
    """BLUESSampler runs the NCMC+MD hybrid simulation.

    This class ties together the two moves classes to execute the NCMC+MD hybrid simulation. One move class is intended to carry out traditional MD and the other is intended carry out the NCMC move proposals which performs the alchemical transformation to given atom subset. This class handles proper metropolization of the NCMC move proposals, while correcting for the switch in integrators.
    """

    def __init__(self,
                 thermodynamic_state=None,
                 alch_thermodynamic_state=None,
                 sampler_state=None,
                 dynamics_move=None,
                 ncmc_move=None,
                 topology=None):
        """Create an NCMC sampler.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to simulate
        alch_thermodynamic_state : CompoundThermodynamicState, optional
            The alchemical thermodynamic state to simulate. If None, one is generated from the thermodynamic state using the default alchemical parameters.
        sampler_state : SamplerState
            The initial sampler state to simulate from.
        dynamics_move : ReportLangevinDynamicsMove
            The move class which propagates traditional dynamics.
        ncmc_move : NCMCMove
            The NCMCMove class which proposes perturbations to the selected atoms.
        topology : openmm.Topology
            A Topology of the system to be simulated.
        """
        if thermodynamic_state is None:
            raise Exception("'thermodynamic_state' must be specified")
        if sampler_state is None:
            raise Exception("'sampler_state' must be specified")

        self.sampler_state = sampler_state
        self.ncmc_move = ncmc_move
        self.dynamics_move = dynamics_move
        # Make a deep copy of the state so that initial state is unchanged.
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        # Generate an alchemical thermodynamic state if none is provided
        if alch_thermodynamic_state:
            self.alch_thermodynamic_state = alch_thermodynamic_state
        else:
            self.alch_thermodynamic_state = self._get_alchemical_state(thermodynamic_state)

        # NML: Attach topology to thermodynamic_states
        self.thermodynamic_state.topology = topology
        self.alch_thermodynamic_state.topology = topology

        # Initialize
        self.accept = False
        self.iteration = 0
        self.n_accepted = 0

    def _get_alchemical_state(self, thermodynamic_state):
        alch_system = generateAlchSystem(thermodynamic_state.get_system(), self.ncmc_move.atom_subset)
        alch_state = alchemy.AlchemicalState.from_system(alch_system)
        alch_thermodynamic_state = ThermodynamicState(alch_system, thermodynamic_state.temperature)
        alch_thermodynamic_state = CompoundThermodynamicState(alch_thermodynamic_state, composable_states=[alch_state])

        return alch_thermodynamic_state

    def _printSimulationTiming(self, n_iterations):
        """Prints the simulation timing and related information."""
        self.ncmc_move.totalSteps = int(self.ncmc_move.n_steps * n_iterations)
        self.dynamics_move.totalSteps = int(self.dynamics_move.n_steps * n_iterations)
        md_timestep = self.dynamics_move.timestep.value_in_unit(unit.picoseconds)
        md_steps = self.dynamics_move.n_steps
        ncmc_timestep = self.ncmc_move.timestep.value_in_unit(unit.picoseconds)
        ncmc_steps = self.ncmc_move.n_steps
        nprop = self.ncmc_move.nprop
        propLambda = self.ncmc_move.propLambda

        force_eval = n_iterations * (ncmc_steps + md_steps)
        time_ncmc_iter = ncmc_steps * ncmc_timestep
        time_ncmc_total = time_ncmc_iter * n_iterations
        time_md_iter = md_steps * md_timestep
        time_md_total = time_md_iter * n_iterations
        time_iter = time_ncmc_iter + time_md_iter
        time_total = time_iter * n_iterations

        msg = 'Total BLUES Simulation Time = %s ps (%s ps/Iter)\n' % (time_total, time_iter)
        msg += 'Total Force Evaluations = %s \n' % force_eval
        msg += 'Total NCMC time = %s ps (%s ps/iter)\n' % (time_ncmc_total, time_ncmc_iter)
        msg += 'Total MD time = %s ps (%s ps/iter)\n' % (time_md_total, time_md_iter)

        # Calculate number of lambda steps inside/outside region with extra propgation steps
        #steps_in_prop = int(nprop * (2 * math.floor(propLambda * nstepsNC)))
        #steps_out_prop = int((2 * math.ceil((0.5 - propLambda) * nstepsNC)))

        #prop_lambda_window = self._ncmc_sim.context._integrator._propLambda
        # prop_lambda_window = round(prop_lambda_window[1] - prop_lambda_window[0], 4)
        # if propSteps != nstepsNC:
        #     msg += '\t%s lambda switching steps within %s total propagation steps.\n' % (nstepsNC, propSteps)
        #     msg += '\tExtra propgation steps between lambda [%s, %s]\n' % (prop_lambda_window[0],
        #                                                                    prop_lambda_window[1])
        #     msg += '\tLambda: 0.0 -> %s = %s propagation steps\n' % (prop_lambda_window[0], int(steps_out_prop / 2))
        #     msg += '\tLambda: %s -> %s = %s propagation steps\n' % (prop_lambda_window[0], prop_lambda_window[1],
        #                                                             steps_in_prop)
        #     msg += '\tLambda: %s -> 1.0 = %s propagation steps\n' % (prop_lambda_window[1], int(steps_out_prop / 2))

        # #Get trajectory frame interval timing for BLUES simulation
        # if 'md_trajectory_interval' in self._config.keys():
        #     frame_iter = nstepsMD / self._config['md_trajectory_interval']
        #     timetraj_frame = (time_ncmc_iter + time_md_iter) / frame_iter
        #     msg += 'Trajectory Interval = %s ps/frame (%s frames/iter)' % (timetraj_frame, frame_iter)

        logger.info(msg)

    def _computeAlchemicalCorrection(self):
        # Create MD context with the final positions from NCMC simulation
        integrator = self.dynamics_move._get_integrator(self.thermodynamic_state)
        context, integrator = cache.global_context_cache.get_context(self.thermodynamic_state, integrator)
        self.thermodynamic_state.apply_to_context(context)
        self.sampler_state.apply_to_context(context, ignore_velocities=True)
        alch_energy = self.thermodynamic_state.reduced_potential(context)
        correction_factor = (self.ncmc_move.initial_energy - self.dynamics_move.initial_energy + alch_energy - self.ncmc_move.final_energy)
        # correction_factor = (self.ncmc_move.initial_energy - self.dynamics_move.final_energy + alch_energy -
        #                      self.ncmc_move.final_energy)
        return correction_factor

    def _acceptRejectMove(self):
        logp_accept = self.ncmc_move.logp_accept
        randnum = numpy.log(numpy.random.random())

        correction_factor = self._computerAlchemicalCorrection()
        logger.debug("logP {} + corr {}".format(logp_accept, correction_factor))
        logp_accept = logp_accept + correction_factor

        if (not numpy.isnan(logp_accept) and logp_accept > randnum):
            logger.debug('NCMC MOVE ACCEPTED: logP {}'.format(logp_accept))
            self.n_accepted += 1
        else:
            logger.debug('NCMC MOVE REJECTED: logP {}'.format(logp_accept))
            # Restore original positions & box vectors
            self.sampler_state.positions = self.ncmc_move.initial_positions
            self.sampler_state.box_vectors = self.ncmc_move.initial_box_vectors

    def equil(self, n_iterations=1):
        """Equilibrate the system for N iterations."""
        # Set initial conditions by running 1 iteration of MD first
        for iteration in range(n_iterations):
            self.dynamics_move.apply(self.thermodynamic_state, self.sampler_state)
        self.dynamics_move.currentStep = 0
        self.iteration += 1

    def run(self, n_iterations=1):
        """Run the sampler for the specified number of iterations.

        descriptive summary here

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        context, integrator = cache.global_context_cache.get_context(self.thermodynamic_state)
        utils.print_host_info(context)
        self._printSimulationTiming(n_iterations)
        if self.iteration == 0:
            # Set initial conditions by running 1 iteration of MD first
            self.equil(1)

        self.iteration = 0
        for iteration in range(n_iterations):

            # print('NCMC Simulation')
            self.ncmc_move.apply(self.alch_thermodynamic_state, self.sampler_state)

            self._acceptRejectMove()

            # print('MD Simulation')
            self.dynamics_move.apply(self.thermodynamic_state, self.sampler_state)

            # Increment iteration count
            self.iteration += 1

        # print('n_accepted', self.n_accepted)
        # print('iteration', self.iteration)
