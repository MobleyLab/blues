"""Provides classes for setting up and running the BLUES simulation."""

import abc
import copy
import logging

import mdtraj
import numpy
from openmmtools import alchemy, cache
from openmmtools.mcmc import LangevinDynamicsMove, MCMCMove
from openmmtools.states import CompoundThermodynamicState, ThermodynamicState
from simtk import unit

from blues import utils
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.systemfactories import generateAlchSystem
import traceback

logger = logging.getLogger(__name__)


class ReportLangevinDynamicsMove(LangevinDynamicsMove):
    """Run Langevin Dynamics and store coordinates.

    This class modiefies the base class so that coordinates are stored at the
    desired intervals.
    """

    def __init__(self,
                 timestep=1.0 * unit.femtosecond,
                 collision_rate=10.0 / unit.picoseconds,
                 n_steps=1000,
                 reassign_velocities=False,
                 reporters=[],
                 **kwargs):
        super(ReportLangevinDynamicsMove, self).__init__(self, **kwargs)
        self.n_steps = n_steps
        self.resassign_velocities = reassign_velocities
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.currentStep = 0
        self.reporters = reporters

    def _before_integration(self, context, thermodynamic_state):
        """Execute code after Context creation and before integration."""
        context_state = context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=thermodynamic_state.is_periodic)
        self.initial_positions = context_state.getPositions()
        self.initial_energy = thermodynamic_state.reduced_potential(context)
        self._usesPBC = thermodynamic_state.is_periodic

    def _after_integration(self, context, thermodynamic_state):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        context_state = context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.final_positions = context_state.getPositions()
        self.final_energy = thermodynamic_state.reduced_potential(context)

    def apply(self, thermodynamic_state, sampler_state):
        """Propagate the state through the integrator.

        This updates the SamplerState after the integration. It also logs
        benchmarking information through the utils.Timer class.

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
        context, integrator = context_cache.get_context(
            thermodynamic_state, integrator)
        thermodynamic_state.apply_to_context(context)

        # If we reassign velocities, we can ignore the ones in sampler_state.
        sampler_state.apply_to_context(
            context, ignore_velocities=self.reassign_velocities)
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
                self.currentStep += stepsToGo

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
            print(e)
            traceback.print_exc()
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
            sampler_state.update_from_context(context_state,
                                              ignore_collective_variables=True)
            # Update only the collective variables from the Context
            sampler_state.update_from_context(context,
                                              ignore_positions=True,
                                              ignore_velocities=True,
                                              ignore_collective_variables=False)


class NCMCMove(MCMCMove):
    """Base Move class.

    Move provides methods for calculating properties and applying the move
    on the set of atoms being perturbed in the NCMC simulation.
    """

    def __init__(self,
                 timestep,
                 n_steps,
                 atom_subset=None,
                 context_cache=None,
                 reporters=[]):
        self.timestep = timestep
        self.n_steps = n_steps
        self.n_accepted = 0
        self.n_proposed = 0
        self.logp_accept = 0
        self.initial_energy = 0
        self.initial_positions = None
        self.final_energy = 0
        self.final_positions = None
        self.proposed_positions = None
        self.atom_subset = atom_subset
        self.context_cache = context_cache
        self.currentStep = 0
        self.reporters = reporters

    @property
    def statistics(self):
        """Statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted,
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
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.initial_positions = context_state.getPositions()
        self.initial_energy = thermodynamic_state.reduced_potential(context)

    def _after_integration(self, context, thermodynamic_state):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        context_state = context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.final_positions = context_state.getPositions()
        self.final_energy = thermodynamic_state.reduced_potential(context)
        self.logp_accept = context._integrator.getLogAcceptanceProbability(
            context)

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
            timestep=1.0*unit.femtoseconds,
            nprop=1,
            prop_lambda=0.3)

    def apply(self, thermodynamic_state, sampler_state):
        """Apply a metropolized move to the sampler state.

        Total number of acceptances and proposed move are updated.

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
        context, integrator = context_cache.get_context(
            thermodynamic_state, integrator)

        # Compute initial energy. We don't need to set velocities to compute the potential.
        # TODO assume sampler_state.potential_energy is the correct potential if not None?
        sampler_state.apply_to_context(context, ignore_velocities=True)
        context.setVelocitiesToTemperature(thermodynamic_state.temperature)

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
                self.currentStep += stepsToGo

                alch_lambda = integrator.getGlobalVariableByName('lambda')
                if alch_lambda == 0.5:
                    # Propose perturbed positions. Modifying the reference changes the sampler state.
                    # print('Proposing move at lambda = %s...' %alch_lambda)
                    sampler_state.update_from_context(context)
                    proposed_positions = self._propose_positions(sampler_state.positions[self.atom_subset])
                    # Compute the energy of the proposed positions.
                    sampler_state.positions[self.atom_subset] = proposed_positions
                    sampler_state.apply_to_context(context)

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
            print(e)
            # Catches particle positions becoming nan during integration.
        else:
            context_state = context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=thermodynamic_state.is_periodic)

            self._after_integration(context, thermodynamic_state)
            # Update everything but the collective variables from the State object
            sampler_state.update_from_context(context_state,
                                              ignore_collective_variables=True,
                                              ignore_velocities=True)
            # Update only the collective variables from the Context
            sampler_state.update_from_context(context,
                                              ignore_positions=True,
                                              ignore_velocities=True,
                                              ignore_collective_variables=False)

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
    """Propose a random rotation about the center of mass.

    This class will propose a random rotation about the ligand's center of mass.
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
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(
            rand_quat)
        # multiply lig coordinates by rot matrix and add back COM translation from origin
        proposed_positions = numpy.dot(reduced_pos, rand_rotation_matrix
                                       ) * positions.unit + center_of_mass

        return proposed_positions


# =============================================================================
# NCMC+MD (BLUES) SAMPLER
# =============================================================================
class BLUESSampler(object):
    """BLUESSampler runs the NCMC+MD hybrid simulation.

    This class will have a descriptive text added.
    """

    def __init__(self,
                 atom_subset=None,
                 thermodynamic_state=None,
                 alch_thermodynamic_state=None,
                 sampler_state=None,
                 dynamics_move=None,
                 ncmc_move=None,
                 platform=None,
                 topology=None,
                 verbose=False):
        """Create an NCMC sampler.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to simulate
        sampler_state : SamplerState
            The initial sampler state to simulate from.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, this platform will be used
        ncfile : netCDF4.Dataset, optional, default=None
            NetCDF storage file.
        """
        if thermodynamic_state is None:
            raise Exception("'thermodynamic_state' must be specified")
        if sampler_state is None:
            raise Exception("'sampler_state' must be specified")

        self.atom_subset = atom_subset
        # Make a deep copy of the state so that initial state is unchanged.
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        # Generate an alchemical thermodynamic state if none is provided
        if not alch_thermodynamic_state:
            self.alch_thermodynamic_state = self._get_alchemical_state(
                thermodynamic_state)
        self.sampler_state = sampler_state

        self.ncmc_move = ncmc_move
        self.dynamics_move = dynamics_move

        # NML: Attach topology to thermodynamic_states
        self.thermodynamic_state.topology = topology
        self.alch_thermodynamic_state.topology = topology

        # Initialize
        self.accept = False
        self.iteration = 0
        self.n_accepted = 0

        self.verbose = verbose
        self.platform = platform

    def _get_alchemical_state(self, thermodynamic_state):
        alch_system = generateAlchSystem(thermodynamic_state.get_system(),
                                         self.atom_subset)
        alch_state = alchemy.AlchemicalState.from_system(alch_system)
        alch_thermodynamic_state = ThermodynamicState(
            alch_system, thermodynamic_state.temperature)
        alch_thermodynamic_state = CompoundThermodynamicState(
            alch_thermodynamic_state, composable_states=[alch_state])

        return alch_thermodynamic_state

    def _acceptRejectMove(self):
        # Create MD context with the final positions from NCMC simulation
        integrator = self.dynamics_move._get_integrator(
            self.thermodynamic_state)
        context, integrator = self.dynamics_move.context_cache.get_context(
            self.thermodynamic_state, integrator)
        self.sampler_state.apply_to_context(context, ignore_velocities=True)
        alch_energy = self.thermodynamic_state.reduced_potential(context)

        correction_factor = (self.ncmc_move.initial_energy - self.dynamics_move.final_energy + alch_energy - self.ncmc_move.final_energy)
        logp_accept = self.ncmc_move.logp_accept
        randnum = numpy.log(numpy.random.random())
        # print("logP {} + corr {}".format(logp_accept, correction_factor))
        logp_accept = logp_accept + correction_factor
        if (not numpy.isnan(logp_accept) and logp_accept > randnum):
            logger.debug('NCMC MOVE ACCEPTED: logP {}'.format(logp_accept))
            self.accept = True
            self.n_accepted += 1
        else:
            logger.debug('NCMC MOVE REJECTED: logP {}'.format(logp_accept))
            self.accept = False

            # Restore original positions.
            self.sampler_state.positions = self.ncmc_move.initial_positions

    def equil(self, n_iterations=1):
        """Equilibrate the system for N iterations."""
        self.dynamics_move.totalSteps = int(self.dynamics_move.n_steps * n_iterations)
        # Set initial conditions by running 1 iteration of MD first
        for iteration in range(n_iterations):
            self.dynamics_move.apply(self.thermodynamic_state,
                                     self.sampler_state)
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
        self.ncmc_move.totalSteps = int(self.ncmc_move.n_steps * n_iterations)
        self.dynamics_move.totalSteps = int(self.dynamics_move.n_steps * n_iterations)

        if self.iteration == 0:
            # Set initial conditions by running 1 iteration of MD first
            self.equil(1)

        self.iteration = 0
        for iteration in range(n_iterations):
            if self.verbose:
                print("." * 80)
                print("BLUES Sampler iteration %d" % self.iteration)

            # print('NCMC Simulation')
            self.ncmc_move.apply(self.alch_thermodynamic_state,
                                 self.sampler_state)

            self._acceptRejectMove()

            # print('MD Simulation')
            self.dynamics_move.apply(self.thermodynamic_state,
                                     self.sampler_state)

            # Increment iteration count
            self.iteration += 1

            if self.verbose:
                print("." * 80)

        # print('n_accepted', self.n_accepted)
        # print('iteration', self.iteration)
