from __future__ import print_function
import numpy as np
import copy
import logging
import traceback
from simtk import openmm, unit
from openmmtools.integrators import GHMCIntegrator

default_functions = {
    'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
    'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
    'lambda_bonds' : '0.9*lambda + 0.1', # don't fully soften bonds
    'lambda_angles' : '0.9*lambda + 0.1', # don't fully soften angles
    'lambda_torsions' : 'lambda'
    }

default_hybrid_functions = {
    'lambda_sterics' : 'lambda',
    'lambda_electrostatics' : 'lambda',
    'lambda_bonds' : 'lambda',
    'lambda_angles' : 'lambda',
    'lambda_torsions' : 'lambda'
    }


default_temperature = 300.0*unit.kelvin
default_nsteps = 1
default_timestep = 1.0 * unit.femtoseconds
default_steps_per_propagation = 1

class NaNException(Exception):
    def __init__(self, *args, **kwargs):
        super(NaNException,self).__init__(*args,**kwargs)

class NCMCEngine(object):
    """
    NCMC switching engine

    Examples
    --------

    Create a transformation for an alanine dipeptide test system where the N-methyl group is eliminated.

    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from perses.rjmc.topology_proposal import TopologyProposal
    >>> new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    >>> topology_proposal = TopologyProposal(old_system=testsystem.system, old_topology=testsystem.topology, old_chemical_state_key='AA', new_chemical_state_key='AA', new_system=testsystem.system, new_topology=testsystem.topology, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())
    >>> ncmc_engine = NCMCEngine(temperature=300.0*unit.kelvin, functions=default_functions, nsteps=50, timestep=1.0*unit.femtoseconds)
    >>> positions = testsystem.positions
    >>> [positions, logP_delete, potential_delete] = ncmc_engine.integrate(topology_proposal, positions, direction='delete')
    >>> [positions, logP_insert, potential_insert] = ncmc_engine.integrate(topology_proposal, positions, direction='insert')

    """

    def __init__(self, temperature=default_temperature, functions=None, nsteps=default_nsteps, steps_per_propagation=default_steps_per_propagation, timestep=default_timestep, constraint_tolerance=None, platform=None, write_ncmc_interval=None, integrator_type='GHMC', storage=None, verbose=False):
        """
        This is the base class for NCMC switching between two different systems.

        Arguments
        ---------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature at which switching is to be run
        functions : dict of str:str, optional, default=default_functions
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int, optional, default=1
            The number of steps to use for switching.
        steps_per_propagation : int, optional, default=1
            The number of intermediate propagation steps taken at each switching step
        timestep : simtk.unit.Quantity with units compatible with femtoseconds, optional, default=1*femtosecond
            The timestep to use for integration of switching velocity Verlet steps.
        constraint_tolerance : float, optional, default=None
            If not None, this relative constraint tolerance is used for position and velocity constraints.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, the platform to use for OpenMM simulations.
        write_ncmc_interval : int, optional, default=None
            If a positive integer is specified, a snapshot frame will be written to storage with the specified interval on NCMC switching.
            'storage' must also be specified.
        integrator_type : str, optional, default='GHMC'
            NCMC internal integrator type ['GHMC', 'VV']
        storage : NetCDFStorageView, optional, default=None
            If specified, write data using this class.
        verbose : bool, optional, default=False
            If True, print debug information.
        """
        # Handle some defaults.
        if functions == None:
            functions = default_functions
        if nsteps == None:
            nsteps = default_nsteps
        if timestep == None:
            timestep = default_timestep
        if temperature == None:
            temperature = default_temperature

        self.temperature = temperature
        self.functions = copy.deepcopy(functions)
        self.nsteps = nsteps
        self.timestep = timestep
        self.constraint_tolerance = constraint_tolerance
        self.platform = platform
        self.integrator_type = integrator_type
        self.steps_per_propagation = steps_per_propagation
        self.verbose = verbose

        if steps_per_propagation != 1:
            raise Exception('steps_per_propagation must be 1 until CustomIntegrator is debugged')

        self.nattempted = 0

        self._storage = None
        if storage is not None:
            self._storage = NetCDFStorageView(storage, modname=self.__class__.__name__)
        self.write_ncmc_interval = write_ncmc_interval

    @property
    def beta(self):
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * self.temperature
        beta = 1.0 / kT
        return beta

    def _getAvailableParameters(self, system, prefix='lambda'):
        """
        Return a list of available alchemical context parameters defined in the system

        Parameters
        ----------
        system : simtk.openmm.System
            The system for which available context parameters are to be determined
        prefix : str, optional, default='lambda'
            Prefix required for parameters to be returned.

        Returns
        -------
        parameters : list of str
            The list of available context parameters in the system

        """
        parameters = list()
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if hasattr(force, 'getNumGlobalParameters'):
                for parameter_index in range(force.getNumGlobalParameters()):
                    parameter_name = force.getGlobalParameterName(parameter_index)
                    if parameter_name[0:(len(prefix)+1)] == (prefix + '_'):
                        parameters.append(parameter_name)
        return parameters

    def _updateAlchemicalState(self, context, functions, value):
        """
        Update alchemical state using the specified lambda value.

        Parameters
        ----------
        context : simtk.openmm.Context
            The Context
        functions : dict
            A dictionary of functions
        value : float
            The alchemical lambda value
 
        TODO: Improve function evaluation to better match Lepton and be more flexible in exact replacement of 'lambda' tokens

        """
        from parsing import NumericStringParser
        nsp = NumericStringParser()
        for parameter in functions:
            function = functions[parameter]
            evaluated = nsp.eval(function.replace('lambda', str(value)))
            context.setParameter(parameter, evaluated)
 
    def _computeAlchemicalCorrection(self, unmodified_system, alchemical_system, initial_positions, final_positions, direction='insert'):
        """
        Compute log probability for correction from transforming real system to/from alchemical system.

        If direction is `insert`, the contribution at `final_positions` is computed as (real - alchemical).
        If direction is `delete`, the contribution at `initial_positions` is computed as (alchemical - real).

        Parameters
        ----------
        unmodified_system : simtk.unit.System
            Real fully-interacting system.
        alchemical_system : simtk.unit.System
            Alchemically modified system in fully-interacting form.
        initial_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The initial positions before NCMC switching.
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The final positions after NCMC switching.
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])

        Returns
        -------
        logP_alchemical_correction : float
            The log acceptance probability of the switch

        """

        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        def computePotentialEnergy(system, positions):
            """
            Compute potential energy of the specified system object at the specified positions.

            Constraints are applied before the energy is computed.

            Parameters
            ----------
            system : simtk.openmm.System
                The System object for which the potential energy is to be computed.
            positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
                Positions of the atoms for which energy is to be computed.

            Returns
            -------
            potential : simtk.unit.Quantity with units of energy
                The computed potential energy

            """
            # Create dummy integrator.
            integrator = openmm.VerletIntegrator(self.timestep)
            # Set the constraint tolerance if specified.
            if self.constraint_tolerance is not None:
                integrator.setConstraintTolerance(self.constraint_tolerance)
            # Create a context on the specified platform.
            if self.platform is not None:
                context = openmm.Context(system, integrator, self.platform)
            else:
                context = openmm.Context(system, integrator)
            context.setPositions(positions)
            context.applyConstraints(integrator.getConstraintTolerance())
            # Compute potential energy.
            potential = context.getState(getEnergy=True).getPotentialEnergy()
            # Clean up context and integrator.
            del context, integrator
            # Return potential energy.
            return potential

        # Compute correction from transforming real system to/from alchemical system
        if direction == 'delete':
            alchemical_potential_correction = computePotentialEnergy(alchemical_system, initial_positions) - computePotentialEnergy(unmodified_system, initial_positions)
        elif direction == 'insert':
            alchemical_potential_correction = computePotentialEnergy(unmodified_system, final_positions) - computePotentialEnergy(alchemical_system, final_positions)
        logP_alchemical_correction = -self.beta * alchemical_potential_correction

        return logP_alchemical_correction

    def make_alchemical_system(self, topology_proposal, direction='insert'):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal

        Arguments
        ---------
        topology_proposal : TopologyProposal namedtuple
            Contains old topology, proposed new topology, and atom mapping
        direction : str, optional, default='insert'
            Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])

        Returns
        -------
        unmodified_system : simtk.openmm.System
            Unmodified real system corresponding to appropriate leg of transformation.
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified

        """
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        atom_map = topology_proposal.new_to_old_atom_map

        #take the unique atoms as those not in the {new_atom : old_atom} atom map
        if direction == 'delete':
            unmodified_system = topology_proposal.old_system
            alchemical_atoms = [atom for atom in range(unmodified_system.getNumParticles()) if atom not in atom_map.values()]
        elif direction == 'insert':
            unmodified_system = topology_proposal.new_system
            alchemical_atoms = [atom for atom in range(unmodified_system.getNumParticles()) if atom not in atom_map.keys()]
        else:
            raise Exception("direction must be one of ['delete', 'insert']; found '%s' instead" % direction)

        # Create an alchemical factory.
        from alchemy import AbsoluteAlchemicalFactory
        alchemical_factory = AbsoluteAlchemicalFactory(unmodified_system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=True, alchemical_bonds=None, alchemical_angles=None, softcore_beta=0.0)

        # Return the alchemically-modified system in fully-interacting form.
        alchemical_system = alchemical_factory.createPerturbedSystem()
        return [unmodified_system, alchemical_system]

    def integrate(self, topology_proposal, initial_positions, direction='insert', platform=None, iteration=None):
        """
        Performs NCMC switching to either delete or insert atoms according to the provided `topology_proposal`.

        For `delete`, the system is first modified from fully interacting to alchemically modified, and then NCMC switching is used to eliminate atoms.
        For `insert`, the system begins with eliminated atoms in an alchemically noninteracting form and NCMC switching is used to turn atoms on, followed by making system real.
        The contribution of transforming the real system to/from an alchemical system is included.

        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        initial_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms at the beginning of the NCMC switching.
        direction : str, optional, default='insert'
            Direction of alchemical switching:
                'insert' causes lambda to switch from 0 to 1 over nsteps steps of integration
                'delete' causes lambda to switch from 1 to 0 over nsteps steps of integration
        platform : simtk.openmm.Platform, optional, default=None
            If not None, this platform is used for integration.
        iteration : int, optional, default=None
            Iteration number, for storage purposes.

        Returns
        -------
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
            The final positions after `nsteps` steps of alchemical switching
        logP : float
            The log acceptance probability of the switch
        potential : simtk.unit.Quantity with units compatible with kilocalories_per_mole
            For `delete`, the potential energy of the final (alchemically eliminated) conformation.
            For `insert`, the potential energy of the initial (alchemically eliminated) conformation.

        """
        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

        assert quantity_is_finite(initial_positions) == True

        # Select reference topology, indices, and system based on whether we are deleting or inserting.
        if direction == 'delete':
            topology = topology_proposal.old_topology
            indices = topology_proposal.unique_old_atoms
            system = topology_proposal.old_system
        elif direction == 'insert':
            topology = topology_proposal.new_topology
            indices = topology_proposal.unique_new_atoms
            system = topology_proposal.new_system

        # Handle special case of instantaneous insertion/deletion.
        if (self.nsteps == 0):
            # TODO: Check this is correct.
            # TODO: Can we simplify this so there are not two code branches here?
            logP = 0.0
            final_positions = copy.deepcopy(initial_positions)
            from perses.tests.utils import compute_potential
            potential = self.beta * compute_potential(system, initial_positions, platform=self.platform)
            return [final_positions, logP, potential]

        # Create alchemical system.
        [unmodified_system, alchemical_system] = self.make_alchemical_system(topology_proposal, direction=direction)

        # Select subset of switching functions based on which alchemical parameters are present in the system.
        available_parameters = self._getAvailableParameters(alchemical_system)
        functions = { parameter_name : self.functions[parameter_name] for parameter_name in self.functions if (parameter_name in available_parameters) }

        # Create an NCMC velocity Verlet integrator.
        if self.integrator_type == 'VV':
            integrator = NCMCVVAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, steps_per_propagation=self.steps_per_propagation, timestep=self.timestep, direction=direction)
        elif self.integrator_type == 'GHMC':
            integrator = NCMCGHMCAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, steps_per_propagation=self.steps_per_propagation, timestep=self.timestep, direction=direction)
        else:
            raise Exception("integrator_type '%s' unknown" % self.integrator_type)

        # Set the constraint tolerance if specified.
        if self.constraint_tolerance is not None:
            integrator.setConstraintTolerance(self.constraint_tolerance)

        # Create a context on the specified platform.
        if self.platform is not None:
            context = openmm.Context(alchemical_system, integrator, self.platform)
        else:
            context = openmm.Context(alchemical_system, integrator)
        context.setPositions(initial_positions)
        context.applyConstraints(integrator.getConstraintTolerance())

        # Set velocities to temperature and apply velocity constraints.
        context.setVelocitiesToTemperature(self.temperature)
        context.applyVelocityConstraints(integrator.getConstraintTolerance())

        # Integrate switching
        try:
            # Write atom indices that are changing.
            if self._storage:
                self._storage.write_object('atomindices', indices, iteration=iteration)

            # Allocate storage for work.
            work = np.zeros([self.nsteps+1], np.float64) # work[n] is the accumulated work up to step n

            # Write trajectory frame.
            if self._storage and self.write_ncmc_interval:
                positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                self._storage.write_configuration('positions', positions, topology, iteration=iteration, frame=0, nframes=(nsteps+1))

            # Perform NCMC integration.
            for step in range(self.nsteps):
                # Take a step.
                integrator.step(1)

                # Store accumulated work
                work[step+1] = integrator.getWork(context)

                # Write trajectory frame.
                if self._storage and self.write_ncmc_interval and (self.write_ncmc_interval % (step+1) == 0):
                    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                    assert quantity_is_finite(positions) == True
                    self._storage.write_configuration('positions', positions, topology, iteration=iteration, frame=(step+1), nframes=(nsteps+1))

            # Store work values.
            if self._storage:
                self._storage.write_array('work_%s' % direction, work, iteration=iteration)

        except Exception as e:
            # Trap NaNs as a special exception (allowing us to reject later, if desired)
            if str(e) == "Particle coordinate is nan":
                msg = "Particle coordinate is nan during NCMC integration while using integrator_type '%s'" % self.integrator_type
                if self.integrator_type == 'GHMC':
                    msg += '\n'
                    msg += 'This should NEVER HAPPEN with GHMC!'
                raise NaNException(msg)
            else:
                traceback.print_exc()
                raise e

        # Store final positions and log acceptance probability.
        final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        assert quantity_is_finite(final_positions) == True
        logP_NCMC = integrator.getLogAcceptanceProbability(context)

        # Get initial and final real and alchemical potentials
        from perses.tests.utils import compute_potential
        initial_alchemical_potential = self.beta * integrator.getGlobalVariableByName("Einitial") * unit.kilojoules_per_mole
        final_alchemical_potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        if direction == 'insert':
            final_unmodified_potential = self.beta * compute_potential(system, final_positions, platform=self.platform)
            logP_alchemical_correction = final_unmodified_potential - final_alchemical_potential
            switch_logp = initial_alchemical_potential
        elif direction == 'delete':
            initial_unmodified_potential = self.beta * compute_potential(system, initial_positions, platform=self.platform)
            logP_alchemical_correction = initial_alchemical_potential - initial_unmodified_potential
            switch_logp = final_alchemical_potential

        # Clean up.
        del context, integrator

        # Check potentials are finite
        if np.isnan(initial_alchemical_potential) or np.isnan(final_alchemical_potential):
            msg = "A required potential of %s operation is NaN:\n" % direction
            msg += "initial_alchemical_potential: %.3f kT\n" % initial_alchemical_potential
            msg += "final_alchemical_potential: %.3f kT\n" % final_alchemical_potential
            raise NaNException(msg)

        # Compute total logP
        logP = logP_NCMC + logP_alchemical_correction

        # Clean up alchemical system.
        del alchemical_system

        # Keep track of statistics.
        self.nattempted += 1

        # Return
        return [final_positions, logP, switch_logp]

class NCMCHybridEngine(NCMCEngine):
    """
    NCMC switching engine which switches directly from old to new systems
    via a hybrid alchemical topology

    Examples
    --------
    ## EXAMPLE UNCHANGED FROM BASE CLASS ##
    Create a transformation for an alanine dipeptide test system where the N-methyl group is eliminated.
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from perses.rjmc.topology_proposal import TopologyProposal
    >>> new_to_old_atom_map = { index : index for index in range(testsystem.system.getNumParticles()) if (index > 3) } # all atoms but N-methyl
    >>> topology_proposal = TopologyProposal(old_system=testsystem.system, old_topology=testsystem.topology, old_chemical_state_key='AA', new_chemical_state_key='AA', new_system=testsystem.system, new_topology=testsystem.topology, logp_proposal=0.0, new_to_old_atom_map=new_to_old_atom_map, metadata=dict())
    >>> ncmc_engine = NCMCHybridEngine(temperature=300.0*unit.kelvin, functions=default_functions, nsteps=50, timestep=1.0*unit.femtoseconds)

    positions = testsystem.positions
    (need a geometry proposal in here now)
    [positions, new_old_positions, logP_insert, potential_insert] = ncmc_engine.integrate(topology_proposal, positions, proposed_positions)
    """

    def __init__(self, temperature=default_temperature, functions=None, 
                 nsteps=default_nsteps, timestep=default_timestep, 
                 constraint_tolerance=None, platform=None, 
                 write_ncmc_interval=None, integrator_type='GHMC'):
        """
        Subclass of NCMCEngine which switches directly between two different
        systems using an alchemical hybrid topology.

        Arguments
        ---------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature at which switching is to be run
        functions : dict of str:str, optional, default=default_functions
            functions[parameter] is the function (parameterized by 't' which
            switched from 0 to 1) that controls how alchemical context
            parameter 'parameter' is switched
        nsteps : int, optional, default=1
            The number of steps to use for switching.
        timestep : simtk.unit.Quantity with units compatible with femtoseconds,
            optional, default=1*femtosecond
            The timestep to use for integration of switching velocity
            Verlet steps.
        constraint_tolerance : float, optional, default=None
            If not None, this relative constraint tolerance is used for
            position and velocity constraints.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, the platform to use for OpenMM simulations.
        write_ncmc_interval : int, optional, default=None
            If a positive integer is specified, a PDB frame will be written
            with the specified interval on NCMC switching, with a different
            PDB file generated for each attempt.
        integrator_type : str, optional, default='GHMC'
            NCMC internal integrator type ['GHMC', 'VV']
        """
        if functions is None:
            functions = default_hybrid_functions

        super(NCMCHybridEngine, self).__init__(temperature=temperature, functions=functions, nsteps=nsteps,
                                               timestep=timestep, constraint_tolerance=constraint_tolerance,
                                               platform=platform, write_ncmc_interval=write_ncmc_interval,
                                               integrator_type=integrator_type)

    def compute_logP(self, system, positions, parameter=None):
        """
        Compute potential energy of the specified system object at the specified positions.

        Constraints are applied before the energy is computed.

        Parameters
        ----------
        system : simtk.openmm.System
            The System object for which the potential energy is to be computed.
        positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms for which energy is to be computed.

        Returns
        -------
        potential : simtk.unit.Quantity with units of energy
            The computed potential energy
            
        """
        # Create dummy integrator.
        integrator = openmm.VerletIntegrator(self.timestep)
        # Set the constraint tolerance if specified.
        if self.constraint_tolerance is not None:
            integrator.setConstraintTolerance(self.constraint_tolerance)
        # Create a context on the specified platform.
        if self.platform is not None:
            context = openmm.Context(system, integrator, self.platform)
        else:
            context = openmm.Context(system, integrator)
        context.setPositions(positions)
        context.applyConstraints(integrator.getConstraintTolerance())
        if parameter is not None:
            available_parameters = self._getAvailableParameters(system)
            for parameter_name in available_parameters:
                context.setParameter(parameter_name, parameter)
        # Compute potential energy.
        potential = context.getState(getEnergy=True).getPotentialEnergy()
        # Clean up context and integrator.
        del context, integrator
        # Return potential energy.
        return -self.beta * potential

    def _computeAlchemicalCorrection(self, unmodified_old_system,
                                     unmodified_new_system, alchemical_system,
                                     initial_positions, alchemical_positions,
                                     final_hybrid_positions, final_positions,
                                     direction='insert'):
        """
        Compute log probability for correction from transforming real system
        to AND from alchemical system.

        Parameters
        ----------
        unmodified_old_system : simtk.unit.System
            Real fully-interacting system.
        unmodified_new_system : simtk.unit.System
            Real fully-interacting system.
        alchemical_system : simtk.unit.System
            Alchemically modified system in fully-interacting form.
        initial_positions : simtk.unit.Quantity of dimensions [nparticles,3]
            with units compatible with angstroms
            The initial positions before NCMC switching.
        alchemical_positions : simtk.unit.Quantity of dimensions [nparticles,3]
            with units compatible with angstroms
            The initial positions of hybrid topology before NCMC switching.
        final_hybrid_positions : simtk.unit.Quantity of dimensions
            [nparticles,3] with units compatible with angstroms
            The final positions of hybrid topology after NCMC switching.
        final_positions : simtk.unit.Quantity of dimensions [nparticles,3]
            with units compatible with angstroms
            The final positions after NCMC switching.
        direction : str, optional, default='insert'
            Not used in calculation
        Returns
        -------
        logP_alchemical_correction : float
            The log acceptance probability of the switch
        """

        # Compute correction from transforming real system to/from alchemical system
        initial_logP_correction = self.compute_logP(alchemical_system, alchemical_positions, parameter=0) - self.compute_logP(unmodified_old_system, initial_positions)
        final_logP_correction = self.compute_logP(unmodified_new_system, final_positions) - self.compute_logP(alchemical_system, final_hybrid_positions, parameter=1)
        logP_alchemical_correction = initial_logP_correction + final_logP_correction
        return logP_alchemical_correction

    def _compute_switch_logP(self, unmodified_old_system, unmodified_new_system, initial_positions, final_positions):
        return self.compute_logP(unmodified_new_system, final_positions) - self.compute_logP(unmodified_old_system, initial_positions)

    def make_alchemical_system(self, topology_proposal, old_positions,
                               new_positions):
        """
        Generate an alchemically-modified system at the correct atoms
        based on the topology proposal
        Arguments
        ---------
        topology_proposal : TopologyProposal namedtuple
            Contains old topology, proposed new topology, and atom mapping
        Returns
        -------
        unmodified_system : simtk.openmm.System
            Unmodified real system corresponding to appropriate leg of
            transformation.
        alchemical_system : simtk.openmm.System
            The system with appropriate atoms alchemically modified
        """

        atom_map = topology_proposal.old_to_new_atom_map

        #take the unique atoms as those not in the {new_atom : old_atom} atom map
        unmodified_old_system = copy.deepcopy(topology_proposal.old_system)
        unmodified_new_system = copy.deepcopy(topology_proposal.new_system)
        old_topology = topology_proposal.old_topology
        new_topology = topology_proposal.new_topology

        # Create an alchemical factory.
        from perses.annihilation.relative import HybridTopologyFactory
        alchemical_factory = HybridTopologyFactory(unmodified_old_system,
                                                   unmodified_new_system,
                                                   old_topology, new_topology,
                                                   old_positions,
                                                   new_positions, atom_map)

        # Return the alchemically-modified system in fully-interacting form.
#        alchemical_system, _, alchemical_positions, final_atom_map, initial_atom_map = alchemical_factory.createPerturbedSystem()
        alchemical_system, alchemical_topology, alchemical_positions, final_atom_map, initial_atom_map = alchemical_factory.createPerturbedSystem()
        return [unmodified_old_system, unmodified_new_system,
                alchemical_system, alchemical_topology, alchemical_positions, final_atom_map,
                initial_atom_map]

    def _convert_hybrid_positions_to_final(self, positions, atom_map):
        final_positions = unit.Quantity(np.zeros([len(atom_map.keys()),3]), unit=unit.nanometers)
        for finalatom, hybridatom in atom_map.items():
            final_positions[finalatom] = positions[hybridatom]
        return final_positions

    def integrate(self, topology_proposal, initial_positions, proposed_positions, platform=None):
        """
        Performs NCMC switching to either delete or insert atoms according to the provided `topology_proposal`.
        The contribution of transforming the real system to/from an alchemical system is included.
        Parameters
        ----------
        topology_proposal : TopologyProposal
            Contains old/new Topology and System objects and atom mappings.
        initial_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms at the beginning of the NCMC switching.
        proposed_positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the new system atoms proposed by geometry engine.
        platform : simtk.openmm.Platform, optional, default=None
            If not None, this platform is used for integration.
        Returns
        -------
        final_positions : simtk.unit.Quantity of dimensions [natoms, 3] with units of distance
            The final positions after `nsteps` steps of alchemical switching
        new_old_positions : simtk.unit.Quantity of dimensions [natoms, 3] with units of distance.
            The final positions of the atoms of the old system after `nsteps`
            steps of alchemical switching
        logP : float
            The log acceptance probability of the switch
        potential : simtk.unit.Quantity with units compatible with kilocalories_per_mole
            For `delete`, the potential energy of the final (alchemically eliminated) conformation.
            For `insert`, the potential energy of the initial (alchemically eliminated) conformation.
        """
        direction = 'insert'
        if (self.nsteps == 0):
            # Special case of instantaneous insertion/deletion.
            logP = 0.0
            final_positions = copy.deepcopy(proposed_positions)
            from perses.tests.utils import compute_potential
            potential_del = self.beta * compute_potential(topology_proposal.old_system, initial_positions, platform=self.platform)
            potential_ins = self.beta * compute_potential(topology_proposal.new_system, proposed_positions, platform=self.platform)
            potential = potential_del - potential_ins
            return [final_positions, logP, potential]

########################################################################
        # Create alchemical system.
        [unmodified_old_system,
         unmodified_new_system,
         alchemical_system,
         alchemical_topology,
         alchemical_positions,
         final_to_hybrid_atom_map,
         initial_to_hybrid_atom_map] = self.make_alchemical_system(
                                            topology_proposal, initial_positions,
                                            proposed_positions)
########################################################################

        # Select subset of switching functions based on which alchemical parameters are present in the system.
        available_parameters = self._getAvailableParameters(alchemical_system)
        functions = { parameter_name : self.functions[parameter_name] for parameter_name in self.functions if (parameter_name in available_parameters) }

        # Create an NCMC velocity Verlet integrator.
        if self.integrator_type == 'VV':
            integrator = NCMCVVAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, timestep=self.timestep, direction='insert')
        elif self.integrator_type == 'GHMC':
            integrator = NCMCGHMCAlchemicalIntegrator(self.temperature, alchemical_system, functions, nsteps=self.nsteps, timestep=self.timestep, direction='insert')
        else:
            raise Exception("integrator_type '%s' unknown" % self.integrator_type)
        # Set the constraint tolerance if specified.
        if self.constraint_tolerance is not None:
            integrator.setConstraintTolerance(self.constraint_tolerance)
        # Create a context on the specified platform.
        if self.platform is not None:
            context = openmm.Context(alchemical_system, integrator, self.platform)
        else:
            context = openmm.Context(alchemical_system, integrator)
        context.setPositions(alchemical_positions)
        context.applyConstraints(integrator.getConstraintTolerance())
        # Set velocities to temperature and apply velocity constraints.
        context.setVelocitiesToTemperature(self.temperature)
        context.applyVelocityConstraints(integrator.getConstraintTolerance())

        # Set initial context parameters.
        integrator.setGlobalVariableByName('lambda', 0)

        # Compute initial potential of alchemical state.
        initial_logP = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        if np.isnan(initial_logP):
            raise NaNException("Initial potential of 'insert' operation is NaN")
        from perses.tests.utils import compute_potential_components
        # Take a single integrator step since all switching steps are unrolled in NCMCVVAlchemicalIntegrator.
        try:
            # Write PDB file if requested.
            if self.write_ncmc_interval is not None:

                from simtk.openmm.app import PDBFile
                filename = 'ncmc-%s-%d.pdb' % (direction, self.nattempted)
                outfile = open(filename, 'w')
                PDBFile.writeHeader(alchemical_topology, file=outfile)
                modelIndex = 0
                PDBFile.writeModel(alchemical_topology, context.getState(getPositions=True).getPositions(asNumpy=True), file=outfile, modelIndex=modelIndex)
                try:
                    for step in range(self.nsteps):
                        integrator.step(1)
                        if (step+1)%self.write_ncmc_interval == 0:
                            modelIndex += 1
                            PDBFile.writeModel(alchemical_topology, context.getState(getPositions=True).getPositions(asNumpy=True), file=outfile, modelIndex=modelIndex)
                except ValueError as e:
                    # System is exploding and coordinates won't fit in PDB ATOM fields
                    print(e)

                PDBFile.writeFooter(alchemical_topology, file=outfile)
                outfile.close()
            else:
                for step in range(self.nsteps):
                    integrator.step(1)
                    potential = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
                    current_step = integrator.get_step()

        except Exception as e:
            # Trap NaNs as a special exception (allowing us to reject later, if desired)
            if str(e) == "Particle coordinate is nan":
                raise NaNException(str(e))
            else:
                raise e

        # Set final context parameters.
        integrator.setGlobalVariableByName('lambda', 1)

        # Compute final potential of alchemical state.
        final_logP = self.beta * context.getState(getEnergy=True).getPotentialEnergy()
        if np.isnan(final_logP):
            raise NaNException("Final potential of hybrid switch operation is NaN")

        # Store final positions and log acceptance probability.
        final_hybrid_positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        final_positions = self._convert_hybrid_positions_to_final(final_hybrid_positions, final_to_hybrid_atom_map)
        new_old_positions = self._convert_hybrid_positions_to_final(final_hybrid_positions, initial_to_hybrid_atom_map)

        logP_NCMC = integrator.getLogAcceptanceProbability(context)
        # Clean up NCMC switching integrator.
        del context, integrator

        # Compute contribution from transforming real system to/from alchemical system.
        logP_alchemical_correction = self._computeAlchemicalCorrection(
                                              unmodified_old_system,
                                              unmodified_new_system,
                                              alchemical_system,
                                              initial_positions,
                                              alchemical_positions,
                                              final_hybrid_positions,
                                              final_positions,
                                          )

        # Compute total logP
        logP_ncmc = logP_NCMC + logP_alchemical_correction

        # Clean up alchemical system.
        del alchemical_system

        # Keep track of statistics.
        self.nattempted += 1

        # Return
        return [final_positions, new_old_positions, logP_ncmc]


class NCMCAlchemicalIntegrator(openmm.CustomIntegrator):
    """
    Helper base class for NCMC alchemical integrators.
    """
    def __init__(self, temperature, system, functions, nsteps, steps_per_propagation, timestep, direction):
        """
        Initialize base class for NCMC alchemical integrators.

        Parameters
        ----------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature to use for computing the NCMC acceptance probability.
        system : simtk.openmm.System
            The system to be simulated.
        functions : dict of str : str
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int
            The number of switching timesteps per call to integrator.step(1).
        steps_per_propagation : int
            The number of propagation steps taken at each value of lambda
        timestep : simtk.unit.Quantity with units compatible with femtoseconds
            The timestep to use for each NCMC step.
        direction : str, optional, default='insert'
            One of ['insert', 'delete'].
            For `insert`, the parameter 'lambda' is switched from 0 to 1.
            For `delete`, the parameter 'lambda' is switched from 1 to 0.

        """
        super(NCMCAlchemicalIntegrator, self).__init__(timestep)

        if direction not in ['insert', 'delete']:
            raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)
        self.direction = direction

        # Compute kT in natural openmm units.
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        self.kT = kT

        self.has_statistics = False # no GHMC statistics by default

        self.nsteps = nsteps

        # Make a list of parameters in the system
        self.system_parameters = list()
        self.alchemical_functions = functions
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if hasattr(force, 'getNumGlobalParameters'):
                for parameter_index in range(force.getNumGlobalParameters()):
                    self.system_parameters.append(force.getGlobalParameterName(parameter_index))

    def addAlchemicalResetStep(self):
        """
        Reset alchemical state to initial state.
        """
        # Set the master 'lambda' alchemical parameter to the initial state
        if self.direction == 'insert':
            self.addComputeGlobal('lambda', '0.0')
        elif self.direction == 'delete':
            self.addComputeGlobal('lambda', '1.0')

        # Update all slaved alchemical parameters
        self.addUpdateAlchemicalParametersStep()

    def addAlchemicalPerturbationStep(self):
        """
        Add alchemical perturbation step.
        """
        # Set the master 'lambda' alchemical parameter to the current fractional state
        if self.nsteps == 0:
            # Toggle alchemical state
            if self.direction == 'insert':
                self.addComputeGlobal('lambda', '1.0')
            elif self.direction == 'delete':
                self.addComputeGlobal('lambda', '0.0')
        else:
            # Use fractional state
            if self.direction == 'insert':
                self.addComputeGlobal('lambda', '(step+1)/nsteps')
            elif self.direction == 'delete':
                self.addComputeGlobal('lambda', '(nsteps - step - 1)/nsteps')

        # Update all slaved alchemical parameters
        self.addUpdateAlchemicalParametersStep()

    def addUpdateAlchemicalParametersStep(self):
        """
        Update Context parameters according to provided functions.
        """
        for context_parameter in self.alchemical_functions:
            if context_parameter in self.system_parameters:
                self.addComputeGlobal(context_parameter, self.alchemical_functions[context_parameter])

    def addVelocityVerletStep(self):
        """
        Add velocity Verlet step.
        """
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

    def addGHMCStep(self):
        """
        Add a GHMC step.
        """
        self.hasStatistics = True

        # TODO: This could be precomputed to save time
        self.addComputePerDof("sigma", "sqrt(kT/m)")

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

        #
        # Metropolized symplectic step.
        #
        self.addConstrainPositions()
        self.addConstrainVelocities()

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold_GHMC", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + v*dt")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        self.addConstrainVelocities()
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew_GHMC", "ke + energy")
        # Compute acceptance probability
        # DEBUG: Check positions are finite
        self.addComputeGlobal("accept", "step(exp(-(Enew_GHMC-Eold_GHMC)/kT) - uniform)")
        self.beginIfBlock("accept != 1")
        # Reject sample, inverting velcoity
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.endBlock()

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

        #
        # Accumulate statistics.
        #
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

    def get_step(self):
        return self.getGlobalVariableByName("step")

    def reset(self):
        """
        Reset step counter and total work
        """
        self.setGlobalVariableByName("total_work", 0.0)
        if (self.nsteps > 0):
            self.setGlobalVariableByName("step", 0)
            self.setGlobalVariableByName("pstep", 0)
            if self.has_statistics:
                self.setGlobalVariableByName("naccept", 0)
                self.setGlobalVariableByName("ntrials", 0)

    def getStatistics(self, context):
        if (self.has_statistics):
            return (self.getGlobalVariableByName("naccept"), self.getGlobalVariableByName("ntrials"))
        else:
            return (0,0)

    def getWork(self, context):
        """Retrieve accumulated work (in units of kT)
        """
        return self.getGlobalVariableByName("total_work") * unit.kilojoules_per_mole / self.kT

    def getLogAcceptanceProbability(self, context):
        logp_accept = -1.0*self.getGlobalVariableByName("total_work") * unit.kilojoules_per_mole / self.kT
        return logp_accept

class NCMCVVAlchemicalIntegrator(NCMCAlchemicalIntegrator):
    """
    Use NCMC switching to annihilate or introduce particles alchemically.

    TODO:
    ----
    * We may need to avoid unrolling integration steps.

    Examples
    --------

    Annihilate a Lennard-Jones particle

    >>> # Create an alchemically-perturbed test system
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.LennardJonesCluster()
    >>> from alchemy import AbsoluteAlchemicalFactory
    >>> alchemical_atoms = [0]
    >>> factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms)
    >>> alchemical_system = factory.createPerturbedSystem()
    >>> # Create an NCMC switching integrator.
    >>> temperature = 300.0 * unit.kelvin
    >>> nsteps = 5
    >>> functions = { 'lambda_sterics' : 'lambda' }
    >>> ncmc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nsteps, direction='delete')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Run the integrator
    >>> ncmc_integrator.step(nsteps)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.getLogAcceptanceProbability(context)

    Turn on an atom and its associated angles and torsions in alanine dipeptide

    >>> # Create an alchemically-perturbed test system
    >>> from openmmtools import testsystems
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> from alchemy import AbsoluteAlchemicalFactory
    >>> alchemical_atoms = [0,1,2,3] # terminal methyl group
    >>> factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)
    >>> alchemical_system = factory.createPerturbedSystem()
    >>> # Create an NCMC switching integrator.
    >>> temperature = 300.0 * unit.kelvin
    >>> nsteps = 10
    >>> functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : 'lambda^2' }
    >>> ncmc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nsteps, direction='delete')
    >>> # Create a Context
    >>> context = openmm.Context(alchemical_system, ncmc_integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Minimize
    >>> openmm.LocalEnergyMinimizer.minimize(context)
    >>> # Run the integrator
    >>> ncmc_integrator.step(nsteps)
    >>> # Retrieve the log acceptance probability
    >>> log_ncmc = ncmc_integrator.getLogAcceptanceProbability(context)

    """

    def __init__(self, temperature, system, functions, nsteps=0, steps_per_propagation=1, timestep=1.0*unit.femtoseconds, direction='insert'):
        """
        Initialize an NCMC switching integrator to annihilate or introduce particles alchemically.

        Parameters
        ----------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature to use for computing the NCMC acceptance probability.
        system : simtk.openmm.System
            The system to be simulated.
        functions : dict of str : str
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int, optional, default=10
            The number of switching timesteps per call to integrator.step(1).
        steps_per_propagation : int, optional, default=1
            The number of propagation steps taken at each value of lambda
        timestep : simtk.unit.Quantity with units compatible with femtoseconds
            The timestep to use for each NCMC step.
        direction : str, optional, default='insert'
            One of ['insert', 'delete'].
            For `insert`, the parameter 'lambda' is switched from 0 to 1.
            For `delete`, the parameter 'lambda' is switched from 1 to 0.

        Note that each call to integrator.step(1) executes the entire integration program; this should not be called with more than one step.

        A symmetric protocol is used, in which the protocol begins and ends with a velocity Verlet step.

        TODO:
        * Add a global variable that causes termination of future calls to step(1) after the first

        """
        super(NCMCVVAlchemicalIntegrator, self).__init__(temperature, system, functions, nsteps, steps_per_propagation, timestep, direction)

        #
        # Initialize global variables
        #

        # NCMC variables
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addGlobalVariable('total_work', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable('kinetic', 0.0) # kinetic energy
        self.addGlobalVariable("Einitial", 0) # initial energy after setting initial alchemical state

        # VV variables
        if (nsteps > 0):
            # VV variables
            self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
            self.addGlobalVariable('step', 0) # current NCMC step number
            self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints
            self.addGlobalVariable('psteps', steps_per_propagation)
            self.addGlobalVariable('pstep', 0)

        # Constrain initial positions and velocities.
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.addUpdateContextState()

        if nsteps == 0:
            self.addAlchemicalResetStep()
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
        if nsteps > 0:
            self.addComputeGlobal('pstep', '0')
            # Initial step only
            self.beginIfBlock('step = 0')
            self.addAlchemicalResetStep()
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Eold", "energy + kinetic")
            self.beginWhileBlock('pstep < psteps')
            self.addVelocityVerletStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Enew", "energy + kinetic")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
            self.endBlock()

            # All steps
            self.beginIfBlock('step < nsteps')
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Eold", "energy + kinetic")
            self.addAlchemicalPerturbationStep()
            self.beginWhileBlock('pstep < psteps')
            self.addVelocityVerletStep()
            self.addComputeGlobal('pstep', 'pstep+1')
            self.endBlock()
            self.addComputeSum('kinetic', '0.5 * m * v^2')
            self.addComputeGlobal("Enew", "energy + kinetic")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
            self.addComputeGlobal('step', 'step+1')
            self.endBlock()

class NCMCGHMCAlchemicalIntegrator(NCMCAlchemicalIntegrator):
    """
    Use NCMC switching to annihilate or introduce particles alchemically.
    """

    def __init__(self, temperature, system, functions, nsteps=0, steps_per_propagation=1, collision_rate=9.1/unit.picoseconds, timestep=1.0*unit.femtoseconds, direction='insert'):
        """
        Initialize an NCMC switching integrator to annihilate or introduce particles alchemically.

        Parameters
        ----------
        temperature : simtk.unit.Quantity with units compatible with kelvin
            The temperature to use for computing the NCMC acceptance probability.
        system : simtk.openmm.System
            The system to be simulated.
        functions : dict of str : str
            functions[parameter] is the function (parameterized by 't' which switched from 0 to 1) that
            controls how alchemical context parameter 'parameter' is switched
        nsteps : int, optional, default=0
            The number of switching timesteps per call to integrator.step(1).
        steps_per_propagation : int, optional, default=1
            The number of propagation steps taken at each value of lambda
        timestep : simtk.unit.Quantity with units compatible with femtoseconds
            The timestep to use for each NCMC step.
        direction : str, optional, default='insert'
            One of ['insert', 'delete'].
            For `insert`, the parameter 'lambda' is switched from 0 to 1.
            For `delete`, the parameter 'lambda' is switched from 1 to 0.

        Note that each call to integrator.step(1) executes the entire integration program; this should not be called with more than one step.

        A symmetric protocol is used, in which the protocol begins and ends with a velocity Verlet step.

        TODO:
        * Add a global variable that causes termination of future calls to step(1) after the first

        """
        super(NCMCGHMCAlchemicalIntegrator, self).__init__(temperature, system, functions, nsteps, steps_per_propagation, timestep, direction)

        gamma = collision_rate

        # NCMC variables
        self.addGlobalVariable('lambda', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalVariable('total_work', 0.0) # initial total energy (kinetic + potential)
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("Einitial", 0) # initial energy after setting initial alchemical state

        if (nsteps > 0):
            # GHMC variables
            self.addGlobalVariable("Eold_GHMC", 0)  # old GHMC energy
            self.addGlobalVariable("Enew_GHMC", 0)  # new GHMC energy
            self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
            self.addGlobalVariable('step', 0) # current NCMC step number
            self.addPerDofVariable("x1", 0) # for velocity Verlet with constraints
            self.addGlobalVariable("kT", self.kT.value_in_unit_system(unit.md_unit_system))  # thermal energy
            self.addGlobalVariable("b", np.exp(-gamma * timestep))  # velocity mixing parameter
            self.addPerDofVariable("sigma", 0)
            self.addGlobalVariable("ke", 0)  # kinetic energy
            self.addPerDofVariable("vold", 0)  # old velocities
            self.addPerDofVariable("xold", 0)  # old positions
            self.addGlobalVariable("accept", 0)  # accept or reject
            self.addGlobalVariable("naccept", 0)  # number accepted
            self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials
            self.addGlobalVariable("pstep", 0) # number of propagation steps taken
            self.addGlobalVariable("psteps", steps_per_propagation) # total number of propagation steps

        # Constrain initial positions and velocities.
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.addUpdateContextState()

        if nsteps == 0:
            self.addAlchemicalResetStep()
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
        if nsteps > 0:
            #self.addComputeGlobal('pstep', '0')

            # Initial step only
            self.beginIfBlock('step = 0')
            #self.beginWhileBlock('pstep < psteps')
            self.addAlchemicalResetStep()
            self.addComputeGlobal('Einitial', 'energy') # store initial energy after setting initial alchemical state
            #self.addGHMCStep()
            #self.addComputeGlobal('pstep', 'pstep+1')
            #self.endBlock()
            self.endBlock()

            # All steps
            self.beginIfBlock('step < nsteps')
            self.addComputeGlobal("Eold", "energy")
            self.addAlchemicalPerturbationStep()
            self.addComputeGlobal("Enew", "energy")
            self.addComputeGlobal("total_work", "total_work + (Enew-Eold)")
            #self.beginWhileBlock('pstep < psteps')
            self.addGHMCStep()
            #self.addComputeGlobal('pstep', 'pstep+1')
            #self.endBlock()
            self.addComputeGlobal('step', 'step+1')
            self.endBlock()


