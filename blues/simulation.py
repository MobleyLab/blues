"""
simulation.py: Provides the Simulation class object that runs the BLUES engine

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""
import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import parmed, math
from openmmtools import alchemy
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues import utils
import logging

logger = logging.getLogger(__name__)

class SystemFactory(object):
    """
    SystemFactory contains methods to generate/modify the OpenMM System object required for
    generating the openmm.Simulation using a given parmed.Structure()

    Example
    --------
    #Generate the reference OpenMM system.
    systems = SystemFactory(structure, ligand.atom_indices, **opt['system'])

    #The MD and alchemical Systems are generated and stored as an attribute
    systems.md
    systems.alch

    #Freeze atoms in the alchemical system
    systems.alch = SystemFactory.freeze_atoms(systems.alch,
                                            freeze_distance=5.0,
                                            freeze_center='LIG'
                                            freeze_solvent='HOH,NA,CL')

    Parameters
    ----------
    structure : parmed.Structure
        A chemical structure composed of atoms, bonds, angles, torsions, and
        other topological features.
    atom_indices : list of int
        Atom indicies of the move or designated for which the nonbonded forces
        (both sterics and electrostatics components) have to be alchemically
        modified.
    """
    def __init__(self, structure, atom_indices, **opt):
        self.structure = structure
        self.atom_indices = atom_indices
        self.opt = opt

        self.alch_opt = self.opt.pop('alchemical')

        self.md = self.generateSystem(self.structure, **self.opt)
        self.alch = self.generateAlchSystem(self.md, self.atom_indices, **self.alch_opt)

    def generateSystem(self, structure, **kwargs):
        """
        Construct an OpenMM System representing the topology described by the
        prmtop file. This function is just a wrapper for parmed Structure.createSystem().

        Parameters
        ----------
        structure : parmed.Structure()
            The parmed.Structure of the molecular system to be simulated

        Kwargs
        -------
        nonbondedMethod : cutoff method
            This is the cutoff method. It can be either the NoCutoff,
            CutoffNonPeriodic, CutoffPeriodic, PME, or Ewald objects from the
            simtk.openmm.app namespace
        nonbondedCutoff : float or distance Quantity
            The nonbonded cutoff must be either a floating point number
            (interpreted as nanometers) or a Quantity with attached units. This
            is ignored if nonbondedMethod is NoCutoff.
        switchDistance : float or distance Quantity
            The distance at which the switching function is turned on for van
            der Waals interactions. This is ignored when no cutoff is used, and
            no switch is used if switchDistance is 0, negative, or greater than
            the cutoff
        constraints : None, app.HBonds, app.HAngles, or app.AllBonds
            Which type of constraints to add to the system (e.g., SHAKE). None
            means no bonds are constrained. HBonds means bonds with hydrogen are
            constrained
        rigidWater : bool=True
            If True, water is kept rigid regardless of the value of constraints.
            A value of False is ignored if constraints is not None.
        implicitSolvent : None, app.HCT, app.OBC1, app.OBC2, app.GBn, app.GBn2
            The Generalized Born implicit solvent model to use.
        implicitSolventKappa : float or 1/distance Quantity = None
            This is the Debye kappa property related to modeling saltwater
            conditions in GB. It should have units of 1/distance (1/nanometers
            is assumed if no units present). A value of None means that kappa
            will be calculated from implicitSolventSaltConc (below)
        implicitSolventSaltConc : float or amount/volume Quantity=0 moles/liter
            If implicitSolventKappa is None, the kappa will be computed from the
            salt concentration. It should have units compatible with mol/L
        temperature : float or temperature Quantity = 298.15 kelvin
            This is only used to compute kappa from implicitSolventSaltConc
        soluteDielectric : float=1.0
            The dielectric constant of the protein interior used in GB
        solventDielectric : float=78.5
            The dielectric constant of the water used in GB
        useSASA : bool=False
            If True, use the ACE non-polar solvation model. Otherwise, use no
            SASA-based nonpolar solvation model.
        removeCMMotion : bool=True
            If True, the center-of-mass motion will be removed periodically
            during the simulation. If False, it will not.
        hydrogenMass : float or mass quantity = None
            If not None, hydrogen masses will be changed to this mass and the
            difference subtracted from the attached heavy atom (hydrogen mass
            repartitioning)
        ewaldErrorTolerance : float=0.0005
            When using PME or Ewald, the Ewald parameters will be calculated
            from this value
        flexibleConstraints : bool=True
            If False, the energies and forces from the constrained degrees of
            freedom will NOT be computed. If True, they will (but those degrees
            of freedom will *still* be constrained).
        verbose : bool=False
            If True, the progress of this subroutine will be printed to stdout
        splitDihedrals : bool=False
            If True, the dihedrals will be split into two forces -- proper and
            impropers. This is primarily useful for debugging torsion parameter
            assignments.

        Notes
        -----
        This function calls prune_empty_terms if any Topology lists have changed
        """
        return structure.createSystem(**kwargs)

    def generateAlchSystem(self, system, atom_indices,
                            softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6,
                            softcore_beta=0.0, softcore_d=1, softcore_e=1, softcore_f=2,
                            annihilate_electrostatics=True, annihilate_sterics=False,
                            disable_alchemical_dispersion_correction=True,
                            **kwargs):
        """Returns the OpenMM System for alchemical perturbations.
        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list of int
            Atom indicies of the move or designated for which the nonbonded forces
            (both sterics and electrostatics components) have to be alchemically
            modified.

        Kwargs
        ------
        annihilate_electrostatics : bool, optional
            If True, electrostatics should be annihilated, rather than decoupled
            (default is True).
        annihilate_sterics : bool, optional
            If True, sterics (Lennard-Jones or Halgren potential) will be annihilated,
            rather than decoupled (default is False).
        softcore_alpha : float, optional
            Alchemical softcore parameter for Lennard-Jones (default is 0.5).
        softcore_a, softcore_b, softcore_c : float, optional
            Parameters modifying softcore Lennard-Jones form. Introduced in
            Eq. 13 of Ref. [1] (default is 1).
        softcore_beta : float, optional
            Alchemical softcore parameter for electrostatics. Set this to zero
            to recover standard electrostatic scaling (default is 0.0).
        softcore_d, softcore_e, softcore_f : float, optional
            Parameters modifying softcore electrostatics form (default is 1).

        References
        ----------
        [1] Pham TT and Shirts MR. Identifying low variance pathways for free
        energy calculations of molecular transformations in solution phase.
        JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597
        """
        #Lower logger level to suppress excess warnings
        logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)

        #Disabled correction term due to increased computational cost
        factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=disable_alchemical_dispersion_correction)
        alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices,
                                            softcore_alpha=softcore_alpha,
                                            softcore_a=softcore_a,
                                            softcore_b=softcore_b,
                                            softcore_c=softcore_c,
                                            softcore_beta=softcore_beta,
                                            softcore_d=softcore_d,
                                            softcore_e=softcore_e,
                                            softcore_f=softcore_f,
                                            annihilate_electrostatics=annihilate_electrostatics,
                                            annihilate_sterics=annihilate_sterics)

        alch_system = factory.create_alchemical_system(system, alch_region)
        return alch_system

    def freeze_atoms(self, system, structure=None, freeze_distance=5.0,
                    freeze_center='LIG', freeze_solvent='HOH,NA,CL', **kwargs):
        """
        Function to zero the masses of selected atoms and solvent. Massless atoms
        will be ignored by the integrator and will not change positions.
        Parameters
        ----------
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        system : openmm.System
            The OpenMM System object to be modified.

        Kwargs
        -------
        freeze_center : str
            AmberMask selection for the center in which to select atoms for zeroing their masses. Default: LIG
        freeze_distance : float
            Distance (angstroms) to select atoms for retaining their masses. Atoms outside the set distance will have their masses set to 0.0. Default: 5.0
        freeze_solvent : str
            AmberMask selection in which to select solvent atoms for zeroing their masses. Default: HOH,NA,CL

        References
        -----
        Amber mask syntax: http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax
        """
        if not structure: structure = self.structure

        #Atom selection for zeroing protein atom masses
        mask = parmed.amber.AmberMask(structure,"(:%s<:%f)&!(:%s)" % (freeze_center,freeze_distance._value,freeze_solvent))
        site_idx = [i for i in mask.Selected()]
        logger.info('Zeroing mass of %s atoms %.1f Angstroms from %s on %s' % (len(site_idx), freeze_distance._value, freeze_center, system))
        logger.debug('\nFreezing atom selection = %s' % site_idx)
        freeze_indices = set(range(system.getNumParticles())) - set(site_idx)
        return utils.zero_masses(system, freeze_indices)

class SimulationFactory(object):
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run.

    Example.
    ----
    from blues.ncmc import SimulationFactory
    simulations = SimulationFactory(structure, system, alch_system, ligand_mover, **opt['simulation'])


    Parameters
    ----------
    structure : parmed.Structure
        A chemical structure composed of atoms, bonds, angles, torsions, and
        other topological features.
    system : openmm.System
        The OpenMM System object corresponding to the reference system.
    alch_system : openmm.System
        The OpenMM System object corresponding to the system for alchemical perturbations.
    move_engine : blues.ncmc.MoveEngine object
        MoveProposal object which contains the dict of moves performed
        in the NCMC simulation.
    opt : dict of parameters for the simulation (i.e timestep, temperature, etc.)
    """
    def __init__(self, systems, move_engine, **opt):
        self.structure = systems.structure
        #Atom indicies from move_engine
        #TODO: change atom_indices selection for multiple regions
        self.atom_indices = move_engine.moves[0].atom_indices
        self.move_engine = move_engine
        self._system = systems.md
        self._alch_system = systems.alch
        self.opt = opt
        self.generateSimulationSet()

    def addBarostat(self, system, temperature=300, pressure=1, frequency=25, **kwargs):
        """
        Adds a MonteCarloBarostat to the MD system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.

        Kwargs
        ------
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        pressure : int, optional, default=None
            Pressure (atm) for Barostat for NPT simulations.
        frequency : int, default=25
            Frequency at which Monte Carlo pressure changes should be attempted (in time steps)
        """
        logger.info('Adding MonteCarloBarostat with %s. MD simulation will be NPT.' %(pressure))
        # Add Force Barostat to the system
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, frequency))
        return system

    def generateIntegrator(self, temperature=300, dt=0.002, friction=1, **kwargs):
        """
        Generates a LangevinIntegrator for the Simulations.

        Kwargs
        ----------
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        friction: float, default=1
            friction coefficient which couples to the heat bath, measured in 1/ps
        dt: int, optional, default=0.002
            The timestep of the integrator to use (in ps).
        """
        integrator = openmm.LangevinIntegrator(temperature, friction, dt)
        return integrator

    def generateNCMCIntegrator(self, alch_system, nstepsNC,
                               alchemical_functions={'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                               'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'},
                               splitting="H V R O R V H",
                               temperature=300,
                               dt=0.002,
                               nprop=1,
                               prop_lambda=0.3, **kwargs):
        """
        Generates the AlchemicalExternalLangevinIntegrator using openmmtools.

        Parameters
        -----------
        alch_system : openmm.System
            The OpenMM System object corresponding to the alchemical system.
        nstepsNC : int, optional, default=1000
            The number of NCMC relaxation steps to use.

        Kwargs
        ------
        alchemical_functions : dict of strings,
            key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
            string that depends on the variable "lambda"
            Default = {'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                      'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda)
                                                  + 1/0.2*(lambda-0.8)*step(lambda-0.8)'}
        splitting : string, default: "H V R O R V H"
            Sequence of R, V, O (and optionally V{i}), and { }substeps to be executed each timestep. There is also an H option,
            which increments the global parameter `lambda` by 1/nsteps_neq for each step.
            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            ( will cause metropolization, and must be followed later by a ).
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        dt: int, optional, default=0.002
            The timestep of the integrator to use (in ps).
        nprop : int (Default: 1)
            Controls the number of propagation steps to add in the lambda
            region defined by `prop_lambda`
        prop_lambda: float, optional, default=0.3
            The range which additional propogation steps are added,
            defined by [0.5-prop_lambda, 0.5+prop_lambda].
        """
        #During NCMC simulation, lambda parameters are controlled by function dict below
        # Keys correspond to parameter type (i.e 'lambda_sterics', 'lambda_electrostatics')
        # 'lambda' = step/totalsteps where step corresponds to current NCMC step,
        ncmc_integrator = AlchemicalExternalLangevinIntegrator(
                                alchemical_functions=alchemical_functions,
                               splitting=splitting,
                               temperature=temperature,
                               nsteps_neq=nstepsNC,
                               timestep=dt,
                               nprop=nprop,
                               prop_lambda=prop_lambda)
        return ncmc_integrator

    def generateSimFromStruct(self, structure, system, integrator, platform=None, properties={}, **kwargs):
        """Used to generate the OpenMM Simulation objects from a given parmed.Structure()

        Parameters
        ----------
        structure : parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        integrator : openmm.Integrator
            The OpenMM Integrator object for the simulation.
        platform : str, default = None
            Valid choices: 'Auto', 'OpenCL', 'CUDA'
            If None is specified, the fastest available platform will be used.
        """
        #Specifying platform properties here used for local development.
        if platform is None:
            #Use the fastest available platform
            simulation = app.Simulation(structure.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName(platform)
            #Make sure key/values are strings
            properties = { str(k) : str(v) for k,v in properties.items()}
            simulation = app.Simulation(structure.topology, system, integrator, platform, properties)

        # Set initial positions/velocities
        # Will get overwritten from saved State.
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(integrator.getTemperature())
        simulation.context.setPeriodicBoxVectors(*structure.box_vectors)

        return simulation

    def _simulation_info_(self, simulation):
        # Host information
        from platform import uname
        for k, v in uname()._asdict().items():
            logger.info('{} = {}'.format(k,v))

        # OpenMM platform information
        mmver = openmm.version.version
        mmplat = simulation.context.getPlatform()
        logger.info('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()))
        # Platform properties
        for prop in mmplat.getPropertyNames():
            val = mmplat.getPropertyValue(simulation.context, prop)
            logger.info('{} = {}'.format(prop,val))

    def generateSimulationSet(self):
        """Function used to generate the 3 OpenMM Simulation objects."""
        #Construct MD Integrator and Simulation
        self.integrator = self.generateIntegrator(**self.opt)
        if 'pressure' in self.opt.keys():
            self.system = self.addBarostat(self._system, **self.opt)
            logger.warning('NCMC simulation will NOT have pressure control. NCMC will use pressure from last MD state.')
        else:
            logger.info('MD simulation will be NVT.')
        self.md = self.generateSimFromStruct(self.structure, self._system, self.integrator, **self.opt)

        #Alchemical Simulation is used for computing correction term from MD simulation.
        alch_integrator = self.generateIntegrator(**self.opt)
        self.alch = self.generateSimFromStruct(self.structure, self._system, alch_integrator, **self.opt)

        #Construct NCMC Integrator and Simulation
        self.ncmc_integrator = self.generateNCMCIntegrator(self._alch_system, **self.opt)

        #Initialize the Move Engine with the Alchemical System and NCMC Integrator
        for move in self.move_engine.moves:
            self._alch_system, self.ncmc_integrator = move.initializeSystem(self._alch_system, self.ncmc_integrator)
        self.nc = self.generateSimFromStruct(self.structure, self._alch_system, self.ncmc_integrator, **self.opt)

        self._simulation_info_(self.nc)

class Simulation(object):
    """Simulation class provides the functions that perform the BLUES run.
    """
    def __init__(self, simulations):
        """Initialize the BLUES Simulation object.

        Parameters
        ----------
        simulations : blues.ncmc.SimulationFactory object
            SimulationFactory Object which carries the 3 required
            OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.


        Simulation options
        ------------------
        nIter: int, optional, default=100
            The number of MD + NCMC/MC iterations to perform.
        mc_per_iter: int, optional, default=1
            The number of MC moves to perform during each
            iteration of a MD + MC simulations.

        """
        self.simulations = simulations
        self.md_sim = simulations.md
        self.alch_sim = simulations.alch
        self.nc_sim = simulations.nc
        self.temperature = self.md_sim.integrator.getTemperature()
        self.accept = 0
        self.reject = 0
        self.accept_ratio = 0
        self.opt = simulations.opt

        self.movestep = int(self.opt['nstepsNC']) / 2

        self.current_iter = 0
        self.current_state = { 'md'   : { 'state0' : {}, 'state1' : {} },
                               'nc'   : { 'state0' : {}, 'state1' : {} },
                               'alch' : { 'state0' : {}, 'state1' : {} }
                            }

        #specify nc integrator variables to report in verbose output
        self.work_keys = [ 'lambda', 'shadow_work',
                          'protocol_work', 'Eold', 'Enew']

        self.state_keys = { 'getPositions' : True,
                       'getVelocities' : True,
                       'getForces' : False,
                       'getEnergy' : True,
                       'getParameters': True,
                       'enforcePeriodicBox' : True}


    def setSimState(self, simkey, stateidx, stateinfo):
        """Stores the dict of Positions, Velocities, Potential/Kinetic energies
        of the state before and after a NCMC step or iteration.

        Parameters
        ----------
        simkey : str (key: 'md', 'nc', 'alch')
            Key corresponding to the simulation.
        stateidx : int (key: 'state0' or 'state1')
            Key corresponding to the state information being stored.
        stateinfo : dict
            Dictionary containing the State information.
        """
        self.current_state[simkey][stateidx] = stateinfo

    def setStateConditions(self):
        """Stores the dict of current state of the MD and NCMC simulations.
        Dict contains the Positions, Velocities, Potential/Kinetic Energies
        of the current state.
        Sets the NCMC simulation Positions/Velocities to
        the current state of the MD simulation.
        """
        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        nc_state0 = self.getStateInfo(self.nc_sim.context, self.state_keys)
        self.nc_sim.context.setPeriodicBoxVectors(*md_state0['box_vectors'])
        self.nc_sim.context.setPositions(md_state0['positions'])
        self.nc_sim.context.setVelocities(md_state0['velocities'])
        self.setSimState('md', 'state0', md_state0)
        self.setSimState('nc', 'state0', nc_state0)

    def _getSimulationInfo(self, nIter):
        """logger.infos out simulation timing and related information."""

        total_ncmc_steps = self.opt['integration_steps']

        #Total NCMC simulation time
        time_ncmc_steps = total_ncmc_steps * self.opt['dt'].value_in_unit(unit.picoseconds)
        logger.info('\t%s NCMC ps/iter' % time_ncmc_steps)

        #Total MD simulation time
        time_md_steps = self.opt['nstepsMD'] * self.opt['dt'].value_in_unit(unit.picoseconds)
        logger.info('MD Steps = %s' % self.opt['nstepsMD'])
        logger.info('\t%s MD ps/iter' % time_md_steps)

        #Total BLUES simulation time
        totaltime = (time_ncmc_steps + time_md_steps) * nIter
        logger.info('Total Simulation Time = %s ps' % totaltime)
        logger.info('\tTotal NCMC time = %s ps' % (int(time_ncmc_steps) * int(nIter)))
        logger.info('\tTotal MD time = %s ps' % (int(time_md_steps) * int(nIter)))

        #Get trajectory frame interval timing for BLUES simulation
        frame_iter = self.opt['nstepsMD'] / self.opt['reporters']['trajectory_interval']
        timetraj_frame = (time_ncmc_steps + time_md_steps) / frame_iter
        logger.info('\tTrajectory Interval = %s ps' % timetraj_frame)
        logger.info('\t\t%s frames/iter' % frame_iter )

    def getStateInfo(self, context, parameters):
        """Function that gets the State information from the given context and
        list of parameters to query it with.
        Returns a dict of the data from the State.

        Parameters
        ----------
        context : openmm.Context
            Context of the OpenMM Simulation to query.
        parameters : list
            Default: [ positions, velocities, potential_energy, kinetic_energy ]
            A list that defines what information to get from the context State.
        """
        stateinfo = {}
        state  = context.getState(**parameters)
        stateinfo['iter'] = int(self.current_iter)
        stateinfo['positions'] =  state.getPositions(asNumpy=True)
        stateinfo['velocities'] = state.getVelocities(asNumpy=True)
        stateinfo['potential_energy'] = state.getPotentialEnergy()
        stateinfo['kinetic_energy'] = state.getKineticEnergy()
        stateinfo['box_vectors'] = state.getPeriodicBoxVectors()
        return stateinfo

    def getWorkInfo(self, nc_integrator, parameters):
        """Function that obtains the work and energies from the NCMC integrator.

        Returns a dict of the specified parameters.

        Parameters
        ----------
        nc_integrator : openmm.Context.Integrator
            The integrator from the NCMC Context
        parameters : list
            list containing strings of the values to get from the integrator.
            Default : ['total_work', 'lambda', 'shadow_work',
                       'protocol_work', 'Eold', 'Enew','Epert']
        """
        workinfo = {}
        for param in parameters:
            workinfo[param] = nc_integrator.getGlobalVariableByName(param)
        return workinfo

    def writeFrame(self, simulation, outfname):
        """Extracts a ParmEd structure and writes the frame given
        an OpenMM Simulation object"""
        topology = simulation.topology
        system = simulation.context.getSystem()
        state = simulation.context.getState(getPositions=True,
                                            getVelocities=True,
                                            getParameters=True,
                                            getForces=True,
                                            getParameterDerivatives=True,
                                            getEnergy=True,
                                            enforcePeriodicBox=True)


        # Generate the ParmEd Structure
        structure = parmed.openmm.load_topology(topology, system,
                                   xyz=state.getPositions())

        structure.save(outfname,overwrite=True)
        logger.info('\tSaving Frame to: %s' % outfname)

    def acceptRejectNCMC(self, temperature=300, write_move=False, **opt):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.current_state['md']['state0']
        nc_state0 = self.current_state['nc']['state0']
        nc_state1 = self.current_state['nc']['state1']

        log_ncmc = self.nc_sim.context._integrator.getLogAcceptanceProbability(self.nc_sim.context)
        randnum =  math.log(np.random.random())

        # Compute Alchemical Correction Term
        if np.isnan(log_ncmc) is False:
            self.alch_sim.context.setPeriodicBoxVectors(*nc_state1['box_vectors'])
            self.alch_sim.context.setPositions(nc_state1['positions'])
            alch_state1 = self.getStateInfo(self.alch_sim.context, self.state_keys)
            self.setSimState('alch', 'state1', alch_state1)
            correction_factor = (nc_state0['potential_energy'] - md_state0['potential_energy'] + alch_state1['potential_energy'] - nc_state1['potential_energy']) * (-1.0/self.nc_sim.context._integrator.kT)
            log_ncmc = log_ncmc + correction_factor

        if log_ncmc > randnum:
            self.accept += 1
            logger.info('NCMC MOVE ACCEPTED: log_ncmc {} > randnum {}'.format(log_ncmc, randnum) )
            self.md_sim.context.setPeriodicBoxVectors(*nc_state1['box_vectors'])
            self.md_sim.context.setPositions(nc_state1['positions'])
            if write_move:
            	self.writeFrame(self.md_sim, '{}acc-it{}.pdb'.format(self.opt['outfname'], self.current_iter))

        else:
            self.reject += 1
            logger.info('NCMC MOVE REJECTED: log_ncmc {} < {}'.format(log_ncmc, randnum) )
            self.nc_sim.context.setPositions(md_state0['positions'])

        self.nc_sim.currentStep = 0
        self.nc_sim.context._integrator.reset()
        self.md_sim.context.setVelocitiesToTemperature(temperature)

    def simulateNCMC(self, nstepsNC=5000, **opt):
        """Function that performs the NCMC simulation."""
        logger.info('[Iter %i] Advancing %i NCMC steps...' % (self.current_iter, nstepsNC))
        #choose a move to be performed according to move probabilities
        #TODO: will have to change to work with multiple alch region
        self.simulations.move_engine.selectMove()
        move_idx = self.simulations.move_engine.selected_move
        move_name = self.simulations.move_engine.moves[move_idx].__class__.__name__

        for nc_step in range(int(nstepsNC)):
            try:
                #Attempt anything related to the move before protocol is performed
                if nc_step == 0:
                    self.nc_sim.context = self.simulations.move_engine.moves[self.simulations.move_engine.selected_move].beforeMove(self.nc_sim.context)

                # Attempt selected MoveEngine Move at the halfway point
                #to ensure protocol is symmetric
                if self.movestep == nc_step:
                    #Do move
                    logger.report('Performing %s...' % move_name)
                    self.nc_sim.context = self.simulations.move_engine.runEngine(self.nc_sim.context)

                # Do 1 NCMC step with the integrator
                self.nc_sim.step(1)

                ###DEBUG options at every NCMC step
                logger.debug('%s' % self.getWorkInfo(self.nc_sim.context._integrator, self.work_keys))
                #Attempt anything related to the move after protocol is performed
                if nc_step == nstepsNC-1:
                    self.nc_sim.context = self.simulations.move_engine.moves[self.simulations.move_engine.selected_move].afterMove(self.nc_sim.context)

            except Exception as e:
                logger.error(e)
                self.simulations.move_engine.moves[self.simulations.move_engine.selected_move]._error(self.nc_sim.context)
                break

        nc_state1 = self.getStateInfo(self.nc_sim.context, self.state_keys)
        self.setSimState('nc', 'state1', nc_state1)

    def simulateMD(self, nstepsMD=5000, **opt):
        """Function that performs the MD simulation."""

        logger.info('[Iter %i] Advancing %i MD steps...' % (self.current_iter, nstepsMD))

        md_state0 = self.current_state['md']['state0']
        try:
            self.md_sim.step(nstepsMD)
        except Exception as e:
            logger.error(e, exc_info=True)
            logger.error('potential energy before NCMC: %s' % md_state0['potential_energy'])
            logger.error('kinetic energy before NCMC: %s' % md_state0['kinetic_energy'])
            #Write out broken frame
            self.writeFrame(self.md_sim, 'MD-fail-it%s-md%i.pdb' %(self.current_iter, self.md_sim.currentStep))
            exit()

        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        self.setSimState('md', 'state0', md_state0)
        # Set NC poistions to last positions from MD
        self.nc_sim.context.setPeriodicBoxVectors(*md_state0['box_vectors'])
        self.nc_sim.context.setPositions(md_state0['positions'])
        self.nc_sim.context.setVelocities(md_state0['velocities'])

    def run(self, nIter,**kwargs):
        """Function that runs the BLUES engine to iterate over the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state niter number of times.

        Parameters
        ----------
        nIter: int
            Number of iterations of NCMC+MD to perform.

        """
        logger.info('Running %i BLUES iterations...' % (nIter))
        self._getSimulationInfo(nIter)
        #set inital conditions
        self.setStateConditions()
        for n in range(int(nIter)):
            self.current_iter = int(n)
            self.setStateConditions()
            self.simulateNCMC(**self.opt)
            self.acceptRejectNCMC(**self.opt)
            self.simulateMD(**self.opt)

        # END OF NITER
        self.accept_ratio = self.accept/float(nIter)
        logger.info('Acceptance Ratio: %s' % self.accept_ratio)
        logger.info('nIter: %s ' % nIter)

    def simulateMC(self):
        """Function that performs the MC simulation."""

        #choose a move to be performed according to move probabilities
        self.simulations.move_engine.selectMove()
        #change coordinates according to Moves in MoveEngine
        new_context = self.simulations.move_engine.runEngine(self.md_sim.context)
        md_state1 = self.getStateInfo(new_context, self.state_keys)
        self.setSimState('md', 'state1', md_state1)

    def acceptRejectMC(self, temperature=300, **opt):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.current_state['md']['state0']
        md_state1 = self.current_state['md']['state1']
        log_mc = (md_state1['potential_energy'] - md_state0['potential_energy']) * (-1.0/self.nc_integrator.kT)
        randnum =  math.log(np.random.random())

        if log_mc > randnum:
            self.accept += 1
            logger.info('MC MOVE ACCEPTED: log_mc {} > randnum {}'.format(log_mc, randnum) )
            self.md_sim.context.setPositions(md_state1['positions'])
        else:
            self.reject += 1
            logger.info('MC MOVE REJECTED: log_mc {} < {}'.format(log_mc, randnum) )
            self.md_sim.context.setPositions(md_state0['positions'])
        logger_mc = log_mc
        self.md_sim.context.setVelocitiesToTemperature(temperature)

    def runMC(self, nIter):
        """Function that runs the BLUES engine to iterate over the actions:
        perform proposed move, accepts/rejects move,
        then performs the MD simulation from the accepted or rejected state.

        Parameters
        ----------
        nIter: None or int, optional, default=None
            The number of iterations to perform. If None, then
            uses the nIter specified in the opt dictionary when
            the Simulation class was created.
        """

        #controls how many mc moves are performed during each iteration
        try:
            self.mc_per_iter = self.opt['mc_per_iter']
        except:
            self.mc_per_iter = 1

        self.setStateConditions()
        for n in range(nIter):
            self.current_iter = int(n)
            for i in range(self.mc_per_iter):
                self.setStateConditions()
                self.simulateMC()
                self.acceptRejectMC(**self.opt)
            self.simulateMD(**self.opt)
