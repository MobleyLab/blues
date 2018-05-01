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
import logging
from math import floor, ceil
from collections import OrderedDict

class SimulationFactory(object):
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run.
    Ex.
        from blues.ncmc import SimulationFactory
        sims = SimulationFactory(structure, move_engine, **opt)
        sims.createSimulationSet()
    #TODO: add functionality for handling multiple alchemical regions

    Parameters
    ----------
    simulations : blues.ncmc.SimulationFactory object
        SimulationFactory Object which carries the 3 required
        OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.
    move_engine : blues.ncmc.MoveEngine object
        MoveProposal object which contains the dict of moves performed
        in the NCMC simulation.

    Integrator options
    ------------------
    dt: int, optional, default=0.002
        The timestep of the integrator to use (in ps).
    nprop: int, optional, default=5
        The number of additional propogation steps to be inserted
        during the middle of the NCMC protocol (defined by
        `prop_lambda`)
    prop_lambda: float, optional, default=0.3
        The range which additional propogation steps are added,
        defined by [0.5-prop_lambda, 0.5+prop_lambda].
    nstepsNC: int, optional, default=1000
        The number of NCMC relaxation steps to use.

    System options
    --------------
    Any arguments that are used to create a system from
    parmed Structure.createSystem() can be passed, where
    the string key in the dictionarycorresponds to the particular
    argument (such as nonbondedMethod, hydrogenMass, etc.).
    For arugments that require units, units can be specified,
    or if floats/ints are used, then the default OpenMM units
    are used. These are:
        length: nanometers
        time: picoseconds
        mass: atomic mass units (daltons)
        charge: proton charge
        temperature: Kelvin
    For arguments that require classes from openmm.app,
    such as nonbondedMethod either the class can be used directly,
    or the string corresponding to that class can be used.
    So for the `nonbondedMethod` arugment, for example, either
    openmm.app.PME or 'PME' can be used.

    Simulation options
    ------------------
    nIter: int, optional, default=100
        The number of MD + NCMC/MC iterations to perform.
    mc_per_iter: int, optional, default=1
        The number of MC moves to perform during each
        iteration of a MD + MC simulations.

    Reporter options
    ----------------
    trajectory_interval: int or None, optional, default=None
        Used to calculate the number of trajectory frames
        per iteration. If None, defaults to the value of nstepsNC.
    reporter_interval: int or None, optional, default=None
        Outputs information (steps, speed, accepted moves, iterations)
        about the NCMC simulation every reporter_interval interval.
        If None, defaults to the value of nstepsNC.
    outfname: str
        Prefix of log file to output to.
    Logger: logging.Logger
        Adds a logger that will output relevant non-trajectory
        simulation information to a log.

    Alchemical System options
    -------------------------
    freeze_center : str, optional, default='HOH,NA,CL'
        AmberMask selection for the center in which to select atoms for zeroing their masses. Default: LIG
    freeze_distance : float, optional, default=0
        Distance (angstroms) to select atoms for retaining their masses.
        Atoms outside the set distance will have their masses set to 0.0. Default: 0.0
    freeze_solvent : str
        AmberMask selection in which to select solvent atoms for zeroing their masses. Default: HOH,NA,CL
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
        Eq. 13 of Ref. [1] (default is 1 for a,b and 6 for c).
    softcore_beta : float, optional
        Alchemical softcore parameter for electrostatics. Set this to zero
        to recover standard electrostatic scaling (default is 0.0).
    softcore_d, softcore_e, softcore_f : float, optional
        Parameters modifying softcore electrostatics form (default is 1 for d,e and 2 for f).

    """
    def __init__(self, structure, move_engine,
                #integrator parameters
                dt=0.002, friction=1/unit.picoseconds, temperature=298*unit.kelvin,
                nprop=5, prop_lambda=0.3, nstepsNC=1000,
                nstepsMD=5000,
                alchemical_functions={'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                          'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' },
                trajectory_interval=5000, reporter_interval=5000,
                #createSystem parameters
                nonbondedMethod=None, nonbondedCutoff=8.0*unit.angstroms, switchDistance=0.0*unit.angstroms,
                constraints=None, rigidWater=True, implicitSolvent=None, implicitSolventKappa=None,
                implicitSolventSaltConc=0.0*unit.moles/unit.liters, soluteDielectric=1.0, solventDielectric=78.5,
                useSASA=False, removeCMMotion=True, hydrogenMass=None, ewaldErrorTolerance=0.0005,
                flexibleConstraints=True, verbose=False, splitDihedrals=False,

                #alchemical system parameters
                freeze_distance=0, freeze_center='LIG', freeze_solvent='HOH,NA,CL',
                softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6,
                softcore_beta=0.0, softcore_d=1, softcore_e=1, softcore_f=2,
                annihilate_electrostatics=True, annihilate_sterics=False,

                #mc options
                mc_per_iter=1,

                **opt):
        """Requires a parmed.Structure of the entire system and the ncmc.Model
        object being perturbed.

        Options is expected to be a dict of values. Ex:
        nIter=5, nstepsNC=50, nstepsMD=10000,
        temperature=300, friction=1, dt=0.002,
        nonbondedMethod='PME', nonbondedCutoff=10, constraints='HBonds',
        trajectory_interval=1000, reporter_interval=1000, platform=None"""
        if (nstepsNC % 2) != 0:
            raise Exception('nstepsNC needs to be even to ensure the protocol is symmetric (currently %i)' % (nstepsNC))

        if 'Logger' in opt:
            self.log = opt['Logger']
        else:
            self.log = logging.getLogger(__name__)
            #self.log = init_logger(logger)

        #Structure of entire system
        self.structure = structure
        #Atom indicies from move_engine
        #TODO: change atom_indices selection for multiple regions
        self.atom_indices = move_engine.moves[0].atom_indices
        self.move_engine = move_engine
        self.system = None
        self.alch_system = None
        self.md = None
        self.alch  = None
        self.nc  = None
        self.nstepsNC, self.integration_steps = self.calcNCMCSteps(nstepsNC, nprop, prop_lambda, self.log)
        self.opt = opt
        system_opt = OrderedDict([('nonbondedMethod',nonbondedMethod), ('nonbondedCutoff',nonbondedCutoff), ('switchDistance',switchDistance), ('constraints',constraints),
                            ('rigidWater',rigidWater), ('implicitSolvent',implicitSolvent), ('implicitSolventKappa',implicitSolventKappa),
                            ('implicitSolventSaltConc',implicitSolventSaltConc), ('temperature',temperature), ('soluteDielectric',soluteDielectric), ('useSASA',useSASA),
                            ('removeCMMotion',removeCMMotion), ('hydrogenMass',hydrogenMass), ('ewaldErrorTolerance',ewaldErrorTolerance),
                            ('flexibleConstraints',flexibleConstraints), ('verbose',verbose), ('splitDihedrals',splitDihedrals),
                            ('freeze_distance',freeze_distance), ('freeze_center',freeze_center), ('freeze_solvent',freeze_solvent),
                            ('dt',dt), ('friction',friction), ('temperature',temperature),
                            ('nprop',nprop), ('prop_lambda',prop_lambda), ('nstepsNC',self.nstepsNC), ('nstepsMD',nstepsMD),
                            ('alchemical_functions',alchemical_functions),
                            ('mc_per_iter',mc_per_iter), ('trajectory_interval',trajectory_interval), ('reporter_interval',reporter_interval),
                            ('softcore_alpha',softcore_alpha), ('softcore_a',softcore_a), ('softcore_b',softcore_b), ('softcore_c',softcore_c),
                            ('softcore_beta',softcore_beta), ('softcore_d',softcore_d), ('softcore_e',softcore_e), ('softcore_f',softcore_f),
                            ('annihilate_electrostatics',annihilate_electrostatics), ('annihilate_sterics',annihilate_sterics)])
        self.system_opt = self.add_units(system_opt, self.log)
        for k,v in system_opt.items():
            self.log.info('{} = {}'.format(k,v))
        self.createSimulationSet()

    @staticmethod
    def add_units(system_opt, logger):
        #for system setup portion

        #set unit defaults to OpenMM defaults
        unit_options = {'nonbondedCutoff':unit.nanometers,
                        'switchDistance':unit.nanometers, 'implicitSolventKappa':unit.nanometers,
                        'implicitSolventSaltConc':unit.mole/unit.liters, 'temperature':unit.kelvins,
                        'hydrogenMass':unit.daltons,
                        'dt':unit.picoseconds, 'friction':1/unit.picoseconds
                        }
        app_options = ['nonbondedMethod', 'constraints', 'implicitSolvent']
        scalar_options = ['soluteDielectric', 'solvent', 'ewaldErrorTolerance']
        bool_options = ['rigidWater', 'useSASA', 'removeCMMotion', 'flexibleConstraints', 'verbose',
                        'splitDihedrals']

        combined_options = list(unit_options.keys()) + app_options + scalar_options + bool_options
        for sel in system_opt.keys():
            if sel in combined_options:
                if sel in unit_options:
                    #if the value requires units check that it has units
                    #if it doesn't assume default units are used
                    if system_opt[sel] is None:
                        system_opt[sel] = None
                    else:
                        try:
                            system_opt[sel]._value
                        except:
                            logger.info('Units for {}:{} not specified. Using default units of {}'.format(sel, system_opt[sel], unit_options[sel]))
                            system_opt[sel] = system_opt[sel]*unit_options[sel]
                #if selection requires an OpenMM evaluation do it here
                elif sel in app_options:
                    try:
                        system_opt[sel] = eval("app.%s" % system_opt[sel])
                    except:
                        #if already an app object we can just pass
                        pass
                #otherwise just take the value as is, should just be a bool or float
                else:
                    pass
        return system_opt

    @staticmethod
    def calcNCMCSteps(total_steps, nprop, prop_lambda, log):
        nstepsNC = total_steps/(2*(nprop*prop_lambda+0.5-prop_lambda))
        if int(nstepsNC) % 2 == 0:
            nstepsNC = int(nstepsNC)
        else:
            nstepsNC = int(nstepsNC) + 1

        in_portion =  (prop_lambda)*nstepsNC
        out_portion = (0.5-prop_lambda)*nstepsNC
        if in_portion.is_integer():
            in_portion= int(in_portion)
        if out_portion.is_integer():
            int(out_portion)
        in_prop = int(nprop*(2*floor(in_portion)))
        out_prop = int((2*ceil(out_portion)))
        calc_total = int(in_prop + out_prop)
        if calc_total != total_steps:
            log.info('total nstepsNC requested ({}) does not divide evenly with the chosen values of prop_lambda and nprop. '.format(total_steps)+
                           'Instead using {} total propogation steps, '.format(calc_total)+
                           '({} steps inside `prop_lambda` and {} steps outside `prop_lambda)`.'.format(in_prop, out_prop))
        log.info('NCMC protocol will consist of {} lambda switching steps and {} total integration steps'.format(nstepsNC, calc_total))
        return nstepsNC, calc_total

    def _zero_allother_masses(self, system, indexlist):
        num_atoms = system.getNumParticles()
        for index in range(num_atoms):
            if index in indexlist:
                pass
            else:
                system.setParticleMass(index, 0*unit.daltons)
        return system

    def generateAlchSystem(self, system, atom_indices,
                            freeze_distance=0, freeze_center='LIG', freeze_solvent='HOH,NA,CL',
                            softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6,
                            softcore_beta=0.0, softcore_d=1, softcore_e=1, softcore_f=2,
                            annihilate_electrostatics=True, annihilate_sterics=False,
                            **opt):
        """Returns the OpenMM System for alchemical perturbations.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list
            Atom indicies of the move.
        freeze_center : str
            AmberMask selection for the center in which to select atoms for zeroing their masses. Default: LIG
        freeze_distance : float
            Distance (angstroms) to select atoms for retaining their masses. Atoms outside the set distance will have their masses set to 0.0. Default: 5.0
        freeze_solvent : str
            AmberMask selection in which to select solvent atoms for zeroing their masses. Default: HOH,NA,CL
        """
        logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)
        factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True)
        alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices, softcore_alpha=softcore_alpha,
                                softcore_a=softcore_a, softcore_b=softcore_b, softcore_c=softcore_c,
                                softcore_beta=softcore_beta, softcore_d=softcore_d, softcore_e=softcore_e,
                                softcore_f=softcore_f, annihilate_electrostatics=annihilate_electrostatics,
                                annihilate_sterics=annihilate_sterics)
        alch_system = factory.create_alchemical_system(system, alch_region)

        if freeze_distance:
            #Atom selection for zeroing protein atom masses
            mask = parmed.amber.AmberMask(self.structure,"(:%s<:%f)&!(:%s)" % (freeze_center,freeze_distance,freeze_solvent))
            site_idx = [i for i in mask.Selected()]
            self.log.info('Zeroing mass of %s atoms %.1f Angstroms from %s in alchemical system' % (len(site_idx), freeze_distance, freeze_center))
            self.log.debug('\nFreezing atom selection = %s' % site_idx)
            alch_system = self._zero_allother_masses(alch_system, site_idx)
        else:
            pass

        return alch_system

    def generateSystem(self, structure,
                       nonbondedMethod=None,
                       nonbondedCutoff=8.0*unit.angstroms,
                       switchDistance=0.0*unit.angstroms,
                       constraints=None,
                       rigidWater=True,
                       implicitSolvent=None,
                       implicitSolventKappa=None,
                       implicitSolventSaltConc=0.0*unit.moles/unit.liters,
                       soluteDielectric=1.0,
                       solventDielectric=78.5,
                       useSASA=False,
                       removeCMMotion=True,
                       hydrogenMass=None,
                       ewaldErrorTolerance=0.0005,
                       flexibleConstraints=True,
                       verbose=False,
                       splitDihedrals=False,
                       barostat_frequency=0,
                       barostat_pressure=1*unit.bar,
                       temperature=300,
                       **system_opt):
        """Returns the OpenMM System for the reference system.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system_opt : arguments for createSystem (i.e. cutoffs/constraints)
        """
        #distrubute list of options according to catagory
        system = structure.createSystem(nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                                        switchDistance=switchDistance, constraints=constraints,
                                        rigidWater=rigidWater, implicitSolvent=implicitSolvent,
                                        implicitSolventKappa=implicitSolventKappa,
                                        implicitSolventSaltConc=implicitSolventSaltConc,
                                        soluteDielectric=soluteDielectric, solventDielectric=solventDielectric,
                                        useSASA=useSASA, removeCMMotion=removeCMMotion, hydrogenMass=hydrogenMass,
                                        ewaldErrorTolerance=ewaldErrorTolerance, flexibleConstraints=flexibleConstraints,
                                        verbose=verbose, splitDihedrals=splitDihedrals)
        if barostat_frequency:
            system.addForce(openmm.MonteCarloBarostat(barostat_pressure, temperature, barostat_frequency))

        return system

    def generateSimFromStruct(self, structure, move_engine, system, nstepsNC,
                             temperature=300, dt=0.002, friction=1,
                             nprop=1, prop_lambda=0.3,
                             alchemical_functions={'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                          'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' },
                             ncmc=False, platform=None, **opt):
        """Used to generate the OpenMM Simulation objects given a ParmEd Structure.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system :
        opt : optional parameters (i.e. cutoffs/constraints)
        atom_indices : list
            Atom indicies of the move.
        prop_lambda : float (Default = 0.3)
            Defines the region in which to add extra propagation
            steps during the NCMC simulation from the midpoint 0.5.
            i.e. A value of 0.3 will add extra steps from lambda 0.2 to 0.8.
        nprop : int (Default: 1)
            Controls the number of propagation steps to add in the lambda
            region defined by `prop_lambda`
        """
        if ncmc:
            #During NCMC simulation, lambda parameters are controlled by function dict below
            # Keys correspond to parameter type (i.e 'lambda_sterics', 'lambda_electrostatics')
            # 'lambda' = step/totalsteps where step corresponds to current NCMC step,
            integrator = AlchemicalExternalLangevinIntegrator(
                                    alchemical_functions=alchemical_functions,
                                   splitting= "H V R O R V H",
                                   temperature=temperature,
                                   nsteps_neq=nstepsNC,
                                   timestep=dt,
                                   nprop=nprop,
                                   prop_lambda=prop_lambda
                                   )

            for move in move_engine.moves:
                system, integrator = move.initializeSystem(system, integrator)

        else:
            integrator = openmm.LangevinIntegrator(temperature,
                                                   friction,
                                                   dt)

        #TODO SIMPLIFY TO 1 LINE.
        #Specifying platform properties here used for local development.
        if platform is None:
            #Use the fastest available platform
            simulation = app.Simulation(structure.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName(platform)
            simulation = app.Simulation(structure.topology, system, integrator, platform)

        if ncmc: #Encapsulate so this self.log.infos once
            # OpenMM platform information
            mmver = openmm.version.version
            mmplat = simulation.context.getPlatform()
            self.log.info('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()))
            # Platform properties
            for prop in mmplat.getPropertyNames():
                val = mmplat.getPropertyValue(simulation.context, prop)
                self.log.info('{} = {}'.format(prop,val))

        # Set initial positions/velocities
        # Will get overwritten from saved State.
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)

        return simulation

    def createSimulationSet(self):
        """Function used to generate the 3 OpenMM Simulation objects."""
        self.system = self.generateSystem(self.structure, **self.system_opt)
        self.alch_system = self.generateAlchSystem(self.system, self.atom_indices, **self.system_opt)
        self.md = self.generateSimFromStruct(self.structure, self.move_engine, self.system,
                                            ncmc=False, **self.system_opt)
        self.alch = self.generateSimFromStruct(self.structure, self.move_engine, self.system,
                                            ncmc=False, **self.system_opt)
        self.nc = self.generateSimFromStruct(self.structure, self.move_engine, self.alch_system,
                                            ncmc=True, **self.system_opt)


class Simulation(object):
    """Simulation class provides the functions that perform the BLUES run.

    Ex.
        import blues.ncmc
        blues = ncmc.Simulation(sims, move_engine, **opt)
        blues.run()

    """
    def __init__(self, simulations, **opt
        ):
        """Initialize the BLUES Simulation object.

        Parameters
        ----------
        simulations : blues.ncmc.SimulationFactory object
            SimulationFactory Object which carries the 3 required
            OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.
        move_engine : blues.ncmc.MoveEngine object
            MoveProposal object which contains the dict of moves performed
            in the NCMC simulation.

        Integrator options
        ------------------
        dt: int, optional, default=0.002
            The timestep of the integrator to use (in ps).
        nprop: int, optional, default=5
            The number of additional propogation steps to be inserted
            during the middle of the NCMC protocol (defined by
            `prop_lambda`)
        prop_lambda: float, optional, default=0.3
            The range which additional propogation steps are added,
            defined by [0.5-prop_lambda, 0.5+prop_lambda].
        nstepsNC: int, optional, default=1000
            The number of NCMC relaxation steps to use.

        System options
        --------------
        Any arguments that are used to create a system from
        parmed Structure.createSystem() can be passed, where
        the string key in the dictionarycorresponds to the particular
        argument (such as nonbondedMethod, hydrogenMass, etc.).
        For arugments that require units, units can be specified,
        or if floats/ints are used, then the default OpenMM units
        are used. These are:
            length: nanometers
            time: picoseconds
            mass: atomic mass units (daltons)
            charge: proton charge
            temperature: Kelvin
        For arguments that require classes from openmm.app,
        such as nonbondedMethod either the class can be used directly,
        or the string corresponding to that class can be used.
        So for the `nonbondedMethod` arugment, for example, either
        openmm.app.PME or 'PME' can be used.

        Simulation options
        ------------------
        nIter: int, optional, default=100
            The number of MD + NCMC/MC iterations to perform.
        mc_per_iter: int, optional, default=1
            The number of MC moves to perform during each
            iteration of a MD + MC simulations.

        Reporter options
        ----------------
        trajectory_interval: int or None, optional, default=None
            Used to calculate the number of trajectory frames
            per iteration. If None, defaults to the value of nstepsNC.
        reporter_interval: int or None, optional, default=None
            Outputs information (steps, speed, accepted moves, iterations)
            about the NCMC simulation every reporter_interval interval.
            If None, defaults to the value of nstepsNC.
        outfname: str
            Prefix of log file to output to.
        Logger: logging.Logger
            Adds a logger that will output relevant non-trajectory
            simulation information to a log.

        """
        if 'Logger' in opt:
            self.log = opt['Logger']
        elif simulations.log:
            self.log = simulations.log
        else:
            self.log = logging.getLogger(__name__)
        self.simulations = simulations
        self.md_sim = simulations.md
        self.alch_sim = simulations.alch
        self.nc_sim = simulations.nc
        self.temperature = self.md_sim.integrator.getTemperature()
        self.accept = 0
        self.reject = 0
        self.accept_ratio = 0

        #if nstepsNC not specified, set it to 0
        #will be caught if NCMC simulation is run

        if (self.simulations.system_opt['nstepsNC'] % 2) != 0:
            raise Exception('nstepsNC needs to be even to ensure the protocol is symmetric (currently %i)' % (self.simulations.system_opt['nstepsNC']))
        else:
            self.movestep = int(self.simulations.system_opt['nstepsNC']) / 2

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
        """self.log.infos out simulation timing and related information."""

        total_ncmc_steps = self.simulations.integration_steps

        #Total NCMC simulation time
        time_ncmc_steps = total_ncmc_steps * self.simulations.system_opt['dt'].value_in_unit(unit.picoseconds)
        self.log.info('\t%s NCMC ps/iter' % time_ncmc_steps)

        #Total MD simulation time
        time_md_steps = self.simulations.system_opt['nstepsMD'] * self.simulations.system_opt['dt'].value_in_unit(unit.picoseconds)
        self.log.info('MD Steps = %s' % self.simulations.system_opt['nstepsMD'])
        self.log.info('\t%s MD ps/iter' % time_md_steps)

        #Total BLUES simulation time
        totaltime = (time_ncmc_steps + time_md_steps) * nIter
        self.log.info('Total Simulation Time = %s ps' % totaltime)
        self.log.info('\tTotal NCMC time = %s ps' % (int(time_ncmc_steps) * int(nIter)))
        self.log.info('\tTotal MD time = %s ps' % (int(time_md_steps) * int(nIter)))

        #Get trajectory frame interval timing for BLUES simulation
        frame_iter = self.simulations.system_opt['nstepsMD'] / self.simulations.system_opt['trajectory_interval']
        timetraj_frame = (time_ncmc_steps + time_md_steps) / frame_iter
        self.log.info('\tTrajectory Interval = %s ps' % timetraj_frame)
        self.log.info('\t\t%s frames/iter' % frame_iter )

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
        self.log.info('\tSaving Frame to: %s' % outfname)

    def acceptRejectNCMC(self, temperature=300, write_move=False, **opt):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.current_state['md']['state0']
        nc_state0 = self.current_state['nc']['state0']
        nc_state1 = self.current_state['nc']['state1']

        log_ncmc = self.nc_sim.context._integrator.getLogAcceptanceProbability(self.nc_sim.context)
        randnum =  math.log(np.random.random())

        # Compute Alchemical Correction Term
        if np.isnan(log_ncmc) == False:
            self.alch_sim.context.setPeriodicBoxVectors(*nc_state1['box_vectors'])
            self.alch_sim.context.setPositions(nc_state1['positions'])
            alch_state1 = self.getStateInfo(self.alch_sim.context, self.state_keys)
            self.setSimState('alch', 'state1', alch_state1)
            correction_factor = (nc_state0['potential_energy'] - md_state0['potential_energy'] + alch_state1['potential_energy'] - nc_state1['potential_energy']) * (-1.0/self.nc_sim.context._integrator.kT)
            log_ncmc = log_ncmc + correction_factor

        if log_ncmc > randnum:
            self.accept += 1
            self.log.info('NCMC MOVE ACCEPTED: log_ncmc {} > randnum {}'.format(log_ncmc, randnum) )
            self.md_sim.context.setPeriodicBoxVectors(*nc_state1['box_vectors'])
            self.md_sim.context.setPositions(nc_state1['positions'])
            if write_move:
            	self.writeFrame(self.md_sim, '{}acc-it{}.pdb'.format(self.simulations.system_opt['outfname'], self.current_iter))

        else:
            self.reject += 1
            self.log.info('NCMC MOVE REJECTED: log_ncmc {} < {}'.format(log_ncmc, randnum) )
            self.nc_sim.context.setPositions(md_state0['positions'])

        self.nc_sim.currentStep = 0
        self.nc_sim.context._integrator.reset()
        self.md_sim.context.setVelocitiesToTemperature(temperature)

    def simulateNCMC(self, nstepsNC=5000, **opt):
        """Function that performs the NCMC simulation."""
        self.log.info('[Iter %i] Advancing %i NCMC steps...' % (self.current_iter, nstepsNC))
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
                    self.log.info('Performing %s...' % move_name)
                    self.nc_sim.context = self.simulations.move_engine.runEngine(self.nc_sim.context)

                # Do 1 NCMC step with the integrator
                self.nc_sim.step(1)

                ###DEBUG options at every NCMC step
                self.log.debug('%s' % self.getWorkInfo(self.nc_sim.context._integrator, self.work_keys))
                #Attempt anything related to the move after protocol is performed
                if nc_step == nstepsNC-1:
                    self.nc_sim.context = self.simulations.move_engine.moves[self.simulations.move_engine.selected_move].afterMove(self.nc_sim.context)

            except Exception as e:
                self.log.error(e)
                self.simulations.move_engine.moves[self.simulations.move_engine.selected_move]._error(self.nc_sim.context)
                break

        nc_state1 = self.getStateInfo(self.nc_sim.context, self.state_keys)
        self.setSimState('nc', 'state1', nc_state1)

    def simulateMD(self, nstepsMD=5000, **opt):
        """Function that performs the MD simulation."""

        self.log.info('[Iter %i] Advancing %i MD steps...' % (self.current_iter, nstepsMD))

        md_state0 = self.current_state['md']['state0']
        try:
            self.md_sim.step(nstepsMD)
        except Exception as e:
            self.log.error(e, exc_info=True)
            self.log.error('potential energy before NCMC: %s' % md_state0['potential_energy'])
            self.log.error('kinetic energy before NCMC: %s' % md_state0['kinetic_energy'])
            #Write out broken frame
            self.writeFrame(self.md_sim, 'MD-fail-it%s-md%i.pdb' %(self.current_iter, self.md_sim.currentStep))
            exit()

        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        self.setSimState('md', 'state0', md_state0)
        # Set NC poistions to last positions from MD
        self.nc_sim.context.setPeriodicBoxVectors(*md_state0['box_vectors'])
        self.nc_sim.context.setPositions(md_state0['positions'])
        self.nc_sim.context.setVelocities(md_state0['velocities'])

    def run(self, nIter):
        """Function that runs the BLUES engine to iterate over the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state niter number of times.

        Parameters
        ----------
        nIter: int
            Number of iterations of NCMC+MD to perform.

        """
        self.log.info('Running %i BLUES iterations...' % (nIter))
        self._getSimulationInfo(nIter)
        #set inital conditions
        self.setStateConditions()
        for n in range(int(nIter)):
            self.current_iter = int(n)
            self.setStateConditions()
            self.simulateNCMC(**self.simulations.system_opt)
            self.acceptRejectNCMC(**self.simulations.system_opt)
            self.simulateMD(**self.simulations.system_opt)

        # END OF NITER
        self.accept_ratio = self.accept/float(nIter)
        self.log.info('Acceptance Ratio: %s' % self.accept_ratio)
        self.log.info('nIter: %s ' % nIter)

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
            self.log.info('MC MOVE ACCEPTED: log_mc {} > randnum {}'.format(log_mc, randnum) )
            self.md_sim.context.setPositions(md_state1['positions'])
        else:
            self.reject += 1
            self.log.info('MC MOVE REJECTED: log_mc {} < {}'.format(log_mc, randnum) )
            self.md_sim.context.setPositions(md_state0['positions'])
        self.log_mc = log_mc
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
            self.mc_per_iter = self.simulations.system_opt['mc_per_iter']
        except:
            self.mc_per_iter = 1

        self.setStateConditions()
        for n in range(nIter):
            self.current_iter = int(n)
            for i in range(self.mc_per_iter):
                self.setStateConditions()
                self.simulateMC()
                self.acceptRejectMC(**self.simulations.system_opt)
            self.simulateMD(**self.simulations.system_opt)
