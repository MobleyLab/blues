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
import os, copy, yaml, logging, sys, json
import mdtraj
from blues import utils
from blues import reporters
from math import floor, ceil
from simtk.openmm import app

logger = logging.getLogger(__name__)

def startup(yaml_config):
    """
    Function that will parse the YAML configuration file for setup and running
    BLUES simulations.

    Parameters
    ----------
    yaml_config : filepath to YAML file (or JSON)
    """

    #Default parmed units.
    default_units = {'nonbondedCutoff':unit.angstroms,
                    'switchDistance':unit.angstroms,
                    'implicitSolventKappa':unit.angstroms,
                    'freeze_distance': unit.angstroms,

                    'implicitSolventSaltConc':unit.mole/unit.liters,
                    'temperature':unit.kelvins,

                    'hydrogenMass':unit.daltons,

                    'dt':unit.picoseconds,
                    'friction':1/unit.picoseconds,

                    'pressure': unit.atmospheres,

                    'weight': unit.kilocalories_per_mole/unit.angstroms**2,

                    }

    #System related parameters that require import from the simtk.openmm.app namesapce
    valid_apps = {
                'nonbondedMethod' : ['NoCutoff', 'CutoffNonPeriodic',
                                    'CutoffPeriodic', 'PME', 'Ewald'],
                'constraints' : [None, 'HBonds', 'HAngles', 'AllBonds'],
                'implicitSolvent' : ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']
                }
    #     scalar_options = ['soluteDielectric', 'solvent', 'ewaldErrorTolerance']
    #     bool_options = ['rigidWater', 'useSASA', 'removeCMMotion', 'flexibleConstraints', 'verbose',
    #                     'splitDihedrals']
    #

    def load_yaml(yaml_config):
        """
        Function that reads the YAML configuration file and parameters are
        returned as a dict.
        """
        #Parse input parameters from YAML
        with open(yaml_config, 'r') as stream:
            try:
                config = (yaml.load(stream))
            except Exception as exc:
                print (exc)
        return config

    def check_amber_selection(structure, selection, logger):
        """
        Given a AmberMask selection (str) for selecting atoms to freeze or restrain,
        check if it will actually select atoms. If the selection produces None,
        suggest valid residues or atoms.

        Parameters
        ----------
        structure: parmed.Structure object
        selection: str (Amber syntax), atoms to restrain/freeze during simulation.
        logger: logging.Logger object, records information
        """

        mask_idx = []
        mask = parmed.amber.AmberMask(structure, str(selection))
        mask_idx = [i for i in mask.Selected()]
        if not mask_idx:
            if ':' in selection:
                res_set = set(residue.name for residue in structure.residues)
                logger.error("'{}' was not a valid Amber selection. Valid residues: {}".format(selection, res_set))
            elif '@' in selection:
                atom_set = set(atom.name for atom in structure.atoms)
                logger.error("'{}' was not a valid Amber selection. Valid atoms: {}".format(selection, atom_set))
            sys.exit(1)

    def load_Structure(filename, restart=None, logger=None, *args, **kwargs):
        """
        Load the input/reference files (.prmtop, .inpcrd) into a parmed.Structure. If a `restart` (.rst7)
        file is given, overwrite the reference positions, velocities, and box vectors on the Structure.

        Parameters
        -----------
        filename: str, filepath to input (.prmtop)
        restart: str, file path to Amber restart file (.rst7)
        logger: logging.Logger object, records information

        Args
        ----

        Kwargs
        ------

        """
        structure = parmed.load_file(filename, *args, **kwargs)
        if restart:
            logger.info('Restarting simulation from {}'.format(restart))
            restart = parmed.amber.Rst7(restart)
            structure.positions = restart.positions
            structure.velocities = restart.velocities
            structure.box = restart.box

        return structure

    def parse_unit_quantity(unit_quantity_str):
        """
        Utility for parsing parameters from the YAML file that require units.
        Takes a str, i.e. '3.024 * daltons' and returns as a simtk.unit.Quantity
        `unit.Quantity(3.024, unit=dalton)``
        """
        value, u = unit_quantity_str.replace(' ', '').split('*')
        if '/' in u:
            u = u.split('/')
            return unit.Quantity(float(value), eval('%s/unit.%s' % (u[0],u[1])))
        return unit.Quantity(float(value), eval('unit.%s' % u))

    def set_Output(config):
        """
        Parses/updates the config (dict) with the given path for storing output files.

        """
        #Set file paths
        if 'output_dir' in config.keys():
            output_dir = config['output_dir']
        else:
            output_dir = '.'
        outfname = os.path.join(output_dir, config['outfname'])
        config['simulation']['outfname'] = outfname
        return config

    def set_Logger(config):
        """
        Initializes the logging.Logger modules and parses/updates the
        config (dict) with the logger_level and the file path to store the .log file

        """
        #Initialize root Logger module
        level = config['logger_level'].upper()
        outfname = config['outfname']
        if level == 'DEBUG':
            #Add verbosity if logging is set to DEBUG
            config['verbose'] = True
            config['system']['verbose'] = True
            config['simulation']['verbose'] = True
        else:
            config['verbose'] = False
            config['system']['verbose'] = False
            config['simulation']['verbose'] = False
        logger_level = eval("logging.%s" % level)
        logger = reporters.init_logger(logging.getLogger(), logger_level, outfname)
        config['Logger'] = logger

        return config

    def set_Units(config):
        """
        Parses/updates the config (dict) values with parameters that should have
        units on them. If no unit is provided, the default units are assumed.

            Distances: unit.angstroms
            Temperature: unit.kelvins
            Masses: unit.daltons
            Time: unit.picoseconds
            Pressure: unit.atmospheres
            Force:  unit.kilocalories_per_mole/unit.angstroms**2

        """
        #Loop over parameters which require units
        for param, unit_type in default_units.items():

            #Check each nested subset of parameters
            for setup_keys in ['system', 'simulation', 'freeze', 'restraints']:
                #If the parameter requires units, cheeck if provided by user
                if param in config[setup_keys]:
                    user_input = config[setup_keys][param]

                    if '*' in str(user_input):
                        config[setup_keys][param] =  parse_unit_quantity(user_input)

                    #If not provided, set default units
                    else:
                        config['Logger'].warn("Units for '{} = {}' not specified. Setting units to '{}'".format(param, user_input, unit_type))
                        config[setup_keys][param] = user_input*unit_type

                else:
                    pass

        return config

    def set_Apps(config):
        """
        Check system parameters which require loading from the simtk.openmm.app namespace


        nonbondedMethod : ['NoCutoff', 'CutoffNonPeriodic', 'CutoffPeriodic', 'PME', 'Ewald'],
        constraints : [None, 'HBonds', 'HAngles', 'AllBonds'],
        implicitSolvent : ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']

        """
        for method, app_type in valid_apps.items():
            if method in config['system']:
                user_input = config['system'][method]
                try:
                    config['system'][method] = eval("app.%s" % user_input)
                except:
                    config['Logger'].exception("'{}' was not a valid option for '{}'. Valid options: {}".format(user_input, method, app_type))
        return opt

    def set_Parameters(opt):
        """
        MAIN execution function for updating/correcting (placing units) in the config
        """
        try:
            config = set_Output(config)
            config = set_Logger(config)
            config['Structure'] = load_Structure(logger=config['Logger'],**config['structure'])
            config = set_Units(config)
            config = set_Apps(config)

            #Check Amber Selections
            if 'freeze' in config.keys():
                for sel in ['freeze_center', 'freeze_solvent']:
                    check_amber_selection(config['Structure'], config['freeze'][sel], config['Logger'])

            if 'restraints' in config.keys():
                check_amber_selection(config['Structure'], config['restraints']['selection'], config['Logger'])

            #Calculate NCMC steps with nprop
            config['simulation']['nstepsNC'], config['simulation']['integration_steps'] = calcNCMCSteps(logger=config['Logger'], **config['simulation'])

        except Exception as e:
            logger.exception(e)
            raise

        return config

    def calcNCMCSteps(total_steps, nprop, prop_lambda, logger, **kwargs):
        if (total_steps % 2) != 0:
           logger.exception('`total_steps = %i` must be even for symmetric protocol.' % (total_steps))

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
            logger.warn('total nstepsNC requested ({}) does not divide evenly with the chosen values of prop_lambda and nprop. '.format(total_steps)+
                           'Instead using {} total propogation steps, '.format(calc_total)+
                           '({} steps inside `prop_lambda` and {} steps outside `prop_lambda)`.'.format(in_prop, out_prop))
        logger.info('NCMC protocol will consist of {} lambda switching steps and {} total integration steps'.format(nstepsNC, calc_total))
        return nstepsNC, calc_total


    #Parse YAML into dict
    if config.endswith('.yaml'):
        config = load_yaml(config)

    #Parse the configions dict
    if type(config) is dict:
        config = set_Parameters(config)

    return config

class SystemFactory(object):
    """
    SystemFactory contains methods to generate/modify the OpenMM System object required for
    generating the openmm.Simulation using a given parmed.Structure()

    Usage Example
    -------------
    #Load Parmed Structure
    structure = parmed.load_file('eqToluene.prmtop', xyz='eqToluene.inpcrd')

    #Select move type
    ligand = RandomLigandRotationMove(structure, 'LIG')
    #Iniitialize object that selects movestep
    ligand_mover = MoveEngine(ligand)

    #Generate the openmm.Systems
    systems = SystemFactory(structure, ligand.atom_indices, **config['system'])

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
    def __init__(self, structure, atom_indices, **config):
        self.structure = structure
        self.atom_indices = atom_indices
        self.config = config

        self.alch_config = self.config.pop('alchemical')

        self.md = SystemFactory.generateSystem(self.structure, **self.config)
        self.alch = SystemFactory.generateAlchSystem(self.md, self.atom_indices, **self.alch_config)

    @classmethod
    def generateSystem(cls, structure, **kwargs):
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

    @classmethod
    def generateAlchSystem(cls, system, atom_indices,
                            softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6,
                            softcore_beta=0.0, softcore_d=1, softcore_e=1, softcore_f=2,
                            annihilate_electrostatics=True, annihilate_sterics=False,
                            disable_alchemical_dispersion_correction=True,
                            suppress_warnings=True,
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
        if suppress_warnings:
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

    @staticmethod
    def _amber_selection_to_atom_indices_(structure, selection):
        mask = parmed.amber.AmberMask(structure, str(selection))
        mask_idx = [i for i in mask.Selected()]
        return mask_idx

    @staticmethod
    def _print_atomlist_from_atom_indices_(structure, mask_idx):
        atom_list = []
        for i, at in enumerate(structure.atoms):
            if i in mask_idx:
                atom_list.append(structure.atoms[i])
        logger.debug('\nFreezing {}'.format(atom_list))
        return atom_list

    @classmethod
    def restrain_positions(cls, structure, system, selection="(@CA,C,N)", weight=5.0, **kwargs):
        """
        Applies positional restraints to the given openmm.System.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.

        Kwargs
        -------
        selection : str, Default = "(@CA,C,N)"
            AmberMask selection to apply positional restraints to
        weight : float, Default = 5.0
            Restraint weight for xyz atom restraints in kcal/(mol A^2)

        References
        -----
        Amber mask syntax: http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax
        """
        mask_idx = cls._amber_selection_to_atom_indices_(structure, selection)

        logger.info("{} positional restraints applied to selection: '{}' ({} atoms) on {}".format(weight, selection, len(mask_idx), system))
        # define the custom force to restrain atoms to their starting positions
        force = openmm.CustomExternalForce('k_restr*periodicdistance(x, y, z, x0, y0, z0)^2')
        # Add the restraint weight as a global parameter in kcal/mol/A^2
        force.addGlobalParameter("k_restr", weight)
        #force.addGlobalParameter("k_restr", weight*unit.kilocalories_per_mole/unit.angstroms**2)
        # Define the target xyz coords for the restraint as per-atom (per-particle) parameters
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        for i, atom_crd in enumerate(structure.positions):
            if i in mask_idx:
                logger.debug(i, structure.atoms[i])
                force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
        system.addForce(force)

        return system

    @classmethod
    def freeze_atoms(cls, structure, system, selection=":LIG", **kwargs):
        """
        Function that will zero the masses of atoms from the given selection.
        Massless atoms will be ignored by the integrator and will not change positions.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.

        Kwargs
        -------
        selection : str, Default = ":LIG"
            AmberMask selection for the center in which to select atoms for zeroing their masses.
            Defaults to freezing protein backbone atoms.

        References
        -----
        Amber mask syntax: http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax
        """
        mask_idx = cls._amber_selection_to_atom_indices_(structure, selection)
        logger.info("Freezing selection '{}' ({} atoms) on {}".format(selection, len(mask_idx), system))

        self._print_atomlist_from_atom_indices_(structure, mask_idx)
        system = utils.zero_masses(system, mask_idx)
        return system

    @classmethod
    def freeze_radius(cls, structure, system, freeze_distance=5.0,
                    freeze_center=':LIG', freeze_solvent=':HOH,NA,CL', **kwargs):
        """
        Function that will zero the masses of atoms outside the given raidus of
        the `freeze_center` selection. Massless atoms will be ignored by the
        integrator and will not change positions.This is intended to freeze
        the solvent and protein atoms around the ligand binding site.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.

        Kwargs
        -------
        freeze_center : str, Default = ":LIG"
            AmberMask selection for the center in which to select atoms for zeroing their masses. Default: LIG
        freeze_distance : float, Default = 5.0
            Distance (angstroms) to select atoms for retaining their masses.
            Atoms outside the set distance will have their masses set to 0.0.
        freeze_solvent : str, Default = ":HOH,NA,CL"
            AmberMask selection in which to select solvent atoms for zeroing their masses.

        References
        -----
        Amber mask syntax: http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax
        """
        selection = "(%s<:%f)&!(%s)" % (freeze_center,freeze_distance._value,freeze_solvent)
        site_idx = cls._amber_selection_to_atom_indices_(structure, selection)
        freeze_idx = set(range(system.getNumParticles())) - set(site_idx)

        #Atom selection for zeroing protein atom masses
        logger.info("Freezing {} atoms {} Angstroms from '{}' on {}".format(len(freeze_idx), freeze_distance._value, freeze_center, system))

        cls._print_atomlist_from_atom_indices_(structure, freeze_idx)
        system = utils.zero_masses(system, freeze_idx)
        return system

class SimulationFactory(object):
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run.

    Usage Example
    -------------
    #Load Parmed Structure
    structure = parmed.load_file('eqToluene.prmtop', xyz='eqToluene.inpcrd')

    #Select move type
    ligand = RandomLigandRotationMove(structure, 'LIG')
    #Iniitialize object that selects movestep
    ligand_mover = MoveEngine(ligand)

    #Generate the openmm.Systems
    systems = SystemFactory(structure, ligand.atom_indices, **opt['system'])

    #Generate the OpenMM Simulations
    simulations = SimulationFactory(systems, ligand_mover, **opt['simulation'])

    Parameters
    ----------
    systems : blues.simulation.SystemFactory object
        The object containing the MD and alchemical openmm.Systems
    move_engine : blues.engine.MoveEngine object
        MoveProposal object which contains the dict of moves performed
        in the NCMC simulation.
    opt : dict of parameters for the simulation (i.e timestep, temperature, etc.)
    """
    def __init__(self, systems, move_engine, **config):
        #Hide these properties since they exist on the SystemsFactory object.
        self._structure = systems.structure
        self._system = systems.md
        self._alch_system = systems.alch
        #Atom indicies from move_engine
        #TODO: change atom_indices selection for multiple regions
        self._atom_indices = move_engine.moves[0].atom_indices
        self.move_engine = move_engine

        self.opt = config
        self.generateSimulationSet()

    @classmethod
    def addBarostat(cls, system, temperature=300, pressure=1, frequency=25, **kwargs):
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
        pressure : int, configional, default=None
            Pressure (atm) for Barostat for NPT simulations.
        frequency : int, default=25
            Frequency at which Monte Carlo pressure changes should be attempted (in time steps)
        """
        logger.info('Adding MonteCarloBarostat with %s. MD simulation will be NPT.' %(pressure))
        # Add Force Barostat to the system
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, frequency))
        return system

    @classmethod
    def generateIntegrator(cls, temperature=300, dt=0.002, friction=1, **kwargs):
        """
        Generates a LangevinIntegrator for the Simulations.

        Kwargs
        ----------
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        friction: float, default=1
            friction coefficient which couples to the heat bath, measured in 1/ps
        dt: int, configional, default=0.002
            The timestep of the integrator to use (in ps).
        """
        integrator = openmm.LangevinIntegrator(temperature, friction, dt)
        return integrator

    @classmethod
    def generateNCMCIntegrator(cls, nstepsNC,
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

    @classmethod
    def generateSimFromStruct(cls, structure, system, integrator, platform=None, properties={}, **kwargs):
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
        simulation.context.setPeriodicBoxVectors(*structure.box_vectors)
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(integrator.getTemperature())

        return simulation

    @staticmethod
    def print_simulation_info(simulation):
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
        self.integrator = self.generateIntegrator(**self.config)
        if 'pressure' in self.config.keys():
            self._system = self.addBarostat(self._system, **self.config)
            logger.warning('NCMC simulation will NOT have pressure control. NCMC will use pressure from last MD state.')
        else:
            logger.info('MD simulation will be NVT.')
        self.md = self.generateSimFromStruct(self._structure, self._system, self.integrator, **self.config)

        #Alchemical Simulation is used for computing correction term from MD simulation.
        alch_integrator = self.generateIntegrator(**self.config)
        self.alch = self.generateSimFromStruct(self._structure, self._system, alch_integrator, **self.config)

        #Construct NCMC Integrator and Simulation
        self.ncmc_integrator = self.generateNCMCIntegrator(**self.config)

        #Initialize the Move Engine with the Alchemical System and NCMC Integrator
        for move in self.move_engine.moves:
            self._alch_system, self.ncmc_integrator = move.initializeSystem(self._alch_system, self.ncmc_integrator)
        self.nc = self.generateSimFromStruct(self._structure, self._alch_system, self.ncmc_integrator, **self.config)

        SimulationFactory.print_simulation_info(self.nc)

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
        self.md_sim = self.simulations.md
        self.alch_sim = self.simulations.alch
        self.nc_sim = self.simulations.nc
        self.temperature = self.md_sim.integrator.getTemperature()
        self.accept = 0
        self.reject = 0
        self.accept_ratio = 0
        self.config = self.simulations.config

        self.movestep = int(self.config['nstepsNC']) / 2

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

        total_ncmc_steps = self.config['integration_steps']

        #Total NCMC simulation time
        time_ncmc_steps = total_ncmc_steps * self.config['dt'].value_in_unit(unit.picoseconds)
        logger.info('\t%s NCMC ps/iter' % time_ncmc_steps)

        #Total MD simulation time
        time_md_steps = self.config['nstepsMD'] * self.config['dt'].value_in_unit(unit.picoseconds)
        logger.info('MD Steps = %s' % self.config['nstepsMD'])
        logger.info('\t%s MD ps/iter' % time_md_steps)

        #Total BLUES simulation time
        totaltime = (time_ncmc_steps + time_md_steps) * nIter
        logger.info('Total Simulation Time = %s ps' % totaltime)
        logger.info('\tTotal NCMC time = %s ps' % (int(time_ncmc_steps) * int(nIter)))
        logger.info('\tTotal MD time = %s ps' % (int(time_md_steps) * int(nIter)))

        #Get trajectory frame interval timing for BLUES simulation
        frame_iter = self.config['nstepsMD'] / self.config['reporters']['trajectory_interval']
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

    def acceptRejectNCMC(self, temperature=300, write_move=False, **config):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.current_state['md']['state0']
        nc_state0 = self.current_state['nc']['state0']
        nc_state1 = self.current_state['nc']['state1']

        work_ncmc = self.nc_sim.context._integrator.getLogAcceptanceProbability(self.nc_sim.context)
        randnum =  math.log(np.random.random())

        # Compute Alchemical Correction Term
        if np.isnan(work_ncmc) is False:
            self.alch_sim.context.setPeriodicBoxVectors(*nc_state1['box_vectors'])
            self.alch_sim.context.setPositions(nc_state1['positions'])
            alch_state1 = self.getStateInfo(self.alch_sim.context, self.state_keys)
            self.setSimState('alch', 'state1', alch_state1)
            correction_factor = (nc_state0['potential_energy'] - md_state0['potential_energy'] + alch_state1['potential_energy'] - nc_state1['potential_energy']) * (-1.0/self.nc_sim.context._integrator.kT)
            work_ncmc = work_ncmc + correction_factor

        if work_ncmc > randnum:
            self.accept += 1
            logger.info('NCMC MOVE ACCEPTED: work_ncmc {} > randnum {}'.format(work_ncmc, randnum) )
            self.md_sim.context.setPeriodicBoxVectors(*nc_state1['box_vectors'])
            self.md_sim.context.setPositions(nc_state1['positions'])
            if write_move:
            	self.writeFrame(self.md_sim, '{}acc-it{}.pdb'.format(self.config['outfname'], self.current_iter))

        else:
            self.reject += 1
            logger.info('NCMC MOVE REJECTED: work_ncmc {} < {}'.format(work_ncmc, randnum) )
            self.nc_sim.context.setPositions(md_state0['positions'])

        self.nc_sim.currentStep = 0
        self.nc_sim.context._integrator.reset()
        self.md_sim.context.setVelocitiesToTemperature(temperature)

    def simulateNCMC(self, nstepsNC=5000, **config):
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

    def simulateMD(self, nstepsMD=5000, **config):
        """Function that performs the MD simulation."""

        logger.info('[Iter %i] Advancing %i MD steps...' % (self.current_iter, nstepsMD))

        md_state0 = self.current_state['md']['state0']
        for md_step in range(int(nstepsMD)):
            try:
                self.md_sim.step(1)
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
            self.simulateNCMC(**self.config)
            self.acceptRejectNCMC(**self.config)
            self.simulateMD(**self.config)

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

    def acceptRejectMC(self, temperature=300, **config):
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
            self.mc_per_iter = self.config['mc_per_iter']
        except:
            self.mc_per_iter = 1

        self.setStateConditions()
        for n in range(nIter):
            self.current_iter = int(n)
            for i in range(self.mc_per_iter):
                self.setStateConditions()
                self.simulateMC()
                self.acceptRejectMC(**self.config)
            self.simulateMD(**self.config)
