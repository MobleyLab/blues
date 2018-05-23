import os, sys, logging
from math import ceil, floor
import numpy as np
import parmed
import yaml
from simtk import unit
from simtk.openmm import app
from blues import reporters


def startup(yaml_config):
    """
    Function that will parse the YAML configuration file for setup and running
    BLUES simulations.

    Parameters
    ----------
    yaml_config : filepath to YAML file (or JSON)
    """

    # Default parmed units.
    default_units = {'nonbondedCutoff': unit.angstroms,
                    'switchDistance': unit.angstroms,
                    'implicitSolventKappa': unit.angstroms,
                    'freeze_distance': unit.angstroms,
                    'temperature': unit.kelvins,
                    'hydrogenMass': unit.daltons,
                    'dt': unit.picoseconds,
                    'friction': 1 / unit.picoseconds,
                    'pressure': unit.atmospheres,
                    'implicitSolventSaltConc': unit.mole / unit.liters,
                    'weight': unit.kilocalories_per_mole / unit.angstroms**2,
                    }

    # System related parameters that require import from the simtk.openmm.app namesapce
    valid_apps = {
        'nonbondedMethod': ['NoCutoff', 'CutoffNonPeriodic',
                            'CutoffPeriodic', 'PME', 'Ewald'],
        'constraints': [None, 'HBonds', 'HAngles', 'AllBonds'],
        'implicitSolvent': ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']
        }
    # scalar_options = ['soluteDielectric', 'solvent', 'ewaldErrorTolerance']
    # bool_options = ['rigidWater', 'useSASA', 'removeCMMotion', 'flexibleConstraints', 'verbose',
    #                     'splitDihedrals']

    def load_yaml(yaml_config):
        """
        Function that reads the YAML configuration file and parameters are
        returned as a dict.
        """
        # Parse input parameters from YAML
        with open(yaml_config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except FileNotFoundError:
                raise FileNotFoundError
            except yaml.YAMLError as e:
                yaml_err = 'YAML parsing error in file: {}'.format(yaml_config)
                if hasattr(e, 'problem_mark'):
                    mark = e.problem_mark
                    print(yaml_err + '\nError on Line:{} Column:{}' \
                          .format(mark.line + 1, mark.column + 1))
                    raise e
            else:
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
                logger.error("'{}' was not a valid Amber selection. \n\tValid residue names: {}".format(
                    selection, res_set))
            elif '@' in selection:
                atom_set = set(atom.name for atom in structure.atoms)
                logger.error("'{}' was not a valid Amber selection. Valid atoms: {}".format(
                    selection, atom_set))
            sys.exit(1)

    def parse_unit_quantity(unit_quantity_str):
        """
        Utility for parsing parameters from the YAML file that require units.
        Takes a str, i.e. '3.024 * daltons' and returns as a simtk.unit.Quantity
        `unit.Quantity(3.024, unit=dalton)``
        """
        value, u = unit_quantity_str.replace(' ', '').split('*')
        if '/' in u:
            u = u.split('/')
            return unit.Quantity(float(value), eval('%s/unit.%s' % (u[0], u[1])))
        return unit.Quantity(float(value), eval('unit.%s' % u))

    def set_Structure(config):
        """
        Load the input/reference files (.prmtop, .inpcrd) into a parmed.Structure. If a `restart` (.rst7)
        file is given, overwrite the reference positions, velocities, and box vectors on the Structure.

        Parameters
        -----------
        filename: str, filepath to input (.prmtop)
        restart: str, file path to Amber restart file (.rst7)
        logger: logging.Logger object, records information

        Notes
        -----
        Reference for parmed.load_Structure *args and **kwargs
        https://parmed.github.io/ParmEd/html/structobj/parmed.formats.registry.load_file.html#parmed.formats.registry.load_file
        """
        structure = parmed.load_file(**config['structure'])
        if 'restart' in config['structure'].keys():
            config['Logger'].info('Restarting simulation from {}'.format(restart))
            restart = parmed.amber.Rst7(restart)
            structure.positions = restart.positions
            structure.velocities = restart.velocities
            structure.box = restart.box
        config['Structure'] = structure
        return config

    def set_Output(config):
        """
        Parses/updates the config (dict) with the given path for storing output files.
        """
        # Set file paths
        if 'output_dir' in config.keys():
            output_dir = config['output_dir']
        else:
            output_dir = '.'
        outfname = os.path.join(output_dir, config['outfname'])
        config['outfname'] = outfname
        config['simulation']['outfname'] = outfname
        return config

    def set_Logger(config):
        """
        Initializes the logging.Logger modules and parses/updates the
        config (dict) with the logger_level and the file path to store the .log file
        """
        # Initialize root Logger module
        level = config['logger_level'].upper()
        outfname = config['outfname']
        if level == 'DEBUG':
            # Add verbosity if logging is set to DEBUG
            config['verbose'] = True
            config['system']['verbose'] = True
            config['simulation']['verbose'] = True
        else:
            config['verbose'] = False
            config['system']['verbose'] = False
            config['simulation']['verbose'] = False
        logger_level = eval("logging.%s" % level)
        logger = reporters.init_logger(
            logging.getLogger(), logger_level, outfname)
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
        # Loop over parameters which require units
        for param, unit_type in default_units.items():

            # Check each nested subset of parameters
            for setup_keys in ['system', 'simulation', 'freeze', 'restraints']:
                # If the parameter requires units, cheeck if provided by user
                try:
                    if param in config[setup_keys]:
                        user_input = config[setup_keys][param]

                        if '*' in str(user_input):
                            config[setup_keys][param] = parse_unit_quantity(
                                user_input)

                        # If not provided, set default units
                        else:
                            config['Logger'].warn("Units for '{} = {}' not specified. Setting units to '{}'".format(
                                param, user_input, unit_type))
                            config[setup_keys][param] = user_input * unit_type

                except:
                    pass

        return config

    def check_SystemModifications(config):
        # Check Amber Selections
        if 'freeze' in config.keys():
            freeze_keys = ['freeze_center', 'freeze_solvent', 'freeze_selection']
            for sel in freeze_keys:
                if sel in config['freeze']:
                    check_amber_selection(config['Structure'],
                                          config['freeze'][sel], config['Logger'])

        if 'restraints' in config.keys():
            check_amber_selection(config['Structure'],
                                  config['restraints']['selection'], config['Logger'])

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
                    config['Logger'].exception(
                        "'{}' was not a valid option for '{}'. Valid options: {}".format(user_input, method, app_type))
        return config

    def set_ncmcSteps(config):
        """
        Calculates the number of lambda switching steps and integrator steps
        for the NCMC simulation.
        """

        logger = config['Logger']
        nstepsNC = config['simulation']['nstepsNC']
        try:
            nprop = config['simulation']['nprop']
            prop_lambda = config['simulation']['prop_lambda']
        except KeyError:
            nprop = 1
            prop_lambda = 0.3
            config['simulation']['nprop'] = nprop
            config['simulation']['prop_lambda'] = prop_lambda

        # Make sure provided NCMC steps is even.
        if (nstepsNC % 2) != 0:
            rounded_val = nstepsNC & ~1
            msg = 'nstepsNC=%i must be even for symmetric protocol.' % (nstepsNC)
            if rounded_val:
                logger.warning(msg+' Setting to nstepsNC=%i' % rounded_val)
                nstepsNC = rounded_val
            else:
                logger.error(msg)
                sys.exit(1)

        # Calculate the total number of lambda switching steps
        lambdaSteps = nstepsNC / (2 * (nprop * prop_lambda + 0.5 - prop_lambda))
        if int(lambdaSteps) % 2 == 0:
            lambdaSteps = int(lambdaSteps)
        else:
            lambdaSteps = int(lambdaSteps) + 1

        # Calculate number of lambda steps inside/outside region with extra propgation steps
        in_portion = (prop_lambda) * lambdaSteps
        out_portion = (0.5 - prop_lambda) * lambdaSteps
        in_prop = int(nprop * (2 * floor(in_portion)))
        out_prop = int((2 * ceil(out_portion)))
        propSteps = int(in_prop + out_prop)

        if propSteps != nstepsNC:
            logger.warn("nstepsNC=%s is incompatible with prop_lambda=%s and nprop=%s." % (nstepsNC, prop_lambda, nprop))
            logger.warn("Changing NCMC protocol to %s lambda switching within %s total propagation steps." % (lambdaSteps, propSteps))
            config['simulation']['nstepsNC'] = lambdaSteps

        config['simulation']['propSteps'] = propSteps
        config['simulation']['moveStep'] = int(config['simulation']['nstepsNC']/ 2)

        return config

    def set_Reporters(config):
        """
        Store the openmm.Reporters for the simulations to the configuration
        """
        logger = config['Logger']
        outfname = config['outfname']
        nstepsNC = config['simulation']['nstepsNC']

        if 'md_reporters' in config.keys():
            # Returns a list of Reporter objects, overwrites the configuration parameters
            md_reporter_cfg = reporters.ReporterConfig(outfname, config['md_reporters'], logger)
            config['md_reporters'] = md_reporter_cfg.makeReporters()
            config['simulation']['md_trajectory_interval'] = md_reporter_cfg.trajectory_interval
        else:
            logger.warn('Configuration for MD reporters were not set.')

        # Configure the NCMC simulation reporters
        if 'ncmc_reporters' in config.keys():

            #Update the reporter parameters with the proper NCMC steps
            for rep in config['ncmc_reporters'].keys():

                if 'totalSteps' in config['ncmc_reporters'][rep].keys():
                    config['ncmc_reporters'][rep]['totalSteps'] = nstepsNC

                #If -1 is given in frame_indices, record the last frame
                if 'frame_indices' in config['ncmc_reporters'][rep].keys():
                    frame_indices = config['ncmc_reporters'][rep]['frame_indices']
                    config['ncmc_reporters'][rep]['frame_indices'] = [nstepsNC if x == -1 else x for x in frame_indices]

            ncmc_reporter_cfg = reporters.ReporterConfig(outfname+'-ncmc', config['ncmc_reporters'], logger)
            config['ncmc_reporters'] = ncmc_reporter_cfg.makeReporters()
        else:
            logger.warn('Configuration for NCMC reporters were not set.')

        return config

    def set_Parameters(config):
        """
        MAIN execution function for updating/correcting (placing units) in the config
        """
        try:
            # Set top level configuration parameters
            config = set_Output(config)
            config = set_Logger(config)
            config = set_Structure(config)
            config = set_Units(config)
            check_SystemModifications(config)
            config = set_Apps(config)
            config = set_ncmcSteps(config)
            config = set_Reporters(config)

        except Exception as e:
            config['Logger'].exception(e)
            raise e

        return config

    # Parse YAML into dict
    if yaml_config.endswith('.yaml'):
        config = load_yaml(yaml_config)
    if type(yaml_config) is str:
        config = yaml.safe_load(yaml_config)
    # Parse the configions dict
    if type(config) is dict:
        config = set_Parameters(config)

    return config
