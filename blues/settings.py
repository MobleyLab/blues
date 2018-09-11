import json
import logging
import os

import parmed
import yaml
from simtk import unit
from simtk.openmm import app

from blues import reporters, utils


class Settings(object):
    """
    Function that will parse the YAML configuration file for setup and running
    BLUES simulations.

    Parameters
    ----------
    yaml_config : filepath to YAML file (or JSON)
    """

    def __init__(self, config):
        # Parse YAML or YAML docstr into dict
        config = Settings.load_yaml(config)

        # Parse the config into dict
        if type(config) is dict:
            config = Settings.set_Parameters(config)
            self.config = config

    @staticmethod
    def load_yaml(yaml_config):
        """
        Function that reads the YAML configuration file and parameters are
        returned as a dict.
        """
        # Parse input parameters from YAML
        try:
            if os.path.isfile(yaml_config):
                with open(yaml_config, 'r') as stream:
                    config = yaml.safe_load(stream)
            else:
                config = yaml.safe_load(yaml_config)
        except IOError as e:
            print("Unable to open file:", yaml_config)
            raise e
        except yaml.YAMLError as e:
            yaml_err = 'YAML parsing error in file: {}'.format(yaml_config)
            if hasattr(e, 'problem_mark'):
                mark = e.problem_mark
                print(yaml_err + '\nError on Line:{} Column:{}' \
                      .format(mark.line + 1, mark.column + 1))
                raise e
        else:
            return config

    @staticmethod
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
        if 'restart' in config['structure'].keys():
            rst7 = config['structure']['restart']
            config['Logger'].info('Restarting simulation from {}'.format(rst7))
            restart = parmed.amber.Rst7(rst7)
            config['structure'].pop('restart')

            structure = parmed.load_file(**config['structure'])
            structure.positions = restart.positions
            structure.velocities = restart.velocities
            structure.box = restart.box
        else:
            structure = parmed.load_file(**config['structure'])

        config['Structure'] = structure
        return config

    @staticmethod
    def set_Output(config):
        """
        Parses/updates the config (dict) with the given path for storing output files.
        """
        # Set file paths
        if 'output_dir' in config.keys():
            os.makedirs(config['output_dir'], exist_ok=True)
        else:
            output_dir = '.'
        outfname = os.path.join(config['output_dir'], config['outfname'])
        print(outfname)
        config['outfname'] = outfname
        config['simulation']['outfname'] = outfname
        return config

    @staticmethod
    def set_Logger(config):
        """
        Initializes the logging.Logger modules and parses/updates the
        config (dict) with the logger_level and the file path to store the .log file
        """
        # Initialize root Logger module
        #level = config['logger_level'].upper()
        level = config['Logger']['level'].upper()
        stream = config['Logger']['stream']

        if 'filename' in config['Logger'].keys():
            outfname = config['Logger']['filename']
        else:
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
        logger = reporters.init_logger(logging.getLogger(), logger_level, stream, outfname)
        config['Logger'] = logger

        return config

    @staticmethod
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
        # Default parmed units.
        default_units = {
            'nonbondedCutoff': unit.angstroms,
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

        # Loop over parameters which require units
        for param, unit_type in default_units.items():
            # Check each nested subset of parameters
            for setup_keys in ['system', 'simulation', 'freeze', 'restraints']:
                # If the parameter requires units, cheeck if provided by user
                try:
                    #print(param, config[setup_keys].keys())
                    if str(param) in config[setup_keys].keys():
                        user_input = config[setup_keys][param]

                        if '*' in str(user_input):
                            config[setup_keys][param] = utils.parse_unit_quantity(user_input)
                        # If not provided, set default units
                        else:
                            config['Logger'].warn("Units for '{} = {}' not specified. Setting units to '{}'".format(
                                param, user_input, unit_type))
                            config[setup_keys][param] = user_input * unit_type

                except:
                    pass
        return config

    @staticmethod
    def check_SystemModifications(config):
        """
        Given a dict (config), check the parameters related to freezing or
        restraining the system. Requires loading parmed.Structure from YAML.
        """
        # Check Amber Selections
        if 'freeze' in config.keys():
            freeze_keys = ['freeze_center', 'freeze_solvent', 'freeze_selection']
            for sel in freeze_keys:
                if sel in config['freeze']:
                    utils.check_amber_selection(config['Structure'], config['freeze'][sel])

        if 'restraints' in config.keys():
            utils.check_amber_selection(config['Structure'], config['restraints']['selection'])

    @staticmethod
    def set_Apps(config):
        """
        Check system parameters which require loading from the simtk.openmm.app namespace

        nonbondedMethod : ['NoCutoff', 'CutoffNonPeriodic', 'CutoffPeriodic', 'PME', 'Ewald'],
        constraints : [None, 'HBonds', 'HAngles', 'AllBonds'],
        implicitSolvent : ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']
        """

        # System related parameters that require import from the simtk.openmm.app namesapce
        valid_apps = {
            'nonbondedMethod': ['NoCutoff', 'CutoffNonPeriodic', 'CutoffPeriodic', 'PME', 'Ewald'],
            'constraints': [None, 'HBonds', 'HAngles', 'AllBonds'],
            'implicitSolvent': ['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2']
        }

        for method, app_type in valid_apps.items():
            if method in config['system']:
                user_input = config['system'][method]
                try:
                    config['system'][method] = eval("app.%s" % user_input)
                except:
                    config['Logger'].exception("'{}' was not a valid option for '{}'. Valid options: {}".format(
                        user_input, method, app_type))
        return config

    @staticmethod
    def set_ncmcSteps(config):
        """
        Calculates the number of lambda switching steps and integrator steps
        for the NCMC simulation.
        """
        ncmc_parameters = utils.calculateNCMCSteps(**config['simulation'])
        for k, v in ncmc_parameters.items():
            config['simulation'][k] = v
        return config

    @staticmethod
    def set_Reporters(config):
        """
        Store the openmm.Reporters for the simulations to the configuration
        """
        logger = config['Logger']
        outfname = config['outfname']
        nstepsNC = config['simulation']['nstepsNC']
        moveStep = config['simulation']['moveStep']

        if 'md_reporters' in config.keys():
            # Returns a list of Reporter objects, overwrites the configuration parameters
            md_reporter_cfg = reporters.ReporterConfig(outfname, config['md_reporters'], logger)
            config['md_reporters'] = md_reporter_cfg.makeReporters()
            if md_reporter_cfg.trajectory_interval:
                config['simulation']['md_trajectory_interval'] = md_reporter_cfg.trajectory_interval
        else:
            logger.warn('Configuration for MD reporters were not set.')

        # Configure the NCMC simulation reporters
        if 'ncmc_reporters' in config.keys():

            #Update the reporter parameters with the proper NCMC steps
            for rep in config['ncmc_reporters'].keys():

                if 'totalSteps' in config['ncmc_reporters'][rep].keys():
                    config['ncmc_reporters'][rep]['totalSteps'] = nstepsNC

                #If -1 is given in frame_indices, record at the last frame
                #If 0.5 is given in frame_indices, record at the midpoint/movestep
                if 'frame_indices' in config['ncmc_reporters'][rep].keys():
                    frame_indices = config['ncmc_reporters'][rep]['frame_indices']
                    frame_indices = [moveStep if x == 0.5 else x for x in frame_indices]
                    frame_indices = [nstepsNC if x == -1 else x for x in frame_indices]
                    config['ncmc_reporters'][rep]['frame_indices'] = frame_indices

            ncmc_reporter_cfg = reporters.ReporterConfig(outfname + '-ncmc', config['ncmc_reporters'], logger)
            config['ncmc_reporters'] = ncmc_reporter_cfg.makeReporters()
        else:
            logger.warn('Configuration for NCMC reporters were not set.')

        return config

    @staticmethod
    def set_Parameters(config):
        """
        MAIN execution function for updating/correcting (placing units) in the config
        """
        try:
            # Set top level configuration parameters
            config = Settings.set_Output(config)
            config = Settings.set_Logger(config)
            if 'structure' in config:
                config = Settings.set_Structure(config)
                Settings.check_SystemModifications(config)
            config = Settings.set_Units(config)
            config = Settings.set_Apps(config)
            config = Settings.set_ncmcSteps(config)
            config = Settings.set_Reporters(config)

        except Exception as e:
            print('config', config)
            #config['Logger'].exception(e)
            raise e

        return config

    def asDict(self):
        return self.config

    def asOrderedDict(self):
        from collections import OrderedDict
        return OrderedDict(sorted(self.config.items(), key=lambda t: t[0]))

    def asYAML(self):
        return yaml.dump(self.config)

    def asJSON(self, pprint=False):
        if pprint:
            return json.dumps(self.config, sort_keys=True, indent=2, skipkeys=True, default=str)
        return json.dumps(self.config, default=str)