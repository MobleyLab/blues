import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import parmed, math
from openmmtools import alchemy
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues import utils
import os, copy, yaml, logging, sys, itertools
import mdtraj
from blues import utils
from blues import reporters
from math import floor, ceil
from simtk.openmm import app
from blues.reporters import *


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
                config = yaml.safe_load(stream)
            except FileNotFoundError:
                raise FileNotFoundError
            except yaml.YAMLError as e:
                yaml_err = 'YAML parsing error in file: {}'.format(yaml_config)
                if hasattr(e, 'problem_mark'):
                    mark = e.problem_mark
                    print(yaml_err + '\nError on Line:{} Column:{}'.format(mark.line + 1, mark.column + 1))
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
                logger.error("'{}' was not a valid Amber selection. \n\tValid residue names: {}".format(selection, res_set))
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
        config['outfname'] = outfname
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
                try:
                    if param in config[setup_keys]:
                        user_input = config[setup_keys][param]

                        if '*' in str(user_input):
                            config[setup_keys][param] =  parse_unit_quantity(user_input)

                        #If not provided, set default units
                        else:
                            config['Logger'].warn("Units for '{} = {}' not specified. Setting units to '{}'".format(param, user_input, unit_type))
                            config[setup_keys][param] = user_input*unit_type

                except:
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
        return config

    def set_Reporters(config):
        """
        Initializes the openmm.Reporters for the simulations
        """

        if 'md_reporters' in config.keys():
            #Return and store the Reporter objects
            config['md_reporters']['Reporters'] = ReporterConfig(config['outfname'], config['md_reporters'], config['Logger']).reporters
        else:
            config['Logger'].warn('Configuration for MD reporters were not set.')

        #Configure the NCMC simulation reporters
        if 'ncmc_reporters' in config.keys():
            config['ncmc_reporters']['Reporters'] = ReporterConfig(config['outfname']+'-ncmc', config['ncmc_reporters'], config['Logger']).reporters
        else:
            config['Logger'].warn('Configuration for NCMC reporters were not set.')

        return config

    def set_Parameters(config):
        """
        MAIN execution function for updating/correcting (placing units) in the config
        """
        try:
            #Set top level configuration parameters
            config = set_Output(config)
            config = set_Logger(config)
            config['Structure'] = load_Structure(logger=config['Logger'], **config['structure'])
            config = set_Units(config)
            config = set_Apps(config)

            #Check Amber Selections
            if 'freeze' in config.keys():
                freeze_keys = ['freeze_center', 'freeze_solvent', 'freeze_selection']
                for sel in freeze_keys:
                    if sel in config['freeze']:
                        check_amber_selection(config['Structure'], config['freeze'][sel], config['Logger'])

            if 'restraints' in config.keys():
                check_amber_selection(config['Structure'], config['restraints']['selection'], config['Logger'])

            #Calculate NCMC steps with nprop
            config['simulation']['nstepsNC'], config['simulation']['integration_steps'] = calcNCMCSteps(logger=config['Logger'], **config['simulation'])

            config = set_Reporters(config)

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
    if yaml_config.endswith('.yaml'):
        config = load_yaml(yaml_config)

    #Parse the configions dict
    if type(config) is dict:
        config = set_Parameters(config)

    return config

class ReporterConfig:

    def __init__(self, outfname, reporter_config, logger=None):
        self._outfname = outfname
        self._cfg = reporter_config
        self._logger = logger
        self.reporters = self.get_Reporters()

    def make_StateReporter(self, outfname, reportInterval, step=True, time=True,
                 potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                 temperature=True, volume=False, density=False, separator='\t',
                 systemMass=None, energyUnit=unit.kilocalories_per_mole,
                 timeUnit=unit.picoseconds, volumeUnit=unit.angstroms**3,
                 densityUnit=unit.grams/unit.item/unit.milliliter, **kwargs):
        """
        This class acts as a state data reporter for OpenMM simulations, but it is a
        little more generalized. Notable differences are:

          -  It allows the units of the output to be specified, with defaults being
             those used in Amber (e.g., kcal/mol, angstroms^3, etc.)
          -  It will write to any object containing a 'write' method; not just
             files. This allows, for instance, writing to a GUI window that
             implements the desired 'write' attribute.

        Most of this code is copied from the OpenMM StateDataReporter class, with
        the above-mentioned changes made.

        Parameters
        ----------
        f : str or file-like
            Destination to write the state data (file name or file object)
        reportInterval : int
            Number of steps between state data reports
        step : bool, optional
            Print out the step number (Default True)
        time : bool, optional
            Print out the simulation time (Defaults True)
        potentialEnergy : bool, optional
            Print out the potential energy of the structure (Default True)
        kineticEnergy : bool, optional
            Print out the kinetic energy of the structure (Default True)
        totalEnergy : bool, optional
            Print out the total energy of the system (Default True)
        temperature : bool, optional
            Print out the temperature of the system (Default True)
        volume : bool, optional
            Print out the volume of the unit cell. If the system is not periodic,
            the value is meaningless (Default False)
        density : bool, optional
            Print out the density of the unit cell. If the system is not periodic,
            the value is meaningless (Default False)
        separator : str, optional
            The string to separate data fields (Default ',')
        systemMass : float, optional
            If not None, the density will be computed from this mass, since setting
            a mass to 0 is used to constrain the position of that particle. (Default
            None)
        energyUnit : unit, optional
            The units to print energies in (default unit.kilocalories_per_mole)
        timeUnit : unit, optional
            The units to print time in (default unit.picoseconds)
        volumeUnit : unit, optional
            The units print volume in (default unit.angstroms**3)
        densityUnit : unit, optional
            The units to print density in (default
            unit.grams/unit.item/unit.milliliter)
        """
        return parmed.openmm.reporters.StateDataReporter(outfname+'.ene', reportInterval, **kwargs)

    def make_NetCDF4Reporter(self, outfname, **kwargs):
        return NetCDF4Reporter(outfname+'.nc', **kwargs)

    def make_Rst7Reporter(self, outfname, reportInterval, write_multiple=False, netcdf=True, write_velocities=True, **kwargs):
        return parmed.openmm.reporters.RestartReporter(outfname+'.rst7', reportInterval,
                                                       write_multiple, netcdf, write_velocities,
                                                       **kwargs)
    def make_SpeedReporter(self, outfname, reportInterval, totalSteps, title='', **kwargs):
        return reporters.BLUESStateDataReporter(outfname, title=title,
                                     reportInterval=reportInterval,
                                      step=True, totalSteps=totalSteps,
                                      time=False, speed=True, progress=True, remainingTime=True)


    def make_ProgressReporter(self, outfname, reportInterval, totalSteps, potentialEnergy=True,
                 kineticEnergy=True, totalEnergy=True, temperature=True,
                 volume=True, density=False, systemMass=None, **kwargs):
        """
        A class that prints out a progress report of how much MD (or minimization)
        has been done, how fast the simulation is running, and how much time is left
        (similar to the mdinfo file in Amber)

        Parameters
        ----------
        f : str
            The file name of the progress report file (overwritten each time)
        reportInterval : int
            The step interval between which to write frames
        totalSteps : int
            The total number of steps that will be run in the simulation (used to
            estimate time remaining)
        potentialEnergy : bool, optional
            Whether to print the potential energy (default True)
        kineticEnergy : bool, optional
            Whether to print the kinetic energy (default True)
        totalEnergy : bool, optional
            Whether to print the total energy (default True)
        temperature : bool, optional
            Whether to print the system temperature (default True)
        volume : bool, optional
            Whether to print the system volume (default False)
        density : bool, optional
            Whether to print the system density (default False)
        systemMass : float or :class:`unit.Quantity`
            The mass of the system used when reporting density (useful in instances
            where masses are set to 0 to constrain their positions)

        See Also
        --------
        In addition to the above, ProgressReporter also accepts arguments for
        StateDataReporter
        """
        return parmed.openmm.reporters.ProgressReporter(outfname+'.prog', reportInterval,
                                                        totalSteps, **kwargs)

    def get_Reporters(self):
        reporters = []
        if 'state' in self._cfg.keys():
            reporters.append(self.make_StateReporter(self._outfname, **self._cfg['state']))

        if 'traj_netcdf' in self._cfg.keys():
            reporters.append(self.make_NetCDF4Reporter(self._outfname, **self._cfg['traj_netcdf']))

        #if 'traj_h5' in self._cfg.keys():
        #    reporters.append(self.make_NetCDF4Reporter(self._outfname, **self._cfg['traj_h5']))

        if 'restart' in self._cfg.keys():
            reporters.append(self.make_Rst7Reporter(self._outfname, **self._cfg['restart']))

        if 'progress' in self._cfg.keys():
            reporters.append(self.make_ProgressReporter(self._outfname, **self._cfg['progress']))

        if 'speed' in self._cfg.keys():
            reporters.append(self.make_SpeedReporter(self._logger, **self._cfg['speed']))

        return reporters
