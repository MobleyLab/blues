"""
utils.py: Provides a host of utility functions for the BLUES engine.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function
import os, copy, yaml, logging, sys
import mdtraj
from simtk import unit
from blues import utils
from blues import reporters
from math import floor, ceil
from simtk.openmm import app

def startup(config):
    def load_yaml(yaml_file):
        #Parse input parameters from YAML
        with open(yaml_file, 'r') as stream:
            try:
                opt = (yaml.load(stream))
            except Exception as exc:
                print (exc)
        return opt

    def set_parameters(opt):
        #Set file paths
        try:
            output_dir = opt['options']['output_dir']
        except Exception as exc:
            output_dir = '.'
        outfname = os.path.join(output_dir, opt['options']['outfname'])
        opt['simulation']['outfname'] = outfname

        #Initialize root Logger module
        level = opt['options']['logger_level'].upper()
        if level == 'DEBUG':
            #Add verbosity if logging is set to DEBUG
            opt['options']['verbose'] = True
            opt['system']['verbose'] = True
            opt['simulation']['verbose'] = True
        else:
            opt['options']['verbose'] = False
            opt['system']['verbose'] = False
            opt['simulation']['verbose'] = False


        level = eval("logging.%s" % level)
        logger = reporters.init_logger(logging.getLogger(), level, outfname)
        opt['Logger'] = logger

        #Ensure proper units
        try:
            opt['simulation']['nstepsNC'], opt['simulation']['integration_steps'] = calcNCMCSteps(logger=logger, **opt['simulation'])
            opt['system'] = add_units(opt['system'], logger)
            opt['simulation'] = add_units(opt['simulation'], logger)
            opt['freeze'] = add_units(opt['freeze'], logger)
        except:
            print(sys.exc_info()[0])
            raise


        return opt

    def add_units(opt, logger):
        #for system setup portion
        #set unit defaults to OpenMM defaults
        unit_options = {'nonbondedCutoff':unit.angstroms,
                        'switchDistance':unit.angstroms,
                        'implicitSolventKappa':unit.angstroms,
                        'implicitSolventSaltConc':unit.mole/unit.liters,
                        'temperature':unit.kelvins,
                        'hydrogenMass':unit.daltons,
                        'dt':unit.picoseconds,
                        'friction':1/unit.picoseconds,
                        'freeze_distance': unit.angstroms,
                        'pressure': unit.atmospheres
                        }

        app_options = ['nonbondedMethod', 'constraints', 'implicitSolvent']
        scalar_options = ['soluteDielectric', 'solvent', 'ewaldErrorTolerance']
        bool_options = ['rigidWater', 'useSASA', 'removeCMMotion', 'flexibleConstraints', 'verbose',
                        'splitDihedrals']

        combined_options = list(unit_options.keys()) + app_options + scalar_options + bool_options
        for sel in opt.keys():
            if sel in combined_options:
                if sel in unit_options:
                    #if the value requires units check that it has units
                    #if it doesn't assume default units are used
                    if opt[sel] is None:
                        opt[sel] = None
                    else:
                        try:
                            opt[sel]._value
                        except:
                            logger.warn("Units for '{} = {}' not specified. Setting units to '{}'".format(sel, opt[sel], unit_options[sel]))
                            opt[sel] = opt[sel]*unit_options[sel]
                #if selection requires an OpenMM evaluation do it here
                elif sel in app_options:
                    try:
                        opt[sel] = eval("app.%s" % opt[sel])
                    except:
                        #if already an app object we can just pass
                        pass
                #otherwise just take the value as is, should just be a bool or float
                else:
                    pass
        return opt

    def calcNCMCSteps(total_steps, nprop, prop_lambda, logger, **kwargs):
        if (total_steps % 2) != 0:
           raise Exception('`total_steps = %i` must be even for symmetric protocol.' % (total_steps))

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
        logger.warn('NCMC protocol will consist of {} lambda switching steps and {} total integration steps'.format(nstepsNC, calc_total))
        return nstepsNC, calc_total

    #Parse YAML into dict
    if config.endswith('.yaml'):
        config = load_yaml(config)

    #Parse the options dict
    if type(config) is dict:
        opt = set_parameters(config)

    return opt

def zero_masses(system, atomList=None):
    """
    Zeroes the masses of specified atoms to constrain certain degrees of freedom.
    Arguments
    ---------
    system: simtk.openmm.system
        system to zero masses
    atomList: list of ints
        atom indicies to zero masses
    """
    for index in (atomList):
        system.setParticleMass(index, 0*unit.daltons)
    return system


def atomIndexfromTop(resname, topology):
    """
    Get atom indices of a ligand from OpenMM Topology.
    Arguments
    ---------
    resname: str
        resname that you want to get the atom indicies for (ex. 'LIG')
    topology: str, optional, default=None
        path of topology file. Include if the topology is not included
        in the coord_file
    Returns
    -------
    lig_atoms : list of ints
        list of atoms in the coordinate file matching lig_resname
    """
    lig_atoms = []
    for atom in topology.atoms():
        if str(resname) in atom.residue.name:
            lig_atoms.append(atom.index)
    return lig_atoms


def get_data_filename(package_root, relative_path):
    """Get the full path to one of the reference files in testsystems.
    In the source distribution, these files are in ``blues/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.
    Adapted from:
    https://github.com/open-forcefield-group/smarty/blob/master/smarty/utils.py
    Parameters
    ----------
    package_root : str
        Name of the included/installed python package
    relative_path: str
        Path to the file within the python package
    """

    from pkg_resources import resource_filename
    fn = resource_filename(package_root, os.path.join(relative_path))
    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)
    return fn
