"""
utils.py: Provides a host of utility functions for the BLUES engine.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function
import os, copy, yaml, logging, sys, itertools
import mdtraj
from simtk import unit
from blues import utils
from blues import reporters
from math import floor, ceil

def ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda x, y: y - x ):
        b = list(b)
        yield b[0][1], b[-1][1]

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


def spreadLambdaProtocol(switching_values, steps, switching_types='auto', kind='cubic', return_tab_function=True):
    """
    Takes a list of lambda values (either for sterics or electrostatics) and transforms that list
    to be spread out over a given `steps` range to be easily compatible with the OpenMM Discrete1DFunction
    tabulated function.

    Parameters
    ----------
    switching_values: list
        A list of lambda values decreasing from 1 to 0.
    steps: int
        The number of steps wanted for the tabulated function.
    switching_types: str, optional, default='auto'
        The type of lambda switching the `switching_values` corresponds to, either 'auto', 'electrostatics',
        or 'sterics'. If 'electrostatics' this assumes the inital value immediately decreases from 1.
        If 'sterics' this assumes the inital values stay at 1 for some period.
        If 'auto' this function tries to guess the switching_types based on this, based on typical
        lambda protocols turning off the electrostatics completely, before turning off sterics.
    kind: str, optional, default='cubic'
        The kind of interpolation that should be performed (using scipy.interpolate.interp1d) to
        define the lines between the points of switching_values.
    Returns
    -------
    tab_steps: list or simtk.openmm.openmm.Discrete1DFunction
        List of length `steps` that corresponds to the tabulated-friendly version of the input switching_values.
        If return-tab_function=True

    Ex.
    from simtk.openmm.openmm import Continuous1DFunction, Discrete1DFunction
    sterics = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.8848447462380346,
                        0.8428373352131427, 0.7928373352131427, 0.7490146003095886, 0.6934088361682191,
                        0.6515123083157823, 0.6088924298371354, 0.5588924298371354, 0.5088924298371353,
                        0.4649556683144045, 0.4298606804827029, 0.3798606804827029, 0.35019373288005945,
                        0.31648339779024653, 0.2780498882483276, 0.2521302239477468, 0.23139484523965026,
                        0.18729812232625365, 0.15427643961733822, 0.12153116162972155,
                        0.09632462702545555, 0.06463743549588846, 0.01463743549588846,
                        0.0]

    statics = [1.0, 0.8519493439593149, 0.7142750443470669,
                        0.5385929179832776, 0.3891972949356391, 0.18820309596839535, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    statics_tab = spreadLambdaProtocol(statics, opt['nstepsNC'], switching_types='auto')
    sterics_tab = spreadLambdaProtocol(sterics, opt['nstepsNC'], switching_types='sterics')
    Assuming some Context already exists:
    context._integrator.addTabulatedFunction( 'sterics_tab', sterics_tab)
    context._integrator.addTabulatedFunction( 'electrostatics_tab', statics_tab)


    """
    #In order to symmetrize the interpolation of turning lambda on/off use the 1.0/0.0 values as a guide
    one_counts = switching_values.count(1.0)
    counts = switching_values.index(0.0)
    #symmetrize the lambda values so that the off state is at the middle
    switching_values = switching_values + (switching_values)[::-1][1:]
    #find the original scaling of lambda, from 0 to 1
    x = [float(j) / float(len(switching_values)-1) for j in range(len(switching_values))]
    #find the new scaling of lambda, accounting for the number of steps
    xsteps = np.arange(0, 1.+1./float(steps), 1./float(steps))
    #interpolate to find the intermediate values of lambda
    interpolate = interp1d(x, switching_values, kind=kind)

    #next we check if we're doing a electrostatic or steric protocol
    #interpolation doesn't guarantee
    if switching_types == 'auto':
        if switching_values[1] == 1.0:
            switching_types = 'sterics'
        else:
            switching_types = 'electrostatics'
    if switching_types== 'sterics':
        tab_steps = [1.0 if (xsteps[i] < x[(one_counts-1)] or xsteps[i] > x[-(one_counts)]) else j for i, j in enumerate(interpolate(xsteps))]
    elif switching_types == 'electrostatics':
        tab_steps = [0.0 if (xsteps[i] > x[(counts)] and xsteps[i] < x[-(counts+1)]) else j for i, j in enumerate(interpolate(xsteps))]
    else:
        raise ValueError('`switching_types` should be either sterics or electrostatics, currently '+switching_types)
    tab_steps = [j if i <= floor(len(tab_steps)/2.) else tab_steps[(-i)-1] for i, j in enumerate(tab_steps)]
    for i, j in enumerate(tab_steps):
        if j<0.0 or j>1.0:
            raise ValueError('This function is not working properly.',
            'value %f at index %i is not bounded by 0.0 and 1.0 Please check if your switching_type is correct'%(j,i))
    if return_tab_function:
        tab_steps = Discrete1DFunction(tab_steps)
    return tab_steps
