"""
Provides a host of utility functions for the BLUES engine.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function
import os, logging, sys
import parmed
from simtk import unit, openmm
from math import floor, ceil
from platform import uname
logger = logging.getLogger(__name__)


def saveSimulationFrame(simulation, outfname):
    """Extracts a ParmEd structure and writes the frame given
    an OpenMM Simulation object.

    Parameters
    ----------
    simulation : openmm.Simulation
        The OpenMM Simulation to write a frame from.
    outfname : str
        The output file name to save the simulation frame from. Supported
        extensions:

        - PDB (.pdb, pdb)
        - PDBx/mmCIF (.cif, cif)
        - PQR (.pqr, pqr)
        - Amber topology file (.prmtop/.parm7, amber)
        - CHARMM PSF file (.psf, psf)
        - CHARMM coordinate file (.crd, charmmcrd)
        - Gromacs topology file (.top, gromacs)
        - Gromacs GRO file (.gro, gro)
        - Mol2 file (.mol2, mol2)
        - Mol3 file (.mol3, mol3)
        - Amber ASCII restart (.rst7/.inpcrd/.restrt, rst7)
        - Amber NetCDF restart (.ncrst, ncrst)

    """
    topology = simulation.topology
    system = simulation.context.getSystem()
    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getParameters=True,
        getForces=True,
        getParameterDerivatives=True,
        getEnergy=True,
        enforcePeriodicBox=True)

    # Generate the ParmEd Structure
    structure = parmed.openmm.load_topology(topology, system, xyz=state.getPositions())

    structure.save(outfname, overwrite=True)
    logger.info('\tSaving Frame to: %s' % outfname)


def print_host_info(simulation):
    """Prints hardware related information for the openmm.Simulation

    Parameters
    ----------
    simulation : openmm.Simulation
        The OpenMM Simulation to write a frame from.

    """
    # OpenMM platform information
    mmver = openmm.version.version
    mmplat = simulation.context.getPlatform()
    msg = 'OpenMM({}) simulation generated for {} platform\n'.format(mmver, mmplat.getName())

    # Host information
    for k, v in uname()._asdict().items():
        msg += '{} = {} \n'.format(k, v)

    # Platform properties
    for prop in mmplat.getPropertyNames():
        val = mmplat.getPropertyValue(simulation.context, prop)
        msg += '{} = {} \n'.format(prop, val)
    logger.info(msg)


def calculateNCMCSteps(nstepsNC=0, nprop=1, propLambda=0.3, **kwargs):
    """
    Calculates the number of NCMC switching steps.

    Parameters
    ----------
    nstepsNC : int
        The number of NCMC switching steps
    nprop : int, default=1
        The number of propagation steps per NCMC switching steps
    propLambda : float, default=0.3
        The lambda values in which additional propagation steps will be added
        or 0.5 +/- propLambda. If 0.3, this will add propgation steps at lambda
        values 0.2 to 0.8.

    """
    ncmc_parameters = {}
    # Make sure provided NCMC steps is even.
    if (nstepsNC % 2) != 0:
        rounded_val = nstepsNC & ~1
        msg = 'nstepsNC=%i must be even for symmetric protocol.' % (nstepsNC)
        if rounded_val:
            logger.warning(msg + ' Setting to nstepsNC=%i' % rounded_val)
            nstepsNC = rounded_val
        else:
            logger.error(msg)
            sys.exit(1)
    # Calculate the total number of lambda switching steps
    lambdaSteps = nstepsNC / (2 * (nprop * propLambda + 0.5 - propLambda))
    if int(lambdaSteps) % 2 == 0:
        lambdaSteps = int(lambdaSteps)
    else:
        lambdaSteps = int(lambdaSteps) + 1

    # Calculate number of lambda steps inside/outside region with extra propgation steps
    in_portion = (propLambda) * lambdaSteps
    out_portion = (0.5 - propLambda) * lambdaSteps
    in_prop = int(nprop * (2 * floor(in_portion)))
    out_prop = int((2 * ceil(out_portion)))
    propSteps = int(in_prop + out_prop)

    if propSteps != nstepsNC:
        logger.warn("nstepsNC=%s is incompatible with prop_lambda=%s and nprop=%s." % (nstepsNC, propLambda, nprop))
        logger.warn("Changing NCMC protocol to %s lambda switching within %s total propagation steps." % (lambdaSteps,
                                                                                                          propSteps))
        nstepsNC = lambdaSteps

    moveStep = int(nstepsNC / 2)
    ncmc_parameters = {
        'nstepsNC': nstepsNC,
        'propSteps': propSteps,
        'moveStep': moveStep,
        'nprop': nprop,
        'propLambda': propLambda
    }

    return ncmc_parameters


def check_amber_selection(structure, selection):
    """
    Given a AmberMask selection (str) for selecting atoms to freeze or restrain,
    check if it will actually select atoms. If the selection produces None,
    suggest valid residues or atoms.

    Parameters
    ----------
    structure : parmed.Structure
        The structure of the simulated system
    selection : str
        The selection string uses Amber selection syntax to select atoms to be
        restrained/frozen during simulation.
    logger : logging.Logger
        Records information or streams to terminal.

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
            logger.error("'{}' was not a valid Amber selection. Valid atoms: {}".format(selection, atom_set))
        sys.exit(1)


def parse_unit_quantity(unit_quantity_str):
    """
    Utility for parsing parameters from the YAML file that require units.

    Parameters
    ----------
    unit_quantity_str : str
        A string specifying a quantity and it's units. i.e. '3.024 * daltons'

    Returns
    -------
    unit_quantity : simtk.unit.Quantity
        i.e `unit.Quantity(3.024, unit=dalton)`

    """
    value, u = unit_quantity_str.replace(' ', '').split('*')
    if '/' in u:
        u = u.split('/')
        return unit.Quantity(float(value), eval('%s/unit.%s' % (u[0], u[1])))
    return unit.Quantity(float(value), eval('unit.%s' % u))


def zero_masses(system, atomList=None):
    """
    Zeroes the masses of specified atoms to constrain certain degrees of freedom.

    Arguments
    ---------
    system : penmm.System
        system to zero masses
    atomList : list of ints
        atom indicies to zero masses

    Returns
    -------
    system : openmm.System
        The modified system with massless atoms.
        
    """
    for index in (atomList):
        system.setParticleMass(index, 0 * unit.daltons)
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
    relative_path : str
        Path to the file within the python package

    Returns
    -------
    fn : str
        Full path to file
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
    switching_values : list
        A list of lambda values decreasing from 1 to 0.
    steps : int
        The number of steps wanted for the tabulated function.
    switching_types : str, optional, default='auto'
        The type of lambda switching the `switching_values` corresponds to, either 'auto', 'electrostatics',
        or 'sterics'. If 'electrostatics' this assumes the inital value immediately decreases from 1.
        If 'sterics' this assumes the inital values stay at 1 for some period.
        If 'auto' this function tries to guess the switching_types based on this, based on typical
        lambda protocols turning off the electrostatics completely, before turning off sterics.
    kind : str, optional, default='cubic'
        The kind of interpolation that should be performed (using scipy.interpolate.interp1d) to
        define the lines between the points of switching_values.

    Returns
    -------
    tab_steps : list or simtk.openmm.openmm.Discrete1DFunction
        List of length `steps` that corresponds to the tabulated-friendly version of the input switching_values.
        If return-tab_function=True

    Examples
    --------
    >>> from simtk.openmm.openmm import Continuous1DFunction, Discrete1DFunction
    >>> sterics = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.8848447462380346,
                    0.8428373352131427, 0.7928373352131427, 0.7490146003095886, 0.6934088361682191,
                    0.6515123083157823, 0.6088924298371354, 0.5588924298371354, 0.5088924298371353,
                    0.4649556683144045, 0.4298606804827029, 0.3798606804827029, 0.35019373288005945,
                    0.31648339779024653, 0.2780498882483276, 0.2521302239477468, 0.23139484523965026,
                    0.18729812232625365, 0.15427643961733822, 0.12153116162972155,
                    0.09632462702545555, 0.06463743549588846, 0.01463743549588846,
                    0.0]

    >>> statics = [1.0, 0.8519493439593149, 0.7142750443470669,
                    0.5385929179832776, 0.3891972949356391, 0.18820309596839535, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> statics_tab = spreadLambdaProtocol(statics, opt['nstepsNC'], switching_types='auto')
    >>> sterics_tab = spreadLambdaProtocol(sterics, opt['nstepsNC'], switching_types='sterics')

    >>> # Assuming some Context already exists:
    >>> context._integrator.addTabulatedFunction( 'sterics_tab', sterics_tab)
    >>> context._integrator.addTabulatedFunction( 'electrostatics_tab', statics_tab)


    """
    #In order to symmetrize the interpolation of turning lambda on/off use the 1.0/0.0 values as a guide
    one_counts = switching_values.count(1.0)
    counts = switching_values.index(0.0)
    #symmetrize the lambda values so that the off state is at the middle
    switching_values = switching_values + (switching_values)[::-1][1:]
    #find the original scaling of lambda, from 0 to 1
    x = [float(j) / float(len(switching_values) - 1) for j in range(len(switching_values))]
    #find the new scaling of lambda, accounting for the number of steps
    xsteps = np.arange(0, 1. + 1. / float(steps), 1. / float(steps))
    #interpolate to find the intermediate values of lambda
    interpolate = interp1d(x, switching_values, kind=kind)

    #next we check if we're doing a electrostatic or steric protocol
    #interpolation doesn't guarantee
    if switching_types == 'auto':
        if switching_values[1] == 1.0:
            switching_types = 'sterics'
        else:
            switching_types = 'electrostatics'
    if switching_types == 'sterics':
        tab_steps = [
            1.0 if (xsteps[i] < x[(one_counts - 1)] or xsteps[i] > x[-(one_counts)]) else j
            for i, j in enumerate(interpolate(xsteps))
        ]
    elif switching_types == 'electrostatics':
        tab_steps = [
            0.0 if (xsteps[i] > x[(counts)] and xsteps[i] < x[-(counts + 1)]) else j
            for i, j in enumerate(interpolate(xsteps))
        ]
    else:
        raise ValueError('`switching_types` should be either sterics or electrostatics, currently ' + switching_types)
    tab_steps = [j if i <= floor(len(tab_steps) / 2.) else tab_steps[(-i) - 1] for i, j in enumerate(tab_steps)]
    for i, j in enumerate(tab_steps):
        if j < 0.0 or j > 1.0:
            raise ValueError(
                'This function is not working properly.',
                'value %f at index %i is not bounded by 0.0 and 1.0 Please check if your switching_type is correct' %
                (j, i))
    if return_tab_function:
        tab_steps = Discrete1DFunction(tab_steps)
    return tab_steps
