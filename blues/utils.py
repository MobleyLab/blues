"""
Provides a host of utility functions for the BLUES engine.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

import logging
import os
import sys
import numpy as np
from math import ceil, floor
from platform import uname

import parmed
from simtk import openmm, unit

logger = logging.getLogger(__name__)


def amber_selection_to_atomidx(structure, selection):
    """
    Converts AmberMask selection [amber-syntax]_ to list of atom indices.

    Parameters
    ----------
    structure : parmed.Structure()
        Structure of the system, used for atom selection.
    selection : str
        AmberMask selection that gets converted to a list of atom indices.

    Returns
    -------
    mask_idx : list of int
        List of atom indices.

    References
    ----------
    .. [amber-syntax] J. Swails, ParmEd Documentation (2015). http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax

    """
    mask = parmed.amber.AmberMask(structure, str(selection))
    mask_idx = [i for i in mask.Selected()]
    return mask_idx

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

    try:
        mask = parmed.amber.AmberMask(structure, str(selection))
        mask_idx = [i for i in mask.Selected()]
    except:
        mask_idx = []
    if not mask_idx:
        if ':' in selection:
            res_set = set(residue.name for residue in structure.residues)
            logger.error("'{}' was not a valid Amber selection. \n\tValid residue names: {}".format(
                selection, res_set))
        elif '@' in selection:
            atom_set = set(atom.name for atom in structure.atoms)
            logger.error("'{}' was not a valid Amber selection. Valid atoms: {}".format(selection, atom_set))
        return False
    else:
        return True

def atomidx_to_atomlist(structure, mask_idx):
    """
    Goes through the structure and matches the previously selected atom
    indices to the atom type.

    Parameters
    ----------
    structure : parmed.Structure()
        Structure of the system, used for atom selection.
    mask_idx : list of int
        List of atom indices.

    Returns
    -------
    atom_list : list of atoms
        The atoms that were previously selected in mask_idx.
    """
    atom_list = []
    for i, at in enumerate(structure.atoms):
        if i in mask_idx:
            atom_list.append(structure.atoms[i])
    logger.debug('\nFreezing {}'.format(atom_list))
    return atom_list


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

def getMasses(atom_subset, topology):
    """
    Returns a list of masses of the specified ligand atoms.
    Parameters
    ----------
    topology: parmed.Topology
       ParmEd topology object containing atoms of the system.
    Returns
    -------
    masses: 1xn numpy.array * simtk.unit.dalton
       array of masses of len(self.atom_indices), denoting
       the masses of the atoms in self.atom_indices
    totalmass: float * simtk.unit.dalton
       The sum of the mass found in masses
    """
    if isinstance(atom_subset, slice):
       atoms = list(topology.atoms())[atom_subset]
    else:
       atoms = [ list(topology.atoms())[i] for i in atom_subset]
    masses = unit.Quantity(np.zeros([int(len(atoms)), 1], np.float32), unit.dalton)
    for idx, atom in enumerate(atoms):
       masses[idx] = atom.element._mass
    totalmass = masses.sum()
    return masses, totalmass

def getCenterOfMass(positions, masses):
    """Returns the calculated center of mass of the ligand as a numpy.array
    Parameters
    ----------
    positions: nx3 numpy array * simtk.unit compatible with simtk.unit.nanometers
       ParmEd positions of the atoms to be moved.
    masses : numpy.array
       numpy.array of particle masses
    Returns
    -------
    center_of_mass: numpy array * simtk.unit compatible with simtk.unit.nanometers
       1x3 numpy.array of the center of mass of the given positions
    """
    if isinstance(positions, unit.Quantity):
        coordinates = np.asarray(positions._value, np.float32)
        pos_unit = positions.unit
    else:
        coordinates = np.asarray(positions, np.float32)
        pos_unit = unit.angstroms
    center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * pos_unit
    return center_of_mass

def saveContextFrame(context, topology, outfname):
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
    system = context.getSystem()
    state = context.getState(
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

def print_host_info(context):
    """Prints hardware related information for the openmm.Simulation

    Parameters
    ----------
    simulation : openmm.Simulation
        The OpenMM Simulation to write a frame from.

    """
    # OpenMM platform information
    mmver = openmm.version.version
    mmplat = context.getPlatform()
    msg = 'OpenMM({}) simulation generated for {} platform\n'.format(mmver, mmplat.getName())

    # Host information
    for k, v in uname()._asdict().items():
        msg += '{} = {} \n'.format(k, v)

    # Platform properties
    for prop in mmplat.getPropertyNames():
        val = mmplat.getPropertyValue(context, prop)
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
