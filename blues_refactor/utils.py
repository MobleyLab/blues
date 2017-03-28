"""
utils.py: Provides a host of utility functions for the BLUES engine.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function
import os, copy
import mdtraj

def rand_rotation_matrix():
    """
    Creates a uniform random rotation matrix
    Returns
    -------
    matrix_out: 3x3 np.array
        random rotation matrix
    """
    rand_quat = mdtraj.utils.uniform_quaternion()
    matrix_out = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
    return matrix_out

def get_particle_masses(self, system, residueList=None, set_self=True):
    """
    Finds the mass of each particle given by residueList and returns
    a list of those particle masses as well as the total mass. If
    set_self=True, sets corresponding SimNCMC attributes as well as
    returning them.
    Arguments
    ---------
    system: simtk.openmm.system
        Openmm system object containing the particles of interest
    residueList: list of ints
        particle indices to find the masses of
    set_self: boolean
        if true, sets self.total_mass and self.mass_list to the
        outputs of this function
    """
    if residueList == None:
        residueList = self.residueList
    mass_list = []
    total_mass = 0*unit.dalton
    for index in residueList:
        mass = system.getParticleMass(int(index))
        total_mass = total_mass + mass
        mass_list.append([mass])
    total_mass = np.sum(mass_list)
    mass_list = np.asarray(mass_list)
    mass_list.reshape((-1,1))
    total_mass = np.array(total_mass)
    total_mass = np.sum(mass_list)
    temp_list = np.zeros((len(residueList), 1))
    for index in range(len(residueList)):
        mass_list[index] = (np.sum(mass_list[index])).value_in_unit(unit.daltons)
    mass_list =  mass_list*unit.daltons
    if set_self == True:
        self.total_mass = total_mass
        self.mass_list = mass_list
    return total_mass, mass_list

def zero_masses(self, system, atomList=None):
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


def calculate_com(pos_state, total_mass, mass_list, residueList, rotate=False):
    """
    This function calculates the com of specified residues and optionally
    rotates them around the center of mass.
    Arguments
    ---------
    total_mass: simtk.unit.quantity.Quantity in units daltons
        contains the total masses of the particles for COM calculation
    mass_list:  nx1 np.array in units daltons,
        contains the masses of the particles for COM calculation
    pos_state:  nx3 np. array in units.nanometers
        returned from state.getPositions
    residueList: list of int,
        list of atom indicies which you'll calculate the total com for
    rotate: boolean
        if True, rotates center of mass by random rotation matrix,
        else returns center of mass coordiantes
    Returns
    -------
    if rotate==True
    rotation : nx3 np.array in units.nm
        positions of ligand with or without random rotation (depending on rotate)
    if rotate==False
    com_coord: 1x3 np.array in units.nm
        position of the center of mass coordinate
    """

    #choose ligand indicies
    copy_orig = copy.deepcopy(pos_state)

    lig_coord = np.zeros((len(residueList), 3))
    for index, resnum in enumerate(residueList):
        lig_coord[index] = copy_orig[resnum]
    lig_coord = lig_coord*unit.nanometers
    copy_coord = copy.deepcopy(lig_coord)

    #mass corrected coordinates (to find COM)
    mass_corrected = mass_list / total_mass * copy_coord
    sum_coord = mass_corrected.sum(axis=0).value_in_unit(unit.nanometers)
    com_coord = [0.0, 0.0, 0.0]*unit.nanometers

    #units are funky, so do this step to get them to behave right
    for index in range(3):
        com_coord[index] = sum_coord[index]*unit.nanometers

    if rotate ==True:
        for index in range(3):
            lig_coord[:,index] = lig_coord[:,index] - com_coord[index]
        #multiply lig coordinates by rot matrix and add back COM translation from origin
        rotation =  np.dot(lig_coord.value_in_unit(unit.nanometers), rand_rotation_matrix())*unit.nanometers
        rotation = rotation + com_coord
        return rotation
    else:
    #remove COM from ligand coordinates to then perform rotation
        return com_coord            #remove COM from ligand coordinates to then perform rotation

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
