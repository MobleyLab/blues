"""
utils.py: Provides a host of utility functions for the BLUES engine.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function


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
