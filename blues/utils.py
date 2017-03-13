from __future__ import print_function
import os

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
