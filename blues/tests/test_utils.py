import pytest
import parmed
import fnmatch
import numpy
from blues.systemfactories import *
from simtk.openmm import app
from simtk import unit


@pytest.fixture(scope='session')
def structure():
    # Load the waterbox with toluene into a structure.
    prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
    inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
    structure = parmed.load_file(prmtop, xyz=inpcrd)
    return structure

@pytest.fixture(scope='session')
def tol_atom_indices(structure):
    atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
    return atom_indices


def test_atom_selections(structure, tol_atom_indices):
    atom_indices = utils.amber_selection_to_atomidx(structure, ':LIG')

    print('Testing AMBER selection parser')
    assert isinstance(atom_indices, list)
    assert len(atom_indices) == len(tol_atom_indices)

def test_atomidx_to_atomlist(structure, tol_atom_indices):
    print('Testing atoms from AMBER selection with parmed.Structure')
    atom_list = utils.atomidx_to_atomlist(structure, tol_atom_indices)
    atom_selection = [structure.atoms[i] for i in tol_atom_indices]
    assert atom_selection == atom_list

def test_get_masses(structure, tol_atom_indices):
    print('Testing get masses from a Topology')
    masses, totalmass = utils.getMasses(tol_atom_indices, structure.topology)
    total = numpy.sum(numpy.vstack(masses))
    assert total == totalmass._value

def test_get_center_of_mass(structure, tol_atom_indices):
    print('Testing get center of mass')
    masses, totalmass = utils.getMasses(tol_atom_indices, structure.topology)
    coordinates = numpy.array(structure.positions._value, numpy.float32)[tol_atom_indices]
    com = utils.getCenterOfMass(coordinates, masses)
    assert len(com) == 3
