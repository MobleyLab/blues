import pytest
import parmed
import fnmatch
import numpy
from openmmtools.cache import ContextCache
from openmmtools.states import ThermodynamicState
from blues.systemfactories import *
from simtk.openmm import app
from simtk import unit


@pytest.fixture(scope='session')
def system_cfg():
    system_cfg = {'nonbondedMethod': app.PME, 'nonbondedCutoff': 8.0 * unit.angstroms, 'constraints': app.HBonds}
    return system_cfg

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

@pytest.fixture(scope='function')
def system(structure, system_cfg):
    system = structure.createSystem(**system_cfg)
    return system

@pytest.fixture(scope='function')
def context(system, structure):
    context_cache = ContextCache()
    thermodynamic_state = ThermodynamicState(system, 300*unit.kelvin)
    context, integrator = context_cache.get_context(thermodynamic_state)
    context.setPositions(structure.positions)
    return context

### Utils ###
def test_amber_atom_selections(structure, tol_atom_indices):
    atom_indices = utils.amber_selection_to_atomidx(structure, ':LIG')

    print('Testing AMBER selection parser')
    assert isinstance(atom_indices, list)
    assert len(atom_indices) == len(tol_atom_indices)

def test_amber_selection_check(structure, caplog):
    print('Testing AMBER selection check')
    assert True == utils.check_amber_selection(structure, ':LIG')
    assert True == utils.check_amber_selection(structure, '@1')
    assert False == utils.check_amber_selection(structure, ':XYZ')
    assert False == utils.check_amber_selection(structure, '@999')

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

def test_print_host_info(context, caplog):
    print('Testing Host Printout')
    with caplog.at_level(logging.INFO):
        utils.print_host_info(context)
        assert 'version' in caplog.text

def test_saveContextFrame(context, structure, caplog):
    print('Testing Save Context Frame')
    filename = 'testContext.pdb'
    with caplog.at_level(logging.INFO):
        utils.saveContextFrame(context, structure.topology, filename)
        assert 'Saving Frame to' in caplog.text

### SystemFactories ###

def test_generateAlchSystem(structure, system, tol_atom_indices):
    # Create the OpenMM system
    print('Creating OpenMM Alchemical System')
    alch_system = generateAlchSystem(system, tol_atom_indices)

    # Check that we get an openmm.System
    assert isinstance(alch_system, openmm.System)

    # Check atoms in system is same in input parmed.Structure
    assert alch_system.getNumParticles() == len(structure.atoms)
    assert alch_system.getNumParticles() == system.getNumParticles()

    # Check customforces were added for the Alchemical system
    alch_forces = alch_system.getForces()
    alch_force_names = [force.__class__.__name__ for force in alch_forces]
    assert len(system.getForces()) < len(alch_forces)
    assert len(fnmatch.filter(alch_force_names, 'Custom*Force')) > 0

def test_restrain_postions(structure, system):
    print('Testing positional restraints')
    no_restr = system.getForces()

    md_system_restr = restrain_positions(structure, system, ':LIG')
    restr = md_system_restr.getForces()

    # Check that forces have been added to the system.
    assert len(restr) != len(no_restr)
    # Check that it has added the CustomExternalForce
    assert isinstance(restr[-1], openmm.CustomExternalForce)

def test_zero_masses(system, tol_atom_indices):
    print('Testing zero masses')
    masses = [system.getParticleMass(i)._value for i in tol_atom_indices]
    massless_system = zero_masses(system, tol_atom_indices)
    massless = [massless_system.getParticleMass(i)._value for i in tol_atom_indices]

    # Check that masses have been zeroed
    assert massless != masses
    assert all(m == 0 for m in massless)

def test_freeze_atoms(structure, system, tol_atom_indices):
    print('Testing freeze_atoms')
    masses = [system.getParticleMass(i)._value for i in tol_atom_indices]
    frzn_lig = freeze_atoms(structure, system, ':LIG')
    massless = [frzn_lig.getParticleMass(i)._value for i in tol_atom_indices]

    # Check that masses have been zeroed
    assert massless != masses
    assert all(m == 0 for m in massless)


def test_freeze_radius(structure, system_cfg, caplog):
    print('Testing freeze_radius')
    freeze_cfg = {'freeze_center': ':LIG', 'freeze_solvent': ':Cl-', 'freeze_distance': 100.0 * unit.angstroms}
    # Setup toluene-T4 lysozyme system
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    structure = parmed.load_file(prmtop, xyz=inpcrd)
    atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
    system = structure.createSystem(**system_cfg)

    # Freeze everything around the binding site
    frzn_sys = freeze_radius(structure, system, **freeze_cfg)

    # Check that the ligand has NOT been frozen
    lig_masses = [system.getParticleMass(i)._value for i in atom_indices]
    assert all(m != 0 for m in lig_masses)

    # Check that the binding site has NOT been frozen
    selection = "({freeze_center}<:{freeze_distance._value})&!({freeze_solvent})".format(**freeze_cfg)
    site_idx = utils.amber_selection_to_atomidx(structure, selection)
    masses = [frzn_sys.getParticleMass(i)._value for i in site_idx]
    assert all(m != 0 for m in masses)

    # Check that the selection has been frozen
    # Invert that selection to freeze everything but the binding site.
    freeze_idx = set(range(system.getNumParticles())) - set(site_idx)
    massless = [frzn_sys.getParticleMass(i)._value for i in freeze_idx]
    assert all(m == 0 for m in massless)

    # Check number of frozen atoms is equal to center
    system = structure.createSystem(**system_cfg)
    with caplog.at_level(logging.ERROR):
         frzn_all = freeze_radius(structure, system, freeze_solvent=':WAT', freeze_distance=1*unit.angstrom)
         assert 'ERROR' in caplog.text

    # Check all frozen error
    system = structure.createSystem(**system_cfg)
    with caplog.at_level(logging.ERROR):
         frzn_all = freeze_radius(structure, system, freeze_solvent=':WAT', freeze_distance=0*unit.angstrom)
         assert 'ERROR' in caplog.text

    # Check freeze threshold error
    system = structure.createSystem(**system_cfg)
    with caplog.at_level(logging.ERROR):
         frzn_all = freeze_radius(structure, system, freeze_solvent=':WAT', freeze_distance=2*unit.angstrom)
         assert 'ERROR' in caplog.text

    # Check freeze threshold error
    system = structure.createSystem(**system_cfg)
    with caplog.at_level(logging.WARNING):
         frzn_sys = freeze_radius(structure, system, freeze_solvent=':Cl-', freeze_distance=20*unit.angstrom)
         assert 'WARNING' in caplog.text



def test_addBarostat(system):
    print('Testing MonteCarloBarostat')
    forces = system.getForces()
    npt_system = addBarostat(system)
    npt_forces = npt_system.getForces()

    #Check that forces have been added to the system.
    assert len(forces) != len(npt_forces)
    #Check that it has added the MonteCarloBarostat
    assert isinstance(npt_forces[-1], openmm.MonteCarloBarostat)
