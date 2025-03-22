import pytest
import parmed
import logging
import glob
import os
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.moves import RandomLigandRotationMove, MoveEngine
from blues.reporters import (BLUESStateDataReporter, NetCDF4Reporter, ReporterConfig, init_logger)
from blues.settings import Settings
import openmm
from openmm import app
import numpy as np
import mdtraj as md
from collections import Counter

logger = logging.getLogger("blues.simulation")
logger = init_logger(logger, level=logging.ERROR, stream=True)





@pytest.fixture(scope="session", autouse=True)
def cleanup_propane_files():
    # This fixture will run after all tests in the session
    yield
    # Cleanup: Remove all files matching the pattern
    for filename in glob.glob("propane-test_*.nc"):
        try:
            os.remove(filename)
            print(f"Removed file: {filename}")
        except OSError as e:
            print(f"Error removing file {filename}: {e}")


def runPropaneTest(N, preferred_platform):
    filename = f'propane-test_{N}'
    print(f'Running {filename}...')
    # set to an integer for consisty across development 
    seed = np.random.randint(low=1, high=5000)
    # Example usage:
    print("Using platform:", preferred_platform)
    # Simulation Parameters
    sim_cfg = {
        'platform': preferred_platform,
        'nprop': 1,
        'propLambda': 0.3,
        'dt': 2 * openmm.unit.femtoseconds,
        'friction': 1 / openmm.unit.picoseconds,
        'temperature': 200 * openmm.unit.kelvin,
        'nIter': 500,
        'nstepsMD': 20,
        'nstepsNC': 20,
        'propSteps': 20,
        'moveStep': 10,
    }

    totalSteps = int(sim_cfg['nIter'] * sim_cfg['nstepsMD'])
    reportInterval = 10
    alchemical_atoms = [0, 1,2,3,4,5,6,7,8,9,10]  # Adjusted for propane

    md_reporters = {'traj_netcdf': {'reportInterval': reportInterval}}

    # Load Parmed Structure for Propane
    structure_pdb = utils.get_data_filename('blues', 'tests/data/propane.pdb')
    structure = parmed.load_file(structure_pdb)

    # Initialize Move Proposal Class
    rot_move = RandomLigandRotationMove(structure, 'LIG')  # Propane ligand
    mover = MoveEngine(rot_move)

    # Load OpenMM System and Integrator
    system_xml = utils.get_data_filename('blues', 'tests/data/propane_system.xml')
    with open(system_xml, 'r') as infile:
        xml = infile.read()
        system = openmm.XmlSerializer.deserialize(xml)

    integrator = openmm.LangevinIntegrator(sim_cfg['temperature'], sim_cfg['friction'], sim_cfg['dt'])
    integrator.setRandomNumberSeed(seed)

    alch_integrator = openmm.LangevinIntegrator(sim_cfg['temperature'], sim_cfg['friction'], sim_cfg['dt'])
    alch_integrator.setRandomNumberSeed(seed)

    alch_system = SystemFactory.generateAlchSystem(system, alchemical_atoms)
    ncmc_integrator = AlchemicalExternalLangevinIntegrator(
        nsteps_neq=sim_cfg['nstepsNC'],
        alchemical_functions={},
        splitting="H V R O R V H",
        temperature=sim_cfg['temperature'],
        timestep=sim_cfg['dt']
    )

    alch_system = SystemFactory.generateAlchSystem(system, alchemical_atoms)
    # Pack Systems
    systems = SystemFactory(structure, alchemical_atoms)
    systems.md = system
    systems.alch = alch_system


    # Check if all atoms still exist before running simulation
    # state = systems.md.getState(getPositions=True)
    # positions = state.getPositions(asNumpy=True)
    # Reporters
    md_reporter_cfg = ReporterConfig(filename, md_reporters)
    md_reporters_list = md_reporter_cfg.makeReporters()

    # Simulations
    simulations = SimulationFactory(systems, mover)

    simulations.md = SimulationFactory.generateSimFromStruct(structure, system, integrator, preferred_platform)
    simulations.md = SimulationFactory.attachReporters(simulations.md, md_reporters_list)

    simulations.alch = SimulationFactory.generateSimFromStruct(structure, system, alch_integrator, preferred_platform)
    simulations.ncmc = SimulationFactory.generateSimFromStruct(structure, alch_system, ncmc_integrator, preferred_platform)

    propane_sim = BLUESSimulation(simulations, sim_cfg)
    propane_sim.run()


def getPropanePopulations(traj):
    """Compute trans/gauche populations for propane dihedral"""
    dihedral_indices = [[0, 1, 2, 3]]  # C1-C2-C3 dihedral in propane
    dihedrals = md.compute_dihedrals(traj, dihedral_indices) * (180 / np.pi)  # Convert to degrees


    dihedrals = md.compute_dihedrals(traj, [[0, 1, 2, 3]]) * (180 / 3.14159)

    # Print population distribution
    left = (dihedrals < 0).sum()
    right = (dihedrals > 0).sum()
    total = left + right

    left_fraction = left / total
    right_fraction = right / total

    return [left_fraction, right_fraction]


def test_runPropaneRepeats(preferred_platform):
    """Run 10 separate simulations"""
    [runPropaneTest(i, preferred_platform) for i in range(10)]


def test_runPropaneAnalysis():
    """Analyze trajectory populations for propane"""
    outfnames = [f'propane-test_{i}.nc' for i in range(10)]
    structure_pdb = utils.get_data_filename('blues', 'tests/data/propane.pdb')

    trajs = [md.load(traj, top=structure_pdb) for traj in outfnames]
    freqs = []
    populations = [0.50, 0.50]  # Expected 50/50 trans/gauche

    for traj in trajs:
        freq = getPropanePopulations(traj)
        freqs.append(freq)

    freqs = np.asarray(freqs)
    avg_freq = np.mean(freqs, axis=0)
    avg_err = np.std(freqs, axis=0) / np.sqrt(len(freqs))
    print(f'standard error for these averages is about {np.average(avg_err):.2%}')
    print(f"Average Populations: Trans {avg_freq[0]:.2%}, Gauche {avg_freq[1]:.2%}")
    print(f'The difference for Trans is: {abs(avg_freq[0] - 0.50):.2%}')
    print(f'The difference for Gauche is: {abs(avg_freq[1] - 0.50):.2%}')
    print(f"Standard Error: {avg_err}")
    tol = max(np.average(avg_err), 0.10)
    print(f'Setting the tolerance to: {tol}')
    check = np.allclose(avg_freq, populations, atol=tol)
    assert check == True
    