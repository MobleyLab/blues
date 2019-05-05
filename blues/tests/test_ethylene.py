import pytest
import parmed
import fnmatch
import logging
import os
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.moves import RandomLigandRotationMove, MoveEngine
from blues.reporters import (BLUESStateDataReporter, NetCDF4Reporter, ReporterConfig, init_logger)
from blues.settings import Settings
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np
import mdtraj as md
from collections import Counter

logger = logging.getLogger("blues.simulation")
logger = init_logger(logger, level=logging.ERROR, stream=True)


def runEthyleneTest(N):
    filename = 'ethylene-test_%s' % N
    print('Running %s...' % filename)
    seed = np.random.randint(low=1, high=5000)
    #print('Seed', seed)
    # filename = 'ethylene-test_%s' % N
    # print(filename)

    # Set Simulation parameters
    sim_cfg = {
        'platform': 'CPU',
        'nprop': 1,
        'propLambda': 0.3,
        'dt': 1 * unit.femtoseconds,
        'friction': 1 / unit.picoseconds,
        'temperature': 200 * unit.kelvin,
        'nIter': 1000,
        'nstepsMD': 20,
        'nstepsNC': 20,
        'propSteps': 20,
        'moveStep': 10
    }

    totalSteps = int(sim_cfg['nIter'] * sim_cfg['nstepsMD'])
    reportInterval = 5
    alchemical_atoms = [2, 3, 4, 5, 6, 7]
    alchemical_functions = {
        'lambda_sterics': 'min(1, (1/0.3)*abs(lambda-0.5))',
        'lambda_electrostatics':
        'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
    }

    md_reporters = {'traj_netcdf': {'reportInterval': reportInterval}}

    # Load a Parmed Structure for the Topology and create our openmm.Simulation
    structure_pdb = utils.get_data_filename('blues', 'tests/data/ethylene_structure.pdb')
    structure = parmed.load_file(structure_pdb)

    # Initialize our move proposal class
    rot_move = RandomLigandRotationMove(structure, 'LIG')
    mover = MoveEngine(rot_move)

    # Load our OpenMM System and create Integrator
    system_xml = utils.get_data_filename('blues', 'tests/data/ethylene_system.xml')
    with open(system_xml, 'r') as infile:
        xml = infile.read()
        system = openmm.XmlSerializer.deserialize(xml)
    integrator = openmm.LangevinIntegrator(sim_cfg['temperature'], sim_cfg['friction'], sim_cfg['dt'])
    integrator.setRandomNumberSeed(seed)

    alch_integrator = openmm.LangevinIntegrator(sim_cfg['temperature'], sim_cfg['friction'], sim_cfg['dt'])
    alch_integrator.setRandomNumberSeed(seed)

    alch_system = SystemFactory.generateAlchSystem(system, alchemical_atoms)
    ncmc_integrator = AlchemicalExternalLangevinIntegrator(nsteps_neq=sim_cfg['nstepsNC'],
                                                           alchemical_functions=alchemical_functions,
                                                           splitting="H V R O R V H",
                                                           temperature=sim_cfg['temperature'],
                                                           timestep=sim_cfg['dt'])
    # ncmc_integrator.setRandomNumberSeed(seed)

    # Pack our systems into a single object
    systems = SystemFactory(structure, alchemical_atoms)
    systems.md = system
    systems.alch = alch_system

    # Make our reporters
    md_reporter_cfg = ReporterConfig(filename, md_reporters)
    md_reporters_list = md_reporter_cfg.makeReporters()

    # Pack our simulations into a single object
    simulations = SimulationFactory(systems, mover)

    simulations.md = SimulationFactory.generateSimFromStruct(structure, system, integrator, 'CPU')
    simulations.md = SimulationFactory.attachReporters(simulations.md, md_reporters_list)

    simulations.alch = SimulationFactory.generateSimFromStruct(structure, system, alch_integrator, 'CPU')

    simulations.ncmc = SimulationFactory.generateSimFromStruct(structure, alch_system, ncmc_integrator, 'CPU')

    ethylene_sim = BLUESSimulation(simulations, sim_cfg)
    ethylene_sim.run()


def getPopulations(traj):
    dist = md.compute_distances(traj, [[0, 2]])
    dist[dist <= 0.49] = 0
    dist[dist > 0.49] = 1
    dist = np.hstack(dist)
    counts = Counter(dist)
    total = counts[0] + counts[1]
    freq = [counts[0] / total, counts[1] / total]
    return dist, freq


def graphConvergence(dist, n_points=10):
    bins = len(dist) / n_points
    bin_count = []
    bin_points = []
    for N in range(1, len(dist) + 1, n_points):
        bin_points.append(N)
        counts = Counter(dist[:N])
        total = counts[0] + counts[1]
        freq = [counts[0] / total, counts[1] / total]
        bin_count.append([freq[0], freq[1]])

    bin_count_arr = np.vstack(bin_count)
    bin_err = []
    for i, row in enumerate(bin_count_arr):
        total = row[0] + row[1]
        std0 = np.std(bin_count_arr[:i, 0]) / np.sqrt(total)
        std1 = np.std(bin_count_arr[:i, 1]) / np.sqrt(total)
        bin_err.append([std0, std1])
    bin_err_arr = np.vstack(bin_err)
    return bin_err_arr[-1, :]


def test_runEthyleneRepeats():
    [runEthyleneTest(i) for i in range(10)]


def test_runAnalysis():
    outfnames = ['ethylene-test_%s.nc' % i for i in range(10)]
    structure_pdb = utils.get_data_filename('blues', 'tests/data/ethylene_structure.pdb')
    trajs = [md.load(traj, top=structure_pdb) for traj in outfnames]
    dists = []
    freqs = []
    errs = []
    populations = [0.25, 0.75]
    for traj in trajs:
        dist, freq = getPopulations(traj)
        dists.append(dist)
        errs.append(graphConvergence(dist, n_points=10))
        freqs.append(freq)
    freqs = np.asarray(freqs)
    errs = np.asarray(errs)
    avg_freq = np.mean(freqs, axis=0)
    avg_err = np.mean(errs, axis=0)
    print(avg_freq, avg_err)
    check = np.allclose(avg_freq, populations, rtol=avg_err)
    assert check == True
