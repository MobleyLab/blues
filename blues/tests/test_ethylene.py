import pytest
import parmed
import fnmatch
import logging
import os
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSampler
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.moves import RandomLigandRotationMove, ReportLangevinDynamicsMove
from blues.reporters import (BLUESStateDataStorage, NetCDF4Storage, ReporterConfig, init_logger)
from blues.settings import Settings
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from openmmtools import cache
import numpy as np
import mdtraj as md
from collections import Counter

logger = logging.getLogger(__name__)
#logger = init_logger(logger, level=logging.INFO, stream=True)
seed = np.random.randint(low=1, high=5000)

def runEthyleneTest(N):
    filename = 'ethylene-test_%s' % N
    print('Running %s...' % filename)

    # Set Simulation parameters
    temperature = 200 * unit.kelvin
    collision_rate = 1 / unit.picoseconds
    timestep = 1.0 * unit.femtoseconds
    n_steps = 20
    nIter = 100
    reportInterval = 5
    alchemical_atoms = [2, 3, 4, 5, 6, 7]
    alchemical_functions = {
        'lambda_sterics': 'min(1, (1/0.3)*abs(lambda-0.5))',
        'lambda_electrostatics': 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
    }
    context_cache = cache.ContextCache()

    # Load a Parmed Structure for the Topology and create our openmm.Simulation
    structure_pdb = utils.get_data_filename('blues', 'tests/data/ethylene_structure.pdb')
    structure = parmed.load_file(structure_pdb)

    # Iniitialize our Move set
    rot_move = RandomLigandRotationMove(timestep,
                                         n_steps,
                                         atom_subset=alchemical_atoms,
                                         context_cache=context_cache)
    langevin_move = ReportLangevinDynamicsMove(timestep, collision_rate, n_steps,
                                         reassign_velocities=True,
                                        context_cache=context_cache)

    # Load our OpenMM System and create Integrator
    system_xml = utils.get_data_filename('blues', 'tests/data/ethylene_system.xml')
    with open(system_xml, 'r') as infile:
        xml = infile.read()
        system = openmm.XmlSerializer.deserialize(xml)

    thermodynamic_state = ThermodynamicState(system=system, temperature=temperature)
    sampler_state = SamplerState(positions=structure.positions.in_units_of(unit.nanometers))
    nc_reporter = NetCDF4Storage(filename+'.nc', reportInterval)
    state_reporter = BLUESStateDataStorage(logger, reportInterval,
                                           title='md',
                                           step=True,
                                           speed=True,
                                           totalSteps=int(n_steps*nIter))

    sampler = BLUESSampler(alchemical_atoms,
                      thermodynamic_state,
                      sampler_state,
                      ncmc_move=rot_move,
                      dynamics_move=langevin_move,
                      platform=None,
                      reporter=[state_reporter, nc_reporter],
                      topology=structure.topology)

    sampler.run(nIter)


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
    [runEthyleneTest(i) for i in range(5)]


def test_runAnalysis():
    outfnames = ['ethylene-test_%s.nc' % i for i in range(5)]
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
    print(avg_freq, avg_err, np.absolute(avg_freq - populations))
    check = np.allclose(avg_freq, populations, atol=avg_err)
    assert check == True
