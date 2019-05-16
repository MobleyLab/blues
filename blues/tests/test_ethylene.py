import logging
from collections import Counter

import mdtraj as md
import numpy as np
import parmed
from openmmtools import cache
from openmmtools.states import SamplerState, ThermodynamicState
from simtk import openmm, unit

from blues import utils
from blues.reporters import BLUESStateDataStorage, NetCDF4Storage
from blues.ncmc import RandomLigandRotationMove, ReportLangevinDynamicsMove, BLUESSampler

logger = logging.getLogger(__name__)


def runEthyleneTest(dir, N):
    filename = dir.join('ethylene-test_%s' % N)
    print('Running %s...' % filename)

    # Set Simulation parameters
    temperature = 200 * unit.kelvin
    collision_rate = 1 / unit.picoseconds
    timestep = 1.0 * unit.femtoseconds
    n_steps = 20
    nIter = 100
    reportInterval = 5
    alchemical_atoms = [2, 3, 4, 5, 6, 7]
    platform = openmm.Platform.getPlatformByName('CPU')
    context_cache = cache.ContextCache(platform)

    # Load a Parmed Structure for the Topology and create our openmm.Simulation
    structure_pdb = utils.get_data_filename('blues', 'tests/data/ethylene_structure.pdb')
    structure = parmed.load_file(structure_pdb)

    nc_reporter = NetCDF4Storage(filename + '_MD.nc', reportInterval)
    state_reporter = BLUESStateDataStorage(
        logger, reportInterval, title='md', step=True, speed=True, totalSteps=int(n_steps * nIter))
    nc_reporter1 = NetCDF4Storage(filename + '_NCMC.nc', reportInterval)
    state_reporter1 = BLUESStateDataStorage(
        logger, reportInterval, title='ncmc', step=True, speed=True, totalSteps=int(n_steps * nIter))

    # Iniitialize our Move set
    rot_move = RandomLigandRotationMove(
        timestep,
        n_steps,
        atom_subset=alchemical_atoms,
        context_cache=context_cache,
        reporters=[nc_reporter1, state_reporter1])
    langevin_move = ReportLangevinDynamicsMove(
        timestep,
        collision_rate,
        n_steps,
        reassign_velocities=True,
        context_cache=context_cache,
        reporters=[nc_reporter, state_reporter])

    # Load our OpenMM System and create Integrator
    system_xml = utils.get_data_filename('blues', 'tests/data/ethylene_system.xml')
    with open(system_xml, 'r') as infile:
        xml = infile.read()
        system = openmm.XmlSerializer.deserialize(xml)

    thermodynamic_state = ThermodynamicState(system=system, temperature=temperature)
    sampler_state = SamplerState(positions=structure.positions.in_units_of(unit.nanometers))

    sampler = BLUESSampler(
        atom_subset=alchemical_atoms,
        thermodynamic_state=thermodynamic_state,
        sampler_state=sampler_state,
        ncmc_move=rot_move,
        dynamics_move=langevin_move,
        platform=None,
        topology=structure.topology)
    sampler.run(nIter)

    return filename


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


def test_runEthyleneRepeats(tmpdir):
    dir = tmpdir.mkdir("tmp")
    outfnames = [runEthyleneTest(dir, N=i) for i in range(5)]

    structure_pdb = utils.get_data_filename('blues', 'tests/data/ethylene_structure.pdb')
    trajs = [md.load('%s_MD.nc' % traj, top=structure_pdb) for traj in outfnames]
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
    assert check is True
