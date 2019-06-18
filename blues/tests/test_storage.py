import pytest
import parmed
import fnmatch
import numpy
import logging
from netCDF4 import Dataset
from openmmtools.cache import ContextCache
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from blues.systemfactory import *
from simtk.openmm import app
from openmmtools import cache
from simtk import unit
from blues.storage import *
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.ncmc import RandomLigandRotationMove, ReportLangevinDynamicsMove, BLUESSampler


def get_states():
    # Set Simulation parameters
    temperature = 200 * unit.kelvin
    collision_rate = 1 / unit.picoseconds
    timestep = 1.0 * unit.femtoseconds
    n_steps = 20
    nIter = 100
    alchemical_atoms = [2, 3, 4, 5, 6, 7]
    platform = openmm.Platform.getPlatformByName('CPU')
    context_cache = cache.ContextCache(platform)

    # Load a Parmed Structure for the Topology and create our openmm.Simulation
    structure_pdb = utils.get_data_filename('blues', 'tests/data/ethylene_structure.pdb')
    structure = parmed.load_file(structure_pdb)


    # Load our OpenMM System and create Integrator
    system_xml = utils.get_data_filename('blues', 'tests/data/ethylene_system.xml')
    with open(system_xml, 'r') as infile:
        xml = infile.read()
        system = openmm.XmlSerializer.deserialize(xml)

    thermodynamic_state = ThermodynamicState(system=system, temperature=temperature)
    sampler_state = SamplerState(positions=structure.positions.in_units_of(unit.nanometers))

    alch_system = generateAlchSystem(thermodynamic_state.get_system(),
                                     alchemical_atoms)
    alch_state = alchemy.AlchemicalState.from_system(alch_system)
    alch_thermodynamic_state = ThermodynamicState(
        alch_system, thermodynamic_state.temperature)
    alch_thermodynamic_state = CompoundThermodynamicState(
        alch_thermodynamic_state, composable_states=[alch_state])

    return structure, thermodynamic_state, alch_thermodynamic_state


def test_add_logging_level():
    print("Testing adding logger level")
    addLoggingLevel('TRACE', logging.DEBUG - 5)
    assert True == hasattr(logging, 'TRACE')


def test_init_logger(tmpdir):
    print('Testing logger initialization')
    dir = tmpdir.mkdir("tmp")
    outfname = dir.join('testlog')
    logger = logging.getLogger(__name__)
    level = logger.getEffectiveLevel()
    logger = init_logger(logger, level=logging.INFO, outfname=outfname, stream=False)
    new_level = logger.getEffectiveLevel()
    assert level != new_level


def test_netcdf4storage(tmpdir):
    dir = tmpdir.mkdir("tmp")
    outfname = dir.join('testlog.nc')
    context_cache = ContextCache()
    ncmc_storage = NetCDF4Storage(outfname, 5, crds=True, vels=True, frcs=True,
                                protocolWork=True, alchemicalLambda=True)


    structure, thermodynamic_state, alch_thermodynamic_state = get_states()
    ncmc_integrator = AlchemicalExternalLangevinIntegrator(
        alchemical_functions={
            'lambda_sterics':
            'min(1, (1/0.3)*abs(lambda-0.5))',
            'lambda_electrostatics':
            'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
        },
        splitting="H V R O R V H",
        temperature=alch_thermodynamic_state.temperature,
        nsteps_neq=10,
        timestep=1.0*unit.femtoseconds,
        nprop=1,
        prop_lambda=0.3)
    context, integrator = context_cache.get_context(alch_thermodynamic_state, ncmc_integrator)
    context.setPositions(structure.positions)
    context.setVelocitiesToTemperature(200 * unit.kelvin)
    integrator.step(5)

    context_state = context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        getForces=True,
        enforcePeriodicBox=thermodynamic_state.is_periodic)
    context_state.currentStep = 5
    context_state.system = alch_thermodynamic_state.get_system()

    # Check for preparation of next report
    report = ncmc_storage.describeNextReport(context_state)
    assert len(report) == 5

    # Check that data has been stored in NC file
    ncmc_storage.report(context_state, integrator)
    dataset = Dataset(outfname)
    nc_keys = set(dataset.variables.keys())
    keys = set(['coordinates', 'velocities', 'forces', 'protocolWork', 'alchemicalLambda'])
    assert set() == keys - nc_keys


def test_statedatastorage(tmpdir):
    context_cache = ContextCache()

    dir = tmpdir.mkdir("tmp")
    outfname = dir.join('blues.log')
    setup_logging(filename=outfname)
    state_storage = BLUESStateDataStorage(outfname,
                                         reportInterval=5,
                                         step=True, time=True,
                                         potentialEnergy=True,
                                         kineticEnergy=True,
                                         totalEnergy=True,
                                         temperature=True,
                                         volume=True,
                                         density=True,
                                         progress=True,
                                         remainingTime=True,
                                         speed=True,
                                         elapsedTime=True,
                                         systemMass=True,
                                         totalSteps=20,
                                         protocolWork=True,
                                         alchemicalLambda=True )
    structure, thermodynamic_state, alch_thermodynamic_state = get_states()
    ncmc_integrator = AlchemicalExternalLangevinIntegrator(
        alchemical_functions={
            'lambda_sterics':
            'min(1, (1/0.3)*abs(lambda-0.5))',
            'lambda_electrostatics':
            'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
        },
        splitting="H V R O R V H",
        temperature=alch_thermodynamic_state.temperature,
        nsteps_neq=10,
        timestep=1.0*unit.femtoseconds,
        nprop=1,
        prop_lambda=0.3)
    context, integrator = context_cache.get_context(alch_thermodynamic_state, ncmc_integrator)
    context.setPositions(structure.positions)
    context.setVelocitiesToTemperature(200 * unit.kelvin)
    integrator.step(5)

    context_state = context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        getForces=True,
        enforcePeriodicBox=thermodynamic_state.is_periodic)
    context_state.currentStep = 5
    context_state.system = alch_thermodynamic_state.get_system()

    # Check for preparation of next report
    report = state_storage.describeNextReport(context_state)
    assert len(report) == 5

    #Check fields have been reported
    state_storage.report(context_state, integrator)
    with open(outfname, 'r') as input:
        headers = input.read().splitlines()[0].split('\t')
        assert len(headers) >= 1
