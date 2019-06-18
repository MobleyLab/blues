import pytest, parmed, fnmatch, logging, os
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.moves import RandomLigandRotationMove, MoveEngine
from blues.reporters import ReporterConfig
from blues.settings import Settings
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np



@pytest.fixture(scope='session')
def sim_cfg():
    # os.getenv is equivalent, and can also give a default value instead of `None`
    PLATFORM = os.getenv('OMM_PLATFORM', 'CPU')
    sim_cfg = {
        'nprop': 1,
        'propLambda': 0.3,
        'dt': 0.002 * unit.picoseconds,
        'friction': 1 * 1 / unit.picoseconds,
        'temperature': 300 * unit.kelvin,
        'nIter': 1,
        'nstepsMD': 10,
        'nstepsNC': 10,
        'platform': PLATFORM
    }
    return sim_cfg


@pytest.fixture(scope='session')
def stateinfo_keys():
    stateinfo_keys = ['positions', 'velocities', 'potential_energy', 'kinetic_energy', 'box_vectors']
    return stateinfo_keys


@pytest.fixture(scope='session')
def state_keys():
    state_keys = {
        'getPositions': True,
        'getVelocities': True,
        'getForces': False,
        'getEnergy': True,
        'getParameters': True,
        'enforcePeriodicBox': True
    }
    return state_keys




class NoRandomLigandRotation(RandomLigandRotationMove):
    def move(self, context):
        return context


@pytest.fixture(scope='session')
def move(structure):
    move = NoRandomLigandRotation(structure, 'LIG')
    #move = RandomLigandRotationMove(structure, 'LIG', random_state)
    return move


@pytest.fixture(scope='session')
def engine(move):
    engine = MoveEngine(move)
    return engine


@pytest.fixture(scope='session')
def systems(structure, tol_atom_indices, system_cfg):
    systems = SystemFactory(structure, tol_atom_indices, system_cfg)
    return systems


@pytest.fixture(scope='session')
def simulations(systems, engine, sim_cfg):
    simulations = SimulationFactory(systems, engine, sim_cfg)
    return simulations


@pytest.fixture(scope='session')
def ncmc_integrator(structure, system):
    cfg = {
        'nstepsNC': 10,
        'temperature': 100 * unit.kelvin,
        'dt': 0.001 * unit.picoseconds,
        'nprop': 1,
        'propLambda': 0.3,
        'splitting': 'V H R O R H V',
        'alchemical_functions': {
            'lambda_sterics': '1',
            'lambda_electrostatics': '1'
        }
    }

    ncmc_integrator = SimulationFactory.generateNCMCIntegrator(**cfg)
    return ncmc_integrator


@pytest.fixture(scope='session')
def md_sim(structure, system):
    integrator = openmm.LangevinIntegrator(100 * unit.kelvin, 1, 0.002 * unit.picoseconds)
    md_sim = SimulationFactory.generateSimFromStruct(structure, system, integrator)
    return md_sim


@pytest.fixture(scope='session')
def blues_sim(simulations):
    blues_sim = BLUESSimulation(simulations)
    blues_sim._md_sim.minimizeEnergy()
    blues_sim._alch_sim.minimizeEnergy()
    blues_sim._ncmc_sim.minimizeEnergy()
    return blues_sim





class TestBLUESSimulation(object):
    def test_getStateFromContext(self, md_sim, stateinfo_keys, state_keys):

        stateinfo = BLUESSimulation.getStateFromContext(md_sim.context, state_keys)

        assert isinstance(stateinfo, dict)
        for key in stateinfo_keys:
            assert key in list(stateinfo.keys())
            assert stateinfo[key] is not None

    def test_getIntegratorInfo(self, ncmc_integrator):
        integrator_keys = ['lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew']
        integrator_info = BLUESSimulation.getIntegratorInfo(ncmc_integrator, integrator_keys)

        assert isinstance(integrator_info, dict)

    def test_setContextFromState(self, md_sim, state_keys):
        md_context = md_sim.context

        pos = md_context.getState(getPositions=True).getPositions(asNumpy=True)
        md_state = BLUESSimulation.getStateFromContext(md_context, state_keys)

        # Create an empty array
        zero_arr = np.zeros(md_state['positions'].shape)
        md_state['positions'] = zero_arr * unit.nanometers

        # Check that the positions have been modified
        md_context_0 = BLUESSimulation.setContextFromState(md_context, md_state)
        pos0 = md_context_0.getState(getPositions=True).getPositions(asNumpy=True)

        assert np.not_equal(pos0, pos).any()

    def test_printSimulationTiming(self, blues_sim, caplog):
        caplog.set_level(logging.INFO)
        blues_sim._printSimulationTiming()
        assert 'Total BLUES Simulation Time' in caplog.text
        #assert 'Total Force Evaluations' in caplog.text
        #assert 'Total NCMC time' in caplog.text
        #assert 'Total MD time' in caplog.text

    def test_setStateTable(self, blues_sim, state_keys):
        assert blues_sim.stateTable['md']['state0'] == {}
        md_context = blues_sim._md_sim.context
        md_state = BLUESSimulation.getStateFromContext(md_context, state_keys)
        blues_sim._setStateTable('md', 'state0', md_state)
        assert blues_sim.stateTable['md']['state0'] == md_state

    def test_syncStatesMDtoNCMC(self, blues_sim, state_keys):
        assert blues_sim.stateTable['ncmc']['state0'] == {}
        blues_sim._syncStatesMDtoNCMC()

        md_state = BLUESSimulation.getStateFromContext(blues_sim._md_sim.context, state_keys)
        ncmc_state = BLUESSimulation.getStateFromContext(blues_sim._ncmc_sim.context, state_keys)
        assert np.equal(ncmc_state['positions'], md_state['positions']).all()

    def test_stepNCMC(self, blues_sim, sim_cfg):
        nstepsNC = sim_cfg['nstepsNC']
        moveStep = sim_cfg['moveStep']
        blues_sim._stepNCMC(nstepsNC, moveStep)
        ncmc_state0 = blues_sim.stateTable['ncmc']['state0']['positions']
        ncmc_state1 = blues_sim.stateTable['ncmc']['state1']['positions']
        assert np.not_equal(ncmc_state0, ncmc_state1).all()

    def test_computeAlchemicalCorrection(self, blues_sim):
        correction_factor = blues_sim._computeAlchemicalCorrection()
        assert isinstance(correction_factor, float)

    def test_acceptRejectMove(self, blues_sim, state_keys, caplog):
        # Check positions are different from stepNCMC
        md_state = BLUESSimulation.getStateFromContext(blues_sim._md_sim.context, state_keys)
        ncmc_state = BLUESSimulation.getStateFromContext(blues_sim._ncmc_sim.context, state_keys)
        assert np.not_equal(md_state['positions'], ncmc_state['positions']).all()

        caplog.set_level(logging.INFO)
        blues_sim._acceptRejectMove()
        ncmc_state = BLUESSimulation.getStateFromContext(blues_sim._ncmc_sim.context, state_keys)
        md_state = BLUESSimulation.getStateFromContext(blues_sim._md_sim.context, state_keys)
        if 'NCMC MOVE ACCEPTED' in caplog.text:
            assert np.equal(md_state['positions'], ncmc_state['positions']).all()
        elif 'NCMC MOVE REJECTED' in caplog.text:
            assert np.not_equal(md_state['positions'], ncmc_state['positions']).all()

    def test_resetSimulations(self, blues_sim, state_keys):
        md_state0 = BLUESSimulation.getStateFromContext(blues_sim._md_sim.context, state_keys)

        blues_sim._resetSimulations(100 * unit.kelvin)

        md_state1 = BLUESSimulation.getStateFromContext(blues_sim._md_sim.context, state_keys)
        assert np.not_equal(md_state0['velocities'], md_state1['velocities']).all()

    def test_stepMD(self, blues_sim, state_keys):
        md_state0 = BLUESSimulation.getStateFromContext(blues_sim._md_sim.context, state_keys)

        blues_sim._stepMD(2)

        md_state1 = BLUESSimulation.getStateFromContext(blues_sim._md_sim.context, state_keys)

        # Check positions have changed
        assert np.not_equal(md_state0['positions'], md_state1['positions']).all()

    def test_blues_simulationRunYAML(self, tmpdir, structure, tol_atom_indices, system_cfg, engine):
        yaml_cfg = """
            output_dir: .
            outfname: tol-test
            logger:
              level: info
              stream: True

            system:
              nonbondedMethod: PME
              nonbondedCutoff: 8.0 * angstroms
              constraints: HBonds

            simulation:
              dt: 0.002 * picoseconds
              friction: 1 * 1/picoseconds
              temperature: 300 * kelvin
              nIter: 1
              nstepsMD: 2
              nstepsNC: 2
              platform: CPU

            md_reporters:
              stream:
                title: md
                reportInterval: 1
                totalSteps: 2 # nIter * nstepsMD
                step: True
                speed: True
                progress: True
                remainingTime: True
                currentIter : True
            ncmc_reporters:
              stream:
                title: ncmc
                reportInterval: 1
                totalSteps: 2 # Use nstepsNC
                step: True
                speed: True
                progress: True
                remainingTime: True
                protocolWork : True
                alchemicalLambda : True
                currentIter : True
        """
        print('Testing Simulation.run() from YAML')
        yaml_cfg = Settings(yaml_cfg)
        cfg = yaml_cfg.asDict()
        cfg['output_dir'] = tmpdir
        # os.getenv is equivalent, and can also give a default value instead of `None`
        PLATFORM = os.getenv('OMM_PLATFORM', 'CPU')
        cfg['simulation']['platform'] = PLATFORM
        systems = SystemFactory(structure, tol_atom_indices, cfg['system'])
        simulations = SimulationFactory(systems, engine, cfg['simulation'], cfg['md_reporters'], cfg['ncmc_reporters'])

        blues = BLUESSimulation(simulations)
        blues._md_sim.minimizeEnergy()
        blues._alch_sim.minimizeEnergy()
        blues._ncmc_sim.minimizeEnergy()
        before_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        blues.run()
        after_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        #Check that our system has run dynamics
        pos_compare = np.not_equal(before_iter, after_iter).all()
        assert pos_compare

    def test_blues_simulationRunPython(self, systems, simulations, engine, tmpdir, sim_cfg):
        print('Testing BLUESSimulation.run() from pure python')
        md_rep_cfg = {
            'stream': {
                'title': 'md',
                'reportInterval': 1,
                'totalSteps': 2,
                'step': True,
                'speed': True,
                'progress': True,
                'remainingTime': True,
                'currentIter': True
            }
        }
        ncmc_rep_cfg = {
            'stream': {
                'title': 'ncmc',
                'reportInterval': 1,
                'totalSteps': 2,
                'step': True,
                'speed': True,
                'progress': True,
                'remainingTime': True,
                'currentIter': True
            }
        }

        md_reporters = ReporterConfig(tmpdir.join('tol-test'), md_rep_cfg).makeReporters()
        ncmc_reporters = ReporterConfig(tmpdir.join('tol-test-ncmc'), ncmc_rep_cfg).makeReporters()

        simulations = SimulationFactory(
            systems, engine, sim_cfg, md_reporters=md_reporters, ncmc_reporters=ncmc_reporters)

        blues = BLUESSimulation(simulations)
        blues._md_sim.minimizeEnergy()
        blues._alch_sim.minimizeEnergy()
        blues._ncmc_sim.minimizeEnergy()
        before_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        blues.run()
        after_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        #Check that our system has run dynamics
        pos_compare = np.not_equal(before_iter, after_iter).all()
        assert pos_compare
