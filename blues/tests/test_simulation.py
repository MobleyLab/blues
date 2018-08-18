import pytest, parmed, fnmatch, logging
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
def system_cfg():
    system_cfg = {
        'nonbondedMethod': app.PME,
        'nonbondedCutoff': 8.0 * unit.angstroms,
        'constraints': app.HBonds
    }
    return system_cfg


@pytest.fixture(scope='session')
def sim_cfg():
    sim_cfg = {
        'nprop': 1,
        'prop_lambda': 0.3,
        'dt': 0.001 * unit.picoseconds,
        'friction': 1 * 1 / unit.picoseconds,
        'temperature': 100 * unit.kelvin,
        'nIter': 1,
        'nstepsMD': 10,
        'nstepsNC': 10,
    }
    return sim_cfg


@pytest.fixture(scope='session')
def stateinfo_keys():
    stateinfo_keys = [
        'positions', 'velocities', 'potential_energy', 'kinetic_energy',
        'box_vectors'
    ]
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


@pytest.fixture(scope='session')
def system(structure, system_cfg):
    system = structure.createSystem(**system_cfg)
    return system


@pytest.fixture(scope='session')
def move(structure):
    move = RandomLigandRotationMove(structure, 'LIG')
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
        'nstepsNC': 100,
        'temperature': 100 * unit.kelvin,
        'dt': 0.001 * unit.picoseconds,
        'nprop': 2,
        'propLambda': 0.1,
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
    integrator = openmm.LangevinIntegrator(100 * unit.kelvin, 1,
                                           0.002 * unit.picoseconds)
    md_sim = SimulationFactory.generateSimFromStruct(structure, system,
                                                     integrator)
    return md_sim


@pytest.fixture(scope='session')
def blues_sim(simulations):
    blues_sim = BLUESSimulation(simulations)
    return blues_sim


class TestSystemFactory(object):
    def test_atom_selections(self, structure, tol_atom_indices):
        atom_indices = SystemFactory.amber_selection_to_atomidx(
            structure, ':LIG')

        print('Testing AMBER selection parser')
        assert isinstance(atom_indices, list)
        assert len(atom_indices) == len(tol_atom_indices)

    def test_atomidx_to_atomlist(self, structure, tol_atom_indices):
        print('Testing atoms from AMBER selection with parmed.Structure')
        atom_list = SystemFactory.atomidx_to_atomlist(structure,
                                                      tol_atom_indices)
        atom_selection = [structure.atoms[i] for i in tol_atom_indices]
        assert atom_selection == atom_list

    def test_generateSystem(self, structure, system, system_cfg):
        # Create the OpenMM system
        print('Creating OpenMM System')
        md_system = SystemFactory.generateSystem(structure, **system_cfg)

        # Check that we get an openmm.System
        assert isinstance(md_system, openmm.System)
        # Check atoms in system is same in input parmed.Structure
        assert md_system.getNumParticles() == len(structure.atoms)
        assert md_system.getNumParticles() == system.getNumParticles()

    def test_generateAlchSystem(self, structure, system, tol_atom_indices):
        # Create the OpenMM system
        print('Creating OpenMM Alchemical System')
        alch_system = SystemFactory.generateAlchSystem(system,
                                                       tol_atom_indices)

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

    def test_restrain_postions(self, structure, system):
        print('Testing positional restraints')
        no_restr = system.getForces()

        md_system_restr = SystemFactory.restrain_positions(
            structure, system, ':LIG')
        restr = md_system_restr.getForces()

        # Check that forces have been added to the system.
        assert len(restr) != len(no_restr)
        # Check that it has added the CustomExternalForce
        assert isinstance(restr[-1], openmm.CustomExternalForce)

    def test_freeze_atoms(self, structure, system, tol_atom_indices):
        print('Testing freeze_atoms')
        masses = [system.getParticleMass(i)._value for i in tol_atom_indices]
        frzn_lig = SystemFactory.freeze_atoms(structure, system, ':LIG')
        massless = [
            frzn_lig.getParticleMass(i)._value for i in tol_atom_indices
        ]

        # Check that masses have been zeroed
        assert massless != masses
        assert all(m == 0 for m in massless)

    def test_freeze_radius(self, system_cfg):
        print('Testing freeze_radius')
        freeze_cfg = {
            'freeze_center': ':LIG',
            'freeze_solvent': ':Cl-',
            'freeze_distance': 3.0 * unit.angstroms
        }
        # Setup toluene-T4 lysozyme system
        prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
        structure = parmed.load_file(prmtop, xyz=inpcrd)
        atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
        system = SystemFactory.generateSystem(structure, **system_cfg)

        # Freeze everything around the binding site
        frzn_sys = SystemFactory.freeze_radius(structure, system, **freeze_cfg)

        # Check that the ligand has NOT been frozen
        lig_masses = [system.getParticleMass(i)._value for i in atom_indices]
        assert all(m != 0 for m in lig_masses)

        # Check that the binding site has NOT been frozen
        selection = "({freeze_center}<:{freeze_distance._value})&!({freeze_solvent})".format(
            **freeze_cfg)
        site_idx = SystemFactory.amber_selection_to_atomidx(
            structure, selection)
        masses = [frzn_sys.getParticleMass(i)._value for i in site_idx]
        assert all(m != 0 for m in masses)

        # Check that the selection has been frozen
        # Invert that selection to freeze everything but the binding site.
        freeze_idx = set(range(system.getNumParticles())) - set(site_idx)
        massless = [frzn_sys.getParticleMass(i)._value for i in freeze_idx]
        assert all(m == 0 for m in massless)


class TestSimulationFactory(object):
    def test_addBarostat(self, system):
        print('Testing MonteCarloBarostat')
        forces = system.getForces()
        npt_system = SimulationFactory.addBarostat(system)
        npt_forces = npt_system.getForces()

        #Check that forces have been added to the system.
        assert len(forces) != len(npt_forces)
        #Check that it has added the MonteCarloBarostat
        assert isinstance(npt_forces[-1], openmm.MonteCarloBarostat)

    def test_generateIntegrator(self):
        print('Testing LangevinIntegrator')
        cfg = {
            'temperature': 500 * unit.kelvin,
            'dt': 0.004 * unit.picoseconds
        }
        integrator = SimulationFactory.generateIntegrator(**cfg)
        #Check we made the right integrator
        assert isinstance(integrator, openmm.LangevinIntegrator)
        #Check that the integrator has taken our Parameters
        assert integrator.getTemperature() == cfg['temperature']
        assert integrator.getStepSize() == cfg['dt']

    def test_generateNCMCIntegrator(self):
        print('Testing AlchemicalExternalLangevinIntegrator')
        cfg = {
            'nstepsNC': 100,
            'temperature': 100 * unit.kelvin,
            'dt': 0.001 * unit.picoseconds,
            'nprop': 2,
            'propLambda': 0.1,
            'splitting': 'V H R O R H V',
            'alchemical_functions': {
                'lambda_sterics': '1',
                'lambda_electrostatics': '1'
            }
        }
        ncmc_integrator = SimulationFactory.generateNCMCIntegrator(**cfg)
        #Check we made the right integrator
        assert isinstance(ncmc_integrator,
                          AlchemicalExternalLangevinIntegrator)
        #Check that the integrator has taken our Parameters
        assert round(
            abs(ncmc_integrator.getTemperature()._value -
                cfg['temperature']._value), 7) == 0
        assert ncmc_integrator.getStepSize() == cfg['dt']
        assert ncmc_integrator._n_steps_neq == cfg['nstepsNC']
        assert ncmc_integrator._n_lambda_steps == \
                         cfg['nstepsNC'] * cfg['nprop']
        assert ncmc_integrator._alchemical_functions == \
                         cfg['alchemical_functions']
        assert ncmc_integrator._splitting == cfg['splitting']
        prop_range = (0.5 - cfg['propLambda'], 0.5 + cfg['propLambda'])
        assert ncmc_integrator._prop_lambda == prop_range

    def test_generateSimFromStruct(self, structure, system, tmpdir):
        print('Generating Simulation from parmed.Structure')
        integrator = openmm.LangevinIntegrator(100 * unit.kelvin, 1,
                                               0.002 * unit.picoseconds)
        simulation = SimulationFactory.generateSimFromStruct(
            structure, system, integrator)

        #Check that we've made a Simulation object
        assert isinstance(simulation, app.Simulation)
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True) / unit.nanometers
        box_vectors = state.getPeriodicBoxVectors(
            asNumpy=True) / unit.nanometers
        struct_box = np.array(
            structure.box_vectors.value_in_unit(unit.nanometers))
        struct_pos = np.array(
            structure.positions.value_in_unit(unit.nanometers))

        #Check that the box_vectors/positions in the Simulation
        # have been set from the parmed.Structure
        np.testing.assert_array_almost_equal(positions, struct_pos)
        np.testing.assert_array_equal(box_vectors, struct_box)

        print('Attaching Reporter')
        reporters = [app.StateDataReporter(tmpdir.join('test.log'), 5)]
        assert len(simulation.reporters) == 0
        simulation = SimulationFactory.attachReporters(simulation, reporters)
        assert len(simulation.reporters) == 1

    def test_generateSimulationSet(self, structure, systems, engine, sim_cfg):
        print('Testing generateSimulationSet')
        simulations = SimulationFactory(systems, engine)
        simulations.generateSimulationSet(sim_cfg)
        #Check that we've made the MD/ALCH/NCMC simulation set
        assert hasattr(simulations, 'md')
        assert hasattr(simulations, 'alch')
        assert hasattr(simulations, 'ncmc')
        #Check that the physical parameters are equivalent
        assert simulations.ncmc_integrator.getStepSize() == sim_cfg['dt']
        assert simulations.integrator.getStepSize() == sim_cfg['dt']
        assert round(
            abs(simulations.ncmc_integrator.getTemperature()._value -
                sim_cfg['temperature']._value), 7) == 0
        assert round(
            abs(simulations.integrator.getTemperature()._value -
                sim_cfg['temperature']._value), 7) == 0


class TestBLUESSimulation(object):
    def test_getStateFromContext(self, md_sim, stateinfo_keys, state_keys):

        stateinfo = BLUESSimulation.getStateFromContext(
            md_sim.context, state_keys)

        assert isinstance(stateinfo, dict)
        for key in stateinfo_keys:
            assert key in list(stateinfo.keys())
            assert stateinfo[key] is not None

    def test_getIntegratorInfo(self, ncmc_integrator):
        integrator_keys = [
            'lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew'
        ]
        integrator_info = BLUESSimulation.getIntegratorInfo(
            ncmc_integrator, integrator_keys)

        assert isinstance(integrator_info, dict)

    def test_setContextFromState(self, md_sim, state_keys):
        md_context = md_sim.context

        pos = md_context.getState(getPositions=True).getPositions(asNumpy=True)
        md_state = BLUESSimulation.getStateFromContext(md_context, state_keys)

        # Create an empty array
        zero_arr = np.zeros(md_state['positions'].shape)
        md_state['positions'] = zero_arr * unit.nanometers

        # Check that the positions have been modified
        md_context_0 = BLUESSimulation.setContextFromState(
            md_context, md_state)
        pos0 = md_context_0.getState(getPositions=True).getPositions(
            asNumpy=True)

        assert np.not_equal(pos0, pos).all()

    def test_printSimulationTiming(self, blues_sim, caplog):
        caplog.set_level(logging.INFO)
        blues_sim._printSimulationTiming()
        assert 'Total BLUES Simulation Time' in caplog.text
        assert 'Total Force Evaluations' in caplog.text
        assert 'Total NCMC time' in caplog.text
        assert 'Total MD time' in caplog.text

    def test_setStateTable(self, blues_sim, state_keys):
        assert blues_sim.stateTable['md']['state0'] == {}
        md_context = blues_sim._md_sim.context
        md_state = BLUESSimulation.getStateFromContext(md_context, state_keys)
        blues_sim._setStateTable('md', 'state0', md_state)
        assert blues_sim.stateTable['md']['state0'] == md_state

    def test_syncStatesMDtoNCMC(self, blues_sim, state_keys):
        assert blues_sim.stateTable['ncmc']['state0'] == {}
        blues_sim._syncStatesMDtoNCMC()

        md_state = BLUESSimulation.getStateFromContext(
            blues_sim._md_sim.context, state_keys)
        ncmc_state = BLUESSimulation.getStateFromContext(
            blues_sim._ncmc_sim.context, state_keys)

        assert np.equal(blues_sim.stateTable['ncmc']['state0']['positions'],
                        md_state['positions']).all()

    def test_stepNCMC(self, blues_sim):
        ncmc_state0 = blues_sim.stateTable['ncmc']['state0']['positions']
        blues_sim._stepNCMC(10, 5)
        ncmc_state1 = blues_sim.stateTable['ncmc']['state1']['positions']
        assert np.not_equal(ncmc_state0, ncmc_state1).all()

    def test_computeAlchemicalCorrection(self, blues_sim):
        correction_factor = blues_sim._computeAlchemicalCorrection()
        assert isinstance(correction_factor, float)

    def test_acceptRejectMove(self, blues_sim, state_keys, caplog):
        # Check positions are different from stepNCMC
        md_state = BLUESSimulation.getStateFromContext(
            blues_sim._md_sim.context, state_keys)
        ncmc_state = BLUESSimulation.getStateFromContext(
            blues_sim._ncmc_sim.context, state_keys)
        assert np.not_equal(md_state['positions'],
                            ncmc_state['positions']).all()

        caplog.set_level(logging.INFO)
        blues_sim._acceptRejectMove()
        assert 'NCMC MOVE REJECTED' in caplog.text

        # Check positions have been reset to before move
        md_state = BLUESSimulation.getStateFromContext(
            blues_sim._md_sim.context, state_keys)
        ncmc_state = BLUESSimulation.getStateFromContext(
            blues_sim._ncmc_sim.context, state_keys)
        assert np.equal(md_state['positions'], ncmc_state['positions']).all()

    def test_resetSimulations(self, blues_sim, state_keys):
        md_state0 = BLUESSimulation.getStateFromContext(
            blues_sim._md_sim.context, state_keys)

        blues_sim._resetSimulations(300 * unit.kelvin)

        md_state1 = BLUESSimulation.getStateFromContext(
            blues_sim._md_sim.context, state_keys)
        assert np.not_equal(md_state0['velocities'],
                            md_state1['velocities']).all()

    def test_stepMD(self, blues_sim, state_keys):
        md_state0 = BLUESSimulation.getStateFromContext(
            blues_sim._md_sim.context, state_keys)

        blues_sim._stepMD(2)

        md_state1 = BLUESSimulation.getStateFromContext(
            blues_sim._md_sim.context, state_keys)

        # Check positions have changed
        assert np.not_equal(md_state0['positions'],
                            md_state1['positions']).all()

    def test_blues_simulationRunYAML(self, tmpdir, systems, engine):
        yaml_cfg = """
            output_dir: .
            outfname: tol-test
            logger:
              level: info
              stream: True

            system:
              nonbonded: NoCutoff
              constraints: HBonds

            simulation:
              dt: 0.002 * picoseconds
              friction: 1 * 1/picoseconds
              temperature: 400 * kelvin
              nIter: 1
              nstepsMD: 4
              nstepsNC: 4

            md_reporters:
              stream:
                title: md
                reportInterval: 1
                totalSteps: 4 # nIter * nstepsMD
                step: True
                speed: True
                progress: True
                remainingTime: True
                currentIter : True
            ncmc_reporters:
              stream:
                title: ncmc
                reportInterval: 1
                totalSteps: 4 # Use nstepsNC
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

        simulations = SimulationFactory(systems, engine,
                                        cfg['simulation'], cfg['md_reporters'],
                                        cfg['ncmc_reporters'])

        blues = BLUESSimulation(simulations)
        before_iter = blues._md_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)
        blues.run()
        after_iter = blues._md_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)
        #Check that our system has run dynamics
        pos_compare = np.not_equal(before_iter, after_iter).all()
        assert pos_compare

    def test_blues_simulationRunPure(self, systems, simulations, engine, tmpdir):
        print('Testing BLUESSimulation.run() from pure python')
        md_rep_cfg = {
            'stream': {
                'title': 'md',
                'reportInterval': 1,
                'totalSteps': 4,
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
                'totalSteps': 4,
                'step': True,
                'speed': True,
                'progress': True,
                'remainingTime': True,
                'currentIter': True
            }
        }

        md_reporters = ReporterConfig(tmpdir.join('tol-test'), md_rep_cfg).makeReporters()
        ncmc_reporters = ReporterConfig(tmpdir.join('tol-test-ncmc'),
                                        ncmc_rep_cfg).makeReporters()

        cfg = {
            'nprop': 1,
            'prop_lambda': 0.3,
            'dt': 0.001 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 100 * unit.kelvin,
            'nIter': 1,
            'nstepsMD': 4,
            'nstepsNC': 4,
        }
        simulations = SimulationFactory(
            systems,
            engine,
            cfg,
            md_reporters=md_reporters,
            ncmc_reporters=ncmc_reporters)

        blues = BLUESSimulation(simulations)
        before_iter = blues._md_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)
        blues.run()
        after_iter = blues._md_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)
        #Check that our system has run dynamics
        pos_compare = np.not_equal(before_iter, after_iter).all()
        assert pos_compare
