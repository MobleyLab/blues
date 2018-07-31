import os, unittest, parmed, yaml
from blues import utils
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.simulation import SystemFactory, SimulationFactory
from simtk import openmm, unit
from simtk.openmm import app
import numpy as np


class SimulationFactoryTester(unittest.TestCase):
    """
    Test the SimulationFactory class.
    """

    def setUp(self):
        # Load the waterbox with toluene into a structure.
        self.prmtop = utils.get_data_filename('blues',
                                              'tests/data/TOL-parm.prmtop')
        self.inpcrd = utils.get_data_filename('blues',
                                              'tests/data/TOL-parm.inpcrd')
        self.structure = parmed.load_file(self.prmtop, xyz=self.inpcrd)
        self.atom_indices = utils.atomIndexfromTop('LIG',
                                                   self.structure.topology)

        system_cfg = {
            'nonbondedMethod': app.PME,
            'nonbondedCutoff': 8.0 * unit.angstroms,
            'constraints': app.HBonds
        }
        self.systems = SystemFactory(self.structure, self.atom_indices,
                                     system_cfg)

        move = RandomLigandRotationMove(self.structure, 'LIG')
        self.engine = MoveEngine(move)

        self.simulations = SimulationFactory(self.systems, self.engine)
        self.system = self.simulations._system

    def test_addBarostat(self):
        print('Testing MonteCarloBarostat')
        forces = self.system.getForces()
        npt_system = SimulationFactory.addBarostat(self.system)
        npt_forces = npt_system.getForces()

        #Check that forces have been added to the system.
        self.assertNotEqual(len(forces), len(npt_forces))
        #Check that it has added the MonteCarloBarostat
        self.assertIsInstance(npt_forces[-1], openmm.MonteCarloBarostat)

    def test_generateIntegrator(self):
        print('Testing LangevinIntegrator')
        cfg = {
            'temperature': 500 * unit.kelvin,
            'dt': 0.004 * unit.picoseconds
        }
        integrator = SimulationFactory.generateIntegrator(**cfg)
        #Check we made the right integrator
        self.assertIsInstance(integrator, openmm.LangevinIntegrator)
        #Check that the integrator has taken our Parameters
        self.assertEqual(integrator.getTemperature(), cfg['temperature'])
        self.assertEqual(integrator.getStepSize(), cfg['dt'])

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
        self.assertIsInstance(ncmc_integrator,
                              AlchemicalExternalLangevinIntegrator)
        #Check that the integrator has taken our Parameters
        self.assertAlmostEqual(ncmc_integrator.getTemperature()._value,
                               cfg['temperature']._value)
        self.assertEqual(ncmc_integrator.getStepSize(), cfg['dt'])
        self.assertEqual(ncmc_integrator._n_steps_neq, cfg['nstepsNC'])
        self.assertEqual(ncmc_integrator._n_lambda_steps,
                         cfg['nstepsNC'] * cfg['nprop'])
        self.assertEqual(ncmc_integrator._alchemical_functions,
                         cfg['alchemical_functions'])
        self.assertEqual(ncmc_integrator._splitting, cfg['splitting'])
        prop_range = (0.5 - cfg['propLambda'], 0.5 + cfg['propLambda'])
        self.assertEqual(ncmc_integrator._prop_lambda, prop_range)

    def test_generateSimFromStruct(self):
        print('Generating Simulation from parmed.Structure')
        integrator = openmm.LangevinIntegrator(100 * unit.kelvin, 1,
                                               0.002 * unit.picoseconds)
        simulation = SimulationFactory.generateSimFromStruct(
            self.structure, self.system, integrator)

        #Check that we've made a Simulation object
        self.assertIsInstance(simulation, app.Simulation)
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True) / unit.nanometers
        box_vectors = state.getPeriodicBoxVectors(
            asNumpy=True) / unit.nanometers
        struct_box = np.array(
            self.structure.box_vectors.value_in_unit(unit.nanometers))
        struct_pos = np.array(
            self.structure.positions.value_in_unit(unit.nanometers))

        #Check that the box_vectors/positions in the Simulation
        # have been set from the parmed.Structure
        np.testing.assert_array_almost_equal(positions, struct_pos)
        np.testing.assert_array_equal(box_vectors, struct_box)

        print('Attaching Reporter')
        reporters = [app.StateDataReporter('test.log', 5)]
        self.assertEqual(len(simulation.reporters), 0)
        simulation = SimulationFactory.attachReporters(simulation, reporters)
        self.assertEqual(len(simulation.reporters), 1)
        os.remove('test.log')

    def test_generateSimulationSet(self):
        print('Testing generateSimulationSet')
        cfg = {
            'dt': 0.001 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 100 * unit.kelvin,
            'nIter': 1,
            'nstepsMD': 10,
            'nstepsNC': 10,
        }
        simulations = SimulationFactory(self.systems, self.engine)
        simulations.generateSimulationSet(cfg)
        #Check that we've made the MD/ALCH/NCMC simulation set
        self.assertTrue(hasattr(simulations, 'md'))
        self.assertTrue(hasattr(simulations, 'alch'))
        self.assertTrue(hasattr(simulations, 'ncmc'))
        #Check that the physical parameters are equivalent
        self.assertEqual(simulations.ncmc_integrator.getStepSize(), cfg['dt'])
        self.assertEqual(simulations.integrator.getStepSize(), cfg['dt'])
        self.assertAlmostEqual(
            simulations.ncmc_integrator.getTemperature()._value,
            cfg['temperature']._value)
        self.assertAlmostEqual(simulations.integrator.getTemperature()._value,
                               cfg['temperature']._value)

    def test_initSimulationFactory(self):
        print('Testing initialization of SimulationFactory')
        cfg = {
            'nprop': 1,
            'prop_lambda': 0.3,
            'dt': 0.001 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 100 * unit.kelvin,
            'nIter': 1,
            'nstepsMD': 10,
            'nstepsNC': 10,
        }
        simulations = SimulationFactory(self.systems, self.engine, cfg)
        #Check that we've made the MD/ALCH/NCMC simulation set
        self.assertTrue(hasattr(simulations, 'md'))
        self.assertTrue(hasattr(simulations, 'alch'))
        self.assertTrue(hasattr(simulations, 'ncmc'))
        #Check that the physical parameters are equivalent
        self.assertEqual(simulations.ncmc_integrator.getStepSize(), cfg['dt'])
        self.assertEqual(simulations.integrator.getStepSize(), cfg['dt'])
        self.assertAlmostEqual(
            simulations.ncmc_integrator.getTemperature()._value,
            cfg['temperature']._value)
        self.assertAlmostEqual(simulations.integrator.getTemperature()._value,
                               cfg['temperature']._value)


if __name__ == '__main__':
    unittest.main()
