import unittest, parmed
from blues import utils
from blues.simulation import SimulationFactory, SystemFactory
from blues.moves import SmartDartMove
from blues.engine import MoveEngine
from openmmtools import testsystems
import simtk.unit as unit
import numpy as np
from simtk.openmm import app


class SmartDartTester(unittest.TestCase):
    """
    Test the SmartDartMove.move() function.
    """

    def setUp(self):
        # Obtain topologies/positions
        prmtop = testsystems.get_data_filename(
            "data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop")
        inpcrd = testsystems.get_data_filename(
            "data/alanine-dipeptide-gbsa/alanine-dipeptide.crd")
        testsystem = testsystems.AlanineDipeptideVacuum(constraints=None)
        structure = parmed.openmm.topsystem.load_topology(
            topology=testsystem.topology,
            system=testsystem.system,
            xyz=testsystem.positions)

        #Initialize the Model object
        basis_particles = [0, 2, 7]
        self.move = SmartDartMove(
            structure,
            basis_particles=basis_particles,
            coord_files=[inpcrd, inpcrd],
            topology=prmtop,
            resname='ALA',
            self_dart=True)
        self.move.atom_indices = range(22)
        self.move.topology = structure.topology
        self.move.positions = structure.positions
        self.move_engine = MoveEngine(self.move)

        self.system_cfg = {
            'nonbondedMethod': app.NoCutoff,
            'constraints': app.HBonds
        }
        self.systems = SystemFactory(structure, self.move.atom_indices,
                                     self.system_cfg)

        #Initialize the SimulationFactory object
        self.cfg = {
            'dt': 0.002 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 300 * unit.kelvin,
            'nIter': 10,
            'nstepsMD': 50,
            'nstepsNC': 10,
            'alchemical_functions': {
                'lambda_sterics':
                'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics':
                'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            }
        }
        sims = SimulationFactory(self.systems, self.move_engine, self.cfg)
        self.ncmc_sim = sims.ncmc
        self.move.calculateProperties()
        self.initial_positions = self.ncmc_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)

    def test_smart_dart(self):
        """Creates two darting regions, one at the center of mass origin,
         and one displaced a little way from the center,
         and checks to see if the alainine dipeptide correctly jumps between the two"""
        orig_pos = self.ncmc_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)
        com = self.move.getCenterOfMass(orig_pos[self.move.atom_indices],
                                        self.move.masses)
        basis_part = self.move.basis_particles
        com_new_coord = self.move._findNewCoord(orig_pos[basis_part[0]],
                                                orig_pos[basis_part[1]],
                                                orig_pos[basis_part[2]], com)
        move_coord = com_new_coord[:] + np.array([1, 1, 1]) * unit.nanometers
        #standard_dart is the new positions of the center of mass if a move occurs
        standard_dart = np.array([-0.0096823, 0.50751791, 0.060064])
        #dart_counter checks how many times darting to another dart occurs
        #should be greater than 1
        dart_counter = 0
        #attempt the darting move 20 times
        for i in range(20):
            #set the new darts using the original settings
            self.move.n_dartboard = [com_new_coord, move_coord]
            nc_context = self.move.move(self.ncmc_sim.context)
            pos_new = nc_context.getState(getPositions=True).getPositions(
                asNumpy=True)
            for x in range(3):
                #postions of the coordinates either should be the same, or displaced by the standard_dart value
                if pos_new[0][x]._value - orig_pos[0][x]._value == 0:
                    np.testing.assert_almost_equal(
                        pos_new[0][x]._value, orig_pos[0][x]._value, decimal=1)
                else:
                    np.testing.assert_almost_equal(
                        pos_new[0][x]._value,
                        (orig_pos[0][x]._value + standard_dart[x]),
                        decimal=1)
                    dart_counter = dart_counter + 1
            #reset and rerun
            nc_context.setPositions(orig_pos)
        #if the move worked, dart_counter should be greater than 0
        assert dart_counter > 0

    def test_self_dart(self):
        """same as test_smart_dart, but with self.self_dart=False, which should always cause
        the com to jump to a different dart, as opposed to having a chance to stay where it is"""
        self.move.self_dart = False
        orig_pos = self.ncmc_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)
        com = self.move.getCenterOfMass(orig_pos[self.move.atom_indices],
                                        self.move.masses)
        basis_part = self.move.basis_particles
        com_new_coord = self.move._findNewCoord(orig_pos[basis_part[0]],
                                                orig_pos[basis_part[1]],
                                                orig_pos[basis_part[2]], com)
        move_coord = com_new_coord[:] + np.array([1, 1, 1]) * unit.nanometers
        standard_dart = np.array([-0.0096823, 0.50751791, 0.060064])
        #dart_counter checks how many times darting to another dart occurs
        #should be greater than 1
        dart_counter = 0
        #attempt the darting move 20 times
        for i in range(20):
            #set the new darts using the original settings
            self.move.n_dartboard = [com_new_coord, move_coord]
            nc_context = self.move.move(self.ncmc_sim.context)
            pos_new = nc_context.getState(getPositions=True).getPositions(
                asNumpy=True)
            for x in range(3):
                #postions of the coordinates either should be the same, or displaced by the standard_dart value
                if pos_new[0][x]._value - orig_pos[0][x]._value == 0:
                    np.testing.assert_almost_equal(
                        pos_new[0][x]._value, orig_pos[0][x]._value, decimal=1)
                else:
                    np.testing.assert_almost_equal(
                        pos_new[0][x]._value,
                        (orig_pos[0][x]._value + standard_dart[x]),
                        decimal=1)
                    dart_counter = dart_counter + 1
            #reset and rerun
            nc_context.setPositions(orig_pos)
        #if the move worked, dart_counter should equal 20*3
        assert dart_counter == 60


class DartLoaderTester(unittest.TestCase):
    """
    Test if darts are made successfully
    using SmartDartMove.dartsFromParmEd.
    """

    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
        structure = parmed.load_file(prmtop, xyz=inpcrd)

        self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)

        #Initialize the SmartDartMove object
        self.move = SmartDartMove(
            structure,
            basis_particles=[100, 110, 150],
            coord_files=[inpcrd, inpcrd],
            topology=prmtop,
            self_dart=False,
            resname='LIG',
        )
        self.engine = MoveEngine(self.move)
        self.engine.selectMove()

        self.system_cfg = {
            'nonbondedMethod': app.NoCutoff,
            'constraints': app.HBonds
        }
        systems = SystemFactory(structure, self.move.atom_indices,
                                self.system_cfg)

        #Initialize the SimulationFactory object
        self.cfg = {
            'dt': 0.002 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 300 * unit.kelvin,
            'nIter': 10,
            'nstepsMD': 50,
            'nstepsNC': 10,
            'alchemical_functions': {
                'lambda_sterics':
                'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics':
                'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            }
        }
        sims = SimulationFactory(systems, self.engine, self.cfg)
        self.ncmc_sim = sims.ncmc

        self.initial_positions = self.ncmc_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)

    def test_dartsFromParmEd(self):
        #load files to see if there are any errors
        prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
        self.move.dartsFromParmEd(coord_files=[inpcrd], topology=prmtop)


if __name__ == "__main__":
    unittest.main()
