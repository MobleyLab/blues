import unittest, parmed
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.moves import WaterTranslationMove
from blues.moves import MoveEngine
from simtk.openmm import app
from simtk import unit
import numpy as np
import mdtraj as md

class WaterTranslationTester(unittest.TestCase):
    """
    Test the RandomLigandRotationMove class.
    """

    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.pdb')
        structure = parmed.load_file(prmtop, xyz=inpcrd)


        #Initialize the Move object
        self.move = WaterTranslationMove(structure, protein_selection='(index 1656) or (index 1657)', radius=0.9*unit.nanometer)
        self.atom_indices = self.move.atom_indices

        self.engine = MoveEngine(self.move)
        self.engine.selectMove()

        self.system_cfg = {'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}
        systems = SystemFactory(structure, self.move.atom_indices, self.system_cfg)

        #Initialize the SimulationFactory object
        self.cfg = {
            'dt': 0.002 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 300 * unit.kelvin,
            'nprop': 1,
            'nIter': 1,
            'nstepsMD': 1,
            'nstepsNC': 100,
            'alchemical_functions': {
                'lambda_sterics':
                'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics':
                'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            }
        }
        self.simulations = SimulationFactory(systems, self.engine, self.cfg)
        self.ncmc_sim = self.simulations.ncmc
        self.initial_positions = self.ncmc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    def test_water_translation_before(self):
        self.original_com_position = self.move.traj.xyz[0][self.move.protein_atoms[0]]

        before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[self.atom_indices, :]
        #new_context = self.engine.runEngine(self.simulations.ncmc.context)
        new_context = self.move.beforeMove(self.simulations.ncmc.context)
        after_move = new_context.getState(getPositions=True).getPositions(
            asNumpy=True)[self.atom_indices, :]

        #Check that the ligand has been rotated
        pos_compare = np.not_equal(before_move, after_move).all()
        self.move.traj.xyz[0][self.move.protein_atoms[0]] = self.original_com_position 
        assert pos_compare

    def test_water_translation_move(self):
        #before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
        #    asNumpy=True)[self.atom_indices, :]
        self.original_com_position = self.ncmc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)[self.move.protein_atoms[0]]

        com_protein_before = self.move.traj.xyz[0,self.move.protein_atoms[0],:]
        self.simulations.ncmc.context = self.move.beforeMove(self.simulations.ncmc.context)
        before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[self.atom_indices, :]
        self.simulations.ncmc.context = self.engine.runEngine(self.simulations.ncmc.context)
        after_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[self.atom_indices, :]
        #Check that the ligand has been rotated
        pos_compare = np.not_equal(before_move, after_move).all()
        assert pos_compare

        #check distance
        com_protein_after = self.move.traj.xyz[0,self.move.protein_atoms[0],:]
        #check that the center of mass refereence has changed
        assert np.not_equal(self.original_com_position, com_protein_after).all()

        #check that the new water position is within the specified radius
        pairs = self.move.traj.topology.select_pairs(np.array(self.move.atom_indices[0]).flatten(), np.array(self.move.protein_atoms[0]).flatten())
        water_distance = md.compute_distances(self.move.traj, pairs, periodic=True)
        water_dist = np.linalg.norm(water_distance)
        assert np.less_equal(water_dist, self.move.radius._value)
        self.move.traj.xyz[0][self.move.protein_atoms[0]] = self.original_com_position 



    def test_water_translation_after(self):
        before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[self.atom_indices, :]
        self.simulations.ncmc.context = self.engine.runEngine(self.simulations.ncmc.context)
        new_context = self.move.beforeMove(self.simulations.ncmc.context)
        new_context = self.move.move(self.simulations.ncmc.context)
        after_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)
        #Check that the move treats inbound water correctly
        assert self.simulations.ncmc.context._integrator.getGlobalVariableByName('protocol_work') == 0
        after_move[self.move.atom_indices] = after_move[self.move.atom_indices] + [self.move.radius._value,0,0] * self.move.radius.unit

        self.simulations.ncmc.context.setPositions(after_move)
        new_context = self.move.afterMove(self.simulations.ncmc.context)
        #Check that the move would be rejected in case it's out of bounds
        assert self.simulations.ncmc.context._integrator.getGlobalVariableByName('protocol_work') >= 999999

if __name__ == "__main__":
    unittest.main()
