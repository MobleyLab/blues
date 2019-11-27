import unittest, parmed
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.moves import RandomRotatableBondMove 
from blues.moves import MoveEngine
from simtk.openmm import app
from simtk import unit
import numpy as np


class RandomRotatableBondMove(unittest.TestCase):
    """
    Test the RandomRotatableBondMove class.
    """

    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues', 'tests/data/eqRef_2gmx.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/eqRef_2gmx.inpcrd')
        structure = parmed.load_file(prmtop, xyz=inpcrd)
        dihedral_atoms = ["C10", "C9", "C3", "C2" ]
        alch_list = ['C9', 'H92', 'H93', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'H1', 'H2', 'H4', 'H5', 'H6']

        self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)

        #Initialize the Move object
        self.move = RandomRotatableBondMove(structure, prmtop, inpcrd, dihedral_atoms, alch_list, 'LIG')
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
            'nstepsNC': 10,
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

    def test_random_rotation_bond(self):
        before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[self.atom_indices, :]
        self.simulations.ncmc.context = self.engine.runEngine(self.simulations.ncmc.context)
        after_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[self.atom_indices, :]

        #Check that the ligand has been rotated
        pos_compare = np.not_equal(before_move, after_move).all()
        assert pos_compare


if __name__ == "__main__":
    unittest.main()
