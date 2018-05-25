import unittest, parmed
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from simtk.openmm import app
from simtk import unit
import numpy as np

class MoveEngineTester(unittest.TestCase):
    """
    Test the MoveEngine class.
    """
    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
        structure = parmed.load_file(prmtop, xyz=inpcrd)

        self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)

        #Initialize the Move object
        self.move = RandomLigandRotationMove(structure, 'LIG')
        self.engine = MoveEngine(self.move)

        self.system_cfg = { 'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}
        systems = SystemFactory(structure, self.move.atom_indices, self.system_cfg)

        #Initialize the SimulationFactory object
        self.cfg = { 'dt' : 0.002 * unit.picoseconds,
                'friction' : 1 * 1/unit.picoseconds,
                'temperature' : 300 * unit.kelvin,
                'nprop' : 1,
                'nIter': 1,
                'nstepsMD': 1,
                'nstepsNC': 4,
                'alchemical_functions' : {
                    'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                    'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
            }
        self.simulations = SimulationFactory(systems, self.engine, self.cfg)
        self.ncmc_sim = self.simulations.ncmc
        self.initial_positions = self.ncmc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

    def test_random_rotation(self):
        before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(asNumpy=True)
        blues = BLUESSimulation(self.simulations)
        blues.run()
        after_move = blues._ncmc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

        #Check that our system has run dynamics
        # Integrator must step for context to update positions
        pos_compare = np.not_equal(before_move, after_move).all()
        self.assertTrue(pos_compare)


if __name__ == "__main__":
        unittest.main()
