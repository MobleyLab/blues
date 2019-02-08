import unittest, parmed
from blues import utils
from blues.simulation import SimulationFactory
from blues.moves import SideChainMove
from blues.engine import MoveEngine
from openmmtools import testsystems
import simtk.unit as unit
import numpy as np

class SideChainTester(unittest.TestCase):
    """
    Test the SmartDartMove.move() function.
    """
    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues', 'tests/data/vacDivaline.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/vacDivaline.inpcrd')
        self.struct = parmed.load_file(prmtop, xyz=inpcrd)


        #self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10, 'outfname' : 'vacDivaline',
                'platform' : None,
                'verbose' : False }
        self.sidechain = SideChainMove(self.struct, [1])
        self.mover = MoveEngine(self.sidechain)

    def test_getRotBondAtoms(self):

        vals = [v for v in self.sidechain.rot_atoms[1]['chis'][1]['atms2mv']]
        print(vals)
        self.assertEqual(len(vals), 11)
        #Ensure it selects 1 rotatable bond in Valine
        self.assertEqual(len(self.sidechain.rot_bonds), 1)

    def test_sidechain_move(self):
        simulations = SimulationFactory(self.struct, self.mover, **self.opt)
        simulations.createSimulationSet()


        nc_context = simulations.nc.context
        self.sidechain.beforeMove(nc_context)
        self.sidechain.move(nc_context, verbose=True)

if __name__ == "__main__":
        unittest.main()
