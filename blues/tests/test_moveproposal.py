import unittest, parmed
from blues import utils
from blues.simulation import SimulationFactory
import blues

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
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : False }


        #Initialize the Move object
        self.move = blues.moves.RandomLigandRotationMove(structure, 'LIG')
        self.move.calculateProperties()
        self.engine = blues.engine.MoveEngine(self.move)
        self.engine.selectMove()
        #Initialize the SimulationFactory object
        sims = SimulationFactory(structure, self.move, **self.opt)
        system = sims.generateSystem(structure, **self.opt)
        alch_system = sims.generateAlchSystem(system, self.atom_indices)
        self.nc_sim = sims.generateSimFromStruct(structure, alch_system, ncmc=True, **self.opt)

        self.initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

    def test_random_rotation(self):
        nc_context = self.engine.runEngine(self.nc_sim.context)
        rot_pos = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
        self.assertNotEqual(rot_pos.tolist(), self.initial_positions.tolist())

if __name__ == "__main__":
        unittest.main()
