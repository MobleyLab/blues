
import unittest, os, parmed
import blues.utils as utils
import blues_refactor.ncmc as ncmc
from simtk import openmm

class MoveProposalTester(unittest.TestCase):
    """
    Test the MoveProposal class.
    """
    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues_refactor', 'tests/data/TOL-parm.prmtop')
        inpcrd = utils.get_data_filename('blues_refactor', 'tests/data/TOL-parm.inpcrd')
        structure = parmed.load_file(prmtop, xyz=inpcrd)
        print(structure)
        self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : False }
        sims = ncmc.SimulationFactory(structure, self.atom_indices, **self.opt)
        sims.createSimulationSet()

        self.nc_sim = sims.nc
        self.initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

        #Initialize the ModelProperties object
        self.model = ncmc.ModelProperties(structure, 'LIG')
        self.model.calculateProperties()


    def test_random_rotation(self):
        initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        mover = ncmc.MoveProposal(self.model, 'random_rotation', self.opt['nstepsNC'])
        nc_context = mover.nc_move['method'](self.model, self.nc_sim.context)

        rot_pos = nc_context.getState(getPositions=True).getPositions(asNumpy=True)

        #for idx in self.atom_indices:
        #    print('Initial')
        #    print(initial_positions[idx,:]._value)
        #    print('Rotated')
        #    print(rot_pos[idx,:]._value)
        self.assertNotEqual(rot_pos.tolist(), initial_positions.tolist())

if __name__ == "__main__":
        unittest.main()
