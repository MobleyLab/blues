
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
        self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : True }

        system = ncmc.SimulationFactory.generateSystem(structure, **self.opt)
        alch_system = ncmc.SimulationFactory.generateAlchSystem(system, self.atom_indices)
        self.nc_sim = ncmc.SimulationFactory.generateSimFromStruct(structure, alch_system, self.functions, **self.opt)

        self.initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

        #Initialize the ModelProperties object
        self.model = ncmc.ModelProperties(self.nc_sim, self.atom_indices)
        self.mover = ncmc.MoveProposal(self.nc_sim, self.model,
                                 'random_rotation', self.opt['nstepsNC'])

    def test_random_rotation(self):
        nc_sim = self.mover.random_rotation(self.mover.nc_sim)
        rot_pos = nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

        for idx in self.atom_indices:
            print('Initial')
            print(self.initial_positions[idx,:]._value)
            print('Rotated')
            print(rot_pos[idx,:]._value)
        #self.assertNotEqual(rot_pos.tolist(), self.initial_positions.tolist())

if __name__ == "__main__":
        unittest.main()
