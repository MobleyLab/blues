
import unittest, os, parmed
import blues.utils as utils
import blues_refactor.ncmc as ncmc
from simtk import openmm

class SimulationFactoryTester(unittest.TestCase):
    """
    Test the SimulationFactory class.
    """
    def setUp(self):
        # Obtain topologies/positions
        self.prmtop = utils.get_data_filename('blues_refactor', 'tests/data/TOL-parm.prmtop')
        self.inpcrd = utils.get_data_filename('blues_refactor', 'tests/data/TOL-parm.inpcrd')
        self.structure = parmed.load_file(self.prmtop, xyz=self.inpcrd)
        self.atom_indices = utils.atomIndexfromTop('LIG', self.structure.topology)
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : True }
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        #Initialize the SimulationFactory object
        self.sims = ncmc.SimulationFactory(self.structure, self.atom_indices, **self.opt)

    def test_system_generation(self):
        system = self.sims.generateSystem(self.structure, **self.opt)
        self.assertIsInstance(system, openmm.System)

        alch_system = self.sims.generateAlchSystem(system, self.atom_indices)
        self.assertIsInstance(alch_system, openmm.System)

    def test_md_simulation_generation(self):
        system = self.sims.generateSystem(self.structure, **self.opt)
        md_sim = self.sims.generateSimFromStruct(self.structure, system, self.functions, **self.opt)
        self.assertIsInstance(md_sim, openmm.app.simulation.Simulation)

    def test_nc_simulation_generation(self):
        system = self.sims.generateSystem(self.structure, **self.opt)
        self.assertIsInstance(system, openmm.System)

        alch_system = self.sims.generateAlchSystem(system, self.atom_indices)
        self.assertIsInstance(alch_system, openmm.System)

        nc_sim = self.sims.generateSimFromStruct(self.structure, alch_system, self.functions, ncmc=True,  **self.opt)
        self.assertIsInstance(nc_sim, openmm.app.simulation.Simulation)

if __name__ == "__main__":
        unittest.main()
