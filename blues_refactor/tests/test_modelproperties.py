
import unittest, os, parmed
import blues.utils as utils
import blues_refactor.ncmc as ncmc
from simtk import openmm

class ModelPropertiesTester(unittest.TestCase):
    """
    Test the ModelProperties class.
    """
    def setUp(self):
        # Obtain topologies/positions
        prmtop = 'tests/data/TOL-gaff.prmtop'
        inpcrd = 'tests/data/TOL-gaff.inpcrd'
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

        #Initialize the ModelProperties object
        self.model = ncmc.ModelProperties(self.nc_sim, self.atom_indices)

    def test_get_masses(self):
        masses = self.model.getMasses(self.nc_sim.context, self.atom_indices)
        self.assertEqual(masses.sum(), self.model.totalmass)

    def test_get_positions(self):
        positions = self.model.getPositions(self.nc_sim.context, self.atom_indices)
        self.assertEqual(len(positions), len(self.atom_indices))

    def test_calculate_COM(self):
        self.model.calculateCOM()
        self.assertNotEqual(self.model.center_of_mass)

if __name__ == "__main__":
        unittest.main()
