
import unittest, os, parmed
import blues.utils as utils
import blues_refactor.ncmc as ncmc
from blues_refactor.ncmc import *
from simtk import openmm
from openmmtools import testsystems
import simtk.unit as unit
import numpy as np
class BLUESTester(unittest.TestCase):
    """
    Test the ModelProperties class.
    """
    def setUp(self):
        # Load the waterbox with toluene into a structure.
        self.prmtop = utils.get_data_filename('blues_refactor', 'tests/data/TOL-parm.prmtop')
        self.inpcrd = utils.get_data_filename('blues_refactor', 'tests/data/TOL-parm.inpcrd')
        self.full_struct = parmed.load_file(self.prmtop, xyz=self.inpcrd)

        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : True }

    def test_modelproperties(self):
        # Model.structure must be residue selection.
        model = ModelProperties(self.full_struct, 'LIG')
        self.assertNotEqual(len(model.structure.atoms), len(self.full_struct.atoms))

        # Test each function separately
        masses = model.getMasses(model.structure)
        self.assertNotEqual(len(masses), 0)

        totalmass = model.getTotalMass(masses)
        self.assertEqual(totalmass, masses.sum())

        center_of_mass = model.getCenterOfMass(model.structure, masses)
        self.assertNotEqual(center_of_mass, [0, 0, 0])

        # Test function that calcs all properties
        # Ensure properties are same as returned values
        model.calculateProperties()
        self.assertEqual(masses.tolist(), model.masses.tolist())
        self.assertEqual(totalmass, model.totalmass)
        self.assertEqual(center_of_mass.tolist(), model.center_of_mass.tolist())


if __name__ == "__main__":
        unittest.main()
