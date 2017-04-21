
import unittest, os, parmed
from blues import utils
from blues.ncmc import Model, SimulationFactory
from simtk import openmm
from openmmtools import testsystems
import simtk.unit as unit
import numpy as np
class BLUESTester(unittest.TestCase):
    """
    Test the Model class.
    """
    def setUp(self):
        # Load the waterbox with toluene into a structure.
        self.prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
        self.inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
        self.full_struct = parmed.load_file(self.prmtop, xyz=self.inpcrd)
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : True }


    def test_modelproperties(self):
        # Model.structure must be residue selection.
        model = Model(self.full_struct, 'LIG')
        self.assertNotEqual(model.topology.getNumAtoms(), len(self.full_struct.atoms))

        # Test each function separately
        masses, totalmass = model.getMasses(model.topology)
        self.assertNotEqual(len(masses), 0)
        self.assertEqual(totalmass, masses.sum())

        center_of_mass = model.getCenterOfMass(model.positions, masses)
        self.assertNotEqual(center_of_mass, [0, 0, 0])

        # Test function that calcs all properties
        # Ensure properties are same as returned values
        model.calculateProperties()
        self.assertEqual(masses.tolist(), model.masses.tolist())
        self.assertEqual(totalmass, model.totalmass)
        self.assertEqual(center_of_mass.tolist(), model.center_of_mass.tolist())

    def test_simulationfactory(self):
        #Initialize the SimulationFactory object
        model = Model(self.full_struct, 'LIG')
        sims = SimulationFactory(self.full_struct, model, **self.opt)

        system = sims.generateSystem(self.full_struct, **self.opt)
        self.assertIsInstance(system, openmm.System)

        alch_system = sims.generateAlchSystem(system, model.atom_indices)
        self.assertIsInstance(alch_system, openmm.System)

        md_sim = sims.generateSimFromStruct(self.full_struct, system, **self.opt)
        self.assertIsInstance(md_sim, openmm.app.simulation.Simulation)

        nc_sim = sims.generateSimFromStruct(self.full_struct, alch_system, ncmc=True, **self.opt)
        self.assertIsInstance(nc_sim, openmm.app.simulation.Simulation)

        #sims.createSimulationSet()
if __name__ == "__main__":
        unittest.main()
