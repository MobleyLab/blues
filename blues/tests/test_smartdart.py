import unittest, os, parmed                                                     
from blues.ncmc import Model, SimulationFactory                                 
from blues.models import Model_SmartDart
from simtk import openmm                                                        
from openmmtools import testsystems                                             
import simtk.unit as unit                                                       
import numpy as np                                                              


class MoveProposalTester(unittest.TestCase):
    """
    Test the MoveProposal class.
    """
    def setUp(self):
        # Obtain topologies/positions
        testsystem = testsystems.AlanineDipeptideVacuum()
        structure = parmed.openmm.topsystem.load_topology(topology=testsystem.topology,
                                            system=testsystem.system,
                                            xyz=testsystem.positions)

        self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : False }


        #Initialize the Model object
        basis_particles = [0,2,7]
        self.model = Model_SmartDart(structure, basis_particles=basis_particles,
                                resname='ALA')
        self.model.calculateProperties()
        #Initialize the SimulationFactory object
        sims = ncmc.SimulationFactory(structure, self.model, **self.opt)
        system = sims.generateSystem(structure, **self.opt)
        alch_system = sims.generateAlchSystem(system, self.model.atom_indices)
        self.nc_sim = sims.generateSimFromStruct(structure, alch_system, ncmc=True, **self.opt)

        self.initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        self.mover = ncmc.MoveProposal(self.model, 'smart_dart', self.opt['nstepsNC'])

    def test_smart_dart(self):
        orig_pos = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        com = self.getCenterOfMass(orig_pos, self.masses)
        com_new_coord = findNewCoord(pos[basis_part[0]], pos[basis_part[1]], pos[basis_part[2]], com)
        move_coord = com_new_coord[:] + np.array([1,1,1])*unit.nanometers
        standard_dart = np.array([-0.0096823 ,  0.50751791,  0.060064  ])

        for i in range(20):
            self.model.n_dartboard = [com_new_coord, move_coord]
            model, nc_context = self.mover.moves['method'](self.model, self.nc_sim.context)
            pos_new = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
            for x in range(3):
                if pos_new[0][x]._value - results[0][x]._value == 0:
                    np.testing.assert_almost_equal(pos_new[0][x]._value, results[0][x]._value, decimal=1)
                else:
                    np.testing.assert_almost_equal(pos_new[0][x]._value, (results[0][x]._value + standard_dart[x]), decimal=1)


if __name__ == "__main__":
        unittest.main()
