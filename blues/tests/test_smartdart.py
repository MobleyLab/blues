import unittest, parmed
from blues.simulation import SimulationFactory
from blues.moves import SmartDartMove
from blues.engine import MoveEngine
from openmmtools import testsystems
import simtk.unit as unit
import numpy as np


class MoveProposalTester(unittest.TestCase):
    """
    Test the MoveProposal class.
    """
    def setUp(self):
        # Obtain topologies/positions
        testsystem = testsystems.AlanineDipeptideVacuum(constraints=None)
        structure = parmed.openmm.topsystem.load_topology(topology=testsystem.topology,
                                            system=testsystem.system,
                                            xyz=testsystem.positions,
                                            )

        #self.atom_indices = utils.atomIndexfromTop('LIG', structure.topology)
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
                'nonbondedMethod' : 'NoCutoff', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 10, 'reporter_interval' : 10,
                'platform' : None,
                'verbose' : False }


        #Initialize the Model object
        basis_particles = [0,2,7]
        self.move = SmartDartMove(structure, basis_particles=basis_particles,
                                resname='ALA')
        self.move.atom_indices = range(22)
        self.move.topology = structure.topology
        self.move.positions = structure.positions
        self.move.calculateProperties()
        self.move_engine = MoveEngine(self.move)

        #Initialize the SimulationFactory object
        sims = SimulationFactory(structure, self.move_engine, **self.opt)
        system = sims.generateSystem(structure, **self.opt)
        alch_system = sims.generateAlchSystem(system, self.move.atom_indices)
        self.nc_sim = sims.generateSimFromStruct(structure, alch_system, ncmc=True, **self.opt)
        self.move.calculateProperties()
        self.initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

    def test_smart_dart(self):
        """Creates two darting regions, one at the center of mass origin,
         and one displaced a little way from the center,
         and checks to see if the alainine dipeptide correctly jumps between the two"""
        orig_pos = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        com = self.move.getCenterOfMass(orig_pos[self.move.atom_indices], self.move.masses)
        basis_part = self.move.basis_particles
        com_new_coord = self.move._findNewCoord(orig_pos[basis_part[0]], orig_pos[basis_part[1]], orig_pos[basis_part[2]], com)
        move_coord = com_new_coord[:] + np.array([1,1,1])*unit.nanometers
        standard_dart = np.array([-0.0096823 ,  0.50751791,  0.060064  ])

        for i in range(20):
            #set the new darts using the original settings
            self.move.n_dartboard = [com_new_coord, move_coord]
            nc_context = self.move.move(self.nc_sim.context)
            pos_new = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
            for x in range(3):
                #postions of the coordinates either should be the same, or displaced by the standard_dart value
                if pos_new[0][x]._value - orig_pos[0][x]._value == 0:
                    np.testing.assert_almost_equal(pos_new[0][x]._value, orig_pos[0][x]._value, decimal=1)
                else:
                    np.testing.assert_almost_equal(pos_new[0][x]._value, (orig_pos[0][x]._value + standard_dart[x]), decimal=1)
            print(pos_new)
            #reset and rerun
            nc_context.setPositions(orig_pos)


if __name__ == "__main__":
        unittest.main()

