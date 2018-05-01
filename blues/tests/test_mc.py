
import unittest, parmed
import numpy as np
from openmmtools.testsystems import get_data_filename
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.simulation import Simulation, SimulationFactory
from simtk import openmm
from openmmtools import testsystems
import types
import math

class BLUESTester(unittest.TestCase):
    """
    Test the Simulation class.
    """
    def setUp(self):
        # Load the waterbox with toluene into a structure.
        self.prmtop = get_data_filename("data/alanine-dipeptide-explicit/alanine-dipeptide.prmtop")
        self.inpcrd = get_data_filename("data/alanine-dipeptide-explicit/alanine-dipeptide.crd")
        self.full_struct = parmed.load_file(self.prmtop, xyz=self.inpcrd)
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.00002,
                'nIter' : 2, 'nstepsNC' : 4, 'nstepsMD' : 2, 'nprop' : 1,
                'nonbondedMethod' : 'PME', 'nonbondedCutoff': 1, 'constraints': 'HBonds',
                'trajectory_interval' : 1, 'reporter_interval' : 1, 'outfname' : 'mc-test',
                'platform' : None }


    def test_simulationRun(self):
        """Tests the Simulation.runMC() function"""
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.00002,
                'nIter' : 2, 'nstepsNC' : 2, 'nstepsMD' : 1, 'nprop' : 1,
                'nonbondedMethod' : 'NoCutoff', 'constraints': 'HBonds',
                'trajectory_interval' : 1, 'reporter_interval' : 1,
                'outfname' : 'mc-test',
                'platform' : None,
                'constraints' : 'HBonds',
                'mc_per_iter' : 2 }

        structure = self.full_struct
        class SetRotationMove(RandomLigandRotationMove):
            def __init__(self, structure, resname='LIG'):
                super(SetRotationMove, self).__init__(structure, resname)

            def move(self, context):
                """Function that performs a random rotation about the
                center of mass of the ligand.
                """
               #TODO: check if we need to deepcopy
                positions = context.getState(getPositions=True).getPositions(asNumpy=True)

                self.positions = positions[self.atom_indices]
                self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)
                reduced_pos = self.positions - self.center_of_mass

                # Define random rotational move on the ligand
                #set rotation so that test is reproducible
                set_rotation_matrix = np.array([[-0.62297988, -0.17349253,  0.7627558 ],
                                                [ 0.55082352, -0.78964857,  0.27027502],
                                                [ 0.55541834,  0.58851973,  0.58749893]])


                #set_rotation_matrix = np.array([[1, 0, 0],
                #                                 [0, 1, 0],
                #                                 [0, 0, 1]])

                #multiply lig coordinates by rot matrix and add back COM translation from origin
                rot_move = np.dot(reduced_pos, set_rotation_matrix) * positions.unit + self.center_of_mass

                # Update ligand positions in nc_sim
                for index, atomidx in enumerate(self.atom_indices):
                    positions[atomidx] = rot_move[index]
                context.setPositions(positions)
                positions = context.getState(getPositions=True).getPositions(asNumpy=True)
                self.positions = positions[self.atom_indices]
                return context


        self.model = SetRotationMove(structure, resname='ALA')
        #self.model = RandomLigandRotationMove(structure, resname='ALA')

        self.model.atom_indices = range(22)
        self.model.topology = structure[self.model.atom_indices].topology
        self.model.positions = structure[self.model.atom_indices].positions
        self.model.calculateProperties()

        self.mover = MoveEngine(self.model)
        #Initialize the SimulationFactory object
        sims = SimulationFactory(structure, self.mover, **self.opt)
        #print(sims)
        self.nc_sim = sims.nc
        self.model.calculateProperties()
        self.initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        mc_sim = Simulation(sims, **self.opt)
        #monkeypatch to access acceptance value
        def nacceptRejectMC(self, temperature=300, **opt):
            """Function that chooses to accept or reject the proposed move.
            """
            md_state0 = self.current_state['md']['state0']
            md_state1 = self.current_state['md']['state1']
            log_mc = (md_state1['potential_energy'] - md_state0['potential_energy']) * (-1.0/self.nc_sim.context._integrator.kT)
            randnum =  math.log(np.random.random())

            if log_mc > randnum:
                self.accept += 1
                print('MC MOVE ACCEPTED: log_mc {} > randnum {}'.format(log_mc, randnum) )
                self.md_sim.context.setPositions(md_state1['positions'])
            else:
                self.reject += 1
                print('MC MOVE REJECTED: log_mc {} < {}'.format(log_mc, randnum) )
                self.md_sim.context.setPositions(md_state0['positions'])
            self.log_mc = log_mc
            self.md_sim.context.setVelocitiesToTemperature(temperature)
        mc_sim.acceptRejectMC = nacceptRejectMC
        nacceptRejectMC.__get__(mc_sim)
        mc_sim.acceptRejectMC = types.MethodType(nacceptRejectMC, mc_sim)
        mc_sim.runMC(self.opt['nIter'])
        #get log acceptance
        print(mc_sim.log_mc)
        #if mc is working, should be around -24.1
        assert mc_sim.log_mc <= -23.8 and mc_sim.log_mc >= -24.3

if __name__ == "__main__":
        unittest.main()
