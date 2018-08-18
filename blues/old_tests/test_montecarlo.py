import os, unittest, parmed, yaml
from blues import utils
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.simulation import SystemFactory, SimulationFactory, MonteCarloSimulation
from blues.reporters import ReporterConfig
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import testsystems
import numpy as np
import types, math


class MonteCarloSimulationTester(unittest.TestCase):
    """
    Test the MonteCarloSimulation class.
    """

    def setUp(self):

        prmtop = testsystems.get_data_filename(
            "data/alanine-dipeptide-explicit/alanine-dipeptide.prmtop")
        inpcrd = testsystems.get_data_filename(
            "data/alanine-dipeptide-explicit/alanine-dipeptide.crd")
        self.structure = parmed.load_file(prmtop, xyz=inpcrd)

        class SetRotationMove(RandomLigandRotationMove):
            def __init__(self, structure, resname='ALA'):
                super(SetRotationMove, self).__init__(structure, resname)

            def move(self, context):
                """Function that performs a random rotation about the
                center of mass of the ligand. Define a set rotation
                for reproducibility
                """
                positions = context.getState(getPositions=True).getPositions(
                    asNumpy=True)

                self.positions = positions[self.atom_indices]
                self.center_of_mass = self.getCenterOfMass(
                    self.positions, self.masses)
                reduced_pos = self.positions - self.center_of_mass

                # Define random rotational move on the ligand
                #set rotation so that test is reproducible
                set_rotation_matrix = np.array(
                    [[-0.62297988, -0.17349253,
                      0.7627558], [0.55082352, -0.78964857, 0.27027502],
                     [0.55541834, 0.58851973, 0.58749893]])

                #multiply lig coordinates by rot matrix and add back COM translation from origin
                rot_move = np.dot(reduced_pos, set_rotation_matrix
                                  ) * positions.unit + self.center_of_mass

                # Update ligand positions in nc_sim
                for index, atomidx in enumerate(self.atom_indices):
                    positions[atomidx] = rot_move[index]
                context.setPositions(positions)
                positions = context.getState(getPositions=True).getPositions(
                    asNumpy=True)
                self.positions = positions[self.atom_indices]
                return context

        self.move = SetRotationMove(self.structure, resname='ALA')
        self.move.atom_indices = range(22)
        self.move.topology = self.structure[self.move.atom_indices].topology
        self.move.positions = self.structure[self.move.atom_indices].positions
        self.move.calculateProperties()

        self.engine = MoveEngine(self.move)

        system_cfg = {
            'nonbondedMethod': app.NoCutoff,
            'constraints': app.HBonds
        }

        #self.systems = self.structure.createSystem(**system_cfg)
        self.systems = SystemFactory(self.structure, self.move.atom_indices,
                                     system_cfg)

        mc_rep_cfg = {
            'stream': {
                'title': 'mc',
                'reportInterval': 1,
                'totalSteps': 4,
                'step': True,
                'speed': True,
                'progress': True,
                'remainingTime': True,
                'currentIter': True
            }
        }
        mc_reporters = ReporterConfig('ala-dipep-vac',
                                      mc_rep_cfg).makeReporters()

        cfg = {
            'nprop': 1,
            'dt': 0.000021 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 300 * unit.kelvin,
            'nIter': 2,
            'nstepsMD': 1,
            'mc_per_iter': 2
        }
        self.simulations = SimulationFactory(
            self.systems, self.engine, cfg, md_reporters=mc_reporters)

    def test_montecarlo_simulationRunPure(self):
        montecarlo = MonteCarloSimulation(self.simulations)
        initial_positions = montecarlo._ncmc_sim.context.getState(
            getPositions=True).getPositions(asNumpy=True)

        #monkeypatch to access acceptance value
        def nacceptRejectMC(self, temperature=300, **opt):
            """Function that chooses to accept or reject the proposed move.
            """
            md_state0 = montecarlo.stateTable['md']['state0']
            md_state1 = montecarlo.stateTable['md']['state1']
            work_mc = (md_state1['potential_energy'] -
                       md_state0['potential_energy']) * (
                           -1.0 / montecarlo._ncmc_sim.context._integrator.kT)
            randnum = math.log(np.random.random())

            if work_mc > randnum:
                montecarlo.accept += 1
                print('MC MOVE ACCEPTED: work_mc {} > randnum {}'.format(
                    work_mc, randnum))
                montecarlo._md_sim.context.setPositions(md_state1['positions'])
            else:
                montecarlo.reject += 1
                print('MC MOVE REJECTED: work_mc {} < {}'.format(
                    work_mc, randnum))
                montecarlo._md_sim.context.setPositions(md_state0['positions'])
            montecarlo.work_mc = work_mc
            montecarlo._md_sim.context.setVelocitiesToTemperature(temperature)

        #Overwrite method
        montecarlo._accept_reject_move_ = nacceptRejectMC
        nacceptRejectMC.__get__(montecarlo)
        montecarlo.acceptRejectMC = types.MethodType(nacceptRejectMC,
                                                     montecarlo)
        montecarlo.run(2)
        #get log acceptance
        print(montecarlo.work_mc)
        #if mc is working, should be around -24.1
        assert montecarlo.work_mc <= -23.8 and montecarlo.work_mc >= -24.3


if __name__ == '__main__':
    unittest.main()
