import os, unittest, parmed, yaml
from blues import utils
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.reporters import ReporterConfig
from blues.settings import Settings
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import testsystems
import numpy as np

class BLUESSimulationTester(unittest.TestCase):
    """
    Test the BLUESSimulation class.
    """
    def setUp(self):
        testsystem = testsystems.AlanineDipeptideVacuum(constraints=None)
        self.structure = parmed.openmm.topsystem.load_topology(topology=testsystem.topology,
                                            system=testsystem.system,
                                            xyz=testsystem.positions)
        self.move = RandomLigandRotationMove(self.structure, 'ALA')
        self.engine = MoveEngine(self.move)
        system_cfg = { 'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}
        self.systems = SystemFactory(self.structure, self.move.atom_indices, system_cfg)

    def test_blues_simulationRunYAML(self):
        yaml_cfg = """
            output_dir: .
            outfname: ala-dipep-vac
            logger_level: info

            system:
              nonbonded: NoCutoff
              constraints: HBonds

            simulation:
              dt: 0.002 * picoseconds
              friction: 1 * 1/picoseconds
              temperature: 400 * kelvin
              nIter: 1
              nstepsMD: 4
              nstepsNC: 4

            md_reporters:
              stream:
                title: md
                reportInterval: 1
                totalSteps: 4 # nIter * nstepsMD
                step: True
                speed: True
                progress: True
                remainingTime: True
                currentIter : True
            ncmc_reporters:
              stream:
                title: ncmc
                reportInterval: 1
                totalSteps: 4 # Use nstepsNC
                step: True
                speed: True
                progress: True
                remainingTime: True
                protocolWork : True
                alchemicalLambda : True
                currentIter : True
        """
        print('Testing Simulation.run() from YAML')
        yaml_cfg = Settings(yaml_cfg)
        cfg = yaml_cfg.asDict()

        simulations = SimulationFactory(self.systems, self.engine, cfg['simulation'],
                                    cfg['md_reporters'], cfg['ncmc_reporters'])

        blues = BLUESSimulation(simulations)
        before_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        blues.run()
        after_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        #Check that our system has run dynamics
        pos_compare = np.not_equal(before_iter, after_iter).all()
        self.assertTrue(pos_compare)
        os.remove('ala-dipep-vac.log')

    def test_blues_simulationRunPure(self):
        print('Testing BLUESSimulation.run() from pure python')
        md_rep_cfg = { 'stream': { 'title': 'md',
                                'reportInterval': 1,
                                'totalSteps': 4,
                                'step': True,
                                'speed': True,
                                'progress': True,
                                'remainingTime': True,
                                'currentIter' : True} }
        ncmc_rep_cfg = { 'stream': { 'title': 'ncmc',
                                'reportInterval': 1,
                                'totalSteps': 4,
                                'step': True,
                                'speed': True,
                                'progress': True,
                                'remainingTime': True,
                                'currentIter' : True} }

        md_reporters = ReporterConfig('ala-dipep-vac', md_rep_cfg).makeReporters()
        ncmc_reporters = ReporterConfig('ala-dipep-vac-ncmc', ncmc_rep_cfg).makeReporters()

        cfg = { 'nprop' : 1,
                'prop_lambda' : 0.3,
                'dt' : 0.001 * unit.picoseconds,
                'friction' : 1 * 1/unit.picoseconds,
                'temperature' : 100 * unit.kelvin,
                'nIter': 1,
                'nstepsMD': 4,
                'nstepsNC': 4,}
        simulations = SimulationFactory(self.systems, self.engine, cfg,
                                    md_reporters=md_reporters,
                                    ncmc_reporters=ncmc_reporters)

        blues = BLUESSimulation(simulations)
        before_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        blues.run()
        after_iter = blues._md_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        #Check that our system has run dynamics
        pos_compare = np.not_equal(before_iter, after_iter).all()
        self.assertTrue(pos_compare)


if __name__ == '__main__':
        unittest.main()
