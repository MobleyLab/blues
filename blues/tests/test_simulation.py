import os, unittest, parmed, yaml
from blues import utils
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues.simulation import SystemFactory, SimulationFactory, Simulation
from blues.config import YAMLSettings
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools import testsystems
import numpy as np

class SimulationTester(unittest.TestCase):
    """
    Test the Simulation class.
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

    def test_simulationRunYAML(self):
        yaml_cfg = """
            output_dir: .
            outfname: ala-dipep-vac
            logger_level: info

            system:
              nonbonded: NoCutoff
              constraints: HBonds

            simulation:
              platform: OpenCL
              properties:
                OpenCLPrecision: single
                OpenCLDeviceIndex: 2
              dt: 0.002 * picoseconds
              friction: 1 * 1/picoseconds
              temperature: 400 * kelvin
              nIter: 1
              nstepsMD: 10
              nstepsNC: 10
              nprop: 2

            md_reporters:
              stream:
                title: md
                reportInterval: 1
                totalSteps: 10 # nIter * nstepsMD
                step: True
                speed: True
                progress: True
                remainingTime: True
                currentIter : True
            ncmc_reporters:
              stream:
                title: ncmc
                reportInterval: 1
                totalSteps: 10 # Use nstepsNC
                step: True
                speed: True
                progress: True
                remainingTime: True
                protocolWork : True
                alchemicalLambda : True
                currentIter : True
            """
        yaml_cfg = YAMLSettings(yaml_cfg)
        cfg = yaml_cfg.asDict()
        simulations = SimulationFactory(self.systems, self.engine, cfg['simulation'],
                                    cfg['md_reporters'], cfg['ncmc_reporters'])
        blues = Simulation(simulations, cfg['simulation'])
        blues.run()
        os.remove('ala-dipep-vac.log')

if __name__ == '__main__':
        unittest.main()
