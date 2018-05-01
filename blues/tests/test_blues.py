
import unittest, parmed
from blues import utils
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.simulation import Simulation, SimulationFactory
from simtk import openmm, unit
from openmmtools import testsystems
import logging


class BLUESTester(unittest.TestCase):
    """
    Test the Simulation class.
    """
    def setUp(self):
        # Load the waterbox with toluene into a structure.
        self.prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
        self.inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
        self.full_struct = parmed.load_file(self.prmtop, xyz=self.inpcrd)
        self.opt = SimulationFactory.add_units({ 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 2, 'nstepsNC' : 4, 'nstepsMD' : 2, 'nprop' : 1,
                'nonbondedMethod' : openmm.app.PME, 'nonbondedCutoff': 10*unit.angstroms, 'constraints': 'HBonds',
                'trajectory_interval' : 1, 'reporter_interval' : 1, 'outfname' : 'blues-test',
                'platform' : None}, logging.getLogger(__name__))


    def test_moveproperties(self):
        # RandomLigandRotationMove.structure must be residue selection.
        move = RandomLigandRotationMove(self.full_struct, 'LIG')
        self.assertNotEqual(move.topology.getNumAtoms(), len(self.full_struct.atoms))

        # Test each function separately
        masses, totalmass = move.getMasses(move.topology)
        self.assertNotEqual(len(masses), 0)
        self.assertEqual(totalmass, masses.sum())

        center_of_mass = move.getCenterOfMass(move.positions, masses)
        self.assertNotEqual(center_of_mass, [0, 0, 0])

        # Test function that calcs all properties
        # Ensure properties are same as returned values
        move.calculateProperties()
        self.assertEqual(masses.tolist(), move.masses.tolist())
        self.assertEqual(totalmass, move.totalmass)
        self.assertEqual(center_of_mass.tolist(), move.center_of_mass.tolist())

    def test_simulationfactory(self):
        #Initialize the SimulationFactory object
        move = RandomLigandRotationMove(self.full_struct, 'LIG')
        engine = MoveEngine(move)
        sims = SimulationFactory(self.full_struct, engine, **self.opt)

        system = sims.generateSystem(self.full_struct, **self.opt)
        self.assertIsInstance(system, openmm.System)

        alch_system = sims.generateAlchSystem(system, move.atom_indices)
        self.assertIsInstance(alch_system, openmm.System)

        md_sim = sims.generateSimFromStruct(self.full_struct, engine, system, **self.opt)
        self.assertIsInstance(md_sim, openmm.app.simulation.Simulation)

        nc_sim = sims.generateSimFromStruct(self.full_struct, engine, alch_system, ncmc=True, **self.opt)
        self.assertIsInstance(nc_sim, openmm.app.simulation.Simulation)

    def test_barostat_force(self):
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 2, 'nstepsNC' : 100, 'nstepsMD' : 2, 'nprop' : 1,
                'nonbondedMethod' : 'NoCutoff', 'constraints': 'HBonds',
                'trajectory_interval' : 1, 'reporter_interval' : 1, 'outfname' : 'blues-test',
                'platform' : None, 'write_move' : False, 'barostat_frequency':25}

        #Initialize the SimulationFactory object
        move = RandomLigandRotationMove(self.full_struct, 'LIG')
        engine = MoveEngine(move)
        sims = SimulationFactory(self.full_struct, engine, **self.opt)

        system = sims.generateSystem(self.full_struct, **self.opt)
        self.assertIsInstance(system, openmm.System)

        alch_system = sims.generateAlchSystem(system, move.atom_indices)
        self.assertIsInstance(alch_system, openmm.System)

        md_sim = sims.generateSimFromStruct(self.full_struct, engine, system, **self.opt)
        self.assertIsInstance(md_sim, openmm.app.simulation.Simulation)

        nc_sim = sims.generateSimFromStruct(self.full_struct, engine, alch_system, ncmc=True, **self.opt)
        for asystem in [md_sim.system, nc_sim.system]:
            nc_forces = asystem.getForces()
            barostat = [i for i in nc_forces if isinstance(i, openmm.MonteCarloBarostat)]
            assert len(barostat) == 1
        self.assertIsInstance(nc_sim, openmm.app.simulation.Simulation)


    def test_simulationRun(self):
        """Tests the Simulation.runNCMC() function"""
        self.opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
                'nIter' : 2, 'nstepsNC' : 100, 'nstepsMD' : 2, 'nprop' : 1,
                'nonbondedMethod' : 'NoCutoff', 'constraints': 'HBonds',
                'trajectory_interval' : 1, 'reporter_interval' : 1, 'outfname' : 'blues-test',
                'platform' : None, 'write_move' : False}

        testsystem = testsystems.AlanineDipeptideVacuum(constraints=None)
        structure = parmed.openmm.topsystem.load_topology(topology=testsystem.topology,
                                            system=testsystem.system,
                                            xyz=testsystem.positions,
                                            )

        self.model = RandomLigandRotationMove(structure, resname='ALA')
        self.model.atom_indices = range(22)
        self.model.topology = structure.topology
        self.model.positions = structure.positions
        self.model.calculateProperties()
        self.mover = MoveEngine(self.model)
        #Initialize the SimulationFactory object
        sims = SimulationFactory(structure, self.mover, **self.opt)
        #print(sims)
        self.nc_sim = sims.nc
        self.model.calculateProperties()
        self.initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        asim = Simulation(sims, **self.opt)
        asim.run(self.opt['nIter'])

if __name__ == "__main__":
        unittest.main()
