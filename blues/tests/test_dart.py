from __future__ import print_function
from blues.mold import MolDart
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import Simulation, SimulationFactory
from simtk.openmm.app import DCDReporter
import parmed
import mdtraj as md
from simtk import unit
import simtk.openmm as mm
import numpy as np
import unittest
#from simtk.openmm.app import OBC2

class MolEdit(MolDart):
    def __init__(self, *args, **kwargs):
        super(MolEdit, self).__init__(*args, **kwargs)
    def initializeSystem(self, system, integrator):
        new_sys, new_int = super(MolEdit, self).initializeSystem(system, integrator)
        force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", 5.0*unit.kilocalories_per_mole/unit.angstroms**2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        aatoms = []
        for i, atom_crd in enumerate(self.structure.positions):
            if self.structure.atoms[i].name in ('CA', 'C', 'N'):
                force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
                aatoms.append(i)
        new_sys.addForce(force)
        return new_sys, new_int


class ReverseDartTester(unittest.TestCase):
    """
    Tests that the ic dart move is reversible

    """
    def setUp(self):
        #Define some options
        opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.0005,
                'nIter' : 4, 'nstepsNC' : 10, 'nstepsMD' : 10,
                'nonbondedMethod' : 'CutoffNonPeriodic', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 1000, 'reporter_interval' : 1000,
                'platform' : 'CPU',
                'outfname' : 't4-tol',
                'nprop':1,
                'prop_lambda':0.10,
                'implicitSolvent': None,
                'verbose' : False,
                }

        prmtop = utils.get_data_filename('blues', 'tests/data/vacVA.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/VAn68.pdb')

        self.struct = parmed.load_file(prmtop, xyz=inpcrd)
        pdb_files = [ [utils.get_data_filename('blues', 'tests/data/VA68.pdb')], [utils.get_data_filename('blues', 'tests/data/VAn68.pdb')]]

        fit_atoms = [0, 4, 16, 18, 20, 26]

        self.ligand = MolEdit(structure=self.struct, resname=list(range(29)),
                                          pdb_files=pdb_files,
                                          fit_atoms=fit_atoms,
                                          restraints=False,
                                          restrained_receptor_atoms=[622, 2592, 2425],
                                          rigid_move=False
                                          )

        # Initialize object that proposes moves.
        self.ligand_mover = MoveEngine(self.ligand)

        # Generate the MD, NCMC, ALCHEMICAL Simulation objects
        self.simulations = SimulationFactory(self.struct, self.ligand_mover, **opt)
        self.simulations.createSimulationSet()

        self.blues = Simulation(self.simulations, self.ligand_mover, **opt)

    def test_dartreverse(self):

        #get context and set positions to end, see if get same positions as beginning
        begin_traj = md.load(utils.get_data_filename('blues', 'tests/data/dart_start.pdb' ))
        end_traj = md.load(utils.get_data_filename('blues', 'tests/data/dart_end.pdb'))

        end_pos = end_traj.openmm_positions(0)

        self.blues.md_sim.context.setPositions(end_pos)
        begin_compare = self.ligand.move(self.blues.md_sim.context).getState(getPositions=True).getPositions(asNumpy=True)
        #check that the reverse of the move gives the same positions
        assert np.allclose(begin_compare._value, begin_traj.openmm_positions(0)._value, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
        unittest.main()

