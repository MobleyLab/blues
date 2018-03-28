from __future__ import print_function
from blues.mold import MolDart
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import Simulation, SimulationFactory
from blues.moves import RandomLigandRotationMove
from simtk.openmm.app import DCDReporter
import parmed
from simtk import openmm
from optparse import OptionParser
import mdtraj as md
from simtk import unit
import parmed as pmd
import pickle
import glob
from simtk.openmm.app import DCDReporter
import simtk.openmm as mm
import numpy as np
#from simtk.openmm.app import OBC2

class MolEdit(MolDart):
    def __init__(self, *args, **kwargs):
        super(MolEdit, self).__init__(*args, **kwargs)
    def initializeSystem(self, system, integrator):
        new_sys, new_int = super(MolEdit, self).initializeSystem(system, integrator)
        print(new_sys, new_int)
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




def test_dartreverse(platform_name):
    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.0005,
            'nIter' : 4, 'nstepsNC' : 10, 'nstepsMD' : 10,
            'nonbondedMethod' : 'CutoffNonPeriodic', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'platform' : platform_name,
            'outfname' : 't4-tol',
            'nprop':1,
            'prop_lambda':0.10,
            'implicitSolvent': None,
            'verbose' : False,
            }

    prmtop = 'vacVA.prmtop'
    inpcrd = 'VAn68.pdb'

    struct = parmed.load_file(prmtop, xyz=inpcrd)
    pdb_files = [ ['VA68.pdb'], ['VAn68.pdb']]

    fit_atoms = [0, 4, 16, 18, 20, 26]

    ligand = MolEdit(structure=struct, resname=list(range(29)),
                                      pdb_files=pdb_files,
                                      fit_atoms=fit_atoms,
                                      restraints=False,
                                      restrained_receptor_atoms=[622, 2592, 2425],
                                      rigid_move=False
                                      )

    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    blues = Simulation(simulations, ligand_mover, **opt)
    blues.md_sim.reporters.append(DCDReporter('output.dcd', opt['nstepsMD']))

    #get context and set positions to end, see if get same positions as beginning
    begin_traj = md.load('startdart.pdb' )
    end_traj = md.load('enddart.pdb')

    end_pos = end_traj.openmm_positions(0)

    blues.md_sim.context.setPositions(end_pos)
    begin_compare = ligand.move(blues.md_sim.context).getState(getPositions=True).getPositions(asNumpy=True)
    #check that the reverse of the move gives the same positions
    assert np.allclose(begin_compare._value, begin_traj.openmm_positions(0)._value, rtol=1e-4, atol=1e-4)


test_dartreverse('CPU')
