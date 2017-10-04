"""
example.py: Provides an example script to run BLUES and
benchmark the run on a given platform

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley

* Benchmarking related code adapted from:
https://github.com/pandegroup/openmm/blob/master/examples/benchmark.py
(Authors: Peter Eastman)
"""

from __future__ import print_function
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.moves import SideChainMove
from blues import utils
from blues.simulation import Simulation, SimulationFactory
import parmed
from simtk import openmm
from optparse import OptionParser
import mdtraj as md
import numpy as np

def runNCMC(platform_name, relaxstepsNC, themdsteps):
    #Define some options

    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 5000, 'nstepsNC' : relaxstepsNC, 'nstepsMD' : themdsteps,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'platform' : platform_name,
            'verbose' : False,
            'write_ncmc' : False, 'freeze_distance' : 5
            }

    #Generate the ParmEd Structure
    #prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    #inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    prmtop = '/home/burleyk/burleyk/inputs/lysozyme_pxy.prmtop'
    inpcrd = '/home/burleyk/burleyk/inputs/lysozyme_pxy.inpcrd'
    struct = parmed.load_file(prmtop, xyz=inpcrd)

    #Define the 'model' object we are perturbing here.

    ligand = SideChainMove(struct, [111])

    ligand.atom_indices = ligand.rot_bond_atoms
    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    blues = Simulation(simulations, ligand_mover, **opt)
    #add the reporter here
    blues.md_sim.reporters.append(openmm.app.dcdreporter.DCDReporter('output.dcd', 1000))
    blues.runNCMC()


# Modify runNCMC parameters heree
mdstep = 1000
theseNCsteps = [5000]

# Loop over parameter options for desired # of repeats and runNCMC
for relaxstepsNC in theseNCsteps:
    print('Running blues with %i NC steps and %i MD steps: ' %(relaxstepsNC, mdstep))

    parser = OptionParser()
    parser.add_option('-f', '--force', action='store_true', default=False,
                          help='run BLUES example without GPU platform')
    (options, args) = parser.parse_args()

    platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]

    if 'CUDA' in platformNames:
        runNCMC('CUDA', relaxstepsNC, mdstep)
    else:
        if options.force:
            runNCMC('CPU', relaxstepsNC, mdstep)
        else:
            print('WARNING: Could not find a valid CUDA/OpenCL platform. BLUES is not recommended on CPUs.')
            print("To run on CPU: 'python blues/example.py -f'")


    #Read in output.dcd file, compute dihedral angles, and write out to file 
    di_dataFN = "dihedrals%iNC_gp%i_MD1000step.txt" %(relaxstepsNC, repeat)
    traj = md.load_dcd('output.dcd', top = 'protein.pdb')
    indicies = np.array([[1735, 1737, 1739, 1741]])
    dihedraldata = md.compute_dihedrals(traj, indicies)
    datafile = open(di_dataFN,'w')
    for value in dihedraldata:
        datafile.write("%s\n" % str(value)[1:-1])

    datafile.close()
