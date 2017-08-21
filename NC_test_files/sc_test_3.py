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
#    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
#            'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 5000,
#            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
#            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
#            'platform' : platform_name,
#            'verbose' : True,
#            'write_ncmc' : 2 }

    #relaxstepsNC = 100
    #themdsteps = 1000

    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 5000, 'nstepsNC' : relaxstepsNC, 'nstepsMD' : themdsteps,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'platform' : platform_name,
            'verbose' : True,
            'write_ncmc' : 100
            }

    #Generate the ParmEd Structure
    #prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    #inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    prmtop = 'vacDivaline.prmtop'
    inpcrd = 'vacDivaline.inpcrd'
    struct = parmed.load_file(prmtop, xyz=inpcrd)

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved

    ligand = SideChainMove(struct, [1])

    ligand.atom_indices = ligand.rot_bond_atoms
    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    blues = Simulation(simulations, ligand_mover, **opt)
    #add the reporter here
    blues.md_sim.reporters.append(openmm.app.dcdreporter.DCDReporter('accept.dcd', 50))
    blues.run()


mdstep = 1000
repeats = [3]
theseNCsteps = [10000]


for relaxstepsNC in theseNCsteps:
    for repeat in repeats:
        print('Running blues with %i NC steps.  \nRepeat # %i: ' %(relaxstepsNC, repeat))
        '''runNCMC(platform_name, relaxstepsNC, mdstep):'''

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

        di_dataFN = "dihedrals%iNC_gp%i_MD1000step.txt" %(relaxstepsNC, repeat)
        traj = md.load_dcd('accept.dcd', top = 'protein.pdb')
        indicies = np.array([[0, 4, 6, 8]])
        dihedraldata = md.compute_dihedrals(traj, indicies)
        datafile = open(di_dataFN,'w')
        for value in dihedraldata:
            datafile.write("%s\n" % str(value)[1:-1])

        datafile.close()


# script to pull out acceptance ratio
