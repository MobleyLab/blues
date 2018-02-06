"""
example_sidechain.py: Provides an example script to run BLUES with sidechain rotations

Authors: Samuel C. Gill
Contributors: Kalistyn Burley, Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function
from blues.moves import RandomLigandRotationMove, SideChainMove
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import Simulation, SimulationFactory
import parmed
from simtk import openmm
from optparse import OptionParser
import sys
import logging


def runNCMC(platform_name, nstepsNC, nprop, outfname):

    #Generate the ParmEd Structure
    prmtop = '/home/burleyk/projects/sidechain/inputs/watDivaline.prmtop'
    inpcrd = '/home/burleyk/projects/sidechain/inputs/watDivaline.inpcrd'
    struct = parmed.load_file(prmtop, xyz=inpcrd)
    print('Structure: %s' % struct.topology)

    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'hydrogenMass' : 3.024,
            'nIter' : 50000, 'nstepsNC' : nstepsNC, 'nstepsMD' : 1000, 'nprop' : nprop,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10,
            'constraints': 'HBonds', 'freeze_distance': 0.0,
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'ncmc_traj' : None, 'write_move' : False,
            'platform' : platform_name,
            'outfname' : 'divaline',
            'verbose' : True}
    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    ligand = SideChainMove(struct, [1])

    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    # Add reporters to MD simulation.
    trajfile = outfname+'-nc{}.dcd'.format(nstepsNC)
    traj_reporter = openmm.app.DCDReporter(trajfile, opt['trajectory_interval'])
    progress_reporter = openmm.app.StateDataReporter(sys.stdout, separator="\t",
                                reportInterval=opt['reporter_interval'],
                                step=True, totalSteps=opt['nIter']*opt['nstepsMD'],
                                time=True, speed=True, progress=True,
                                elapsedTime=True, remainingTime=True)
    simulations.md.reporters.append(traj_reporter)
    simulations.md.reporters.append(progress_reporter)

    # Run BLUES Simulation
    blues = Simulation(simulations, ligand_mover, **opt)
    blues.run(opt['nIter'])


    #Analysis
    import mdtraj as md
    import numpy as np
    traj = md.load_dcd(trajfile, top=prmtop)
    # get "BLUES" dihedrals
    indices = np.array([[0,4,6,8]])
    dihedraldata = md.compute_dihedrals(traj, indices)
    with open("dihedrals-%iNC-%s.txt" %(nstepsNC, outfname), 'w') as output:
        for value in dihedraldata:
            output.write("%s\n" % str(value)[1:-1])
    # get "non-BLUES" dihedrals (sidechain that's not rotated)
    indices2 = np.array([[18, 20, 22, 24]])
    dihedral2 = md.compute_dihedrals(traj, indices2)
    with open("dihedrals_nonblues.txt", 'w') as output2:
        for value in dihedral2:
            output2.write("%s\n" % str(value)[1:-1])

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
parser.add_option('-n','--ncmc', dest='nstepsNC', type='int', default=5000,
                  help='number of NCMC steps')
parser.add_option('-p','--nprop', dest='nprop', type='int', default=3,
                  help='number of propgation steps')
parser.add_option('-o','--output', dest='outfname', type='str', default="blues",
                  help='Filename for output DCD')
(options, args) = parser.parse_args()



platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
if 'CUDA' in platformNames:
    runNCMC('CUDA', options.nstepsNC, options.nprop, options.outfname)
elif 'OpenCL' in platformNames:
    runNCMC('OpenCL',options.nstepsNC, options.nprop, options.outfname)
else:
    if options.force:
        runNCMC('CPU', options.nstepsNC, options.outfname)
    else:
        print('WARNING: Could not find a valid CUDA/OpenCL platform. BLUES is not recommended on CPUs.')
        print("To run on CPU: 'python blues/example.py -f'")
