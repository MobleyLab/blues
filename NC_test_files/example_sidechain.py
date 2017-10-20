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
from blues.moves import RandomLigandRotationMove, SideChainMove
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import Simulation, SimulationFactory
import parmed
from simtk import openmm
from optparse import OptionParser
import sys
import logging

def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s-%(asctime)s %(message)s',  "%H:%M:%S")
    # Write to File
    fh = logging.FileHandler('blues-example.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream to terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def runNCMC(platform_name, nstepsNC, nprop, outfname):

    logger = init_logger()

    #Generate the ParmEd Structure
    prmtop = utils.get_data_filename('blues', 'tests/data/vacDivaline.prmtop')
    inpcrd = utils.get_data_filename('blues', 'tests/data/vacDivaline.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)
    logger.info('Structure: %s' % struct.topology)

    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.004,
            'hydrogenMass' : 3.024,
            'nIter' : 100, 'nstepsNC' : nstepsNC, 'nstepsMD' : 1000, 'nprop' : nprop,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10,
            'constraints': 'HBonds',
            'trajectory_interval' : 100, 'reporter_interval' : 250,
            'ncmc_traj' : None, 'write_move' : False,
            'platform' : platform_name,
            'outfname' : 'vacDivaline',
            'verbose' : False}

    for k,v in opt.items():
        logger.debug('Options: {} = {}'.format(k,v))

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    ligand = SideChainMove(struct, [1])
    ligand.atom_indices = ligand.rot_bond_atoms

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
    traj = md.load_dcd(trajfile, top='protein.pdb')
    indicies = np.array([[0, 4, 6, 8]])
    dihedraldata = md.compute_dihedrals(traj, indicies)
    with open("dihedrals-%iNC.txt" %(nstepsNC), 'w') as output:
        for value in dihedraldata:
            output.write("%s\n" % str(value)[1:-1])

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
