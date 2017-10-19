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
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
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
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)
    logger.info('Structure: %s' % struct.topology)

    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 100, 'nstepsNC' : nstepsNC, 'nstepsMD' : 5000, 'nprop' : nprop,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10,
            'constraints': 'HBonds', 'freeze_distance' : 5.0,
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'ncmc_traj' : None, 'write_move' : True,
            'platform' : platform_name,
            'verbose' : False}
            
    for k,v in opt.items():
        logger.debug('Options: {} = {}'.format(k,v))

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    ligand = RandomLigandRotationMove(struct, 'LIG')
    ligand.calculateProperties()

    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    # Add reporters to MD simulation.
    traj_reporter = openmm.app.DCDReporter(outfname+'-nc{}.dcd'.format(nstepsNC), opt['trajectory_interval'])
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

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
parser.add_option('-n','--ncmc', dest='nstepsNC', type='int', default=5000,
                  help='number of NCMC steps')
parser.add_option('-p','--nprop', dest='nprop', type='int', default=5,
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
