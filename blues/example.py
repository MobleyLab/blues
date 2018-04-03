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
from blues.reporters import init_logger, BLUESHDF5Reporter, BLUESStateDataReporter

def runNCMC(platform_name, nstepsNC, nprop, outfname):

    #Generate the ParmEd Structure
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)

    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 100, 'nstepsNC' : 10000, 'nstepsMD' : 10000, 'nprop' : 1,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10,
            'constraints': 'HBonds', 'freeze_distance' : 5.0,
            'trajectory_interval' : 2000, 'reporter_interval' : 2000,
            'ncmc_traj' : None, 'write_move' : False,
            'platform' : platform_name,
            'outfname' : 't4-toluene'}

    logger = init_logger(level=logging.INFO, outfname=opt['outfname'])
    opt['Logger'] = logger

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
    md_progress_reporter = BLUESStateDataReporter(logger, separator="\t", title='md',
                                reportInterval=opt['reporter_interval'],
                                step=True, totalSteps=opt['nIter']*opt['nstepsMD'],
                                time=False, speed=True, progress=True, remainingTime=True)
    simulations.md.reporters.append(traj_reporter)
    simulations.md.reporters.append(md_progress_reporter)

    # Add reporters to NCMC simulation.
    ncmc_reporter = BLUESHDF5Reporter(file='t4tol-pmoves.h5',
                                    reportInterval=1,
                                    coordinates=True, frame_indices=[1,opt['nstepsNC']],
                                    time=False, cell=True, temperature=False,
                                    potentialEnergy=False, kineticEnergy=False,
                                    velocities=False, atomSubset=None,
                                    protocolWork=True, alchemicalLambda=True,
                                    parameters=opt, environment=True)
    ncmc_progress_reporter = BLUESStateDataReporter(logger, separator="\t", title='ncmc',
                                reportInterval=opt['reporter_interval'],
                                step=True, totalSteps=opt['nstepsNC'],
                                time=False, speed=True, progress=True, remainingTime=True)
    simulations.nc.reporters.append(ncmc_reporter)
    simulations.nc.reporters.append(ncmc_progress_reporter)

    # Run BLUES Simulation
    blues = Simulation(simulations, ligand_mover, **opt)
    blues.run(opt['nIter'])

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
parser.add_option('-n','--ncmc', dest='nstepsNC', type='int', default=100,
                  help='number of NCMC steps')
parser.add_option('-p','--nprop', dest='nprop', type='int', default=1,
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
