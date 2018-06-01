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
from blues.moves import MolDartMove
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import Simulation, SimulationFactory
import parmed
from simtk import openmm
from optparse import OptionParser
import mdtraj as md
import logging, sys
from blues.reporters import init_logger, BLUESHDF5Reporter, BLUESStateDataReporter


def runNCMC(platform_name):
    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 10, 'nstepsNC' : 2000, 'nstepsMD' : 5000,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 250,
            'platform' : 'CUDA',
            'outfname' : 't4-tol',
            'write_move' : False,
            'nprop':5,
            'freeze_distance' : 5.0,
            'verbose' : True
 }

    logger = init_logger(logging.getLogger(), level=logging.INFO, outfname=opt['outfname'])
    opt['Logger'] = logger

    #Generate the ParmEd Structure
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    posA = utils.get_data_filename('blues', 'tests/data/posA.pdb')
    posB = utils.get_data_filename('blues', 'tests/data/posB.pdb')
    traj = md.load(inpcrd, top=prmtop)
    fit_atoms = traj.top.select("protein")

    ligand = MolDartMove(structure=struct, resname='LIG',
                                      pdb_files=[posA, posB],
                                      fit_atoms=fit_atoms,
                                      restrained_receptor_atoms=[1605, 1735, 1837],
                                      rigid_ring=True
                                      )

    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    # Add reporters to MD simulation.
    #traj_reporter = openmm.app.DCDReporter(opt['outfname']+'-nc{}.dcd'.format(opt['nstepsNC']), 1)
    from blues.reporters import NetCDF4Reporter
    traj_reporter = NetCDF4Reporter(opt['outfname']+'-nc{}.nc'.format(opt['nstepsNC']), 1000)
    md_progress_reporter = openmm.app.StateDataReporter(sys.stdout, separator="\t",
                                reportInterval=opt['reporter_interval'],
                                step=True, totalSteps=opt['nIter']*opt['nstepsMD'],
                                time=True, speed=True, progress=True,
                                elapsedTime=True, remainingTime=True)
    simulations.md.reporters.append(traj_reporter)
    simulations.md.reporters.append(md_progress_reporter)

    # Add reporters to NCMC simulation.
    ncmc_progress_reporter = openmm.app.StateDataReporter(sys.stdout, separator="\t",
                                reportInterval=opt['reporter_interval'],
                                step=True, totalSteps=opt['nstepsNC'],
                                time=True, speed=True, progress=True,
                                elapsedTime=True, remainingTime=True)
    #simulations.nc.reporters.append(traj_reporter)
    simulations.nc.reporters.append(ncmc_progress_reporter)

    blues = Simulation(simulations, ligand_mover, **opt)
    blues.run(opt['nIter'])

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
(options, args) = parser.parse_args()

platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]

if 'OpenCL' in platformNames:
    runNCMC('OpenCL')
elif 'CUDA' in platformNames:
    runNCMC('CUDA')
else:
    if options.force:
        runNCMC('CPU')
    else:
        print('WARNING: Could not find a valid CUDA/OpenCL platform. BLUES is not recommended on CPUs.')
        print("To run on CPU: 'python blues/example.py -f'")
