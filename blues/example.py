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

def runNCMC(platform_name, nstepsNC, outfname):
    #Generate the ParmEd Structure
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)
    print('Structure:', struct.topology)

    #Atom selection for zeroing masses
    mask = parmed.amber.AmberMask(struct,"(:LIG<:5.0)&!(:HOH,NA,CL)")
    site_idx = [i for i in mask.Selected()]

    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 100, 'nstepsNC' : nstepsNC, 'nstepsMD' : 10000,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 5000, 'write_ncmc' : False,
            'platform' : platform_name, 'zero_list' : site_idx,
            'verbose' : False }
    print('Options:')
    for k,v in opt.items():
        print('\t', k,v)

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    ligand = RandomLigandRotationMove(struct, 'LIG')
    ligand.calculateProperties()

    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    traj_reporter = openmm.app.DCDReporter(outfname+'-nc{}.dcd'.format(nstepsNC), opt['trajectory_interval'])
    simulations.md.reporters.append(traj_reporter)

    blues = Simulation(simulations, ligand_mover, **opt)
    blues.runNCMC()

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
parser.add_option('-n','--ncmc', dest='nstepsNC', type='int', default=5000,
                  help='number of NCMC steps')
parser.add_option('-o','--output', dest='outfname', type='str', default="blues",
                  help='Filename for output DCD')
(options, args) = parser.parse_args()

platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]

if 'CUDA' in platformNames:
    runNCMC('CUDA', options.nstepsNC, options.outfname)
elif 'OpenCL' in platformNames:
    runNCMC('OpenCL',options.nstepsNC, options.outfname)
else:
    if options.force:
        runNCMC('CPU', options.nstepsNC, options.outfname)
    else:
        print('WARNING: Could not find a valid CUDA/OpenCL platform. BLUES is not recommended on CPUs.')
        print("To run on CPU: 'python blues/example.py -f'")
