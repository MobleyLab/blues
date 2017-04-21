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
from blues import ncmc
from blues import utils
from simulation import Simulation
import parmed
from simtk import unit, openmm
from datetime import datetime
from optparse import OptionParser

def runNCMC(platform_name):
    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 100, 'nstepsNC' : 2, 'nstepsMD' : 1,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 1,
            'platform' : platform_name,
            'verbose' : True }

    #Generate the ParmEd Structure
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    ligand = ncmc.Model(struct, 'LIG')
    ligand.calculateProperties()

    # Initialize object that proposes moves.
    ligand_mover = ncmc.MoveProposal(ligand, 'random_rotation', opt['nstepsNC'])

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = ncmc.SimulationFactory(struct, ligand, **opt)
    simulations.createSimulationSet()

    blues = Simulation(simulations, ligand, ligand_mover, **opt)
    blues.run()

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
