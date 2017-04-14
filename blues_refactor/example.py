"""
example.py: Provides an example script to run BLUES and
benchmark the run on a given platform

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley

* Benchmarking related code adapted from:
https://github.com/pandegroup/openmm/blob/master/examples/benchmark.py
(Authors: Peter Eastman)

version: 0.0.2 (WIP-Refactor)
"""
from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState

import blues.utils as utils

import blues.ncmc_switching as ncmc_switching
from blues.smartdart import SmartDarting

import blues_refactor.ncmc as ncmc

import sys, os, parmed
import numpy as np
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
from optparse import OptionParser

def runNCMC(platform_name):
    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 5, 'nstepsNC' : 10, 'nstepsMD' : 50,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 10, 'reporter_interval' : 10,
            'platform' : platform_name,
            'verbose' : True }

    # Obtain topologies/positions
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)
    atom_indices = utils.atomIndexfromTop('LIG', struct.topology)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    sims = ncmc.SimulationFactory(struct, atom_indices, **opt)
    sims.createSimulationSet()

    # Calculate particle masses of object to be moved
    model = ncmc.ModelProperties(struct, 'LIG')
    model.calculateProperties()

    # Initialize object that proposes moves.
    mover = ncmc.MoveProposal(model, 'random_rotation', opt['nstepsNC'])

    blues = ncmc.Simulation(sims, model, mover, **opt)
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
