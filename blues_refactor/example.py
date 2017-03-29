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
from blues_refactor.simulationfactory import SimulationFactory

import sys, os, parmed
import numpy as np
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
from optparse import OptionParser

def runNCMC(platform_name):
    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'numIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 50,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 10, 'reporter_interval' : 10, 'platform' : platform_name,
            'verbose' : True }

    #Defines ncmc move eqns for lambda peturbation of sterics/electrostatics
    opt['functions'] = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }

    # Obtain topologies/positions
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)
    atom_indices = utils.atomIndexfromTop('LIG', struct.topology)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    sims = SimulationFactory(struct, atom_indices)
    sims.createSimulationSet(opt)

    # Calculate particle masses of object to be moved
    from blues.modeller import LigandModeller
    model = LigandModeller(sims.nc, atom_indices)
    model.calculateCOM()

    # Propse some move
    rot_move = ncmc.ProposeMove(sims.nc, model)
    rot_step = (opt['nstepsNC']  / 2) - 1
    nc_move = { 'type' : 'rotation',
                'function': rot_move.rotation(),
                'step' : int(rot_step) }

    blues = ncmc.Simulation(sims, model, **opt)
    blues.run(nc_move=nc_move, residueList=atom_indices, alchemical_correction=True, **opt)

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
