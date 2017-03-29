"""
example.py: Provides an example script to run BLUES and
benchmark the run on a given platform

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley

* Benchmarking related code adapted from:
https://github.com/pandegroup/openmm/blob/master/examples/benchmark.py
(Authors: Peter Eastmen)
"""
from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState

import blues.utils as utils
import blues.ncmc as ncmc
import blues.ncmc_switching as ncmc_switching
from blues.smartdart import SmartDarting

import sys, os
import numpy as np
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
from optparse import OptionParser

def runNCMC(platform_name):
    # Define some constants
    temperature = 300.0*unit.kelvin
    friction = 1/unit.picosecond
    dt = 0.002*unit.picoseconds
    # set nc attributes
    numIter = 5
    nstepsNC = 10
    nstepsMD = 10
    #functions here defines the equation of the lambda peturbation of the sterics and electrostatics over the course of a ncmc move
    functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
    # Initialize integrators
    md_integrator = openmm.LangevinIntegrator(temperature, friction, dt)
    alch_integrator = openmm.LangevinIntegrator(temperature, friction, dt)
    # Define platform
    platform = openmm.Platform.getPlatformByName(platform_name)

    # Obtain topologies/positions
    prmtop = app.AmberPrmtopFile(utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop'))
    inpcrd = app.AmberInpcrdFile(utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd'))

    # Generate OpenMM System
    system = prmtop.createSystem(nonbondedMethod=app.PME,
                                nonbondedCutoff=1*unit.nanometer,
                                constraints=app.HBonds)

    # Initailize MD Simulation
    md_sim = app.Simulation(prmtop.topology, system,
                            md_integrator, platform)
    mmver = openmm.version.version
    print('OpenMM({}) simulation generated for {} platform'.format(mmver, platform_name))
    md_sim.context.setPositions(inpcrd.positions)
    md_sim.context.setVelocitiesToTemperature(temperature)
    md_sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # Add reporters for MD simulation
    #md_sim.reporters.append(app.dcdreporter.DCDReporter('traj.dcd', nstepsMD))
    md_sim.reporters.append(HDF5Reporter('traj.h5', nstepsMD))
    md_sim.reporters.append(app.StateDataReporter(sys.stdout, separator="\t",
                                    reportInterval=10,
                                    step=True, totalSteps=numIter*nstepsMD,
                                    time=True, speed=True, progress=True,
                                    elapsedTime=True, remainingTime=True))

    # Get ligand atom indices
    ligand_atoms = utils.atomIndexfromTop('LIG', prmtop.topology)

    #Initialize Alchemical Simulation
    # performs alchemical corrections
    # Reporter for NCMC moves
    alch_sim = app.Simulation(prmtop.topology, system,
                              alch_integrator, platform)
    alch_sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # Generate Alchemical System
    factory = AbsoluteAlchemicalFactory(system, ligand_atoms,
                                        annihilate_sterics=True,
                                        annihilate_electrostatics=True)

    alch_system = factory.createPerturbedSystem()

    # Generate NC Integrator/Contexts
    nc_integrator = ncmc_switching.NCMCVVAlchemicalIntegrator(temperature,
                                               alch_system,
                                               functions,
                                               nsteps=nstepsNC,
                                               direction='insert',
                                               timestep=0.001*unit.picoseconds,
                                               steps_per_propagation=1)

    nc_context = openmm.Context(alch_system, nc_integrator, platform)
    nc_context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    print(dir(nc_context))
    print(dir(nc_integrator))
    # Initialize BLUES engine
    blues_run = ncmc.SimNCMC(temperature, ligand_atoms)

    #during the ncmc move, perform a rotation around the center of mass at the start of step 49 (again to maintain symmetry of ncmc move
    rot_step = (nstepsNC/2) -1
    nc_move = [[blues_run.rotationalMove, [rot_step]]]

    # actually run
    blues_run.get_particle_masses(system, ligand_atoms)
    blues_run.runSim(md_sim, nc_context, nc_integrator,
                    alch_sim, movekey=nc_move,
                    niter=numIter, nstepsNC=nstepsNC, nstepsMD=nstepsMD,
                    alchemical_correction=True)

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
(options, args) = parser.parse_args()

platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print(platformNames)
if set(['CUDA', 'OpenCL']).issubset( platformNames ):
    runNCMC('CUDA')
    runNCMC('OpenCL')

else:
    if options.force:
        runNCMC('CPU')
    else:
        print('WARNING: Could not find a valid CUDA/OpenCL platform. BLUES is not recommended on CPUs.')
        print("To run on CPU: 'python blues/example.py -f'")
