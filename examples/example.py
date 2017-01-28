"""
example.py: Provides an example script to run BLUES and
benchmark the run on a given platform

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley

* Benchmarking related code adapted from:
https://github.com/pandegroup/openmm/blob/master/examples/benchmark.py
(Authors: Peter Eastmen)
"""

from simtk import unit, openmm
from simtk.openmm import app
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState

import blues.utils as utils
import blues.ncmc as ncmc
import blues.ncmc_switching as ncmc_switching
from blues.smartdart import SmartDarting

import sys
import numpy as np
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
from datetime import datetime
from optparse import OptionParser

def timeIntegration(context, steps, initialSteps):
    # Adapated from OpenMM benchmark.py
    """Integrate a Context for a specified number of steps, then return how many seconds it took."""
    context.getIntegrator().step(initialSteps) # Make sure everything is fully initialized
    context.getState(getEnergy=True)
    start = datetime.now()
    context.getIntegrator().step(steps)
    context.getState(getEnergy=True)
    end = datetime.now()
    elapsed = end -start
    return elapsed.seconds + elapsed.microseconds*1e-6

def runNCMC(options):
    # Define some constants
    temperature = 300.0*unit.kelvin
    friction = 1/unit.picosecond
    dt = 0.002*unit.picoseconds
    # set nc attributes
    numIter = 10
    nstepsNC = 10
    nstepsMD = 50
    #functions here defines the equation of the lambda peturbation of the sterics and electrostatics over the course of a ncmc move
    functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
    # Initialize integrators
    md_integrator = openmm.LangevinIntegrator(temperature, friction, dt)
    alch_integrator = openmm.LangevinIntegrator(temperature, friction, dt)
    # Define platform
    platform = openmm.Platform.getPlatformByName(options.platform)

    # Obtain topologies/positions
    prmtop = app.AmberPrmtopFile('eqToluene.prmtop')
    inpcrd = app.AmberInpcrdFile('eqToluene.inpcrd')

    # Generate OpenMM System
    system = prmtop.createSystem(nonbondedMethod=app.PME,
                                nonbondedCutoff=1*unit.nanometer,
                                constraints=app.HBonds)

    # Initailize MD Simulation
    md_sim = app.Simulation(prmtop.topology, system,
                            md_integrator, platform)
    md_sim.context.setPositions(inpcrd.positions)
    md_sim.context.setVelocitiesToTemperature(temperature)
    md_sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # Add reporters for MD simulation
    md_sim.reporters.append(app.dcdreporter.DCDReporter('traj.dcd', nstepsMD))
    md_sim.reporters.append(HDF5Reporter('traj.h5', nstepsMD))

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

    # Adapted from OpenMM benchmark.py
    nc_time = timeIntegration(nc_context, nstepsNC, 5)
    md_time = timeIntegration(md_sim.context, nstepsMD, 25)

    nc_steps = int(nstepsNC*1.0/nc_time)
    md_steps = int(nstepsMD*1.0/md_time)

    print('NC: Integrated %d steps in %g seconds' % (nstepsNC, nc_time))
    print('MD: Integrated %d steps in %g seconds' % (nstepsMD, md_time))

    print('NC: %g ns/day' % (dt/2*nstepsNC*86400/nc_time).value_in_unit(unit.nanoseconds))
    print('MD: %g ns/day' % (dt*nstepsMD*86400/nc_time).value_in_unit(unit.nanoseconds))

parser = OptionParser()
platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
parser.add_option('--platform', dest='platform', choices=platformNames, help='name of the platform to benchmark')
(options, args) = parser.parse_args()
if len(args) > 0:
    parser.error('Unknown argument: '+args[0])
if options.platform is None:
    parser.error('No platform specified')
else:
    print('Platform:', options.platform)
    runNCMC(options)
