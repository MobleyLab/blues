import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as unit
from simtk.openmm.app import Simulation
#from openmmtools.testsystems import TestSystem
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState

from blues.ncmc import *
from blues.ncmc_switching import *
from blues.smartdart import SmartDarting

import sys
import numpy as np
import mdtraj as md
from mdtraj.reporters import HDF5Reporter
from datetime import datetime
from optparse import OptionParser

def timeIntegration(context, steps, initialSteps):
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
    platform = mm.Platform.getPlatformByName(options.platform)
    ## load systems and make alchemical systems
    coord_file = 'eqToluene.inpcrd'
    top_file =   'eqToluene.prmtop'
    prmtop = app.AmberPrmtopFile(top_file)
    inpcrd = app.AmberInpcrdFile(coord_file)
    testsystem = prmtop.createSystem(nonbondedMethod=app.PME,
                                      nonbondedCutoff=1*unit.nanometer,
                                      constraints=app.HBonds)

    # helper function to get list of ligand atoms
    lig_atoms = get_lig_residues(lig_resname='LIG',
                                 coord_file=coord_file,
                                 top_file=top_file)

    # create alchemical system using alchemy functions
    factory = AbsoluteAlchemicalFactory(testsystem,
                                        ligand_atoms=lig_atoms,
                                        annihilate_sterics=True,
                                        annihilate_electrostatics=True)

    alchemical_system = factory.createPerturbedSystem()

    ## set up OpenMM simulations
    temperature = 300.0*unit.kelvin
    friction = 1/unit.picosecond
    dt = 0.002*unit.picoseconds
    # functions describes how lambda scales with nstepsNC
    md_integrator = mm.LangevinIntegrator(temperature, friction, dt)
    dummy_integrator = mm.LangevinIntegrator(temperature, friction, dt)

    md_sim = Simulation(topology=prmtop.topology,
                        system=testsystem,
                        integrator=md_integrator,
                        platform=platform)

    # dummy_simulation is used to perform alchemical corrections and serves as a reporter for ncmc moves
    dummy_sim = Simulation(topology=prmtop.topology,
                           system=testsystem,
                           integrator=dummy_integrator,
                           platform=platform)


    md_sim.context.setPositions(inpcrd.positions)
    md_sim.context.setVelocitiesToTemperature(temperature)
    if inpcrd.boxVectors is not None:
        md_sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        dummy_sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)


    # set nc attributes
    numIter = 50
    nstepsNC = 100
    nstepsMD = 1000

    # add reporters
    md_sim.reporters.append(app.dcdreporter.DCDReporter('traj.dcd', nstepsMD))
    md_sim.reporters.append(HDF5Reporter('traj.h5', nstepsMD))
    practice_run = SimNCMC(temperature=temperature, residueList=lig_atoms)

    #functions here defines the equation of the lambda peturbation of the sterics and electrostatics over the course of a ncmc move
    functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)', 
                'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
    nc_integrator = NCMCVVAlchemicalIntegrator(temperature,
                                               alchemical_system,
                                               functions,
                                               nsteps=nstepsNC,
                                               direction='insert',
                                               timestep=0.001*unit.picoseconds,
                                               steps_per_propagation=1)

    nc_context = mm.Context(alchemical_system, nc_integrator, platform)
    if inpcrd.boxVectors is not None:
        nc_context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    #during the ncmc move, perform a rotation around the center of mass at the start of step 49 (again to maintain symmetry of ncmc move
    ncmove = [[practice_run.rotationalMove, [49]]]

    # actually run
    practice_run.get_particle_masses(testsystem, residueList=lig_atoms)
    practice_run.runSim(md_sim, nc_context, nc_integrator,
                        dummy_sim, movekey=ncmove,
                        niter=numIter, nstepsNC=nstepsNC, nstepsMD=nstepsMD,
                        alchemical_correction=True)

    nc_time = timeIntegration(nc_context, nstepsNC, 5)
    md_time = timeIntegration(md_sim.context, nstepsMD, 20)

    nc_steps = int(nstepsNC*1.0/nc_time)
    md_steps = int(nstepsMD*1.0/md_time)

    print('NC: Integrated %d steps in %g seconds' % (nstepsNC, nc_time))
    print('MD: Integrated %d steps in %g seconds' % (nstepsMD, md_time))

    print('NC: %g ns/day' % (dt/2*nstepsNC/nc_time).value_in_unit(unit.nanoseconds))
    print('MD: %g ns/day' % (dt*nstepsMD/nc_time).value_in_unit(unit.nanoseconds))

parser = OptionParser()
platformNames = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
parser.add_option('--platform', dest='platform', choices=platformNames, help='name of the platform to benchmark')
(options, args) = parser.parse_args()
if len(args) > 0:
    parser.error('Unknown argument: '+args[0])
if options.platform is None:
    parser.error('No platform specified')
print('Platform:', options.platform)

import cProfile, pstats, io
cProfile.run('runNCMC(options)', '{}.profile'.format(options.platform), sort=-1)
