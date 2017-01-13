from simtk.openmm.app import *
from simtk.openmm import *
from simtk import unit
from blues.ncmc_switching import *
import mdtraj as md
from openmmtools import testsystems
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np
from mdtraj.reporters import HDF5Reporter
from blues.smartdart import SmartDarting
from blues.ncmc import *

## load systems and make alchemical systems
coord_file = 'eqToluene.inpcrd'
top_file =   'eqToluene.prmtop'
prmtop = openmm.app.AmberPrmtopFile(top_file)
inpcrd = openmm.app.AmberInpcrdFile(coord_file)
temp_system = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*unit.nanometer, constraints=openmm.app.HBonds)
testsystem = testsystems.TestSystem
testsystem.system = temp_system
testsystem.topology = prmtop.topology
testsystem.positions = inpcrd.positions
# helper function to get list of ligand atoms
lig_atoms = get_lig_residues(lig_resname='LIG', coord_file=coord_file, top_file=top_file)
# create alchemical system using alchemy functions
factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=lig_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
alchemical_system = factory.createPerturbedSystem()
## set up OpenMM simulations
temperature = 300.0*unit.kelvin
# functions describes how lambda scales with nstepsNC
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
md_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=md_integrator)
# dummy_simulation is used to perform alchemical corrections and serves as a reporter for ncmc moves
dummy_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=dummy_integrator)
md_simulation.context.setPositions(testsystem.positions)
md_simulation.context.setVelocitiesToTemperature(temperature)
# add reporters
md_simulation.reporters.append(openmm.app.dcdreporter.DCDReporter('traj.dcd', 1000))
md_simulation.reporters.append(HDF5Reporter('traj.h5', 1000))
#enforce peroidc box from inpcrd file
if inpcrd.boxVectors is not None:
    md_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    dummy_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

practice_run = SimNCMC(temperature=temperature, residueList=lig_atoms)
# set nc attributes
numIter = 100
nstepsNC = 100
nstepsMD = 1000
#functions here defines the equation of the lambda peturbation of the sterics and electrostatics over the course of a ncmc move
functions = { 'lambda_sterics' : 'step(0.2-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.8)', 
			'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert', timestep=0.001*unit.picoseconds)
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
nc_context = openmm.Context(alchemical_system, nc_integrator)
if inpcrd.boxVectors is not None:
    nc_context.setPeriodicBoxVectors(*inpcrd.boxVectors)
#during the ncmc move, perform a rotation around the center of mass at the start of step 1
ncmove = [[practice_run.rotationalMove, [50]]]
# actually run
practice_run.get_particle_masses(testsystem.system, residueList = lig_atoms)
practice_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, movekey=ncmove, niter=numIter, nstepsNC=nstepsNC, nstepsMD=nstepsMD)


