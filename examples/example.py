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
practice_run = SimNCMC(temperature=temperature, residueList=lig_atoms)
# set nc attributes
numIter = 100
nstepsNC = 100
nstepsMD = 1000
#functions here defines the equation of the lambda peturbation of the sterics and electrostatics over the course of a ncmc move
functions = { 'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))', 'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)' }
nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert', timestep=0.001*unit.picoseconds)
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
nc_context = openmm.Context(alchemical_system, nc_integrator)
#during the ncmc move, perform a rotation around the center of mass at the start of step 1
ncmove = [[practice_run.rotationalMove, [1]]]
# actually run
practice_run.get_particle_masses(testsystem.system, residueList = lig_atoms)
practice_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, movekey=ncmove, niter=numIter, nstepsNC=nstepsNC, nstepsMD=nstepsMD)


