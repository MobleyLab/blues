# `BLUES`: Binding modes of Ligands Using Enhanced Sampling

This package takes advantage of non-candidate equilibrium monte carlo moves (NCMC) to help sample between different ligand binding modes.

This also provides a prototype and validation of the SMIRFF SMIRKS-based force field format, along with classes to parameterize OpenMM systems given [SMIRFF `.ffxml` format files](https://github.com/open-forcefield-group/smarty/blob/master/The-SMIRFF-force-field-format.md) as provided here.

## Manifest

* `utils/` - some helper scripts for various things (not directly associated with BLUES)
* `blues/` - simple toolkit illustrating the use of RJMCMC to sample over SMARTS-specified atom types; also contains forcefield.py for handling SMIRFF forcefield format.
* `run_scripts/` - example scripts to run blues
* `systems/` - some example systems to run blues on.

## Prerequisites

Install [miniconda](http://conda.pydata.org/miniconda.html) first. On `osx` with `bash`, this is:
```

Install other conda dependencies:
```
conda install --yes omnia mdtraj
conda install --yes omnia openmmtools
conda install --yes omnia alchemy
```

NOTE: We'll add a better way to install these dependencies via `conda` soon.

## Installation

Install `blues` from the `blues/` directory with:
```bash
pip install .
```

## Documentation

## BLUES using NCMC

This package takes advantage of non-candidate equilibrium monte carlo moves (NCMC) to help sample between different ligand binding modes using the OpenMM simulation package. Currently the innate functionality is found in `blues/ncmc.py`, which utilizes a lambda coupling to alter the sterics and electrostatics of the ligand over the course of the ncmc move. One goal for this package is to allow for easy additions of other moves of interest, which will be covered below.

## Actually using BLUES
The heart of the package is found in `blues/ncmc_switching.py`. This holds the framework for NCMC. Particularly, it contains the integrator class that calculates the work done during the NCMC move. It also controls the lambda scaling of parameters. Currently the alchemy package is used to generate the lambda parameters for the ligand, which can potentially modify 5 parameters–sterics, electrostatics, bonds, angles, and torsions. 
The class SimNCMC in `blues/ncmc.py` serves as a wrapper for running ncmc simulations. SimNCMC.runSim() actually runs the simulation.

## Example Use
The following is an example of how to set up a simulation sampling the binding modes of toluene bound to T4 lysozyme using NCMC and a rotational move.
This example can also be found in 'examples/example.py'
```
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

##load systems and make alchemical systems
coord_file = 'eqToluene.inpcrd'
top_file =   'eqToluene.prmtop'
prmtop = openmm.app.AmberPrmtopFile(top_file)
inpcrd = openmm.app.AmberInpcrdFile(coord_file)
temp_system = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*unit.nanometer, constraints=openmm.app.HBonds)
testsystem = testsystems.TestSystem
testsystem.system = temp_system
testsystem.topology = prmtop.topology
testsystem.positions = inpcrd.positions
#helper function to get list of ligand atoms
lig_atoms = get_lig_residues(lig_resname='LIG', coord_file=coord_file, top_file=top_file)
#create alchemical system using alchemy functions
factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=lig_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
alchemical_system = factory.createPerturbedSystem()
##set up OpenMM simulations
temperature = 300.0*unit.kelvin
#functions describes how lambda scales with nstepsNC
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
md_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=md_integrator)
#dummy_simulation is used to perform alchemical corrections and serves as a reporter for ncmc moves
dummy_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=dummy_integrator)
md_simulation.context.setPositions(testsystem.positions)
md_simulation.context.setVelocitiesToTemperature(temperature)
#add reporters
md_simulation.reporters.append(openmm.app.dcdreporter.DCDReporter('traj.dcd', 1000))
md_simulation.reporters.append(HDF5Reporter('traj.h5', 1000))
practice_run = SimNCMC(temperature=temperature, residueList=lig_atoms)
#set nc attributes
numIter = 100
nstepsNC = 100
nstepsMD = 1000
functions = { 'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))', 'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)' }
nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert', timestep=0.001*unit.picoseconds)
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
nc_context = openmm.Context(alchemical_system, nc_integrator)

ncmove = [[practice_run.rotationalMove, [1]]]
#actually run
practice_run.get_particle_masses(testsystem.system, residueList = lig_atoms)
practice_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, movekey=ncmove, niter=numIter, nstepsNC=nstepsNC, nstepsMD=nstepsMD)



```




## Implementing other non-equilibrium moves
An optional argument for SimNCMC.runSim() , movekey allows users to implement their own moves into the simulation workflow. To add your own moves make a subclass of SimNCMC. The custom method can use self.nc_context or self.nc_integrator to access the NCMC context or NCMC integrator in the runSim() method. Then, insert the method into the movekey. As an example, say you implemented a smart darting move in a method called smartdart(). Then for the keymove argument use [[smartdart, [range]]], where range is a list of integers that specify when you want to apply the move. In general the keymove argument takes a list of lists. The first item in the list references the function, while the second references a list which specifies what steps you want to apply the function.

