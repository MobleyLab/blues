from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import Simulation, SimulationFactory, SystemFactory
import parmed
from simtk import openmm
from optparse import OptionParser
import sys
import logging, yaml,json
from blues.reporters import init_logger, BLUESHDF5Reporter, BLUESStateDataReporter, getReporters
from utils import startup

opt = startup('blues.yaml')
print(json.dumps(opt, sort_keys=True, indent=2, skipkeys=True, default=str))

logger = opt['Logger']

#Load Parmed Structure
prmtop = opt['structure']['prmtop']
inpcrd = opt['structure']['inpcrd']
structure = parmed.load_file(prmtop, xyz=inpcrd)

#Select move type
ligand = RandomLigandRotationMove(structure, 'LIG')

#Iniitialize object that selects movestep
ligand_mover = MoveEngine(ligand)

#Generate the openmm.Systems outside SimulationFactory to allow modifications
system = SystemFactory.generateSystem(structure, **opt['system'])
alch_system = SystemFactory.generateAlchSystem(system, ligand.atom_indices, **opt['alchemical'])

#Freeze atoms in the alchemical system
alch_system = SystemFactory.freeze_atoms(structure, alch_system, **opt['freeze'])

#Generate the OpenMM Simulations
simulations = SimulationFactory(structure, system, alch_system, ligand_mover, **opt['simulation'])

# Add reporters to MD simulation.
outfname = opt['options']['outfname']
totalSteps = opt['simulation']['nIter']*opt['simulation']['nstepsMD']
reportInterval = opt['simulation']['reporters']['reporter_interval']
reporters = getReporters(totalSteps, outfname, **opt['simulation']['reporters'])

md_progress_reporter = BLUESStateDataReporter(logger, separator="\t", title='md',
                             reportInterval=reportInterval,
                             step=True, totalSteps=totalSteps,
                             time=False, speed=True, progress=True, remainingTime=True)
reporters.append(md_progress_reporter)

for rep in reporters:
    simulations.md.reporters.append(rep)

# Add reporters to NCMC simulation.
###TODO: Recommended to only write to HDF5 at the last frame.
ncmc_reporter = BLUESHDF5Reporter(file=outfname+'-pmoves.h5',
                                 reportInterval=opt['simulation']['nstepsNC'],
                                 coordinates=True, frame_indices=[1,opt['simulation']['nstepsNC']],
                                 time=False, cell=True, temperature=False,
                                 potentialEnergy=True, kineticEnergy=False,
                                 velocities=False, atomSubset=None,
                                 protocolWork=True, alchemicalLambda=True,
                                 parameters=None, environment=True)

ncmc_progress_reporter = BLUESStateDataReporter(logger, separator="\t", title='ncmc',
                             reportInterval=reportInterval,
                             step=True, totalSteps=opt['simulation']['nstepsNC'],
                             time=False, speed=True, progress=True, remainingTime=True)

simulations.nc.reporters.append(ncmc_reporter)
simulations.nc.reporters.append(ncmc_progress_reporter)

# Run BLUES Simulation
blues = Simulation(simulations)
blues.run(**opt['simulation'])
