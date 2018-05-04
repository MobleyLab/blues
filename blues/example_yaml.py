from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import *
import parmed
from simtk import openmm
from optparse import OptionParser
import sys
import logging, yaml,json
from blues.reporters import init_logger, BLUESHDF5Reporter, BLUESStateDataReporter, getReporters


opt = startup('blues.yaml')
print(json.dumps(opt, sort_keys=True, indent=2, skipkeys=True, default=str))
logger = opt['Logger']
structure = opt['Structure']

#Select move type
ligand = RandomLigandRotationMove(structure, 'LIG')
#Iniitialize object that selects movestep
ligand_mover = MoveEngine(ligand)

#Generate the openmm.Systems outside SimulationFactory to allow modifications
systems = SystemFactory(structure, ligand.atom_indices, **opt['system'])

#Apply positional restraints
#systems.md = systems.restrain_positions(structure, systems.md, **opt['restraints'])

#Freeze atoms in the alchemical system
#systems.md = systems.freeze_atoms(structure, systems.md, **opt['freeze'])
systems.alch = systems.freeze_radius(structure, systems.alch, **opt['freeze'])

#Generate the OpenMM Simulations
simulations = SimulationFactory(systems, ligand_mover, **opt['simulation'])

# Run BLUES Simulation
blues = Simulation(simulations)
blues.run(**opt['simulation'])
