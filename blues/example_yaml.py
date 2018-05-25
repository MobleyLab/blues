from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues.simulation import *
import json
from blues.settings import *

opt = Settings('blues_cuda.yaml').asDict()
structure = opt['Structure']

#Select move type
ligand = RandomLigandRotationMove(structure, 'LIG')
#Iniitialize object that selects movestep
ligand_mover = MoveEngine(ligand)

#Generate the openmm.Systems outside SimulationFactory to allow modifications
systems = SystemFactory(structure, ligand.atom_indices, opt['system'])

#Freeze atoms in the alchemical system
#systems.alch = systems.freeze_atoms(structure, systems.alch, **opt['freeze'])

#Generate the OpenMM Simulations
simulations = SimulationFactory(systems, ligand_mover, opt['simulation'], opt['md_reporters'], opt['ncmc_reporters'])

# Run BLUES Simulation
blues = BLUESSimulation(simulations)
blues.run(**opt['simulation'])
