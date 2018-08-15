from blues.moves import RandomLigandRotationMove, MoveEngine
from blues.simulation import *
import json
from blues.settings import *

# Parse a YAML configuration, return as Dict
cfg = Settings('rotmove_cuda.yaml').asDict()
structure = cfg['Structure']

#Select move type
ligand = RandomLigandRotationMove(structure, 'LIG')
#Iniitialize object that selects movestep
ligand_mover = MoveEngine(ligand)

#Generate the openmm.Systems outside SimulationFactory to allow modifications
systems = SystemFactory(structure, ligand.atom_indices, cfg['system'])

#Freeze atoms in the alchemical system to speed up alchemical calculation
systems.alch = systems.freeze_radius(structure, systems.alch, **cfg['freeze'])

#Generate the OpenMM Simulations
simulations = SimulationFactory(systems, ligand_mover, cfg['simulation'],
                                cfg['md_reporters'], cfg['ncmc_reporters'])

# Run BLUES Simulation
blues = BLUESSimulation(simulations, cfg['simulation'])
blues.run()
