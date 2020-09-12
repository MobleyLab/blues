from blues.moves import WaterTranslationMove, MoveEngine
from blues.simulation import *
import json
from blues.settings import Settings

# Parse a YAML configuration, return as Dict
opt = Settings('water_cuda.yaml').asDict()
structure = opt['Structure']
print(json.dumps(opt, sort_keys=True, indent=2, skipkeys=True, default=str))

# Select move type
water = WaterTranslationMove(structure, water_name='WAT', protein_selection='index 9', radius=2*unit.nanometers)

# Initialize object that selects movestep
water_mover = MoveEngine(water)

 #Generate the openmm.Systems outside SimulationFactory to allow modifications
systems = SystemFactory(structure, water.atom_indices, opt['system'])

# Restrain atoms in the MD and alchemical system
systems.md = systems.restrain_positions(structure, systems.md, **opt['restraints'])
systems.alch = systems.restrain_positions(structure, systems.alch, **opt['restraints'])

# Generate the OpenMM Simulations
simulations = SimulationFactory(systems, water_mover, opt['simulation'], opt['md_reporters'], opt['ncmc_reporters'])

blues = BLUESSimulation(simulations, opt['simulation'])
blues.run()
