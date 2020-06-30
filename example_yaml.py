from WaterTranslationRotation2 import WaterTranslationRotationMove
from blues.moves import MoveEngine
from blues.simulation import *
import json
from blues.settings import Settings
import scipy
#from blues.engine import MoveEngine
#from simulation import *
#import json
#from blues.settings import *
#import scipy

opt = Settings('blues_cuda.yaml').asDict()

structure = opt['Structure']
#restart = parmed.amber.Rst7('/oasis/tscc/scratch/bergazin/water/unequal_density_2wall/2w_144_10E_r1.rst7')#change to dir where rst is
restart = parmed.amber.Rst7('2wall-prod-500k-part2.rst7')#-1-500k.restrt')#wall_2000.restrt')#2wall-equil.rst7')

structure.positions = restart.positions
#structure.velocities = restart.velocities
#structure.box = restart.box

structure.box = restart.box
print("structure.box",structure.box)
structure.box = np.array([ 31.23,  30.52,  85.916,  90.,      90.,      90.,    ])
print("structure.box",structure.box)
print(json.dumps(opt, sort_keys=True, indent=2, skipkeys=True, default=str))



#Select move type
ligand = WaterTranslationRotationMove(structure, water_name='WAT')

#Iniitialize object that selects movestep
ligand_mover = MoveEngine(ligand)

#Generate the openmm.Systems outside SimulationFactory to allow modifications
systems = SystemFactory(structure, ligand.atom_indices, opt['system'])

# Restrain atoms
systems.md = systems.restrain_positions(structure, systems.md, **opt['restraints'])
systems.alch = systems.restrain_positions(structure, systems.alch, **opt['restraints'])

#Generate the OpenMM Simulations
simulations = SimulationFactory(systems, ligand_mover, opt['simulation'], opt['md_reporters'], opt['ncmc_reporters'])

#Energy minimize system
simulations.md.minimizeEnergy(maxIterations=0)
#simulations.md.step(50000)
state = simulations.md.context.getState(getPositions=True, getEnergy=True)
print('Minimized energy = {}'.format(state.getPotentialEnergy().in_units_of(unit.kilocalorie_per_mole)))

# MD simulation
#simulations.md.step(opt['simulation']['nstepsMD'])

# BLUES Simulation
blues = BLUESSimulation(simulations)
blues.run(**opt['simulation'])

# MC Simulation
#mc = MonteCarloSimulation(simulations, opt['simulation'])
