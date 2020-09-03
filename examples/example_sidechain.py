from blues.moves import SideChainMove
from blues.moves import MoveEngine
from blues.simulation import *
import json
from blues.settings import *

# Parse a YAML configuration, return as Dict
cfg = Settings('sidechain_cuda.yml').asDict()
structure = cfg['Structure']

#Select move type
sidechain = SideChainMove(structure, [1])
#Iniitialize object that selects movestep
sidechain_mover = MoveEngine(sidechain)

#Generate the openmm.Systems outside SimulationFactory to allow modifications
systems = SystemFactory(structure, sidechain.atom_indices, cfg['system'])

#Generate the OpenMM Simulations
simulations = SimulationFactory(systems, sidechain_mover, cfg['simulation'], cfg['md_reporters'],
                                cfg['ncmc_reporters'])

# Run BLUES Simulation
blues = BLUESSimulation(simulations, cfg['simulation'])
blues.run()

#Analysis
import mdtraj as md
import numpy as np

traj = md.load_netcdf('vacDivaline-test/vacDivaline.nc', top='../blues/tests/data/vacDivaline.prmtop')
indicies = np.array([[0, 4, 6, 8]])
dihedraldata = md.compute_dihedrals(traj, indicies)
with open("vacDivaline-test/dihedrals.txt", 'w') as output:
    for value in dihedraldata:
        output.write("%s\n" % str(value)[1:-1])
