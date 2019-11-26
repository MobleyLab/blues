from blues.moves import RandomRotatableBondFlipMove, MoveEngine
from blues.simulation import *
import json
from blues.settings import *


def rotbondmove_cuda(yaml_file):
    # Parse a YAML configuration, return as Dict
    cfg = Settings('rotbondmove_cuda.yaml').asDict()
    structure = cfg['Structure']

    #Select move type
    prmtop = cfg['structure']['filename']
    inpcrd = cfg['structure']['xyz']
    dihedral_atoms = cfg['rotbond_info']['dihedral_atoms']
    alch_list = cfg['rotbond_info']['alch_list']
    ligand = RandomRotatableBondFlipMove(structure, prmtop, inpcrd, dihedral_atoms, alch_list, 'LIG')

    #Iniitialize object that selects movestep
    ligand_mover = MoveEngine(ligand)

    #Generate the openmm.Systems outside SimulationFactory to allow modifications
    systems = SystemFactory(structure, ligand.atom_indices, cfg['system'])

    #Freeze atoms in the alchemical system to speed up alchemical calculation
    systems.alch = systems.freeze_radius(structure, systems.alch, **cfg['freeze'])

    #Generate the OpenMM Simulations
    simulations = SimulationFactory(systems, ligand_mover, cfg['simulation'], cfg['md_reporters'],
                                    cfg['ncmc_reporters'])

    # Run BLUES Simulation
    blues = BLUESSimulation(simulations, cfg['simulation'])
    blues.run()


if __name__ == "__main__":
    rotbondmove_cuda('rotbondmove_cuda.yaml')
