from blues.moves import MoveEngine, RandomLigandRotationMove, SideChainMove
from blues.settings import Settings
from blues.simulation import *
from blues.utils import get_data_filename


def ligrot_example(yaml_file):
    # Parse a YAML configuration, return as Dict
    cfg = Settings(yaml_file).asDict()
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
    simulations = SimulationFactory(systems, ligand_mover, cfg['simulation'], cfg['md_reporters'],
                                    cfg['ncmc_reporters'])

    # Run BLUES Simulation
    blues = BLUESSimulation(simulations, cfg['simulation'])
    blues.run()


def sidechain_example(yaml_file):
    # Parse a YAML configuration, return as Dict
    cfg = Settings(yaml_file).asDict()
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

    traj = md.load_netcdf('vacDivaline-test/vacDivaline.nc', top='tests/data/vacDivaline.prmtop')
    indicies = np.array([[0, 4, 6, 8]])
    dihedraldata = md.compute_dihedrals(traj, indicies)
    with open("vacDivaline-test/dihedrals.txt", 'w') as output:
        for value in dihedraldata:
            output.write("%s\n" % str(value)[1:-1])


ligrot_example(get_data_filename('blues', '../examples/rotmove_cuda.yml'))
#sidechain_example(get_data_filename('blues', '../examples/sidechain_cuda.yml'))
