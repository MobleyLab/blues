from simtk.openmm.app import *
from simtk.openmm import *
#from simtk.unit import *
import simtk.unit as unit
#from ncmc_switching import *
import numpy as np
from blues.ncmc import *

def test_getMasses():
    pdb_file = 'squareB2.pdb' 
    if 1: #if cluster test system
        periodic=False
        pdb = PDBFile(pdb_file)
        forcefield = ForceField('circle.xml')
        system = forcefield.createSystem(pdb.topology,
                 constraints=HBonds)

    ###set values here
    temperature = 100.0 * unit.kelvin
    ###

    # Create a Context
    platform = Platform.getPlatformByName('CPU')
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=md_integrator, platform=platform)
    test_class = SimNCMC(residueList=[847, 848, 849], temperature=100*unit.kelvin)
    md_simulation.context.setPositions(pdb.positions)
    total_mass, mass_list = test_class.get_particle_masses(md_simulation.system)
    constant_mass = 14.00672*unit.dalton
    np.testing.assert_almost_equal(mass_list[1]._value, constant_mass._value, decimal=4)
    np.testing.assert_almost_equal(total_mass._value, (constant_mass*3)._value, decimal=4)

def test_getCOM():
    pdb_file = 'squareB2.pdb' 
    if 1: #if cluster test system
        periodic=False
        pdb = PDBFile(pdb_file)
        forcefield = ForceField('circle.xml')
        system = forcefield.createSystem(pdb.topology,
                 constraints=HBonds)

    ###set values here
    temperature = 100.0 * unit.kelvin
    ###

    # Create a Context
    platform = Platform.getPlatformByName('CPU')
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=md_integrator, platform=platform)
    test_class = SimNCMC(residueList=[847, 848, 849], temperature=100*unit.kelvin)
    md_simulation.context.setPositions(pdb.positions)
    total_mass, mass_list = test_class.get_particle_masses(md_simulation.system)
    npos = md_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    com_coord = test_class.calculate_com(npos)
    constant_mass = 14.00672*unit.dalton
    np.testing.assert_almost_equal(com_coord._value, np.array([0.9239666, 1.39193333, 1.3373]), decimal=4)
    rotated_particles = test_class.calculate_com(npos, rotate=True)
    for index, i in enumerate([847, 848, 849]):
        npos[i] = rotated_particles[index]
    r_com_coord = test_class.calculate_com(npos)
    for i in range(3):
        np.testing.assert_almost_equal(np.asarray(r_com_coord._value)[i], np.asarray(com_coord._value)[i], decimal=4)

test_getMasses()
test_getCOM()


