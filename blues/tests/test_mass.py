from simtk.openmm.app import *
from simtk.openmm import *
import simtk.unit as unit
import numpy as np
from blues.ncmc import *
from blues.utils import get_data_filename


def test_getMasses():
    '''Tests if the ncmc.SimNCMC.get_particle_masses() works properly.'''
    #set up OpenMM system for testing on
    pdb_file = get_data_filename('squareB2.pdb')
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(get_data_filename('circle.xml'))
    system = forcefield.createSystem(pdb.topology,
             constraints=HBonds)
    temperature = 100.0 * unit.kelvin
    # Create a Context
    platform = Platform.getPlatformByName('CPU')
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=md_integrator, platform=platform)
    test_class = SimNCMC(residueList=[847, 848, 849], temperature=100*unit.kelvin)
    md_simulation.context.setPositions(pdb.positions)
    total_mass, mass_list = test_class.get_particle_masses(md_simulation.system)
    #constant_mass is the expected mass of a single particle
    constant_mass = 14.00672*unit.dalton
    #check whether get_particle_masses() returns the expected masses and totals
    np.testing.assert_almost_equal(mass_list[1]._value, constant_mass._value, decimal=4)
    np.testing.assert_almost_equal(total_mass._value, (constant_mass*3)._value, decimal=4)

def test_getCOM():
    '''Tests if the ncmc.SimNCMC.get_particle_masses works properly.'''
    #set up OpenMM system for testing on
    pdb_file = get_data_filename('squareB2.pdb')
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(get_data_filename('circle.xml'))
    system = forcefield.createSystem(pdb.topology,
             constraints=HBonds)
    temperature = 100.0 * unit.kelvin
    # Create a Context
    platform = Platform.getPlatformByName('CPU')
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=md_integrator, platform=platform)
    test_class = SimNCMC(residueList=[847, 848, 849], temperature=100*unit.kelvin)
    md_simulation.context.setPositions(pdb.positions)
    total_mass, mass_list = test_class.get_particle_masses(md_simulation.system)
    npos = md_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    #check to see if the expected com is found
    com_coord = test_class.calculate_com(npos)
    constant_mass = 14.00672*unit.dalton
    np.testing.assert_almost_equal(com_coord._value, np.array([0.9239666, 1.39193333, 1.3373]), decimal=4)
    #perform a rotation around the center of mass and see if the resulting center of mass
    #is relatively unchanged
    rotated_particles = test_class.calculate_com(npos, rotate=True)
    for index, i in enumerate([847, 848, 849]):
        npos[i] = rotated_particles[index]
    r_com_coord = test_class.calculate_com(npos)
    for i in range(3):
        np.testing.assert_almost_equal(np.asarray(r_com_coord._value)[i], np.asarray(com_coord._value)[i], decimal=4)

