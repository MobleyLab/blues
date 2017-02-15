from simtk.openmm.app import *
from simtk.openmm import *
import simtk.unit as unit
import numpy as np
from blues.ncmc import *
from blues.utils import get_data_filename


def test_getMasses():
    '''Tests if the ncmc.SimNCMC.get_particle_masses() works properly.'''
    #set up OpenMM system for testing on
    test_system = testsystems.AlanineDipeptideVacuum()
    #the 'ligand' of the system is the dipeptide
    residueList = range(22)
    #set up the ncmc simulation
    temperature = 300*unit.kelvin
    test_class = SimNCMC(residueList=range(22), temperature=300*unit.kelvin)
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=md_integrator)
    md_simulation.context.setPositions(test_system.positions)
    total_mass, mass_list = test_class.get_particle_masses(md_simulation.system)
    #constant_total_mass is the expected mass of all the particles
    constant_total_mass = 144.17600000000002*unit.dalton
    #constant_mass_list is the expected mass of all the particles
    constant_mass_list = np.array([[1.008], [12.01], [1.008], [1.008], [12.01], [16.0],[14.01], [1.008],[12.01], [1.008], [12.01],
                        [1.008], [1.008], [1.008], [12.01], [16.0], [14.01],[1.008],[12.01], [1.008], [1.008], [1.008]])
    #check whether get_particle_masses() returns the expected masses and totals
    for index, entry in enumerate(constant_mass_list):
        np.testing.assert_almost_equal(entry, mass_list._value[index], decimal=2)
    np.testing.assert_almost_equal(total_mass._value, constant_total_mass._value, decimal=4)

def test_getCOM():
    '''Tests if the ncmc.SimNCMC.get_particle_masses works properly.'''
    #set up OpenMM system for testing on
    test_system = testsystems.AlanineDipeptideVacuum()
    #the 'ligand' of the system is the dipeptide
    residueList = range(22)
    #set up the ncmc simulation
    temperature = 300*unit.kelvin
    test_class = SimNCMC(residueList=range(22), temperature=300*unit.kelvin)
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=md_integrator)
    md_simulation.context.setPositions(test_system.positions)
    total_mass, mass_list = test_class.get_particle_masses(md_simulation.system)
    npos = md_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    #check to see if the expected com is found
    com_coord = test_class.calculate_com(npos)
    constant_com = np.array([0.43921147, 0.47429394, -0.01284274])
    np.testing.assert_almost_equal(com_coord._value, constant_com, decimal=4)
    #perform a rotation around the center of mass and see if the resulting center of mass
    #is relatively unchanged
    rotated_particles = test_class.calculate_com(npos, rotate=True)
    for index, i in enumerate(residueList):
        npos[i] = rotated_particles[index]
    r_com_coord = test_class.calculate_com(npos)
    for i in range(3):
        np.testing.assert_almost_equal(np.asarray(r_com_coord._value)[i], np.asarray(com_coord._value)[i], decimal=4)