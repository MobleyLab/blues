from simtk.openmm.app import *
from simtk.openmm import *
import simtk.unit as unit
import numpy as np
from blues.ncmc import *
from blues.ncmc_switching import *
from blues.smartdart import SmartDarting, findNewCoord
from blues.utils import get_data_filename
from openmmtools import testsystems

def test_dartMove():
    ''' Tests whether a very short SimNCMC.runSim() simulation with a
    goes to completion without error.
    '''
    ###sets up system
    test_system = testsystems.AlanineDipeptideVacuum()
    #the 'ligand' of the system is the dipeptide
    residueList = range(22)
    basis_part = [0, 2, 7]
    #set up the ncmc simulation
    temperature = 300*unit.kelvin
    test_run = SmartDarting(residueList=residueList, temperature=300*unit.kelvin)
    test_run.basis_particles = basis_part
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.0000002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=md_integrator)
    functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}
    nc_integrator = NCMCVVAlchemicalIntegrator(300*unit.kelvin, test_system.system, functions, nsteps=2, timestep=0.0002*unit.femtoseconds, direction='flux')
    nc_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=nc_integrator)
    md_simulation.context.setPositions(test_system.positions)
    md_simulation.context.setVelocitiesToTemperature(temperature)
    #set up center of mass coordinates
    md_state = md_simulation.context.getState(getPositions=True)
    pos = md_state.getPositions(asNumpy=True)
    test_run.get_particle_masses(system=test_system.system)
    com = test_run.calculate_com(pos)
    #find inital com in new basis
    com_new_coord = findNewCoord(pos[basis_part[0]], pos[basis_part[1]], pos[basis_part[2]], com)
    #make sure findNewCoord is behaving correctly
    standard_new_coord = np.array([-0.96649944,  1.49404527, -2.52902291])
    np.testing.assert_almost_equal(com_new_coord._value, standard_new_coord, decimal=2)
    move_coord = com_new_coord[:] + np.array([1,1,1])*unit.nanometers
    test_run.n_dartboard = [com_new_coord, move_coord]
    #add rotational move
    nc_move = None
    #see if simulation runs
    results = test_run.runSim(md_simulation, nc_simulation, nstepsNC=2, nstepsMD=2, niter=1, alchemical_correction=True, movekey=nc_move)
    assert type(results) == type(test_system.positions)
    standard_dart = np.array([-0.0096823 ,  0.50751791,  0.060064  ])
    for i in range(20):
        pos_new = test_run.updateDartMove(context=md_simulation.context)
        for x in range(3):
            if pos_new[0][x]._value - results[0][x]._value == 0:
                np.testing.assert_almost_equal(pos_new[0][x]._value, results[0][x]._value, decimal=1)
            else:
                np.testing.assert_almost_equal(pos_new[0][x]._value, (results[0][x]._value + standard_dart[x]), decimal=1)
        md_simulation.context.setPositions(results)

