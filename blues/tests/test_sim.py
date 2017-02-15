from simtk.openmm.app import *
from simtk.openmm import *
import simtk.unit as unit
import numpy as np
from blues.ncmc import *
from blues.ncmc_switching import *
from blues.utils import get_data_filename
from openmmtools import testsystems


def test_runSim():
    ''' Tests whether a very short SimNCMC.runSim() simulation goes to completion
    without error.
    '''
    ###sets up system
    test_system = testsystems.AlanineDipeptideVacuum()
    #the 'ligand' of the system is the dipeptide
    residueList = range(22)
    #set up the ncmc simulation
    temperature = 300*unit.kelvin
    test_run = SimNCMC(residueList=residueList, temperature=300*unit.kelvin)
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=md_integrator)
    dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    dummy_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=dummy_integrator)
    functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}
    nc_integrator = NCMCVVAlchemicalIntegrator(300*unit.kelvin, test_system.system, functions, nsteps=2, direction='flux')
    nc_context = openmm.Context(test_system.system, nc_integrator)
    md_simulation.context.setPositions(test_system.positions)
    md_simulation.context.setVelocitiesToTemperature(temperature)
    results = test_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, nstepsNC=2, nstepsMD=2, niter=1, alchemical_correction=True)
    assert type(results) == type(pdb.positions)


def test_rotationalMove():
    ''' Tests whether a very short SimNCMC.runSim() simulation with a
    goes to completion without error.
    '''
    ###sets up system
    test_system = testsystems.AlanineDipeptideVacuum()
    #the 'ligand' of the system is the dipeptide
    residueList = range(22)
    #set up the ncmc simulation
    temperature = 300*unit.kelvin
    test_run = SimNCMC(residueList=residueList, temperature=300*unit.kelvin)
    md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    md_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=md_integrator)
    dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    dummy_simulation = openmm.app.simulation.Simulation(topology=test_system.topology, system=test_system.system, integrator=dummy_integrator)
    functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}
    nc_integrator = NCMCVVAlchemicalIntegrator(300*unit.kelvin, test_system.system, functions, nsteps=2, direction='flux')
    nc_context = openmm.Context(test_system.system, nc_integrator)
    md_simulation.context.setPositions(test_system.positions)
    md_simulation.context.setVelocitiesToTemperature(temperature)
    #add rotational move
    nc_move = [[test_run.rotationalMove, [1]]]
    #see if simulation runs
    results = test_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, nstepsNC=2, nstepsMD=2, niter=1, alchemical_correction=True, movekey=nc_move)
    assert type(results) == type(pdb.positions)


