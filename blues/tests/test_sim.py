from simtk.openmm.app import *
from simtk.openmm import *
#from simtk.unit import *
import simtk.unit as unit
#from ncmc_switching import *
import numpy as np
from blues.ncmc import *
from blues.ncmc_switching import *
from blues.utils import get_data_filename

def test_runSim():
        temperature = 300*unit.kelvin
        periodic=False
        pdb_file = get_data_filename('squareB2.pdb')
        pdb = PDBFile(pdb_file)
        forcefield = ForceField(get_data_filename('circle.xml'))
        system = forcefield.createSystem(pdb.topology,
                 constraints=HBonds)
        system.removeForce(1)
        system.removeForce(0)
        numParticles = system.getNumParticles()
        ###custom nonbonded
        pairwiseForce = CustomNonbondedForce("q/(r^2) + 4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); q = q1*q2")
        rangeparticles = range(numParticles)
        pairwiseForce.addInteractionGroup(rangeparticles[-3:], rangeparticles[:-3])
        pairwiseForce.addPerParticleParameter("sigma")
        pairwiseForce.addPerParticleParameter("epsilon")
        pairwiseForce.addPerParticleParameter("q")
        for i in rangeparticles:
            pairwiseForce.addParticle()
            pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 0])
        for i in rangeparticles[-3:]:
            pairwiseForce.setParticleParameters(i,[0.32,0.7, -10.0])
        for i in rangeparticles[-2:]:
            pairwiseForce.setParticleParameters(i,[0.15,0.50, 0.25])
        for i in [788, 782]:
            pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 5])
        system.addForce(pairwiseForce)
        ###

        ###harmonic
        harmonic = HarmonicBondForce()
        harmonic.addBond(rangeparticles[-3], rangeparticles[-2], 0.30, 40000)
        harmonic.addBond(rangeparticles[-3], rangeparticles[-1], 0.30, 40000)
        harmonic.addBond(rangeparticles[-2], rangeparticles[-1], 0.30, 40000)
        system.addForce(harmonic)

        ##angle
        angleForce = CustomAngleForce("0.5*k*(theta-theta0)^2")
        angleForce.addPerAngleParameter("k")
        angleForce.addPerAngleParameter("theta0")
        angleForce.addAngle(rangeparticles[-1], rangeparticles[-3], rangeparticles[-2], [5000, 1.5707963268])
        system.addForce(angleForce)
        #since you'll use lastres to denote the end of indices
        ligand_atoms = range(numParticles-3, numParticles)
        zero_masses(system, 0, numParticles-3)
        residueList = rangeparticles[-3:]
        test_run = SimNCMC(residueList=residueList, temperature=300*unit.kelvin)
        md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
        md_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=md_integrator)
        dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
        dummy_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=dummy_integrator)
        functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}
        nc_integrator = NCMCVVAlchemicalIntegrator(300*unit.kelvin, system, functions, nsteps=2, direction='flux')
        nc_context = openmm.Context(system, nc_integrator)
        md_simulation.context.setPositions(pdb.positions)
        md_simulation.context.setVelocitiesToTemperature(temperature)
        results = test_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, nstepsNC=2, nstepsMD=2, niter=1, alchemical_correction=True)
        assert type(results) == type(pdb.positions)

def test_rotationalMove():
        temperature = 300*unit.kelvin
        periodic=False
        pdb_file = get_data_filename('squareB2.pdb')
        pdb = PDBFile(pdb_file)
        forcefield = ForceField(get_data_filename('circle.xml'))
        system = forcefield.createSystem(pdb.topology,
                 constraints=HBonds)
        system.removeForce(1)
        system.removeForce(0)
        numParticles = system.getNumParticles()
        ###custom nonbonded
        pairwiseForce = CustomNonbondedForce("q/(r^2) + 4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); q = q1*q2")
        rangeparticles = range(numParticles)
        pairwiseForce.addInteractionGroup(rangeparticles[-3:], rangeparticles[:-3])
        pairwiseForce.addPerParticleParameter("sigma")
        pairwiseForce.addPerParticleParameter("epsilon")
        pairwiseForce.addPerParticleParameter("q")
        for i in rangeparticles:
            pairwiseForce.addParticle()
            pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 0])
        for i in rangeparticles[-3:]:
            pairwiseForce.setParticleParameters(i,[0.32,0.7, -10.0])
        for i in rangeparticles[-2:]:
            pairwiseForce.setParticleParameters(i,[0.15,0.50, 0.25])
        for i in [788, 782]:
            pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 5])
        system.addForce(pairwiseForce)
        ###

        ###harmonic
        harmonic = HarmonicBondForce()
        harmonic.addBond(rangeparticles[-3], rangeparticles[-2], 0.30, 40000)
        harmonic.addBond(rangeparticles[-3], rangeparticles[-1], 0.30, 40000)
        harmonic.addBond(rangeparticles[-2], rangeparticles[-1], 0.30, 40000)
        system.addForce(harmonic)

        ##angle
        angleForce = CustomAngleForce("0.5*k*(theta-theta0)^2")
        angleForce.addPerAngleParameter("k")
        angleForce.addPerAngleParameter("theta0")
        angleForce.addAngle(rangeparticles[-1], rangeparticles[-3], rangeparticles[-2], [5000, 1.5707963268])
        system.addForce(angleForce)
        #since you'll use lastres to denote the end of indices
        ligand_atoms = range(numParticles-3, numParticles)
        zero_masses(system, 0, numParticles-3)
        residueList = rangeparticles[-3:]
        test_run = SimNCMC(residueList=residueList, temperature=300*unit.kelvin)
        md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
        md_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=md_integrator)
        dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
        dummy_simulation = openmm.app.simulation.Simulation(topology=pdb.topology, system=system, integrator=dummy_integrator)
        functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}
        nc_integrator = NCMCVVAlchemicalIntegrator(300*unit.kelvin, system, functions, nsteps=2, direction='flux')
        nc_context = openmm.Context(system, nc_integrator)
        md_simulation.context.setPositions(pdb.positions)
        md_simulation.context.setVelocitiesToTemperature(temperature)
        nc_move = [[test_run.rotationalMove, [1]]]
        results = test_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, nstepsNC=2, nstepsMD=2, niter=1, alchemical_correction=True, movekey=nc_move)
        assert type(results) == type(pdb.positions)



