from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from ncmc_switching import *
import mdtraj as md
#from alCorrect import *
from openmmtools import testsystems
import math
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np
from mdtraj.reporters import HDF5Reporter
from smartdart import SmartDarting
from ncmc import *

if 0:
     # Create a reference system.
     from openmmtools import testsystems
     testsystem = testsystems.LysozymeImplicit()
     print(type(testsystem))
     [reference_system, positions] = [testsystem.system, testsystem.positions]
     # Create a factory to produce alchemical intermediates.
     receptor_atoms = range(0,2603) # T4 lysozyme L99A
     ligand_atoms = range(2603,2621) # p-xylene
     factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
     full_alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_electrostatics=1, lambda_sterics=1))
     firstres = 2603
     lastres  = 2621


     # Get the default protocol for 'denihilating' in complex in explicit solvent.
#     protocol = factory.defaultComplexProtocolImplicit()
     # Create the perturbed systems using this protocol.
     alchemical_system = factory.createPerturbedSystem()
if 0: #if alanine test system
    testsystem = testsystems.AlanineDipeptideVacuum()
    alchemical_atoms = [0,1,2,3] # terminal methyl group
    factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)
    alchemical_system = factory.createPerturbedSystem()
if 0: #if cluster test system
    periodic = False

    coord_file = 'eqToluene.inpcrd'
    top_file =   'eqToluene.prmtop'
    prmtop = openmm.app.AmberPrmtopFile(top_file)
    inpcrd = openmm.app.AmberInpcrdFile(coord_file)
    temp_system = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*unit.nanometer, constraints=openmm.app.HBonds)
    testsystem = testsystems.TestSystem
    testsystem.system = temp_system
    testsystem.topology = prmtop.topology
    testsystem.positions = inpcrd.positions
    firstres, lastres, lig_atoms = get_lig_residues(lig_resname='LIG', coord_file=coord_file, top_file=top_file)
    #since you'll use lastres to denote the end of indices
    lastres = lastres + 1
    ligand_atoms = lig_atoms
    print(ligand_atoms)
    recepotr_atoms = range(0,2634)
    factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
#    factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)

    full_alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_electrostatics=1, lambda_sterics=1))
    alchemical_system = factory.createPerturbedSystem()

if 0: #if cluster test system implicit
    periodic = False
    coord_file = 'tolimp1.inpcrd'
    top_file =   'tolimp1.prmtop'
    prmtop = openmm.app.AmberPrmtopFile(top_file)
    inpcrd = openmm.app.AmberInpcrdFile(coord_file)
    temp_system = prmtop.createSystem(implicitSolvent=openmm.app.OBC2, soluteDielectric=1, solventDielectric=80.0, constraints=openmm.app.HBonds)
    testsystem = testsystems.TestSystem
    testsystem.system = temp_system
    testsystem.topology = prmtop.topology
    testsystem.positions = inpcrd.positions
    firstres, lastres, lig_atoms = get_lig_residues(lig_resname='LIG', coord_file=coord_file, top_file=top_file)
    #since you'll use lastres to denote the end of indices
    lastres = lastres + 1
    ligand_atoms = lig_atoms
    print(ligand_atoms)
    recepotr_atoms = range(0,2634)
    factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
#    factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)

    full_alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_electrostatics=1, lambda_sterics=1))
    alchemical_system = factory.createPerturbedSystem()


if 1: #if cluster test system
    periodic=False
    pdb = PDBFile('./systems/circle.pdb')
    forcefield = ForceField('systems/circle.xml')
    system = forcefield.createSystem(pdb.topology,
             constraints=HBonds)
    print 'removing', system.getForce(0)
    system.removeForce(1)
    system.removeForce(0)
    numParticles = system.getNumParticles()
    ###custom nonbonded
    pairwiseForce = CustomNonbondedForce("q/(r^2) + 4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); q = q1*q2")
    print('energy', pairwiseForce.getEnergyFunction())
    rangeparticles = range(numParticles)
    pairwiseForce.addInteractionGroup(rangeparticles[-3:], rangeparticles[:-3])
    print 'Force num particles', pairwiseForce.getNumParticles()
    pairwiseForce.addPerParticleParameter("sigma")
    pairwiseForce.addPerParticleParameter("epsilon")
    pairwiseForce.addPerParticleParameter("q")
    for i in rangeparticles:
        pairwiseForce.addParticle()
        pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 0])
    #    print pairwiseForce.getParticleParameters(i)
    #    print pairwiseForce.getPerParticleParameterName(i)
    for i in rangeparticles[-3:]:
        pairwiseForce.setParticleParameters(i,[0.32,0.7, -1])
    for i in rangeparticles[-2:]:
        pairwiseForce.setParticleParameters(i,[0.15,0.50, 0.75])
    #pairwiseForce.setParticleParameters(140,[0.324999852378,0.71128, 1])
    for i in [125, 128, 131, 134, 126, 129, 132, 135]:
        pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 1])

    pairwiseForce.setParticleParameters(132,[0.324999852378,0.71128, 1])
    pairwiseForce.setParticleParameters(122,[0.324999852378,0.71128, 1])



    system.addForce(pairwiseForce)
    ###

    ###harmonic
    if 1:
        harmonic = HarmonicBondForce()
        harmonic.addBond(rangeparticles[-3], rangeparticles[-2], 0.30, 10000)
        harmonic.addBond(rangeparticles[-3], rangeparticles[-1], 0.30, 10000)
        system.addForce(harmonic)


    ##angle
    if 1:
        angleForce = CustomAngleForce("0.5*k*(theta-theta0)^2")
        angleForce.addPerAngleParameter("k")
        angleForce.addPerAngleParameter("theta0")
        angleForce.addAngle(rangeparticles[-1], rangeparticles[-3], rangeparticles[-2], [1000, 1.5707963268])
        system.addForce(angleForce)

    #since you'll use lastres to denote the end of indices
    ligand_atoms = [141,142,143]
    zero_masses(system, 0, numParticles-3)
    testsystem = testsystems.TestSystem
    testsystem.system = system
    testsystem.topology = pdb.topology
    testsystem.positions = pdb.positions
    print(ligand_atoms)
    factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
#    factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)

    full_alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_electrostatics=1, lambda_sterics=1))
    alchemical_system = factory.createPerturbedSystem()

# Create an NCMC switching integrator.
if 1: #alchemical system
    alchemicalsystem = forcefield.createSystem(pdb.topology,
             constraints=HBonds)
    print 'removing', alchemicalsystem.getForce(0)
    alchemicalsystem.removeForce(1)
    alchemicalsystem.removeForce(0)
    numParticles = alchemicalsystem.getNumParticles()
    ###custom nonbonded
    pairwiseForce = CustomNonbondedForce("q/(r^2) + 4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2)*lambda_sterics; epsilon=sqrt(epsilon1*epsilon2)*lambda_electrostatics; q = q1*q2")
    print('energy', pairwiseForce.getEnergyFunction())
    rangeparticles = range(numParticles)
    pairwiseForce.addInteractionGroup(rangeparticles[-3:], rangeparticles[:-3])
    print 'Force num particles', pairwiseForce.getNumParticles()
    pairwiseForce.addPerParticleParameter("sigma")
    pairwiseForce.addPerParticleParameter("epsilon")
    pairwiseForce.addPerParticleParameter("q")
    pairwiseForce.addGlobalParameter("lambda_sterics", 1)
    pairwiseForce.addGlobalParameter("lambda_electrostatics", 1)
    for i in rangeparticles:
        pairwiseForce.addParticle()
        pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 0])
    #    print pairwiseForce.getParticleParameters(i)
    #    print pairwiseForce.getPerParticleParameterName(i)
    for i in rangeparticles[-3:]:
        pairwiseForce.setParticleParameters(i,[0.32,0.7, -1])
    for i in rangeparticles[-2:]:
        pairwiseForce.setParticleParameters(i,[0.15,0.50, 0.75])
    #pairwiseForce.setParticleParameters(140,[0.324999852378,0.71128, 1])
    for i in [125, 128, 131, 134, 126, 129, 132, 135]:
        pairwiseForce.setParticleParameters(i,[0.324999852378,0.71128, 1])

    pairwiseForce.setParticleParameters(132,[0.324999852378,0.71128, 1])
    pairwiseForce.setParticleParameters(122,[0.324999852378,0.71128, 1])



    alchemicalsystem.addForce(pairwiseForce)
    ###

    ###harmonic
    if 1:
        harmonic = HarmonicBondForce()
        harmonic.addBond(rangeparticles[-3], rangeparticles[-2], 0.30, 10000)
        harmonic.addBond(rangeparticles[-3], rangeparticles[-1], 0.30, 10000)
        alchemicalsystem.addForce(harmonic)


    ##angle
    if 1:
        angleForce = CustomAngleForce("0.5*k*(theta-theta0)^2")
        angleForce.addPerAngleParameter("k")
        angleForce.addPerAngleParameter("theta0")
        angleForce.addAngle(rangeparticles[-1], rangeparticles[-3], rangeparticles[-2], [1000, 1.5707963268])
        alchemicalsystem.addForce(angleForce)

###set values here
temperature = 100.0 * unit.kelvin
nstepsNC = 1000
#functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : '0.9*(lambda^2) + 0.1' }
#functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5'}
functions = { 'lambda_sterics' : 'min(1, lambda*1000/900)', 'lambda_electrostatics' : 'min(1, (lambda*1000/900)^0.5)'}

#functions = { 'lambda_sterics' : 'lambda^0.5', 'lambda_electrostatics' : 'lambda^0.25'}

#functions = { 'lambda_sterics' : 'lambda^0.2', 'lambda_electrostatics' : 'lambda^0.1'}
#functions = { 'lambda_sterics' : 'lambda^6', 'lambda_electrostatics' : 'lambda^3'}

#functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}
#functions = { 'lambda_sterics' : '0', 'lambda_electrostatics' : '0'}


#functions = { 'lambda_sterics' : '0.5*lambda + 0.5', 'lambda_electrostatics' : '0.5*(lambda^0.5) + 0.5'}

###

#nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='delete')
nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert', timestep=0.001*unit.picoseconds)
#nc_integrator = NCMCGHMCAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert', timestep=0.001*unit.picoseconds)

print 'kt', nc_integrator.kT
#nc_integrator = NCMCGHMCAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert')


#md_integrator = openmm.openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
# Create a Context
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
md_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=md_integrator)
dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
dummy_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=dummy_integrator)

#pdb = openmm.app.PDBFile('equil.pdb')
#if inpcrd.boxVectors is not None:
#    md_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
print 'testsystem Forces', testsystem.system.getForces()
print 'alchemical Forces', alchemical_system.getForces()

md_simulation.context.setPositions(testsystem.positions)
#openmm.LocalEnergyMinimizer.minimize(md_simulation.context)
md_simulation.context.setVelocitiesToTemperature(temperature)
print('Equilibrating...')
md_simulation.step(5000)
#md_simulation.step(100000)
if 0:
    md_info = md_simulation.context.getState(True, False, False, False, False, periodic)

    equilPos = md_info.getPositions(asNumpy=True)
    md_simulation.reporters.append(openmm.app.dcdreporter.DCDReporter('youtput_debug1.dcd', 500))
    md_simulation.reporters.append(HDF5Reporter('youtput_debug1.h5', 500))
    md_simulation.reporters.append(openmm.app.statedatareporter.StateDataReporter('info.csv', 500, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))
    md_simulation.step(1)

#    zero_masses(alchemical_system, firstres, lastres)
#    reslist = ['151', '153', '132', '116', '135', '112', '144', '104', '102', '88', '89', '111', '110', '113', '103', '82', '83', '80', '81', '86', '87', '84', '85', '123', '106', '164', '105', '107', '2', '5', '97', '6', '128', '96', '115', '145', '122', '137', '120', '119', '148', '149', '77', '76', '108', '74', '73', '125', '129', '91', '90', '100', '101', '95', '94', '79', '78', '118', '99', '121', '98', '117', '131', '109', '114', '124', '152']

#    reslist = [int(i) for i in reslist]
#    reslist = range(165)
#    all_res = selectres(reslist, inpcrd=coord_file, prmtop=top_file)
#    print all_res
#    all_res.extend(range(firstres, lastres))
#    print 'new', all_res
#    zero_masses(alchemical_system, firstres, lastres)
#    zero_allother_masses(alchemical_system, all_res)

    firstres = 141
    lastres  = 143
    lastres = lastres + 1

    nc_context = openmm.Context(alchemical_system, nc_integrator)

    nc_context.setPositions(equilPos)

    numIter = 10
    test_class = testintegrator()
    test_class.set_res(firstres, lastres)
    test_class.get_particle_masses(testsystem.system, firstres, lastres)
    test_class.testintegrator(md_simulation, nc_context, nc_integrator, dummy_simulation, nstepsNC=nstepsNC, nstepsMD=1000, niter=1, periodic=False, verbose=False, alchemical_correction=True, ncmc_report=False, rot_report=False)
if 0:
    md_info = md_simulation.context.getState(True, False, False, False, False, periodic)
    equilPos = md_info.getPositions(asNumpy=True)
    md_simulation.reporters.append(openmm.app.dcdreporter.DCDReporter('youtput_debug1.dcd', 100000))
    md_simulation.reporters.append(HDF5Reporter('youtput_debug1.h5', 100000))
    md_simulation.step(10000000000)

if 1:
    md_info = md_simulation.context.getState(True, False, False, False, False, periodic)

    equilPos = md_info.getPositions(asNumpy=True)
    md_simulation.reporters.append(openmm.app.dcdreporter.DCDReporter('youtput_debug1.dcd', 1000))
    md_simulation.reporters.append(HDF5Reporter('youtput_debug1.h5', 1000))
    md_simulation.reporters.append(openmm.app.statedatareporter.StateDataReporter('info.csv', 1000, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True))
    dboard = SmartDarting(temperature=300*unit.kelvin, residueList=[141, 142, 143])
    dboard.get_particle_masses(system=md_simulation.system)
    dboard.dart_size = 0.20*unit.nanometers
    dboard.add_dart((np.array([3.37, 2.64, 3.43]))*unit.nanometers)
    dboard.add_dart((np.array([3.40, 3.34, 2.61]))*unit.nanometers)
    dboard.add_dart((np.array([2.59, 2.63, 3.44]))*unit.nanometers)
    dboard.add_dart((np.array([2.59, 3.34, 2.63]))*unit.nanometers)

    firstres = 141
    lastres  = 143
    lastres = lastres + 1
    nc_context = openmm.Context(alchemical_system, nc_integrator)

    nc_context.setPositions(equilPos)

    numIter = 10
    test_class = testintegrator()
    test_class.set_res(ligand_atoms)
    test_class.get_particle_masses(testsystem.system, residueList = ligand_atoms)
    for i in range(5000):
        md_info = md_simulation.context.getState(True, False, False, False, False, periodic)
        origPos = md_info.getPositions(asNumpy=True)
        dboard.justdartmove(md_simulation.context)
        test_class.testintegrator(md_simulation, nc_context, nc_integrator, dummy_simulation, nstepsNC=nstepsNC, nstepsMD=500, niter=1, periodic=False, verbose=False, alchemical_correction=True, ncmc_report=False, rot_report=False, origPos=origPos)
        md_simulation.step(2500)
