from ncmc_switching import *
import mdtraj as md
from alCorrect import *
from openmmtools import testsystems
import math
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np
from mdtraj.reporters import HDF5Reporter
def fix_units(np_array, unit_type):
    for x in np.nditer(np_array, op_flags=['readwrite']):
        x[...] = np.sum(x).value_in_unit(unit_type)
        return np_array
        
def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

class testintegrator:
    def __init__(self):
        """
        This controls the ability to run a ncmc simulation with MD

        Arguments
        ---------
        md_simulation: a Simulation object containing the system of interest
        nc_context:  openmm.context
        nc_integrator: nc integrator
        nstepsNC: int, number of steps of NC to perform
        nstepsMD: int, number of steps of nstepsMD to perform
        niter:    int, number of iterations of NC/MD to perform in total
        """
        self.total_mass = 0
        self.mass_list = None
        self.firstres = None
        self.lastres = None

    def set_res(self, firstres, lastres):
        self.firstres = firstres
        self.lastres = lastres

    def get_particle_masses(self, system, firstres=None, lastres=None):
        if firstres == None:
            firstres = self.firstres
        if lastres == None:
            lastres = self.lastres
        mass_list = []
        total_mass = 0*unit.dalton
        for index in range(firstres, lastres):
            mass = system.getParticleMass(index)
            total_mass = total_mass + mass
            mass_list.append([mass])
        print('MASS TEST\n')
        total_mass = np.sum(mass_list)
#        print mass_list
        mass_list = np.asarray(mass_list)
        mass_list.reshape((-1,1))
#        print mass_list
        total_mass = np.array(total_mass)
#        print 'masslist', mass_list
        total_mass = np.sum(mass_list)
#        print 'total-mass', total_mass
#        print np.sum(mass_list[0])
        temp_list = np.zeros(((lastres-firstres), 1))
#        print temp_list
        for index in range(lastres-firstres):
            #print (np.sum(mass_list[index])).__str__
            #help(np.sum(mass_list[index]))

            mass_list[index] = (np.sum(mass_list[index])).value_in_unit(unit.daltons)
#        print np.shape(mass_list), np.shape(temp_list)
        mass_list =  mass_list*unit.daltons
        print('mass_list', mass_list)
            

        self.total_mass = total_mass
        self.mass_list = mass_list
        return total_mass, mass_list

    def calculate_com(self, total_mass, mass_list, pos_state, firstres=None, lastres=None):
        if firstres == None:
            firstres = self.firstres
        if lastres == None:
            lastres = self.lastres

#        print total_mass
#        print mass_list, np.size(mass_list)
        #print pos_state
        #find ligand com coordinates
        lig_coord = pos_state[firstres:lastres]
#        print lig_coord.value_in_unit(unit.nanometers)
#        print 'final output'
        lig_coord = lig_coord.value_in_unit(unit.nanometers)*unit.nanometers
        copy_coord = copy.deepcopy(lig_coord)
#        print lig_coord
        #print pos_state[firstres:lastres], np.shape(pos_state[firstres:lastres])
#        print 'mass_list / total mass'
#        print mass_list / total_mass
        mass_corrected = mass_list / total_mass * copy_coord
        print(mass_corrected)
        print('test')
#        for index, entry in enumerate(mass_list):
#            print entry
#            print mass_list[0]
#            print total_mass
#            print( mass_list[0] / total_mass)
        #mass corrected is the com coordinates
#        mass_corrected.sum(axis=0)
        print('summed mass')
        #subtract from original coordinates to find translated com coordinates
#        print lig_coord
#        help(mass_corrected)
        sum_coord = mass_corrected.sum(axis=0).value_in_unit(unit.nanometers)
#        help(sum_coord)
        temp_coord = [0.0, 0.0, 0.0]*unit.nanometers
        for index in range(3):
            temp_coord[index] = sum_coord[index]*unit.nanometers
            print temp_coord

        print('sumcoord', sum_coord)
#        new_coord = (temp_coord*unit.nanometers).value_in_unit(unit.nanometers)
#        print('new_coord', new_coord)
        #print test
        #temp_coord = temp_coord.value_in_unit(unit.nanometers)*unit.nanometers
#        print temp_coord
        #print('tempcoord', temp_coord)
        

        print('before', lig_coord)
        for index in range(3):
            print(lig_coord[:,index])
            print(sum_coord, np.shape(sum_coord))
#            lig_coord[:,index] = lig_coord[:,index] - sum_coord[index]
            lig_coord[:,index] = lig_coord[:,index] - temp_coord[index]
            
#        print('after', lig_coord)
#        print('rotation')
#        print lig_coord
#        print rand_rotation_matrix()*unit.nanometers
        xy = rand_rotation_matrix()*unit.nanometers
#        print 'xy', xy
#        print 'lig_coord', lig_coord
        rotation =  np.dot(lig_coord.value_in_unit(unit.nanometers), rand_rotation_matrix())*unit.nanometers
#        print 'rotation, before', rotation
        rotation = rotation + temp_coord
#        print 'rotation, after', rotation
        #print 'original lig point', lig_coord + sum_coord
        return rotation    
        #print(mass_corrected.sum(axis=0))


    def testintegrator(self, md_simulation, nc_context, nc_integrator,  nstepsNC=25, nstepsMD=1000, niter=10, verbose=False, firstres=None, lastres=None):
        if firstres == None:
            firstres = self.firstres
        if lastres == None:
            lastres = self.lastres

        accCounter = 0
        otherCounter = 0
        lowest_difference = 1000000
        nc_stateinfo = nc_context.getState(True, False, False, False, False, False)
        tempPos = nc_stateinfo.getPositions(asNumpy=True)

        #oldPos = stateinfo.getPositions()
        last_x, last_y = np.shape(tempPos)
        ncmc_frame = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
#        print 'ncmc_frame start', ncmc_frame

        for stepsdone in range(niter):
#            ncmc_frame = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)

            #get original state info, no periodic boundries
            nc_stateinfo = nc_context.getState(True, True, False, True, True, False)
            oldPos = nc_stateinfo.getPositions(asNumpy=True)
#            print 'oldPos', oldPos[firstres:lastres]
            oldVel = nc_stateinfo.getVelocities(asNumpy=True)
            #set original position
            md_simulation.context.setPositions(oldPos)
            md_simulation.context.setVelocities(oldVel)

            oldPE =  nc_stateinfo.getPotentialEnergy()
            oldKE =  nc_stateinfo.getKineticEnergy()

#            oldPos[2634:2649] = self.calculate_com(total_mass=self.total_mass, mass_list=self.mass_list, pos_state=oldPos, firstres=2634, lastres=2649)
            if 1:
                rot_output = self.calculate_com(total_mass=self.total_mass, mass_list=self.mass_list, pos_state=oldPos, firstres=firstres, lastres=lastres)
#               print 'output', np.shape(output)
#               print 'oldPos after', np.shape(oldPos[firstres:lastres])
                rot_output = rot_output.value_in_unit(unit.nanometers)
                rotPos = oldPos.value_in_unit(unit.nanometers)
#               print 'rotPos', rotPos[firstres:lastres]
#               print 'rot_output', rot_output
                rotPos[firstres:lastres] = rot_output
                rotPos=rotPos*unit.nanometers
#               print 'final oldPos', rotPos
                nc_context.setPositions(rotPos)


            print('performing ncmc step')
            print('accCounter =', accCounter)
            mdinfo = md_simulation.context.getState(True, True, False, True, True, False)
            oldPE =  mdinfo.getPotentialEnergy()
            oldKE =  mdinfo.getKineticEnergy()
            original_energy = oldPE + oldKE
#            nc_integrator.getEnergyBeforeRotation(nc_context, original_energy)
#            print 'original energy'
#            print nc_integrator.getGlobalVariableByName('Eorig')
            if verbose == True:
                log_ncmc = nc_integrator.getLogAcceptanceProbability(nc_context)
                print('before ncmc move log_ncmc', log_ncmc, np.isnan(log_ncmc))

            for stepscarried in range(nstepsNC):
                #get original state info, no periodic boundries
    #            stateinfo = nc_context.getState(True, True, False, False, False, False)
    #            oldPos = stateinfo.getPositions(asNumpy=True)
    #            oldVel = stateinfo.getVelocities(asNumpy=True)
                try:
                    nc_integrator.step(1)
                except:
                    print('nan, breaking')
                    break
 
            #    st =  ncmc_integrator.getstep()
                if verbose == True:
                    log_ncmc = nc_integrator.getLogAcceptanceProbability(nc_context)
                    print('log_ncmc', log_ncmc, stepscarried, np.isnan(log_ncmc))
                    nc_stateinfo = nc_context.getState(True, False, False, False, False, False)
                    tempPos = nc_stateinfo.getPositions(asNumpy=True)
                    #connected
                    newinfo = nc_context.getState(True, True, False, True, True, False)
                    newPos = newinfo.getPositions(asNumpy=True)
                    correction = computeAlchemicalCorrection(testsystem, nc_context.getSystem(), oldPos, newPos, direction='insert')
                    print('correction', correction)
                    ######
                    print nc_integrator.getGlobalVariableByName('lambda')
                    #print nc_integrator.getGlobalVariableByName('lambda_sterics')




                    last_x, last_y = np.shape(tempPos)
                    print np.reshape(tempPos, (1, last_x, last_y))
                    temp_frame = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                    if True not in np.isnan(temp_frame):
                        ncmc_frame = np.vstack((ncmc_frame, temp_frame))

                    if np.isnan(log_ncmc) == True:
                        print('nan, breaking')
                        break
                

        # Retrieve the log acceptance probability
            log_ncmc = nc_integrator.getLogAcceptanceProbability(nc_context)
            newinfo = nc_context.getState(True, True, False, True, True, False)
            newPos = newinfo.getPositions(asNumpy=True)
#            print newPos[1]
            newVel = newinfo.getVelocities(asNumpy=True)


#            correction = computeAlchemicalCorrection(testsystem.system, nc_context.getSystem(), oldPos, newPos, direction='insert')
#            print('correction', correction)

        #    enOld, enNew = ncmc_integrator.getEnergy(context)
        #    print('Energy:', enOld, enNew)
            randnum =  math.log(np.random.random())
            print(log_ncmc, randnum)
#            if log_ncmc > randnum and newinfo.getPotentialEnergy()/unit.kilocalorie_per_mole < 0:
            log_ncmc_edit = nc_integrator.getLogAcceptanceProbabilityEdit(nc_context)

            if log_ncmc > randnum:
                print('ncmc move accepted!')
                print 'Eorig', nc_integrator.getGlobalVariableByName('Eorig')
                print 'ncmc PE', newinfo.getPotentialEnergy()
                print 'ncmc KE', newinfo.getKineticEnergy()
                print 'old PE', oldPE
                print 'old KE', oldKE
                print newinfo.getPotentialEnergy() + newinfo.getKineticEnergy()
                print oldKE + oldPE
                total_diff =  ((newinfo.getPotentialEnergy() + newinfo.getKineticEnergy() - (oldKE + oldPE))/nc_integrator.kT)
                PE_diff = newinfo.getPotentialEnergy() - oldPE                 
                print 'PE_diff', PE_diff
                if np.isnan(total_diff) == False:
                    print factory._checkEnergyIsFinite(full_alchemical_system, newPos)
                    print nc_integrator.kT
                    al_correction = factory._checkEnergyIsFinite(full_alchemical_system, newPos)*(1/nc_integrator.kT)
                    print 'al_correction', al_correction


                randnum1 =  math.log(np.random.random())
                log_afterRot = -1.0*PE_diff / nc_integrator.kT
                print 'log_afterRot', log_afterRot
                otherCounter = otherCounter + 1
                print 'otherCounter', otherCounter

                if log_afterRot > randnum1:
                    print 'its cool!', log_afterRot, '>', randnum1
                    print('move accepted!')
                    accCounter = accCounter + 1.0
                    print('accCounter', float(accCounter)/float(stepsdone+1), accCounter)
#                   print('correction', correction)
    
                    nc_stateinfo = nc_context.getState(True, True, False, False, False, False)
    
                    oldPos = nc_stateinfo.getPositions(asNumpy=True)
                    oldVel = nc_stateinfo.getVelocities(asNumpy=True)
                else:
                    print 'bummer', log_afterRot, '<', randnum1
                    #TODO may want to think about velocity switching on rejection

                  



                
                

 
#            if log_ncmc_edit > randnum and newinfo.getPotentialEnergy()/unit.kilocalorie_per_mole < 0:
            else:
                print('move rejected, reversing velocities')
                nc_context.setPositions(oldPos)
                nc_context.setVelocities(-oldVel)
        #        stateinfo = nc_context.getState(False, True)
        #        velinfo = -stateinfo.getVelocities(asNumpy=True)
#            print 'total work', log_ncmc
#            log_ncmc_edit = nc_integrator.getLogAcceptanceProbabilityEdit(nc_context)
#
#            print 'total work_edit', log_ncmc_edit
#            print 'log original energy =', nc_integrator.getLogAcceptanceProbabilityOrig(nc_context)
#            
#            print 'ncmc PE', newinfo.getPotentialEnergy()
#            print 'ncmc KE', newinfo.getKineticEnergy()
#            print 'old PE', oldPE
#            print 'old KE', oldKE
#            print newinfo.getPotentialEnergy() + newinfo.getKineticEnergy()
#            print oldKE + oldPE
#            total_diff =  ((newinfo.getPotentialEnergy() + newinfo.getKineticEnergy() - (oldKE + oldPE))/nc_integrator.kT)
#            print 'total_diff', total_diff
#            if np.isnan(total_diff) == False:
#                print factory._checkEnergyIsFinite(full_alchemical_system, newPos)
#                print nc_integrator.kT
#                al_correction = factory._checkEnergyIsFinite(full_alchemical_system, newPos)*(1/nc_integrator.kT)
#                print 'al_correction', al_correction
            
            
#            if  total_diff < lowest_difference and newinfo.getPotentialEnergy()/unit.kilocalorie_per_mole < 0:
#                lowest_difference = total_diff
#            print 'lowest_difference', lowest_difference
#            if newinfo.getPotentialEnergy() + newinfo.getKineticEnergy() < oldKE + oldPE:
#                otherCounter = otherCounter + 1
#            print 'otherCounter', otherCounter
#            print log_ncmc

            nc_integrator.reset()
            md_simulation.context.setPositions(oldPos)
            md_simulation.context.setVelocities(oldVel)
            try:
                md_simulation.step(nstepsMD)
            except:
                stateinfo = md_simulation.context.getState(True, True, False, False, False, False)
                #oldPos = stateinfo.getPositions()
                print oldPos
                last_x, last_y = np.shape(oldPos)
                print np.reshape(oldPos, (1, last_x, last_y))
                reshape = (np.reshape(oldPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                print np.shape(reshape)
                print newinfo.getPotentialEnergy()
                print newinfo.getKineticEnergy()
                print oldPE
                print oldKE

#                last_top = md.load_prmtop('Cluster1.prmtop')
                last_top = md.Topology.from_openmm(testsystem.topology)
                broken_frame = md.Trajectory(xyz=reshape, topology=last_top)
                broken_frame.save_pdb('broken.pdb')
                print 'np.shape', np.shape(broken_frame)
                broken_ncmc = md.Trajectory(xyz=ncmc_frame, topology=last_top)
                try:
                    broken_ncmc.save_gro('broken_ncmc.gro')
                except ValueError:
                    print 'couldnt output gro, values too large'
                try:
                    broken_ncmc.save_dcd('broken_ncmc.dcd')
                except ValueError:
                    print 'couldnt output dcd, values too large'
                try:
                    broken_ncmc.save_pdb('broken_ncmc.pdb')
                except ValueError:
                    print 'couldnt output pdb, values too large'

                exit()


            md_stateinfo = md_simulation.context.getState(True, True, False, False, False, False)
            oldPos = md_stateinfo.getPositions(asNumpy=True)
            oldVel = md_stateinfo.getVelocities(asNumpy=True)
            nc_context.setPositions(oldPos)
            nc_context.setVelocities(oldVel)
            
        acceptRatio = accCounter/float(niter)
        print acceptRatio
        print(functions)
        print('numsteps ', nstepsNC)



print rand_rotation_matrix()
if 1:
     # Create a reference system.
     from openmmtools import testsystems
     testsystem = testsystems.LysozymeImplicit()
     [reference_system, positions] = [testsystem.system, testsystem.positions]
     # Create a factory to produce alchemical intermediates.
     receptor_atoms = range(0,2603) # T4 lysozyme L99A
     ligand_atoms = range(2603,2621) # p-xylene
     factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=ligand_atoms)
     full_alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_electrostatics=1, lambda_sterics=1))

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
    prmtop = openmm.app.AmberPrmtopFile('Cluster1.prmtop')
    inpcrd = openmm.app.AmberInpcrdFile('Cluster1.inpcrd')
    testsystem = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*unit.nanometer, constraints=openmm.app.HBonds)
    ligand_atoms = range(2634, 2649)
    recepotr_atoms = range(0,2634)
    factory = AbsoluteAlchemicalFactory(testsystem, ligand_atoms=ligand_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
    full_alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_electrostatics=1, lambda_sterics=1))

    #factory = AbsoluteAlchemicalFactory(testsystem, ligand_atoms=ligand_atoms, alchemical_torsions=True, alchemical_angles=True, annihilate_sterics=True, annihilate_electrostatics=True)
    alchemical_system = factory.createPerturbedSystem()
#    particle_mass(2634, 2649, alchemical_system)
 
# Create an NCMC switching integrator.
temperature = 300.0 * unit.kelvin
nstepsNC = 200
#functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5', 'lambda_torsions' : 'lambda', 'lambda_angles' : '0.9*(lambda^2) + 0.1' }
functions = { 'lambda_sterics' : 'lambda', 'lambda_electrostatics' : 'lambda^0.5'}
#functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}

#functions = { 'lambda_sterics' : '0.5*lambda + 0.5', 'lambda_electrostatics' : '0.5*(lambda^0.5) + 0.5'}


#nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='delete')
nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert', timestep=0.001*unit.picoseconds)
print 'kt', nc_integrator.kT
#nc_integrator = NCMCGHMCAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert')


#md_integrator = openmm.openmm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
# Create a Context
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
md_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=md_integrator)
md_simulation.context.setPositions(testsystem.positions)
md_simulation.context.setVelocitiesToTemperature(temperature)
print('Equilibrationg...')
md_simulation.step(1000)
md_info = md_simulation.context.getState(True, False, False, False, False, False)
equilPos = md_info.getPositions(asNumpy=True)
md_simulation.reporters.append(openmm.app.pdbreporter.PDBReporter('youtput_debug1.pdb', 10000))
md_simulation.reporters.append(HDF5Reporter('youtput_debug1.h5', 10000))
nc_context = openmm.Context(alchemical_system, nc_integrator)
nc_context.setPositions(equilPos)

#md_simulation.reporters.append(openmm.app.dcdreporter.DCDReporter('output.dcd', 1000))

# Minimize
openmm.LocalEnergyMinimizer.minimize(nc_context)
# Run the integrator
#ncmc_integrator.step(nsteps)
# Retrieve the log acceptance probability
#log_ncmc = ncmc_integrator.getLogAcceptanceProbability(context)
accCounter = 0.0
numIter = 10
test_class = testintegrator()
test_class.set_res(2603, 2621)
test_class.get_particle_masses(alchemical_system, 2603, 2621)
test_class.testintegrator(md_simulation, nc_context, nc_integrator, nstepsNC=nstepsNC, nstepsMD=1000, niter=10000, verbose=False)
    


