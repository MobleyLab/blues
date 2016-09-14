from ncmc_switching import *
import mdtraj as md
import math
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np

def get_lig_residues(lig_resname, coord_file, top_file=None):
    """
    This controls the ability to run a ncmc simulation with MD

    Arguments
    ---------
    lig_resname: str, resname that you want to get the atom indicies for (ex. 'LIG')
    coord_file:  file, cooridnate file (.pdb, .gro, .h5 etc)
    top_file: file, if topology isn't innately included in coordinates, include topology here

    Returns
    -------
    rotation : nx3 np.array in units.nm
        positions of ligand after random rotation

    """

    if top_file == None:
        traj = md.load(coord_file)
    else:
        traj = md.load(coord_file, top=top_file)

    lig_atoms = traj.top.select(("resname =~ %s") % lig_resname)
    first_res = min(lig_atoms)
    last_res = max(lig_atoms)
    return first_res, last_res, lig_atoms

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    # and http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    
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
class md_reporter:
    """
    class to handle error reporting
    """

    def __init__(self):
        self.traj = None
        self.shape = None
        self.temp_traj = None
        self.first = 0
        self.last  = -1

    def initialize(self, nc_context):

        nc_stateinfo = nc_context.getState(True, False, False, False, False, True)
        tempPos = nc_stateinfo.getPositions(asNumpy=True)

        last_x, last_y = np.shape(tempPos)
        self.shape = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)

    def add_frame(self, nc_context):
        if self.traj == None:
            nc_stateinfo = nc_context.getState(True, False, False, False, False, True)
            tempPos = nc_stateinfo.getPositions(asNumpy=True)

            last_x, last_y = np.shape(tempPos)
            self.traj = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
        else:
            nc_stateinfo = nc_context.getState(True, False, False, False, False, True)
            tempPos = nc_stateinfo.getPositions(asNumpy=True)

            last_x, last_y = np.shape(tempPos)
            temp = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
            self.traj = np.concatenate((self.traj, temp))
    def add_temp_traj(self, nc_context):
        if self.temp_traj == None:
            nc_stateinfo = nc_context.getState(True, False, False, False, False, True)
            tempPos = nc_stateinfo.getPositions(asNumpy=True)

            last_x, last_y = np.shape(tempPos)
            self.temp_traj = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
        else:
            nc_stateinfo = nc_context.getState(True, False, False, False, False, True)
            tempPos = nc_stateinfo.getPositions(asNumpy=True)

            last_x, last_y = np.shape(tempPos)
            temp = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
            if True not in np.isnan(temp):
                self.temp_traj = np.concatenate((self.temp_traj, temp))
    def combine_traj(self):
        if self.traj != None and self.temp_traj != None:
            if True not in np.isnan(self.temp_traj):
                self.traj = np.concatenate((self.traj, self.temp_traj))
            self.temp_traj = None
        if self.traj == None and self.temp_traj != None:
            self.traj = self.temp_traj[:]

    def save_traj(self, output_name, topology, traj=None):
        if traj == None:
            traj = self.traj
        last_top = md.Topology.from_openmm(topology)
        broken_ncmc = md.Trajectory(xyz=traj, topology=last_top)
        try:
            broken_ncmc.save_dcd(output_name)
        except ValueError:
            print 'couldnt output', output_name, 'values too large'

         
def rand_rotation_matrix2():
    rand_quat = md.utils.uniform_quaternion()
    matrix_out = md.utils.rotation_matrix_from_quaternion(rand_quat)
    return matrix_out

def zero_masses( system, firstres, lastres):
    for index in range(firstres, lastres):
        system.setParticleMass(index, 0*unit.daltons)

def selectres(reslist, inpcrd, prmtop):
    t = md.load(inpcrd, top=prmtop)
    output_list = []
    for entry in reslist:
        output_list.extend(t.top.select(("resid %s") % (entry)))
    return output_list

def zero_allother_masses( system, indexlist):
    num_atoms = system.getNumParticles()
    for index in range(num_atoms):
        if index in indexlist:
            pass
        else:
            system.setParticleMass(index, 0*unit.daltons)


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
        self.residueList = None

    def set_res(self, residueList):
        self.residueList = residueList

    def get_particle_masses(self, system, residueList=None):
        if residueList == None:
            residueList = self.residueList
        mass_list = []
        total_mass = 0*unit.dalton
        for index in residueList:
            mass = system.getParticleMass(index)
            total_mass = total_mass + mass
            print('mass', mass, 'total_mass', total_mass)
            mass_list.append([mass])
        total_mass = np.sum(mass_list)
        mass_list = np.asarray(mass_list)
        mass_list.reshape((-1,1))
        total_mass = np.array(total_mass)
        total_mass = np.sum(mass_list)
        temp_list = np.zeros((len(residueList), 1))
        for index in range(len(residueList)):
            mass_list[index] = (np.sum(mass_list[index])).value_in_unit(unit.daltons)
        mass_list =  mass_list*unit.daltons
        self.total_mass = total_mass
        self.mass_list = mass_list
        return total_mass, mass_list
            
    def zero_masses(self, system, residueList=None):
        if residueList == None:
            residueList = self.residueList
        for index in (residueList):
            system.setParticleMass(index, 0*unit.daltons)
        

    def calculate_com(self, total_mass, mass_list, pos_state, residueList=None, rotate=True):
        """
        This function calculates the com of specified residues and optionally rotates them around the center of mass

        Arguments
        ---------
        total_mass: simtk.unit.quantity.Quantity in units daltons, contains the total masses of the particles for COM calculation
        mass_list:  nx1 np.array in units daltons, contains the masses of the particles for COM calculation
        pos_state:  nx3 np. array in units.nanometers, returned from state.getPositions
        residueList: list of int, index of atoms which you'll calculate the total com for 
        rotate: boolean, if True, rotates center of mass by random rotation matrix
        Returns
        -------
        rotation : nx3 np.array in units.nm
            positions of ligand after random rotation

        """
        if residueList == None:
            residueList = self.residueList
        if mass_list == None:
            mass_list = self.mass_list
        if total_mass == None:
            total_mass = self.total_mass
        if mass_list == None:
            mass_list = self.mass_list

        #choose ligand indicies
        copy_orig = copy.deepcopy(pos_state)
        lig_coord = np.zeros((len(residueList), 3))
        for index, resnum in enumerate(residueList):
            lig_coord[index] = copy_orig[resnum]
            print lig_coord
        lig_coord = lig_coord*unit.nanometers
        copy_coord = copy.deepcopy(lig_coord)
        #mass corrected coordinates (to find COM)
#        print('mass_list', mass_list)
#        print('total_mass', total_mass)
#        print('copy_coord', copy_coord)
        mass_corrected = mass_list / total_mass * copy_coord
        sum_coord = mass_corrected.sum(axis=0).value_in_unit(unit.nanometers)
        com_coord = [0.0, 0.0, 0.0]*unit.nanometers
        #units are funky, so do this step to get them to behave right
        for index in range(3):
            com_coord[index] = sum_coord[index]*unit.nanometers
        if rotate ==True:
            for index in range(3):
                lig_coord[:,index] = lig_coord[:,index] - com_coord[index]
            #multiply lig coordinates by rot matrix and add back COM translation from origin
            rotation =  np.dot(lig_coord.value_in_unit(unit.nanometers), rand_rotation_matrix2())*unit.nanometers
            rotation = rotation + com_coord
            return rotation
        else:    
        #remove COM from ligand coordinates to then perform rotation
            return com_coord            #remove COM from ligand coordinates to then perform rotation


    def testintegrator(self, md_simulation, nc_context, nc_integrator, dummy_simulation, nstepsNC=25, nstepsMD=1000, niter=10, periodic=True, verbose=False, residueList=None, alchemical_correction=False, rot_report=False, ncmc_report=False, origPos=None):
        if residueList == None:
            residueList = self.residueList
        #set up initial counters/ inputs
        accCounter = 0
        otherCounter = 0
        nc_stateinfo = nc_context.getState(True, False, False, False, False, periodic)
        tempPos = nc_stateinfo.getPositions(asNumpy=True)

        last_x, last_y = np.shape(tempPos)
        ncmc_frame = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
        tempTraj = np.array(())
        if rot_report == True:
            rot_reporter = md_reporter()
        if ncmc_report == True:
            ncmc_reporter = md_reporter()


        #set inital conditions
        md_stateinfo = md_simulation.context.getState(True, True, False, True, True, periodic)
        oldPos = md_stateinfo.getPositions(asNumpy=True)
        oldVel = md_stateinfo.getVelocities(asNumpy=True)

        oldPE =  md_stateinfo.getPotentialEnergy()
        oldKE =  md_stateinfo.getKineticEnergy()
        nc_context.setPositions(oldPos)
        nc_context.setVelocities(oldVel)
        #TODO
        
        first_ncmc = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)




        for stepsdone in range(niter):
            if 1:
            #if active, performs the rotation of the ligand
                if rot_report == True:
                    rot_reporter.add_frame(nc_context)
                rot_output = self.calculate_com(total_mass=self.total_mass, mass_list=self.mass_list, pos_state=oldPos, residueList=residueList)
                rot_output = rot_output[:].value_in_unit(unit.nanometers)
                rotPos = oldPos.value_in_unit(unit.nanometers)
                for index, resnum in enumerate(residueList):
                    rotPos[resnum] = rot_output[index]
                rotPos[:] = rotPos*unit.nanometers
                nc_context.setPositions(rotPos)
                if rot_report == True:
                    rot_reporter.add_frame(nc_context)



            print('performing ncmc step')
            print('accCounter =', accCounter)
            mdinfo = md_simulation.context.getState(True, True, False, True, True, periodic)
            oldPE =  mdinfo.getPotentialEnergy()
            oldKE =  mdinfo.getKineticEnergy()
            if verbose == True:
                log_ncmc = nc_integrator.getLogAcceptanceProbability(nc_context)
                print('before ncmc move log_ncmc', log_ncmc, np.isnan(log_ncmc))
            prev = np.random.randint(2)
            if prev == 1:
                nc_context.setVelocities(-oldVel)

            for stepscarried in range(nstepsNC):
#                tempTraj = np.array(())
                if verbose == True and stepscarried == 0:
                    tempPos = nc_stateinfo.getPositions(asNumpy=True)
                    last_x, last_y = np.shape(tempPos)
                    print np.reshape(tempPos, (1, last_x, last_y))
                    tempTraj = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                    first_ncmc = np.concatenate((first_ncmc, tempTraj))
                    print np.shape(first_ncmc)
#                    tempTraj = np.concatenate((tempTraj, temp_frame))


                try:
                    nc_integrator.step(1)
                except:
                    print('nan, breaking')
                    break
                if ncmc_report == True:
                    #edit
                    if stepscarried % 10 == 0:
                        ncmc_reporter.add_temp_traj(nc_context)
                if verbose == True:
                    log_ncmc = nc_integrator.getLogAcceptanceProbability(nc_context)
                    print('log_ncmc', log_ncmc, stepscarried, np.isnan(log_ncmc))
                    nc_stateinfo = nc_context.getState(True, False, False, False, False, periodic)
##                    tempPos = nc_stateinfo.getPositions(asNumpy=True)
                    #connected
                    newinfo = nc_context.getState(True, True, False, True, True, periodic)
                    newPos = newinfo.getPositions(asNumpy=True)
                    #correction = computeAlchemicalCorrection(testsystem, nc_context.getSystem(), oldPos, newPos, direction='insert')
                    #print('correction', correction)
                    ######
                    print nc_integrator.getGlobalVariableByName('lambda')
##                    last_x, last_y = np.shape(tempPos)
#                    print np.reshape(tempPos, (1, last_x, last_y))
                    temp_frame = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                    if True not in np.isnan(temp_frame) and stepscarried != 0:
#                        ncmc_frame = np.vstack((ncmc_frame, temp_frame))
                        tempTraj = np.concatenate((tempTraj, temp_frame))


                    if np.isnan(log_ncmc) == True:
                        print('nan, breaking')
                        break
                

        # Retrieve the log acceptance probability
##            if verbose == True:
##                ncmc_frame = np.concatenate((ncmc_frame, tempTraj))
            if ncmc_report == True:
                ncmc_reporter.combine_traj()

            log_ncmc = nc_integrator.getLogAcceptanceProbability(nc_context)
            newinfo = nc_context.getState(True, True, False, True, True, periodic)
            newPos = newinfo.getPositions(asNumpy=True)
            newVel = newinfo.getVelocities(asNumpy=True)
            randnum =  math.log(np.random.random())
            if alchemical_correction == True and np.isnan(log_ncmc) == False:
                print('performing correction')
                dummy_simulation.context.setPositions(oldPos)
                dummy_simulation.context.setVelocities(oldVel)
                dummy_info = dummy_simulation.context.getState(True, True, False, True, True, False)#*might want to make this periodic
                md_simulation.context.setPositions(newPos)
                md_simulation.context.setVelocities(newVel)
                md_info_new = md_simulation.context.getState(True, True, False, True, True, False)#*might want to make this periodic

                print('md_energy_orig', oldPE)

                print('md_energy', md_info_new.getPotentialEnergy())               
                print('dummy_energy', dummy_info.getPotentialEnergy())

                dummy_simulation.context.setPositions(newPos)
                reshape = (np.reshape(oldPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                dummy_simulation.context.setVelocities(newVel)
                dummy_info = dummy_simulation.context.getState(True, True, False, True, True, False)#*might want to make this periodic
                print('dummy_energy', dummy_info.getPotentialEnergy())
                print('difference', (-newinfo.getPotentialEnergy() + md_info_new.getPotentialEnergy())*(1/nc_integrator.kT))
                log_ncmc = log_ncmc + (-newinfo.getPotentialEnergy() + md_info_new.getPotentialEnergy())*(1/nc_integrator.kT)

            print(log_ncmc, randnum)
            yesAccept = False
            if log_ncmc > randnum:
                print('ncmc move accepted!')
                print 'ncmc PE', newinfo.getPotentialEnergy(), 'old PE', oldPE
                print newinfo.getPotentialEnergy() + newinfo.getKineticEnergy()
                PE_diff = newinfo.getPotentialEnergy() - oldPE                 
                print 'PE_diff', PE_diff


                randnum1 =  math.log(np.random.random())
                if alchemical_correction == True:
                    log_afterRot = -1.0*(md_info_new.getPotentialEnergy() - oldPE)/nc_integrator.kT
                else:
                    log_afterRot = -1.0*PE_diff / nc_integrator.kT
                print 'log_afterRot', log_afterRot
                otherCounter = otherCounter + 1
                print 'otherCounter', otherCounter

                if log_afterRot > randnum1:
                    print 'its cool!', log_afterRot, '>', randnum1
                    print('move accepted!')
                    accCounter = accCounter + 1.0
                    print('accCounter', float(accCounter)/float(stepsdone+1), accCounter)
                    yesAccept = True
    
                    nc_stateinfo = nc_context.getState(True, True, False, False, False, periodic)
    
                    oldPos = newPos[:]
                    oldVel = newVel[:]
                else:
                    print 'bummer', log_afterRot, '<', randnum1
                    #TODO may want to think about velocity switching on rejection
            else:
                print('move rejected, reversing velocities')
                nc_context.setPositions(oldPos)
                nc_context.setVelocities(-oldVel)
            if prev == 1:
                oldVel = -oldVel
            if yesAccept==False and origPos != None:
                oldPos = origPos

            print('accCounter:', accCounter, 'otherCounter:', otherCounter, 'iter:', stepsdone)
            nc_integrator.reset()
            md_simulation.context.setPositions(oldPos)
            md_simulation.context.setVelocities(oldVel)
#            md_simulation.context.setVelocitiesToTemperature(100)
            if nstepsMD > 0:
                try:
                    md_simulation.step(nstepsMD)
                except Exception as e:
                    print('Error:', e)
                    stateinfo = md_simulation.context.getState(True, True, False, False, False, periodic)
                    #oldPos = stateinfo.getPositions()
                    if ncmc_report == True:
                        print 'shape', np.shape(ncmc_reporter.traj)
                        ncmc_reporter.save_traj(output_name='broken_ncmc.dcd', topology=testsystem.topology)
                    if rot_report == True:
                        print np.shape(rot_reporter.traj)
                
                        rot_reporter.save_traj(output_name='ncmc_rot.gro', topology=testsystem.topology)

                    print oldPos
                    last_x, last_y = np.shape(oldPos)
                    print np.reshape(oldPos, (1, last_x, last_y))
                    reshape = (np.reshape(oldPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                    print np.shape(reshape)
                    print newinfo.getPotentialEnergy()
                    print newinfo.getKineticEnergy()
                    print oldPE
                    print oldKE

                    last_top = md.Topology.from_openmm(testsystem.topology)
                    broken_frame = md.Trajectory(xyz=reshape, topology=last_top)
                    broken_frame.save_pdb('broken.pdb')
                    print 'np.shape', np.shape(broken_frame)
                    broken_ncmc = md.Trajectory(xyz=ncmc_frame, topology=last_top)
                    try:
                        broken_ncmc.save_gro('broken_last.gro')
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



            md_stateinfo = md_simulation.context.getState(True, True, False, False, False, periodic)
            oldPos = md_stateinfo.getPositions(asNumpy=True)
            oldVel = md_stateinfo.getVelocities(asNumpy=True)
            nc_integrator.reset() #TODO
            nc_context.setPositions(oldPos)
            nc_context.setVelocities(oldVel)
            
        #print 'traj', ncmc_reporter.traj
        #print 'traj2', rot_reporter.traj
#        if ncmc_report == True:
        if ncmc_report == True:
            print 'shape', np.shape(ncmc_reporter.traj)
            ncmc_reporter.save_traj(output_name='broken_ncmc.dcd', topology=testsystem.topology)
        if rot_report == True:
            print np.shape(rot_reporter.traj)

            rot_reporter.save_traj(output_name='ncmc_rot.gro', topology=testsystem.topology)

        acceptRatio = accCounter/float(niter)
        print acceptRatio
        print otherCounter

        print('numsteps ', nstepsNC)
        return oldPos
