from simtk.openmm.app import *
from simtk.openmm import *
from blues.ncmc_switching import *
import simtk.unit as unit
import mdtraj as md
import math
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np
from openmmtools import testsystems

def get_lig_residues(lig_resname, coord_file, top_file=None):
    """
    Helper function to get atom indices of a ligand from a coordinate 
    and/or topology file.
    Arguments
    ---------
    lig_resname: str 
        resname that you want to get the atom indicies for (ex. 'LIG')
    coord_file:  str 
        path of coordinate file (.pdb, .gro, .h5 etc)
    top_file: str, optional, default=None
        path of topology file. Include if the topology is not included
        in the coord_file
    Returns
    -------
    lig_atoms : list of ints
        list of atoms in the coordinate file matching lig_resname
    """

    if top_file == None:
        traj = md.load(coord_file)
    else:
        traj = md.load(coord_file, top=top_file)

    lig_atoms = traj.top.select(("resname =~ %s") % lig_resname)
    first_res = min(lig_atoms)
    last_res = max(lig_atoms)
    return lig_atoms

def quantity_is_finite(quantity):
    """
    Check that elements in quantity are all finite.
    Arguments
    ----------
    quantity : simtk.unit.Quantity
        The quantity to check
    Returns
    -------
    is_finite : bool
        If quantity is finite, returns True; otherwise False.
    """
    if np.any( np.isnan( np.array(quantity / quantity.unit) ) ):
        return False
    return True


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
            print('couldnt output', output_name, 'values too large')
         
def rand_rotation_matrix():
    """
    Creates a uniform random rotation matrix
    Returns
    -------
    matrix_out: 3x3 np.array
        random rotation matrix
    """
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

class SimNCMC(object):
    def __init__(self, temperature, residueList, **kwds):
        """
        Stores parameters and methods relevant to NCMC and runs NCMC simulations.
        Arguments
        ---------
        temperature: simtk.unit.kelvin 
            desired temperature of simulation in Kelvins
        residueList: list of ints 
            residue numbers that specify all the ligand atoms
        """
        super(SimNCMC, self).__init__(**kwds)

        self.total_mass = 0
        self.mass_list = None
        self.residueList = residueList
        self.acceptance = 0
        self.md_simulation = None
        self.dummy_simulation = None
        self.nc_context = None
        self.nc_integrator = None
        self.nc_pos = None
        self._storage = None
        self.temperature = temperature
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        beta = 1.0 / kT
        self.beta = beta

    def get_particle_masses(self, system, residueList=None):
        if residueList == None:
            residueList = self.residueList
        mass_list = []
        total_mass = 0*unit.dalton
        for index in residueList:
            mass = system.getParticleMass(int(index))
            total_mass = total_mass + mass
#            print('mass', mass, 'total_mass', total_mass)
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
            
    def zero_masses(self, system, atomList=None):
        """
        Zeroes the masses of specified atoms to constrain certain degrees of freedom
        Arguments
        ---------
        system: simtk.openmm.system 
            system to zero masses
        atomList: list of ints 
            atom indicies to zero masses
        """
        for index in (atomList):
            system.setParticleMass(index, 0*unit.daltons)
        

    def calculate_com(self,  pos_state, total_mass=None, mass_list=None, residueList=None, rotate=False):
        """
        This function calculates the com of specified residues and optionally 
        rotates them around the center of mass.
        Arguments
        ---------
        total_mass: simtk.unit.quantity.Quantity in units daltons
            contains the total masses of the particles for COM calculation
        mass_list:  nx1 np.array in units daltons, 
            contains the masses of the particles for COM calculation
        pos_state:  nx3 np. array in units.nanometers 
            returned from state.getPositions
        residueList: list of int, 
            list of atom indicies which you'll calculate the total com for 
        rotate: boolean
            if True, rotates center of mass by random rotation matrix, 
            else returns center of mass coordiantes
        Returns
        -------
        if rotate==True
        rotation : nx3 np.array in units.nm
            positions of ligand with or without random rotation (depending on rotate)
        if rotate==False
        com_coord: 1x3 np.array in units.nm
            position of the center of mass coordinate
        """
        if residueList == None:
            residueList = self.residueList
        if mass_list == None:
            mass_list = self.mass_list
        if mass_list == None:
#            self.get_particle_masses(system=self.md_simulation.system)
            mass_list = self.mass_list

        if total_mass == None:
            total_mass = self.total_mass

        #choose ligand indicies
        copy_orig = copy.deepcopy(pos_state)
        lig_coord = np.zeros((len(residueList), 3))
        for index, resnum in enumerate(residueList):
            lig_coord[index] = copy_orig[resnum]
        lig_coord = lig_coord*unit.nanometers
        copy_coord = copy.deepcopy(lig_coord)
        #mass corrected coordinates (to find COM)
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
            rotation =  np.dot(lig_coord.value_in_unit(unit.nanometers), rand_rotation_matrix())*unit.nanometers
            rotation = rotation + com_coord
            return rotation
        else:    
        #remove COM from ligand coordinates to then perform rotation
            return com_coord            #remove COM from ligand coordinates to then perform rotation

    def create_alchemicalSystem(self, coord_file, top_file, residueList=None):
        """
        Helper function to create alchemical system.
        Arguments
        ---------
        coord_file: str
            Path of amber coordinate file
        top_file: str
            path of amber topology file
        residueList: list of ints
            Indices of ligand atoms for alchemical transformation. If none are
            specified uses the residueList in self.residueList
        Returns
        ---------    
        alchemical_system: simtk.openmm.System
            The alchemically transformed system
        """

        if residueList == None:
            residueList = self.residueList
        prmtop = openmm.app.AmberPrmtopFile(top_file)
        inpcrd = openmm.app.AmberInpcrdFile(coord_file)
        temp_system = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*unit.nanometer, constraints=openmm.app.HBonds)
        testsystem = testsystems.TestSystem
        testsystem.system = temp_system
        testsystem.topology = prmtop.topology
        testsystem.positions = inpcrd.positions
        ligand_atoms = get_lig_residues(lig_resname='LIG', coord_file=coord_file, top_file=top_file)
        factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
        self.alchemical_system = factory.createPerturbedSystem()
        return self.alchemical_system

    def create_normalSystem(self, coord_file, top_file):
        """
        Helper function to create normal system.
        Arguments
        ---------
        coord_file: str
            Path of amber coordinate file
        top_file: str
            path of amber topology file
            Specified uses the residueList in self.residueList
        Returns
        ---------    
        system: simtk.openmm.System
            The normal system
        """

        prmtop = openmm.app.AmberPrmtopFile(top_file)
        inpcrd = openmm.app.AmberInpcrdFile(coord_file)
        temp_system = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*unit.nanometer, constraints=openmm.app.HBonds)
        testsystem = testsystems.TestSystem
        testsystem.system = temp_system 
        testsystem.topology = prmtop.topology
        testsystem.positions = inpcrd.positions
        self.normalsystem = testsystem
        return self.normalsystem

#    def createNormalSimulation(self, friction=1/unit.picosecond, timestep=0.002*unit.picoseconds, temperature=None):
#        if temperature == None:
#            temperature = self.temperature
#        self.md_integrator = openmm.openmm.LangevinIntegrator(temperature, friction, timestep)
#        self.dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, friction, timestep)
#        self.md_simulation = openmm.app.simulation.Simulation(topology=self.normalsystem.topology, system=self.normalsystem.system, integrator=self.md_integrator)
#        self.dummy_simulation = openmm.app.simulation.Simulation(topology=self.normalsystem.topology, system=self.normalsystem.system, integrator=self.dummy_integrator)
   
    def rotationalMove(self, residueList=None):
        """
        Function to be used in movekey. Performs a rotation around the center of mass
        of the ligand.
        Arguments
        ---------
        generally none
        """    

        if residueList == None:
            residueList = self.residueList
        rot_output = self.calculate_com(total_mass=self.total_mass, mass_list=self.mass_list, pos_state=self.nc_pos, residueList=residueList, rotate=True)
        rot_output = rot_output[:].value_in_unit(unit.nanometers)
        print(rot_output, 'rot_output')
        rotPos = self.nc_pos.value_in_unit(unit.nanometers)
        for index, resnum in enumerate(residueList):
            print(rotPos[resnum], rot_output[index])
            rotPos[resnum] = rot_output[index]
        rotPos[:] = rotPos*unit.nanometers
        self.nc_context.setPositions(rotPos)

    def runSim(self, md_simulation, nc_context, nc_integrator, dummy_simulation, movekey=None, nstepsNC=25, nstepsMD=1000, niter=10, periodic=True, verbose=False, residueList=None, alchemical_correction=False, ncmc_storage='out_ncmc.h5', write_ncmc_interval=None):
        """
        Runs a ncmc+MD simulation
        Arguments
        ---------
        md_simulation: simtk.openmm.simulation
            normal interacting simulation
        nc_context: simtk.openmm.context
            Context containing the alchemical system with a NCMCAlchemicalIntegrator (currently
            either GHMC or VV based)
        nc_intgrator: blues.ncmc_switching.NCMCAlchemicalIntegrator
            integrator used for the NCMC steps
        dummy_simulation: simtk.openmm.simulation
            A copy of the md_simulation. Is used for alchemical corrections and for writing ncmc frames
            if write_ncmc_interval is specified
        movekey: list of 2 item lists, the first item is a reference to a function and the second is a list of ints
            This will perform the given function at the start of a NCMC step, given that step is listed in
            the list of ints. In the case where mulitple moves are being performed at the same step they are performed
            in list order.
        nstepsNC: int, optional, default=25
            Number of NC steps to be performed during an iteration
        nstepsMD: int, optional, default=1000
            Number of MD steps to be performed during an iteration
        niter: int, optional, default=10
            Number of total iterations to be performed
        alchemical_correction: boolean, optional, default=False
            Whether or not to perform alchemical correction
        ncmc_storage: str, optional, default='out_ncmc.h5'
            Name of hdf5 file to store positions of system during NCMC step
            if write_ncmc_interval is specified.
        write_ncmc_interval: int or None, optional, default=None
            If int is used, specifies the interval which NCMC positions are written.
            Also writes out after every function applied by movekey
        """
        if residueList == None:
            residueList = self.residueList
        self.md_simulation = md_simulation
        self.dummy_simulation = dummy_simulation
        self.nc_context = nc_context
        self.nc_integrator = nc_integrator
        self.get_particle_masses(self.md_simulation.system)
        if type(write_ncmc_interval) == int:
            h5reporter = md.reporters.HDF5Reporter(file=ncmc_storage, reportInterval=100, 
                coordinates=True, time=False, cell=False, potentialEnergy=False, kineticEnergy=False, 
                temperature=False, velocities=False, atomSubset=None)
        #set up initial counters/ inputs
        accCounter = 0
        nc_stateinfo = nc_context.getState(True, False, False, False, False, periodic)
        tempPos = nc_stateinfo.getPositions(asNumpy=True)

        last_x, last_y = np.shape(tempPos)
        ncmc_frame = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
        tempTraj = np.array(())

        #set inital conditions
        md_stateinfo = md_simulation.context.getState(True, True, False, True, True, periodic)
        oldPos = md_stateinfo.getPositions(asNumpy=True)
        oldVel = md_stateinfo.getVelocities(asNumpy=True)

        oldPE =  md_stateinfo.getPotentialEnergy()
        oldKE =  md_stateinfo.getKineticEnergy()
        nc_context.setPositions(oldPos)
        self.nc_pos = oldPos
        nc_context.setVelocities(oldVel)
        first_ncmc = (np.reshape(tempPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)

        for stepsdone in range(niter):
            print('performing ncmc step')
            print('accCounter =', accCounter)
            mdinfo = md_simulation.context.getState(True, True, False, True, True, periodic)
            oldPE =  mdinfo.getPotentialEnergy()
            oldKE =  mdinfo.getKineticEnergy()
            if alchemical_correction == True:
                alc_oldPE = nc_context.getState(True, True, False, True, True, periodic).getPotentialEnergy()

            for stepscarried in range(nstepsNC):
                if movekey != None:
                    for func in movekey:
                        if stepscarried in func[1]:
                            print('doing the move')
                            print('before step', nc_integrator.getGlobalVariableByName('lambda'))

                            func[0]()
                            if write_ncmc_interval:
                                positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
                                dummy_simulation.context.setPositions(positions)
                                h5reporter.report(dummy_simulation, dummy_simulation.context.getState(True, True))

                try:
                    if verbose:
                        print('work_before', nc_integrator.getGlobalVariableByName("total_work"))
                        print('lambda_before', nc_integrator.getGlobalVariableByName("lambda"))
                        print('shadow_before', nc_integrator.getGlobalVariableByName("shadow_work"))
                        print('protocol_before', nc_integrator.getGlobalVariableByName("protocol_work"))
                        print('Epert_before', nc_integrator.getGlobalVariableByName("Epert"))

                    if write_ncmc_interval and (stepscarried+1) % write_ncmc_interval == 0:
                        if verbose:
                            print('writing coordinates')
                        positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
                        dummy_simulation.context.setPositions(positions)
                        h5reporter.report(dummy_simulation, dummy_simulation.context.getState(True, True))

                    nc_integrator.step(1)
                    if verbose:
                        print('work_after', nc_integrator.getGlobalVariableByName("total_work"))
                        print('lambda_after', nc_integrator.getGlobalVariableByName("lambda"))
                        print('shadow_after', nc_integrator.getGlobalVariableByName("shadow_work"))
                        print('protocol_after', nc_integrator.getGlobalVariableByName("protocol_work"))
                        print('Eold', nc_integrator.getGlobalVariableByName("Eold"))
                        print('Enew', nc_integrator.getGlobalVariableByName("Enew"))
                        print('Epert_after', nc_integrator.getGlobalVariableByName("Epert"))

                except Exception as e:
                    if str(e) == "Particle coordinate is nan":
                        print('nan, breaking')
                        break                

            log_ncmc = nc_integrator.getLogAcceptanceProbability(nc_context)
            newinfo = nc_context.getState(True, True, False, True, True, periodic)
            newPos = newinfo.getPositions(asNumpy=True)
            newVel = newinfo.getVelocities(asNumpy=True)
            randnum =  math.log(np.random.random())
            if alchemical_correction == True and np.isnan(log_ncmc) == False:
                alc_newPE = newinfo.getPotentialEnergy()
                dummy_simulation.context.setPositions(newPos)
                dummy_info = dummy_simulation.context.getState(True, True, False, True, True, periodic)
                norm_newPE = dummy_info.getPotentialEnergy()
                correction_factor = -1.0*((norm_newPE - alc_newPE) - (oldPE - alc_oldPE))*(1/nc_integrator.kT)
                print('correction_factor', correction_factor)
                log_ncmc = log_ncmc + correction_factor

            if log_ncmc > randnum:
                print('ncmc move accepted')
                if verbose:
                    print('ncmc PE', newinfo.getPotentialEnergy(), 'old PE', oldPE)
                    print('ncmc Total energy', newinfo.getPotentialEnergy() + newinfo.getKineticEnergy())
                    PE_diff = newinfo.getPotentialEnergy() - oldPE                 
                    print('PE_diff', PE_diff)
                print('accepted since', log_ncmc, '>', randnum)
                print('log_ncmc > randnum')
                print('move accepted')
                accCounter = accCounter + 1.0
                print('accCounter', float(accCounter)/float(stepsdone+1), accCounter)
    
                nc_stateinfo = nc_context.getState(True, True, False, False, False, periodic)
    
                oldPos = newPos[:]
                oldVel = newVel[:]
            else:
                print('ncmc PE', newinfo.getPotentialEnergy(), 'old PE', oldPE)
                print('rejected', log_ncmc, '<', randnum)
                print('log_ncmc > randnum')
                print('move rejected')
                nc_context.setPositions(oldPos)
                nc_context.setVelocities(-oldVel)

            print('accCounter:', accCounter,  'iter:', stepsdone+1)
            nc_integrator.reset()
            md_simulation.context.setPositions(oldPos)
            md_simulation.context.setVelocities(oldVel)
            md_simulation.context.setVelocitiesToTemperature(self.temperature)
            if nstepsMD > 0:
                try:
                    md_simulation.step(nstepsMD)
                except Exception as e:
                    print('Error:', e)
                    stateinfo = md_simulation.context.getState(True, True, False, False, False, periodic)
                    print(oldPos)
                    last_x, last_y = np.shape(oldPos)
                    print(np.reshape(oldPos, (1, last_x, last_y)))
                    reshape = (np.reshape(oldPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                    print(np.shape(reshape))
                    print(newinfo.getPotentialEnergy())
                    print(newinfo.getKineticEnergy())
                    print(oldPE)
                    print(oldKE)

                    last_top = md.Topology.from_openmm(md_simulation.topology)
                    broken_frame = md.Trajectory(xyz=reshape, topology=last_top)
                    broken_frame.save_pdb('broken.pdb')
                    print('np.shape', np.shape(broken_frame))
                    broken_ncmc = md.Trajectory(xyz=ncmc_frame, topology=last_top)
                    try:
                        broken_ncmc.save_gro('broken_last.gro')
                    except ValueError:
                        print('couldnt output gro, values too large')
                    try:
                        broken_ncmc.save_dcd('broken_ncmc.dcd')
                    except ValueError:
                        print('couldnt output dcd, values too large')
                    try:
                        broken_ncmc.save_pdb('broken_ncmc.pdb')
                    except ValueError:
                
                        print('couldnt output pdb, values too large')
                    exit()

            md_stateinfo = md_simulation.context.getState(True, True, False, False, False, periodic)
            oldPos = md_stateinfo.getPositions(asNumpy=True)
            oldVel = md_stateinfo.getVelocities(asNumpy=True)
            nc_integrator.reset()
            nc_context.setPositions(oldPos)
            self.nc_pos = oldPos
            nc_context.setVelocities(oldVel)
            
        acceptRatio = accCounter/float(niter)
        print(acceptRatio)
        print('numsteps ', nstepsNC)
        return oldPos
