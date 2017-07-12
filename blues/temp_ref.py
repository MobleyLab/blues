from __future__ import print_function
import sys
from simtk.openmm.app import *
from simtk.openmm import *
from blues.ncmc_switching import *
import simtk.unit as unit
import mdtraj as md
import math
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np

def random_sphere_point(radius):
    r = radius * ( np.random.random()**(1./3.) )
    phi = np.random.uniform(0,2*np.pi)
    costheta = np.random.uniform(-1,1)
    u = np.random.random()
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    sphere_point = np.array([x, y, z]) * r
    return sphere_point

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

class WaterNCMC(object):
    def __init__(self, temperature, residueList, radius=2*unit.nanometers, **kwds):
        """
        Stores parameters and methods relevant to NCMC and runs NCMC simulations.
        Arguments
        ---------
        temperature: simtk.unit.kelvin
            desired temperature of simulation in Kelvins
        residueList: list of ints
            Residue index numbers that specify the alchemical water indicies.
        alch_water: list of 3 ints
            List of ints corresponding to the indices of a single alchemical water.
            The first index corresponds to the O, the second to H1 and last to H2.
        """
        super(WaterNCMC, self).__init__(**kwds)

        self.total_mass = 0
        self.mass_list = None
        self.residueList = residueList
        self.acceptance = 0
        self.md_simulation = None
        self.dummy_simulation = None
        self.nc_context = None
        self.nc_integrator = None
        self._storage = None
        self.radius = radius
        self.temperature = temperature
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        beta = 1.0 / kT
        self.beta = beta

    def get_particle_masses(self, system, residueList=None, set_self=True):
        """
        Finds the mass of each particle given by residueList and returns
        a list of those particle masses as well as the total mass. If
        set_self=True, sets corresponding SimNCMC attributes as well as
        returning them.
        Arguments
        ---------
        system: simtk.openmm.system
            Openmm system object containing the particles of interest
        residueList: list of ints
            particle indices to find the masses of
        set_self: boolean
            if true, sets self.total_mass and self.mass_list to the
            outputs of this function
        """
        if residueList == None:
            residueList = self.residueList
        mass_list = []
        total_mass = 0*unit.dalton
        for index in residueList:
            mass = system.getParticleMass(int(index))
            total_mass = total_mass + mass
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
        if set_self == True:
            self.total_mass = total_mass
            self.mass_list = mass_list
        return total_mass, mass_list

    def zero_masses(self, system, atomList=None):
        """
        Zeroes the masses of specified atoms to constrain certain degrees of freedom.
        Arguments
        ---------
        system: simtk.openmm.system
            system to zero masses
        atomList: list of ints
            atom indicies to zero masses
        """
        for index in (atomList):
            system.setParticleMass(index, 0*unit.daltons)


    def calculate_com(self, pos_state, total_mass=None, mass_list=None, residueList=None, rotate=False):
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
        print('mass_list', mass_list)
        print('total_mass', total_mass)
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

    def rotationalMove(self, context=None, residueList=None):
        """
        Function to be used in movekey. Performs a rotation around the center of mass
        of the ligand. Generally no arguments are specified, since this takes the
        context and residueList attributes from the SimNCMC class.
        Arguments
        ---------
        context: openmm.context
            variable referring to the ncmc context. If none are
            specified uses the context in self.nc_context
        residueList: list of ints
            Indices of ligand atoms for alchemical transformation. If none are
            specified uses the residueList in self.residueList
        """

        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context
        before_rot_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        rot_output = self.calculate_com(total_mass=self.total_mass, mass_list=self.mass_list, pos_state=before_rot_pos, residueList=residueList, rotate=True)
        rot_output = rot_output[:].value_in_unit(unit.nanometers)
        rotPos = before_rot_pos.value_in_unit(unit.nanometers)
        for index, resnum in enumerate(residueList):
            rotPos[resnum] = rot_output[index]
        rotPos[:] = rotPos*unit.nanometers
        context.setPositions(rotPos)

    def waterMove(self, context=None, residueList=None):
        """
        Function to be used in movekey. Performs a rotation around the center of mass
        of the ligand. Generally no arguments are specified, since this takes the
        context and residueList attributes from the SimNCMC class.
        Arguments
        ---------
        context: openmm.context
            variable referring to the ncmc context. If none are
            specified uses the context in self.nc_context
        residueList: list of ints
            Indices of ligand atoms for alchemical transformation. If none are
            specified uses the residueList in self.residueList
        """

        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context
        before_move_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        prot_com = self.calculate_com(total_mass=self.prot_total_mass, mass_list=self.prot_mass_list, pos_state=before_move_pos, residueList=self.protein_residues, rotate=False)
        sphere_displacement = random_sphere_point(self.radius)
        movePos = before_move_pos[:]
        water_dist = movePos[residueList[0]] - prot_com
        if np.linalg.norm(water_dist._value) <= self.radius._value:
            for index, resnum in enumerate(residueList):
                movePos[resnum] = movePos[resnum] - water_dist + sphere_displacement
            #TODO check units, rotate water molecule
            #TODO make sure
            #movePos[:] = movePos*unit.nanometers
            context.setPositions(movePos)

    def runSim(self, md_simulation, nc_context, nc_integrator, dummy_simulation, movekey=None, nstepsNC=25, nstepsMD=1000, niter=10,
                periodic=True, verbose=False, print_output=sys.stdout, residueList=None, alchemical_correction=False,
                ncmc_storage='out_ncmc.h5', write_ncmc_interval=None):
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
        verbose: boolean, default=False
            print more detailed information on each step
        print_output: sys.stdout or str, default=sys.stdout:
            If str outputs print statemetents to file matching str.
            Otherwise outputs to sys.stdout
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
        if print_output == sys.stdout:
            print_file = sys.stdout
        else:
            print_file = open(print_output, 'wb')
        #find water molecules
        res = md_simulation.topology.residues()
        water_residues = []
        for x in res:
            if x.name == 'HOH':
                temp_wat = []
                for atom in x.atoms():
                    temp_wat.append(atom.index)
                water_residues.append(temp_wat)
        #get protein atom indices by iterating over first chain
        protein_residues = []
        chain = md_simulation.topology.chains()
        first_chain = chain.next()
        for atom in first_chain.atoms():
            protein_residues.append(atom.index)
        self.protein_residues = protein_residues
        prot_total_mass, prot_mass_list = self.get_particle_masses(system=self.md_simulation.system,
                                            residueList = protein_residues, set_self=False)
        self.prot_total_mass = prot_total_mass
        self.prot_mass_list = prot_mass_list
        if type(write_ncmc_interval) == int:
            h5reporter = md.reporters.HDF5Reporter(file=ncmc_storage, reportInterval=100,
                coordinates=True, time=False, cell=False, potentialEnergy=False, kineticEnergy=False,
                temperature=False, velocities=False, atomSubset=None)
        #set up initial counters/ inputs
        accCounter = 0
        #set inital conditions
        md_stateinfo = md_simulation.context.getState(True, True, False, True, True, periodic)
        oldPos = md_stateinfo.getPositions(asNumpy=True)
        oldVel = md_stateinfo.getVelocities(asNumpy=True)

        oldPE =  md_stateinfo.getPotentialEnergy()
        oldKE =  md_stateinfo.getKineticEnergy()
        nc_context.setPositions(oldPos)
        nc_context.setVelocities(oldVel)
        nc_stateinfo = nc_context.getState(True, False, False, False, False, periodic)

        for stepsdone in range(niter):
            print('performing ncmc step', file=print_file)
            print('accCounter =', accCounter, file=print_file)
            mdinfo = md_simulation.context.getState(True, True, False, True, True, periodic)
            oldPE =  mdinfo.getPotentialEnergy()
            oldKE =  mdinfo.getKineticEnergy()
            if alchemical_correction == True:
                alc_oldPE = nc_context.getState(True, True, False, True, True, periodic).getPotentialEnergy()

            for stepscarried in range(nstepsNC):
                #at start switch alchemical water with water in sphere distance
                if 1:
                    if stepscarried == 0:
                        start_state = md_simulation.context.getState(getPositions=True, getVelocities=True)
                        start_pos = start_state.getPositions(asNumpy=True)
                        print('start_pos', start_pos[residueList[0]])
                        start_vel = start_state.getVelocities(asNumpy=True)
                        switch_pos = np.copy(start_pos)
                        switch_vel = np.copy(start_vel)
                        prot_com = self.calculate_com(switch_pos, total_mass = prot_total_mass,
                                            mass_list = prot_mass_list, residueList= protein_residues)
                        #pick random water within the sphere radius
                        dist_boolean = 0
                        #TODO use random.shuffle to pick random particles (limits upper bound)
                        while dist_boolean == 0:
                            #water_choice = np.random.choice(water_residues)
                            water_index = np.random.choice(range(len(water_residues)))
                            water_choice = water_residues[water_index]
                            oxygen_pos = start_pos[water_choice[0]]
                            water_distance = np.linalg.norm(oxygen_pos._value - prot_com._value)
                            print('distance', water_distance)
                            if water_distance <= (self.radius.value_in_unit(unit.nanometers)):
                                dist_boolean = 1
                            print('water_choice', water_choice)
                        #replace chosen water's positions/velocities with alchemical water
                        for i in range(3):
                            switch_pos[residueList[i]] = start_pos[water_choice[i]]
                            switch_vel[residueList[i]] = start_vel[water_choice[i]]
                            switch_pos[water_choice[i]] = start_pos[residueList[i]]
                            switch_vel[water_choice[i]] = start_vel[residueList[i]]
                        print('after_switch', switch_pos[residueList[0]])
                        if write_ncmc_interval:
                            positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
                            dummy_simulation.context.setPositions(positions)
                            h5reporter.report(dummy_simulation, dummy_simulation.context.getState(True, True))

                        nc_context.setPositions(switch_pos)
                        nc_context.setVelocities(switch_vel)



                if movekey != None:
                    for func in movekey:
                        if stepscarried in func[1]:
                            print('doing the move', file=print_file)
                            print('before step', nc_integrator.getGlobalVariableByName('lambda'), file=print_file)

                            func[0]()
                            if write_ncmc_interval:
                                positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
                                dummy_simulation.context.setPositions(positions)
                                h5reporter.report(dummy_simulation, dummy_simulation.context.getState(True, True))

                try:
                    if verbose:
                        print('work_before', nc_integrator.getGlobalVariableByName("total_work"), file=print_file)
                        print('lambda_before', nc_integrator.getGlobalVariableByName("lambda"), file=print_file)
                        print('shadow_before', nc_integrator.getGlobalVariableByName("shadow_work"), file=print_file)
                        print('protocol_before', nc_integrator.getGlobalVariableByName("protocol_work"), file=print_file)
                        print('Epert_before', nc_integrator.getGlobalVariableByName("Epert"), file=print_file)

                    if write_ncmc_interval and (stepscarried+1) % write_ncmc_interval == 0:
                        if verbose:
                            print('writing coordinates')
                        positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
                        dummy_simulation.context.setPositions(positions)
                        h5reporter.report(dummy_simulation, dummy_simulation.context.getState(True, True))

                    nc_integrator.step(1)
                    if verbose:
                        print('work_after', nc_integrator.getGlobalVariableByName("total_work"), file=print_file)
                        print('lambda_after', nc_integrator.getGlobalVariableByName("lambda"), file=print_file)
                        print('shadow_after', nc_integrator.getGlobalVariableByName("shadow_work"), file=print_file)
                        print('protocol_after', nc_integrator.getGlobalVariableByName("protocol_work"), file=print_file)
                        print('Eold', nc_integrator.getGlobalVariableByName("Eold"), file=print_file)
                        print('Enew', nc_integrator.getGlobalVariableByName("Enew"), file=print_file)
                        print('Epert_after', nc_integrator.getGlobalVariableByName("Epert"), file=print_file)

                except Exception as e:
                    if str(e) == "Particle coordinate is nan":
                        print('nan, breaking', file=print_file)
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
                print('correction_factor', correction_factor, file=print_file)
                log_ncmc = log_ncmc + correction_factor

            if log_ncmc > randnum:
                print('ncmc move accepted', file=print_file)
                if verbose:
                    print('ncmc PE', newinfo.getPotentialEnergy(), 'old PE', oldPE, file=print_file)
                    print('ncmc Total energy', newinfo.getPotentialEnergy() + newinfo.getKineticEnergy(), file=print_file)
                    PE_diff = newinfo.getPotentialEnergy() - oldPE
                    print('PE_diff', PE_diff, file=print_file)
                print('accepted since', log_ncmc, '>', randnum, file=print_file)
                print('log_ncmc > randnum', file=print_file)
                print('move accepted', file=print_file)
                accCounter = accCounter + 1.0
                print('accCounter', float(accCounter)/float(stepsdone+1), accCounter, file=print_file)
                nc_stateinfo = nc_context.getState(True, True, False, False, False, periodic)
                oldPos = newPos[:]
                oldVel = newVel[:]

            else:
                print('ncmc PE', newinfo.getPotentialEnergy(), 'old PE', oldPE, file=print_file)
                print('rejected', log_ncmc, '<', randnum, file=print_file)
                print('log_ncmc > randnum', file=print_file)
                print('move rejected', file=print_file)
                nc_context.setPositions(oldPos)
                nc_context.setVelocities(-oldVel)

            print('accCounter:', accCounter,  'iter:', stepsdone+1, file=print_file)
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
                    last_x, last_y = np.shape(oldPos)
                    reshape = (np.reshape(oldPos, (1, last_x, last_y))).value_in_unit(unit.nanometers)
                    print('potential energy before NCMC', oldPE, file=print_file)
                    print('kinetic energy before NCMC', oldKE, file=print_file)

                    last_top = md.Topology.from_openmm(md_simulation.topology)
                    broken_frame = md.Trajectory(xyz=reshape, topology=last_top)
                    broken_frame.save_pdb('broken.pdb')
                    exit()

            md_stateinfo = md_simulation.context.getState(True, True, False, False, False, periodic)
            oldPos = md_stateinfo.getPositions(asNumpy=True)
            oldVel = md_stateinfo.getVelocities(asNumpy=True)
            nc_integrator.reset()
            nc_context.setPositions(oldPos)
            nc_context.setVelocities(oldVel)

        acceptRatio = accCounter/float(niter)
        print('acceptance ratio', acceptRatio, file=print_file)
        print('numsteps ', nstepsNC, file=print_file)
        if print_output != sys.stdout:
            print_file.close()
        return oldPos
