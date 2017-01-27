from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as unit
import numpy as np
from blues.ncmc import SimNCMC
import mdtraj as md

#TODO consider throwing a warning if particle choices
#are not dispersed enough (by some angle cutoff)
def changeBasis(a, b):
    '''
    a is 3x3 np.array defining vectors that create a basis
    b is 1x3 np.array define position of particle to be transformed into
        new coordinate system
    '''
    ainv = np.linalg.inv(a.T)
    print('ainv', ainv)
    print('b.T', b.T)
    changed_coord = np.dot(ainv,b.T)*unit.nanometers
    return changed_coord
def undoBasis(a, b):
    '''
    a is 3x3 np.array defining vectors that create a basis
    b is 1x3 np.array define position of particle to be transformed into
        new coordinate system
    '''
    a = a.T
    print('a', a.T)
    print('b.T', b.T)
    changed_coord = np.dot(a,b.T)*unit.nanometers
    return changed_coord

    
def normalize(a):
    magnitude = np.sqrt(np.sum(a*a))
    unit_vec = a / magnitude
    return unit_vec
def localcoord(particle1, particle2, particle3):
    '''Defines a new coordinate system using 3 particles
    returning the new basis set vectors
    '''
    part1 = particle1 - particle1
    part2 = particle2 - particle1
    part3 = particle3 - particle1
#    vec1 = normalize(part2)
#    vec2 = normalize(part3)
    vec1 = part2
    vec2= part3
    vec3 = np.cross(vec1,vec2)*unit.nanometers
    print('vec3', vec3, normalize(vec3))
    print('vec1', vec1, 'vec2', vec2, 'vec3', vec3)
    return vec1, vec2, vec3
def findNewCoord(particle1, particle2, particle3, center):
    '''Finds the coordinates of a given center in a new coordinate
        system defined by particles1-3
    '''
    #calculate new basis set
    vec1, vec2, vec3 = localcoord(particle1, particle2, particle3)
    basis_set = np.zeros((3,3))*unit.nanometers
    basis_set[0] = vec1
    basis_set[1] = vec2
    print('vec3', vec3)
    basis_set[2] = vec3
    print('basis_set', basis_set)
    #since the origin is centered at particle1 by convention
    #subtract to account for this
    recenter = center - particle1
    #find coordinate in new coordinate system
    new_coord = changeBasis(basis_set, recenter)
    print('new_coord', new_coord)
    old_coord = undoBasis(basis_set, new_coord)
    print('old_coord', old_coord)
    print('old_recenter', recenter)
    return new_coord
def findOldCoord(particle1, particle2, particle3, center):
    '''Finds the coordinates of a given center (defined by a different basis
    given by particles1-3) back in euclidian coordinates
        system defined by particles1-3
    '''

    print ('particles', particle1, particle2, particle3)
    vec1, vec2, vec3 = localcoord(particle1, particle2, particle3)
    basis_set = np.zeros((3,3))*unit.nanometers
    basis_set[0] = vec1
    basis_set[1] = vec2
    print('vec3', vec3)
    basis_set[2] = vec3
    print('basis_set', basis_set)
    #since the origin is centered at particle1 by convention
    #subtract to account for this
    old_coord = undoBasis(basis_set, center)
    print('old coord before adjustment', old_coord)
    print ('particles', particle1, particle2, particle3)
    adjusted_center = old_coord + particle1
    print('adjusted coord', adjusted_center)
    return adjusted_center


def zero_masses( system, firstres, lastres):
    for index in range(firstres, lastres):
        system.setParticleMass(index, 0*daltons)

def beta(temperature):
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    beta = 1.0 / kT
    return beta



def forcegroupify(system):
    forcegroups = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
        forcegroups[force] = i
    return forcegroups

def getEnergyDecomposition(context, forcegroups):
    energies = {}
    for f, i in forcegroups.items():
        energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
    return energies

class SmartDarting(SimNCMC):
    """
    Class for performing smart darting moves during an NCMC simulation.
    """
    def __init__(self, *args, **kwds):
        super(SmartDarting, self).__init__(*args, **kwds)
        self.dartboard = []
        self.n_dartboard = []
        self.particle_pairs = []
        self.particle_weights = []
        self.basis_particles = []
        self.dart_size = 0.2*unit.nanometers
        self.virtual_particles = []

    def setDartUpdates(self, particle_pairs, particle_weights):
        self.particle_pairs = particle_pairs
        self.particle_weights = particle_weights

    def get_particle_masses(self, system, set_self=True, residueList=None):
        if residueList == None:
            residueList = self.residueList
        mass_list = []
        total_mass = 0*unit.dalton
        for index in residueList:
            mass = system.getParticleMass(index)
            print('getting')
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
        if set_self == True:
            self.total_mass = total_mass
            self.mass_list = mass_list
        return total_mass, mass_list

    def dartsFromPDB(self, pdb_list, forcefield, residueList=None, basis_particles=None):
        """
        Used to setup darts from a pdb, using the particle_list to define 
        a new coordinate system. ResidueList corresponds to indicies of
        the ligand residues.
        """
        if residueList == None:
            residueList = self.residueList
        if basis_particles == None:
            basis_particles = self.basis_particles
        n_dartboard = []
        dartboard = []
        for pdb_file in pdb_list:
            pdb = PDBFile(pdb_file)
            pdb.positions = pdb.positions.value_in_unit(unit.nanometers)
            #print('pdbpositions,', pdb.positions)
            #print('pdbtype', type(pdb.positions))
            #print('pdbtype', type(pdb.positions._value))

            force = ForceField(forcefield)
            system = force.createSystem(pdb.topology)
            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            sim = Simulation(pdb.topology, system, integrator)
            sim.context.setPositions(pdb.positions)
            context_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
            total_mass, mass_list = self.get_particle_masses(system, set_self=False, residueList=residueList)
            com = self.calculate_com(pos_state=context_pos,
                                    total_mass=total_mass,
                                    mass_list=mass_list,
                                    residueList=residueList)
            #get particle positions
            particle_pos = []
            for particle in basis_particles:
                print('particle %i position' % (particle), context_pos[particle])
                particle_pos.append(context_pos[particle])
            new_coord = findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], com)
            #keep this in for now to check code is correct
            #old_coord should be equal to com
            old_coord = findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)
        print('n_dartboard from pdb', n_dartboard)
        print('dartboard from pdb', dartboard)

        self.n_dartboard = n_dartboard
        self.dartboard = dartboard

    def dartsFromMDTraj(self, system, file_list, topology=None, residueList=None, basis_particles=None):
        """
        Used to setup darts from a pdb, using the particle_list to define 
        a new coordinate system. ResidueList corresponds to indicies of
        the ligand residues.
        """
        if residueList == None:
            residueList = self.residueList
        if basis_particles == None:
            basis_particles = self.basis_particles
        n_dartboard = []
        dartboard = []
        for md_file in file_list:
            if topology == None:
                temp_md = md.load(md_file)
            else:
                temp_md = md.load(md_file, top=topology)
            context_pos = temp_md.openmm_positions(0)
            print('context_pos', context_pos, 'context_pos')
            print('context_pos type', type(context_pos._value))
            print('temp_md', temp_md)
            context_pos = np.asarray(context_pos._value)*unit.nanometers
            total_mass, mass_list = self.get_particle_masses(system, set_self=False, residueList=residueList)
            com = self.calculate_com(pos_state=context_pos,
                                    total_mass=total_mass,
                                    mass_list=mass_list,
                                    residueList=residueList)
            #get particle positions
            particle_pos = []
            for particle in basis_particles:
                print('particle %i position' % (particle), context_pos[particle])
                particle_pos.append(context_pos[particle])
            new_coord = findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], com)
            #keep this in for now to check code is correct
            #old_coord should be equal to com
            old_coord = findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)
        print('n_dartboard from pdb', n_dartboard)
        print('dartboard from pdb', dartboard)

        self.n_dartboard = n_dartboard
        self.dartboard = dartboard

    def dartsFromAmber(self, inpcrd_list, prmtop, residueList=None, basis_particles=None):
        """
        Used to setup darts from a pdb, using the particle_list to define 
        a new coordinate system. ResidueList corresponds to indicies of
        the ligand residues.
        """
        if residueList == None:
            residueList = self.residueList
        if basis_particles == None:
            basis_particles = self.basis_particles
        n_dartboard = []
        dartboard = []
        prmtop=AmberPrmtopFile(prmtop)
        for inpcrd_file in inpcrd_list:
            amber = AmberInpcrdFile(inpcrd_file)
            system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
                                        constraints=HBonds)
            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            sim = Simulation(prmtop.topology, system, integrator)
            sim.context.setPositions(pdb.positions)
            context_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
            total_mass, mass_list = self.get_particle_masses(system, set_self=False, residueList=residueList)
            com = self.calculate_com(pos_state=context_pos,
                                    total_mass=total_mass,
                                    mass_list=mass_list,
                                    residueList=residueList)
            #get particle positions
            particle_pos = []
            for particle in basis_particles:
                print('particle %i position' % (particle), context_pos[particle])
                particle_pos.append(context_pos[particle])
            new_coord = findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], com)
            #keep this in for now to check code is correct
            #old_coord should be equal to com
            old_coord = findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)
        print('n_dartboard from pdb', n_dartboard)
        print('dartboard from pdb', dartboard)

        self.n_dartboard = n_dartboard
        self.dartboard = dartboard 


    def add_dart(self, dart):
        self.dartboard.append(dart)
    def addPart(self, particles):
        self.basis_particles = particles[:]

    def findDart(self, particle_pairs=None, particle_weights=None):
        """ 
        For dynamically updating dart positions based on positions
        of other particles.
        This takes the weighted average of the specified particles
        and changes the dartboard of the object
        
        Arguments
        ---------
        particle_pairs: list of list of ints
            each list defines the pairs to define darts
        particle_weights: list of list of floats
            each list defines the weights assigned to each particle positions
        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights

        """
        if particle_pairs == None:
            particle_pairs = self.particle_pairs
        if particle_weights == None:
            particle_weights = self.particle_weights
        #make sure there's an equal number of particle pair lists 
        #and particle weight lists
        assert len(particle_pairs) == len(particle_weights)

        dart_list = []
        state_info = self.nc_context.getState(True, True, False, True, True, False)
        temp_pos = state_info.getPositions(asNumpy=True)
        #find particles positions and multiply by weights
        for i, ppair in enumerate(particle_pairs):
            temp_array = np.array([0, 0, 0]) * unit.nanometers
            #weighted average
            temp_wavg = 0
            for j, particle in enumerate(ppair):
                print('temp_pos', particle, temp_pos[particle])
                temp_array += (temp_pos[particle] * float(particle_weights[i][j]))
                temp_wavg += float(particle_weights[i][j])
                print(temp_array)
            #divide by total number of particles in a list and append 
            #calculated postion to dart_list
            dart_list.append(temp_array[:] / temp_wavg)
        self.dartboard = dart_list[:]
        return dart_list

    def n_findDart(self, basis_particles=None):
        """ 
        For dynamically updating dart positions based on positions
        of other particles.
        This takes the weighted average of the specified particles
        and changes the dartboard of the object
        
        Arguments
        ---------
        particle_pairs: list of list of ints
            each list defines the pairs to define darts
        particle_weights: list of list of floats
            each list defines the weights assigned to each particle positions
        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights

        """
        if basis_particles == None:
            basis_particles = self.basis_particles
        #make sure there's an equal number of particle pair lists 
        #and particle weight lists
        dart_list = []
        state_info = self.nc_context.getState(True, True, False, True, True, False)
        temp_pos = state_info.getPositions(asNumpy=True)
        part1 = temp_pos[basis_particles[0]]
        part2 = temp_pos[basis_particles[1]]
        part3 = temp_pos[basis_particles[2]]
        print('n_findDart before dartboard', self.dartboard)
        for dart in self.n_dartboard:
            print('particles', part1, part2, part3)
            old_center = findOldCoord(part1, part2, part3, dart)
            dart_list.append(old_center)
        self.dartboard = dart_list[:]
        print('n_findDart dartboard', self.dartboard)
        return dart_list

    def virtualDart(self, virtual_particles=None):
        """ 
        For dynamically updating dart positions based on positions
        of other particles.
        This takes the weighted average of the specified particles
        and changes the dartboard of the object
        
        Arguments
        ---------
        virtual_particles: list of ints
            Each int in the list specifies a particle
        particle_weights: list of list of floats
            each list defines the weights assigned to each particle positions
        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights

        """
        if virtual_particles == None:
            virtual_particles = self.virtual_particles

        dart_list = []
        state_info = self.nc_context.getState(True, True, False, True, True, False)
        temp_pos = state_info.getPositions(asNumpy=True)
        #find virtual particles positions and add to dartboard
        for particle in virtual_particles:
            print('temp_pos', particle, temp_pos[particle])
            dart_list.append(temp_pos[particle])
        self.dartboard = dart_list[:]
        return dart_list

    def calc_from_center(self, com):


        distList = []
        diffList = []
        indexList = []
        #Find the distances of the COM to each dart, appending 
        #the results to distList
        for dart in self.dartboard:
            diff = com - dart
            dist = np.sqrt(np.sum((diff)*(diff)))*unit.nanometers
            distList.append(dist)
            diffList.append(diff)
        selected = []
        #Find the dart(s) less than self.dart_size
        for index, entry in enumerate(distList):
            if entry <= self.dart_size:
                selected.append(entry)
                diff = diffList[index]
                indexList.append(index)
        #Dart error checking
        #to ensure reversibility the COM should only be 
        #within self.dart_size of one dart
        if len(selected) == 1:
            return selected[0], diffList[indexList[0]]
        elif len(selected) == 0:
            return None, diff
        elif len(selected) >= 2:
            print(selected)
            #COM should never be within two different darts
            raise ValueError('sphere size overlap, check darts')


    def reDart(self, changevec):
        """
        Helper function to choose a random dart and determine the vector
        that would translate the COM to that dart center
        """ 
        dartindex = np.random.randint(len(self.dartboard))
        dvector = self.dartboard[dartindex]
        chboard = dvector + changevec   
        print('chboard', chboard)
        return chboard

    def dartmove(self, context=None, residueList=None):
        """
        Obsolete function kept for reference. 
        """
        if residueList == None:
            residueList = self.residueList
        if context == None:
            self.nc_context

        stateinfo = self.context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('changevec', changevec)
        if selectedboard != None:
        #notes
        #comMove is where the com ends up after accounting from where 
        #it was from the original dart center
        #basically it's final displacement location
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            vecMove = comMove - center
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            context.setPositions(newDartPos)
            newDartInfo = context.getState(True, True, False, True, True, False)
            newDartPE = newDartInfo.getPotentialEnergy()
            logaccept = -1.0*(newDartPE - oldDartPE) * self.beta
            randnum = math.log(np.random.random())
            print('logaccept', logaccept, randnum)
            print('old/newPE', oldDartPE, newDartPE)
            if logaccept >= randnum:
                print('move accepted!')
                self.acceptance = self.acceptance+1
            else:
                print('rejected')
                context.setPositions(oldDartPos)
            dartInfo = context.getState(True, False, False, False, False, False)

            return newDartInfo.getPositions(asNumpy=True)

    def justdartmove(self, context=None, residueList=None):
        """
        Function for performing smart darting move with fixed coordinate darts
        """
        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context


        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('selectedboard', selectedboard)
        print('changevec', changevec)
        print('centermass', center)
        if selectedboard != None:
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            print('worked')
            print(newDartPos)
            context.setPositions(newDartPos)
            newDartInfo = context.getState(True, True, False, True, True, False)
#            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)

    def updateDartMove(self, context=None, residueList=None):
        """
        Function for performing smart darting move with darts that 
        depend on particle positions in the system
        """

        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context


        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        self.n_findDart()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('selectedboard', selectedboard)
        print('changevec', changevec)
        print('centermass', center)
        if selectedboard != None:
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            print('worked')
            print('old', oldDartPos)
            print('new', newDartPos)
            context.setPositions(newDartPos)
            newDartInfo = context.getState(True, True, False, True, True, False)
#            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)

    def virtualDartMove(self, context=None, residueList=None):
        """
        Function for performing smart darting move with darts that 
        depend on particle positions in the system
        """

        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context


        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        self.virtualDart()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('selectedboard', selectedboard)
        print('changevec', changevec)
        print('centermass', center)
        if selectedboard != None:
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            print('worked')
            print(newDartPos)
            context.setPositions(newDartPos)
            newDartInfo = self.nc_context.getState(True, True, False, True, True, False)
#            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)



