from blues.ncmc import Model
import simtk.unit as unit
import numpy as np
import parmed
#import mdtraj as md

def changeBasis(a, b):
    '''
    Changes positions of a particle (b) in the regular basis set to
    another basis set (a).
    Arguments
    ---------
    a: 3x3 np.array
        Defines vectors that will create the new basis.
    b: 1x3 np.array
        Defines position of particle to be transformed into
        new basis set.
    Returns
    -------
    changed_coord: 1x3 np.array
        Coordinates of b in new basis.
    '''
    ainv = np.linalg.inv(a.T)
    changed_coord = np.dot(ainv,b.T)*unit.nanometers
    return changed_coord
def undoBasis(a, b):
    '''
    Transforms positions in a transformed basis (b) to the regular
    basis set.
    Arguments
    ---------
    a: 3x3 np.array
        Defines vectors that defined the new basis.
    b: 1x3 np.array
        Defines position of particle to be transformed into
        regular basis set.
    Returns
    -------
    changed_coord: 1x3 np.array
        Coordinates of b in new basis.
    '''
    a = a.T
    changed_coord = np.dot(a,b.T)*unit.nanometers
    return changed_coord


def normalize(vector):
    '''Normalize a given vector
    Arguemnts
    ---------
    vector: 1xn np.array
        Vector to be normalized.
    Returns
    -------
    unit_vec: 1xn np.array
        Normalized vector.
    '''
    magnitude = np.sqrt(np.sum(vector*vector))
    unit_vec = vector / magnitude
    return unit_vec

def localcoord(particle1, particle2, particle3):
    '''Defines a new coordinate system using 3 particles
    returning the new basis set vectors
    '''
    part1 = particle1 - particle1
    part2 = particle2 - particle1
    part3 = particle3 - particle1
    vec1 = part2
    vec2= part3
    vec3 = np.cross(vec1,vec2)*unit.nanometers
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
    basis_set[2] = vec3
    #since the origin is centered at particle1 by convention
    #subtract to account for this
    recenter = center - particle1
    #find coordinate in new coordinate system
    new_coord = changeBasis(basis_set, recenter)
    old_coord = undoBasis(basis_set, new_coord)
    return new_coord

def findOldCoord(particle1, particle2, particle3, center):
    '''Finds the coordinates of a given center (defined by a different basis
    given by particles1-3) back in euclidian coordinates
        system defined by particles1-3
    '''

    vec1, vec2, vec3 = localcoord(particle1, particle2, particle3)
    basis_set = np.zeros((3,3))*unit.nanometers
    basis_set[0] = vec1
    basis_set[1] = vec2
    basis_set[2] = vec3
    #since the origin is centered at particle1 by convention
    #subtract to account for this
    old_coord = undoBasis(basis_set, center)
    adjusted_center = old_coord + particle1
    return adjusted_center

class Model_SmartDart(Model):
    def __init__(self, structure, basis_particles, dart_size=0.2*unit.nanometers, resname='LIG'):
        super(Model_SmartDart, self).__init__(structure, resname=resname)
        self.dartboard = []
        self.n_dartboard = []
        self.particle_pairs = []
        self.particle_weights = []
        self.basis_particles = basis_particles
        self.dart_size = dart_size
        self.calculateProperties()

    def dartsFromParmEd(self, coord_files, topology=None):
        """
        Used to setup darts from a generic coordinate file, through MDtraj using the basis_particles to define
        new basis vectors, which allows dart centers to remain consistant through a simulation.
        This adds to the self.n_dartboard, which defines the centers used for smart darting.
        Arguments
        ---------
        system: simtk.openmm.system
            Openmm System corresponding to the system to smart dart.
        coord_files: list of str
            List containing coordinate files of the system for smart darting.
        atom_indices: list of ints
            List containing the ligand atom indices. If None uses self.atom_indices instead.
        basis_particles: list of 3 ints
            Specifies the 3 indices of particles whose coordinates will be used
            as basis vectors. If None is specified, uses those found in basis particles.
            If None uses self.basis_particles instead.
        Returns
        -------
        n_dartboard: list of 1x3 np.arrays
            Center of mass coordinates of atom_indices particles in new basis set.
        """
        atom_indices = self.atom_indices
        basis_particles = self.basis_particles
        n_dartboard = []
        dartboard = []
        for coord_file in coord_files:
            if topology == None:
                temp_md = parmed.load_file(coord_file)
            else:
                temp_md = parmed.load_file(topology, xyz=coord_file)
            context_pos = temp_md.positions.in_units_of(unit.nanometers)
            lig_pos = np.asarray(context_pos._value)[self.atom_indices]*unit.nanometers
            particle_pos = np.asarray(context_pos._value)[self.basis_particles]*unit.nanometers
            self.calculateProperties()
            self.center_of_mass = self.getCenterOfMass(lig_pos, self.masses)
            #get particle positions
            new_coord = findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], self.center_of_mass)
            #keep this in for now to check code is correct
            #old_coord should be equal to com
            old_coord = findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)

        self.n_dartboard = n_dartboard
        self.dartboard = dartboard

    def dartsFromMDTraj(self, system, file_list, topology=None):
        """
        Used to setup darts from a generic coordinate file, through MDtraj using the basis_particles to define
        new basis vectors, which allows dart centers to remain consistant through a simulation.
        This adds to the self.n_dartboard, which defines the centers used for smart darting.
        Arguments
        ---------
        system: simtk.openmm.system
            Openmm System corresponding to the system to smart dart.
        file_list: list of str
            List containing coordinate files of the system for smart darting.
        atom_indices: list of ints
            List containing the ligand atom indices. If None uses self.atom_indices instead.
        basis_particles: list of 3 ints
            Specifies the 3 indices of particles whose coordinates will be used
            as basis vectors. If None is specified, uses those found in basis particles.
            If None uses self.basis_particles instead.
        Returns
        -------
        n_dartboard: list of 1x3 np.arrays
            Center of mass coordinates of atom_indices particles in new basis set.
        """
        atom_indices = self.atom_indices
        basis_particles = self.basis_particles
        n_dartboard = []
        dartboard = []
        for md_file in file_list:
            if topology == None:
                temp_md = md.load(md_file)
            else:
                temp_md = md.load(md_file, top=topology)
            context_pos = temp_md.openmm_positions(0)
            context_pos = np.asarray(context_pos._value)*unit.nanometers
            total_mass, mass_list = self.get_particle_masses(system, set_self=False, atom_indices=atom_indices)
            com = self.calculate_com(pos_state=context_pos,
                                    total_mass=total_mass,
                                    mass_list=mass_list,
                                    atom_indices=atom_indices)
            #get particle positions
            particle_pos = []
            for particle in basis_particles:
                particle_pos.append(context_pos[particle])
            new_coord = findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], com)
            #keep this in for now to check code is correct
            #old_coord should be equal to com
            old_coord = findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)

        self.n_dartboard = n_dartboard
        self.dartboard = dartboard

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
            #COM should never be within two different darts
            raise ValueError('sphere size overlap, check darts')
            #TODO can treat cases using appropriate probablility correction

    def n_findDart(self, nc_context):
        """
        Helper function to dynamically update dart positions based on positions
        of other particles.
        Arguments
        ---------
        basis_particles: list of 3 ints
            Specifies the 3 indices of particles whose coordinates will be used
            as basis vectors. If None is specified, uses those found in basis particles
        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights
        """
        basis_particles = self.basis_particles
        #make sure there's an equal number of particle pair lists
        #and particle weight lists
        dart_list = []
        state_info = nc_context.getState(True, True, False, True, True, False)
        temp_pos = state_info.getPositions(asNumpy=True)
        part1 = temp_pos[basis_particles[0]]
        part2 = temp_pos[basis_particles[1]]
        part3 = temp_pos[basis_particles[2]]
        for dart in self.n_dartboard:
            old_center = findOldCoord(part1, part2, part3, dart)
            dart_list.append(old_center)
        self.dartboard = dart_list[:]
        return dart_list

    def reDart(self, changevec):
        """
        Helper function to choose a random dart and determine the vector
        that would translate the COM to that dart center
        """
        dartindex = np.random.randint(len(self.dartboard))
        dvector = self.dartboard[dartindex]
        chboard = dvector + changevec
        return chboard

    def smartDartMove(self, nc_context):
        """
        Function for performing smart darting move with darts that
        depend on particle positions in the system
        """

        atom_indices = self.atom_indices
        context = nc_context

        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        lig_pos = np.asarray(oldDartPos._value)[self.atom_indices]*unit.nanometers
        oldDartPE = stateinfo.getPotentialEnergy()
        self.n_findDart(context)
        center = self.getCenterOfMass(lig_pos, self.masses)
        selectedboard, changevec = self.calc_from_center(com=center)
        if selectedboard != None:
            #TODO just use oldDartPos instead of using temp newDartPos
            newDartPos = np.copy(oldDartPos)
            comMove = self.reDart(changevec)
            vecMove = comMove - center
            for residue in atom_indices:
                newDartPos[residue] = newDartPos[residue] + vecMove._value
            context.setPositions(newDartPos)
            newDartInfo = context.getState(True, True, False, True, True, False)

            return newDartInfo.getPositions(asNumpy=True)

