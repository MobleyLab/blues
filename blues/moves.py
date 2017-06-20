"""
moves.py: Provides the two main classes, Move and MoveEngine
which allow altering the positions of a
subset of atoms in a context during a BLUES simulation to
increase sampling.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

import parmed
from simtk import unit
import mdtraj
import numpy as np
import sys, traceback
import mdtraj as md


class Move(object):

    """Move provides methods for calculating properties on the
    object 'move' (i.e ligand) being perturbed in the NCMC simulation.
    This is the base Move class.
    Ex.
        from blues.ncmc import Model
        ligand = Model(structure, 'LIG')
        ligand.calculateProperties()
    Attributes
    ----------
    """

    def __init__(self):
        """Initialize the Move object
        Currently empy.
        """

class RandomLigandRotationMove(Move):
    """Move that provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.
    Current methods calculate the object's atomic masses and center of masss.
    Calculating the object's center of mass will get the positions
    and total mass.
    Ex.
        from blues.move import RandomLigandRotationMove
        ligand = RandomLigandRotationMove(structure, 'LIG')
        ligand.resname : string specifying the residue name of the ligand
        ligand.calculateProperties()
    Attributes
    ----------
    ligand.topology : openmm.topology of ligand
    ligand.atom_indices : list of atom indicies of the ligand.
    ligand.masses : list of particle masses of the ligand with units.
    ligand.totalmass : integer of the total mass of the ligand.
    #Dynamic attributes that must be updated with each iteration
    ligand.center_of_mass : np.array of calculated center of mass of the ligand
    ligand.positions : np.array of ligands positions
    """

    def __init__(self, structure, resname='LIG'):
        """Initialize the model.
        Parameters
        ----------
        resname : str
            String specifying the resiue name of the ligand.
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        """

        self.resname = resname
        self.atom_indices = self.getAtomIndices(structure, self.resname)
        self.topology = structure[self.atom_indices].topology
        self.totalmass = 0
        self.masses = []

        self.center_of_mass = None
        self.positions = structure[self.atom_indices].positions

    def getAtomIndices(self, structure, resname):
        """
        Get atom indices of a ligand from ParmEd Structure.
        Arguments
        ---------
        resname : str
            String specifying the resiue name of the ligand.
        structure: parmed.Structure
            ParmEd Structure object of the atoms to be moved.
        Returns
        -------
        atom_indices : list of ints
            list of atoms in the coordinate file matching lig_resname
        """
#       TODO: Add option for resnum to better select residue names
        atom_indices = []
        topology = structure.topology
        for atom in topology.atoms():
            if str(resname) in atom.residue.name:
                atom_indices.append(atom.index)
        return atom_indices

    def getMasses(self, topology):
        """Returns a list of masses of the specified ligand atoms.
        Parameters
        ----------
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        """
        masses = unit.Quantity(np.zeros([int(topology.getNumAtoms()),1],np.float32), unit.dalton)
        for idx,atom in enumerate(topology.atoms()):
            masses[idx] = atom.element._mass
        totalmass = masses.sum()
        return masses, totalmass

    def getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a np.array
        Parameters
        ----------
        structure: parmed.Structure
            ParmEd positions of the atoms to be moved.
        masses : numpy.array
            np.array of particle masses
        """
        coordinates = np.asarray(positions._value, np.float32)
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
        return center_of_mass

    def calculateProperties(self):
        """Function to quickly calculate available properties."""
        self.masses, self.totalmass = self.getMasses(self.topology)
        self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)

    def move(self, context):
        """Function that performs a random rotation about the
        center of mass of the ligand.
        """
       #TODO: check if we need to deepcopy
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        self.positions = positions[self.atom_indices]
        self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)
        reduced_pos = self.positions - self.center_of_mass

        # Define random rotational move on the ligand
        rand_quat = mdtraj.utils.uniform_quaternion()
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
        #multiply lig coordinates by rot matrix and add back COM translation from origin
        rot_move = np.dot(reduced_pos, rand_rotation_matrix) * positions.unit + self.center_of_mass

        # Update ligand positions in nc_sim
        for index, atomidx in enumerate(self.atom_indices):
            positions[atomidx] = rot_move[index]
        context.setPositions(positions)
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices]
        return context


class CombinationMove(Move):
    """Move object that allows Move object moves to be performed sequentially.
    These moves are performed according to the order in move_list. To ensure
    detailed balance, the moves have an equal chance to be performed in listed
    or reverse order.
    Parameters
    ----------
    move_list : list of blues.move.Move-like objects
    """
    def __init__(self, move_list):
        self.move_list = move_list

    def move(self, context):
        rand = np.random.random()
        #to maintain detailed balance this executes both
        #the forward and reverse order moves with equal probability
        if rand > 0.5:
            for single_move in self.move_list:
                 single_move.move(context)
        else:
            for single_move in reverse(self.move_list):
                 single_move.move(context)


class MoveEngine(object):
    """MoveEngine provides perturbation functions for the context during the NCMC
    simulation.
    Ex.
        from blues.ncmc import MoveEngine
        probabilities = [0.25, 0.75]
        #Where move is a list of two Move objects
        mover = MoveEngine(move, probabilities)
        #Get the dictionary of proposed moves
        mover.moves
    """
    def __init__(self, moves, probabilities=None):
        """Initialize the MovePropsal object that contains functions to perturb
        the context in the NCMC simulation.
        Parameters
        ----------
        moves : blues.ncmc.Model object or list of n blues.ncmc.Model-like objects
            Specifies the possible moves to be performed.

        probabilities: list of floats, default=None
            A list of n probabilities,
            where probabilities[i] corresponds to the probaility of moves[i]
            being selected to perform its associated move() method.

            If None, uniform probabilities are assigned.
        """

    #make a list from moves if not a list
        if isinstance(moves,list):
            self.moves = moves
        else:
            self.moves = [moves]
        #normalize probabilities
        if probabilities is None:
            single_prob = 1. / len(self.moves)
            self.probs = [single_prob for x in (self.moves)]
        else:
            prob_sum = float(sum(probabilities))
            self.probs = [x/prob_sum for x in probabilities]
        #if move and probabilitiy lists are different lengths throw error
        if len(self.moves) != len(self.probs):
            print('moves and probability list lengths need to match')
            raise IndexError


    def runEngine(self, context):
        """Selects a random Move object based on its
        assigned probability and and performs its move() function
        on a context.
        Parameters
        ----------
        context : openmm.context object
        OpenMM context whose positions should be moved.
        """
        rand_num = np.random.choice(len(self.probs), p=self.probs)
        try:
            new_context = self.moves[rand_num].move(context)
        except Exception as e:
            #In case the move isn't properly implemented, print out useful info
            print('Error: move not implemented correctly, printing traceback:')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(e)
            raise SystemExit

        return new_context


class SmartDartMove(RandomLigandRotationMove):
    def __init__(self, structure, basis_particles, dart_size=0.2*unit.nanometers, resname='LIG'):
        super(SmartDartMove, self).__init__(structure, resname=resname)
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
        basis_particles: list of 3 ints
            Specifies the 3 indices of particles whose coordinates will be used
            as basis vectors. If None is specified, uses those found in basis particles.
            If None uses self.basis_particles instead.
        Returns
        -------
        n_dartboard: list of 1x3 np.arrays
            Center of mass coordinates of atom_indices particles in new basis set.
        """
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
            new_coord = self._findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], self.center_of_mass)
            #keep this in for now to check code is correct
            #old_coord should be equal to com
            old_coord = self._findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
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
            new_coord = self._findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], com)
            #keep this in for now to check code is correct
            #old_coord should be equal to com
            old_coord = self._findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)

        self.n_dartboard = n_dartboard
        self.dartboard = dartboard

    def move(self, nc_context):
        """
        Function for performing smart darting move with darts that
        depend on particle positions in the system
        """

        atom_indices = self.atom_indices
        context = nc_context

        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        lig_pos = np.asarray(oldDartPos._value)[self.atom_indices]*unit.nanometers
        self._findDart(context)
        center = self.getCenterOfMass(lig_pos, self.masses)
        selectedboard, changevec = self._calc_from_center(com=center)
        if selectedboard != None:
            #TODO just use oldDartPos instead of using temp newDartPos
            newDartPos = np.copy(oldDartPos)
            comMove = self._reDart(changevec)
            vecMove = comMove - center
            for residue in atom_indices:
                newDartPos[residue] = newDartPos[residue] + vecMove._value
            context.setPositions(newDartPos)

            return context

    def _calc_from_center(self, com):

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

    def _findDart(self, nc_context):
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
            old_center = self._findOldCoord(part1, part2, part3, dart)
            dart_list.append(old_center)
        self.dartboard = dart_list[:]
        return dart_list

    def _reDart(self, changevec):
        """
        Helper function to choose a random dart and determine the vector
        that would translate the COM to that dart center
        """
        dartindex = np.random.randint(len(self.dartboard))
        dvector = self.dartboard[dartindex]
        chboard = dvector + changevec
        return chboard

    def _changeBasis(self, a, b):
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

    def _undoBasis(self, a, b):
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


    def _normalize(self, vector):
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

    def _localCoord(self, particle1, particle2, particle3):
        '''Defines a new coordinate system using 3 particles
        returning the new basis set vectors
        '''
        part2 = particle2 - particle1
        part3 = particle3 - particle1
        vec1 = part2
        vec2= part3
        vec3 = np.cross(vec1,vec2)*unit.nanometers
        return vec1, vec2, vec3

    def _findNewCoord(self, particle1, particle2, particle3, center):
        '''Finds the coordinates of a given center in a new coordinate
            system defined by particles1-3
        '''
        #calculate new basis set
        vec1, vec2, vec3 = self._localCoord(particle1, particle2, particle3)
        basis_set = np.zeros((3,3))*unit.nanometers
        basis_set[0] = vec1
        basis_set[1] = vec2
        basis_set[2] = vec3
        #since the origin is centered at particle1 by convention
        #subtract to account for this
        recenter = center - particle1
        #find coordinate in new coordinate system
        new_coord = self._changeBasis(basis_set, recenter)
        return new_coord

    def _findOldCoord(self, particle1, particle2, particle3, center):
        '''Finds the coordinates of a given center (defined by a different basis
        given by particles1-3) back in euclidian coordinates
            system defined by particles1-3
        '''

        vec1, vec2, vec3 = self._localCoord(particle1, particle2, particle3)
        basis_set = np.zeros((3,3))*unit.nanometers
        basis_set[0] = vec1
        basis_set[1] = vec2
        basis_set[2] = vec3
        #since the origin is centered at particle1 by convention
        #subtract to account for this
        old_coord = self._undoBasis(basis_set, center)
        adjusted_center = old_coord + particle1
        return adjusted_center
