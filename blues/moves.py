"""
moves.py: Provides the main Move class which allows definition of moves
which alter the positions of subsets of atoms in a context during a BLUES
simulation, in order to increase sampling.
Also provides functionality for CombinationMove definitions which consist of
a combination of other pre-defined moves such as via instances of Move.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

import parmed
from simtk import unit
import mdtraj
import numpy as np


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
            ParmEd Structure object of the relevant system to be moved.
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
        """
        Returns a list of masses of the specified ligand atoms.

        Parameters
        ----------
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.

        Returns
        -------
        masses: 1xn numpy.array * simtk.unit.dalton
            array of masses of len(self.atom_indices), denoting
            the masses of the atoms in self.atom_indices
        totalmass: float* simtk.unit.dalton
            The sum of the mass found in masses

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
        positions: nx3 numpy array * simtk.unit compatible with simtk.unit.nanometers
            ParmEd positions of the atoms to be moved.
        masses : numpy.array
            np.array of particle masses

        Returns
        -------
        center_of_mass: numpy array * simtk.unit compatible with simtk.unit.nanometers
            1x3 np.array of the center of mass of the given positions

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

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose positions were changed by this function.

        """
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
    """Move object that allows Move object moves to be performed according to
    the order in move_list.
    To ensure detailed balance, the moves have an equal chance to be performed
    in listed or reverse order.

    Parameters
    ----------
    move_list : list of blues.move.Move-like objects
    """
    def __init__(self, move_list):
        self.move_list = move_list

    def move(self, context):
        """Performs the move() functions of the Moves in move_list on
        a context.

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose positions were changed by this function.

        """
        rand = np.random.random()
        #to maintain detailed balance this executes both
        #the forward and reverse order moves with equal probability
        if rand > 0.5:
            for single_move in self.move_list:
                 single_move.move(context)
        else:
            for single_move in reverse(self.move_list):
                 single_move.move(context)


class SmartDartMove(RandomLigandRotationMove):
    """
    Move object that allows center of mass smart darting moves to be performed on a ligand,
    allowing translations of a ligand between pre-defined regions in space. The
    `SmartDartMove.move()` method translates the ligand to the locations of the ligand
    found in the coord_files. These locations are defined in terms of the basis_particles.
    These locations are picked with a uniform probability.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    basis_particles: list of 3 ints
        Specifies the 3 indices of the protein whose coordinates will be used
        to define a new set of basis vectors.
    coord_files: list of str
        List containing paths to coordinate files of the whole system for smart darting.
    topology: str, optional, default=None
        A path specifying a topology file matching the files in coord_files. Not
        necessary if the coord_files already contain topologies (ex. PDBs).
    dart_radius: simtk.unit float object compatible with simtk.unit.nanometers unit,
        optional, default=0.2*simtk.unit.nanometers
        The radius of the darting region around each dart.
    self_dart: boolean, optional, default='False'
        When performing the center of mass darting in `SmartDartMove.move()`,this
        specifies whether or not to include the darting region where the center
        of mass currently resides as an option to dart to.
    resname : str, optional, default='LIG'
        String specifying the residue name of the ligand.

    References:
    (1) I. Andricioaei, J. E. Straub, and A. F. Voter, J. Chem. Phys. 114, 6994 (2001).
        https://doi.org/10.1063/1.1358861

    """
    def __init__(self, structure, basis_particles, coord_files,
                 topology=None, dart_radius=0.2*unit.nanometers,
                 self_dart=False, resname='LIG'):

        super(SmartDartMove, self).__init__(structure, resname=resname)
        self.dartboard = []
        self.n_dartboard = []
        self.particle_pairs = []
        self.particle_weights = []
        self.basis_particles = basis_particles
        self.dart_radius = dart_radius
        self.calculateProperties()
        self.self_dart = self_dart
        self.dartsFromParmEd(coord_files, topology)

    def dartsFromParmEd(self, coord_files, topology=None):
        """
        Used to setup darts from a generic coordinate file, through MDtraj using the basis_particles to define
        new basis vectors, which allows dart centers to remain consistant through a simulation.
        This adds to the self.n_dartboard, which defines the centers used for smart darting.

        Parameters
        ---------
        system: simtk.openmm.system
            Openmm System corresponding to the whole system to smart dart.
        coord_files: list of str
            List containing coordinate files of the whole system for smart darting.
        topology: str, optional, default=None
            A path specifying a topology file matching the files in coord_files. Not
            necessary if the coord_files already contain topologies.

        """

        n_dartboard = []
        dartboard = []
        #loop over specified files and generate parmed structures from each
        #then the center of masses of the ligand in each structureare found
        #finally those center of masses are added to the `self.dartboard`s to
        #be used in the actual smart darting move to define darting regions
        for coord_file in coord_files:
            if topology == None:
                #if coord_file contains topology info, just load coord file
                temp_md = parmed.load_file(coord_file)
            else:
                #otherwise load file specified in topology
                temp_md = parmed.load_file(topology, xyz=coord_file)
            #get position values in terms of nanometers
            context_pos = temp_md.positions.in_units_of(unit.nanometers)
            lig_pos = np.asarray(context_pos._value)[self.atom_indices]*unit.nanometers
            particle_pos = np.asarray(context_pos._value)[self.basis_particles]*unit.nanometers
            #calculate center of mass of ligand
            self.calculateProperties()
            center_of_mass = self.getCenterOfMass(lig_pos, self.masses)
            #get particle positions
            new_coord = self._findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], center_of_mass)
            #old_coord should be equal to com
            old_coord = self._findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            np.testing.assert_almost_equal(old_coord._value, center_of_mass._value, decimal=1)
            #add the center of mass in euclidian and new basis set (defined by the basis_particles)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)
        self.n_dartboard = n_dartboard
        self.dartboard = dartboard


    def move(self, nc_context):
        """
        Function for performing smart darting move with darts that
        depend on particle positions in the system.

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.

        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose positions were changed by this function.

        """

        atom_indices = self.atom_indices
        if len(self.n_dartboard) == 0:
            raise ValueError('No darts are specified. Make sure you use ' +
                'SmartDartMove.dartsFromParmed() before using the move() function')
        context = nc_context
        #get state info from context
        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        #get the ligand positions
        lig_pos = np.asarray(oldDartPos._value)[self.atom_indices]*unit.nanometers
        #updates the darting regions based on the current position of the basis particles
        self._findDart(context)
        #find the ligand's current center of mass position
        center = self.getCenterOfMass(lig_pos, self.masses)
        #calculate the distance of the center of mass to the center of each darting region
        selected_dart, changevec = self._calc_from_center(com=center)
        #selected_dart is the selected darting region

        #if the center of mass was within one darting region, move the ligand to another region
        if selected_dart != None:
            newDartPos = np.copy(oldDartPos)
            #find the center of mass in the new darting region
            dart_switch = self._reDart(selected_dart, changevec)
            #find the vector that will translate the ligand to the new darting region
            vecMove = dart_switch - center
            #apply that vector to the ligand to actually translate the coordinates
            for atom in atom_indices:
                newDartPos[atom] = newDartPos[atom] + vecMove._value
            #set the positions after darting
            context.setPositions(newDartPos)

            return context

    def _calc_from_center(self, com):
        """
        Helper function that finds the distance of the current center of
        mass to each dart center in self.dartboard

        Parameters
        --------
        com: 1x3 np.array*simtk.unit.nanometers
            Current center of mass coordinates of the ligand.
        Returns
        -------
        selected_dart: simtk.unit.nanometers, or None
            The distance of a dart to a center. Returns
            None if the distance is greater than the darting region.
        changevec: 1x3 np.array*simtk.unit.nanometers,
            The vector from the ligand center of mass
            to the center of a darting region.

        """

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
        selected_dart = []
        #Find the dart(s) less than self.dart_radius
        for index, entry in enumerate(distList):
            if entry <= self.dart_radius:
                selected_dart.append(index)
                diff = diffList[index]
                indexList.append(index)
        #Dart error checking
        #to ensure reversibility the COM should only be
        #within self.dart_radius of one dart
        if len(selected_dart) == 1:
            return selected_dart[0], diffList[indexList[0]]
        elif len(selected_dart) == 0:
            return None, diff
        elif len(selected_dart) >= 2:
            #COM should never be within two different darts
            raise ValueError(' The spheres defining two darting regions have overlapped, ' +
                             'which results in potential problems with detailed balance. ' +
                             'We are terminating the simulation. Please check the size and ' +
                             'identity of your darting regions defined by dart_radius.')
            #TODO can treat cases using appropriate probablility correction
            #see https://doi.org/10.1016/j.patcog.2011.02.006

    def _findDart(self, nc_context):
        """
        Helper function to dynamically update dart positions based on the current positions
        of the basis particles.
        Arguments
        ---------
        nc_context: Context object from simtk.openmm
            Context from the ncmc simulation.

        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights.

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

    def _reDart(self, selected_dart, changevec):
        """
        Helper function to choose a random dart and determine the vector
        that would translate the COM to that dart center + changevec.
        This is called reDart in the sense that it helps to switch
        the ligand to another darting region.

        Parameters
        ---------
        changevec: 1x3 np.array * simtk.unit.nanometers
            The vector difference of the ligand center of mass
            to the closest dart center (if within the dart region).


        Returns
        -------
        dart_switch: 1x3 np.array * simtk.unit.nanometers

        """
        dartindex = range(len(self.dartboard))
        if self.self_dart == False:
            dartindex.pop(selected_dart)
        dartindex = np.random.choice(dartindex)
        dvector = self.dartboard[dartindex]
        dart_switch = dvector + changevec
        return dart_switch

    def _changeBasis(self, a, b):
        """
        Changes positions of a particle (b) in the regular basis set to
        another basis set (a). Used to recalculate the center of mass
        in terms of the local coordinates defined by self.basis_particles.
        Used to change between the basis sets defined from the basis_particles
        and the normal euclidian basis set.

        Parameters
        ----------
        a: 3x3 np.array
            Defines vectors that will create the new basis.
        b: 1x3 np.array
            Defines position of particle to be transformed into
            new basis set.
        Returns
        -------
        changed_coord: 1x3 np.array
            Coordinates of b in new basis.

        """

        ainv = np.linalg.inv(a.T)
        changed_coord = np.dot(ainv,b.T)*unit.nanometers
        return changed_coord

    def _undoBasis(self, a, b):
        """
        Transforms positions in a transformed basis (b) to the regular
        basis set. Used to transform the dart positions in the local
        coordinate basis set to the cartesian basis set.

        Parameters
        ----------
        a: 3x3 np.array
            Defines vectors that defined the new basis.
        b: 1x3 np.array
            Defines position of particle to be transformed into
            regular basis set.
        Returns
        -------
        changed_coord: 1x3 np.array
            Coordinates of b in new basis.
        """

        a = a.T
        changed_coord = np.dot(a,b.T)*unit.nanometers
        return changed_coord

    def _normalize(self, vector):
        """Normalize a given vector

        Parameters
        ----------
        vector: 1xn np.array
            Vector to be normalized.
        Returns
        -------
        unit_vec: 1xn np.array
            Normalized vector.

        """

        magnitude = np.sqrt(np.sum(vector*vector))
        unit_vec = vector / magnitude
        return unit_vec

    def _localCoord(self, particle1, particle2, particle3):
        """
        Defines a new coordinate system using 3 particles
        returning the new basis set vectors

        Parameters
        ----------
        particle1, particle2, particle3: 1x3 np.array
            np.array corresponding to a given particle's positions

        Returns
        -------
        vec1, vec2, vec3: 1x3 np.array
            Basis vectors of the coordinate system defined
            by particles1-3.

        """

        part2 = particle2 - particle1
        part3 = particle3 - particle1
        vec1 = part2
        vec2= part3
        vec3 = np.cross(vec1,vec2)*unit.nanometers
        return vec1, vec2, vec3

    def _findNewCoord(self, particle1, particle2, particle3, center):
        """
        Finds the coordinates of a given center in the standard basis
            in terms of a new basis defined by particles1-3

        Parameters
        ----------
        particle1, particle2, particle3: 1x3 np.array
            np.array corresponding to a given particle's positions
        center: 1x3 np.array * simtk.unit compatible with simtk.unit.nanometers
            Coordinate of the center of mass in the standard basis set.

        """

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
        """
        Finds the coordinates of a given center (defined by a different basis
        given by particles1-3) back in the euclidian coordinates

        Parameters
        ----------
        particle1, particle2, particle3: 1x3 np.array
            np.array corresponding to a given particle's positions
        center: 1x3 np.array * simtk.unit compatible with simtk.unit.nanometers
            Coordinate of the center of mass in the non-standard basis set.

        """

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
