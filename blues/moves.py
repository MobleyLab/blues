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
        self.before_ncmc = False
        #self.atom_indices = []

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
        positions: parmed.Structure
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


class WaterTranslationMove(Move):

    def __init__(self, structure, water_name='WAT', radius=2*unit.nanometers, before_ncmc_check=True):
        self.radius = radius
        self.water_name = water_name
        self.water_residues = []
        self.protein_atoms = []
        self.before_ncmc_check = before_ncmc_check
        residues = structure.topology.residues()
        for res in residues:
            if res.name == self.water_name:
                water_mol = []
                for atom in res.atoms():
                    water_mol.append(atom.index)
                self.water_residues.append(water_mol)
        residues = structure.topology.residues()

        for res in residues:
            atom_names = []
            atom_index = []
            for atom in res.atoms():
                atom_names.append(atom.name)
                atom_index.append(atom.index)
            if 'CA' in atom_names:
                self.protein_atoms = self.protein_atoms+atom_index
        self.atom_indices = self.water_residues[0]
        self.topology_protein = structure[self.protein_atoms].topology
        self.topology_water = structure[self.atom_indices].topology
        self.water_mass = self.getMasses(self.topology_water)
        self.protein_mass = self.getMasses(self.topology_protein)

    def _random_sphere_point(self, radius):
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
        return masses


    def getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a np.array
        Parameters
        ----------
        positions: parmed.Structure
            ParmEd positions of the atoms to be moved.
        masses : numpy.array
            np.array of particle masses
        """
        print('masses', masses)
        print('type', type(masses))
        print('type2', type(masses[0]))
        print('masses[0]', masses[0]/ unit.dalton * unit.dalton)
        print('dir', dir(masses))
        #print('value', positions.value_in_unit(positions.unit))
        print(positions)
        coordinates = np.asarray(positions._value, np.float32)
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
        return center_of_mass

    def before_ncmc(self, nc_context):
        start_state = nc_context.getState(getPositions=True, getVelocities=True)
        start_pos = start_state.getPositions(asNumpy=True)
        print('start_pos', start_pos[self.atom_indices[0]])
        start_vel = start_state.getVelocities(asNumpy=True)
        switch_pos = np.copy(start_pos)*start_pos.unit
        switch_vel = np.copy(start_vel)*start_vel.unit
        print('switch_pos', switch_pos)
        prot_com = self.getCenterOfMass(switch_pos[self.protein_atoms],
                            masses = self.protein_mass)
        #pick random water within the sphere radius
        dist_boolean = 0
        #TODO use random.shuffle to pick random particles (limits upper bound)
        while dist_boolean == 0:
            #water_choice = np.random.choice(water_residues)
            water_index = np.random.choice(range(len(self.water_residues)))
            water_choice = self.water_residues[water_index]
            oxygen_pos = start_pos[water_choice[0]]
            water_distance = np.linalg.norm(oxygen_pos._value - prot_com._value)
            #print('distance', water_distance)
            if water_distance <= (self.radius.value_in_unit(unit.nanometers)):
                dist_boolean = 1
            print('water_choice', water_choice)
        #replace chosen water's positions/velocities with alchemical water
        for i in range(3):
            switch_pos[self.atom_indices[i]] = start_pos[water_choice[i]]
            switch_vel[self.atom_indices[i]] = start_vel[water_choice[i]]
#            switch_pos[self.atom_indices[i]] = start_pos[self.atom_indices[i]]
#            switch_vel[self.atom_indices[i]] = start_vel[self.atom_indices[i]]
            switch_pos[water_choice[i]] = start_pos[self.atom_indices[i]]
            switch_vel[water_choice[i]] = start_vel[self.atom_indices[i]]

        print('after_switch', switch_pos[self.atom_indices[0]])
        nc_context.setPositions(switch_pos)
        nc_context.setVelocities(switch_vel)

        return nc_context


    def move(self, context):
        before_move_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        protein_pos = before_move_pos[self.protein_atoms]
        prot_com = self.getCenterOfMass(positions=protein_pos, masses=self.protein_mass)
        sphere_displacement = self._random_sphere_point(self.radius)
        movePos = np.copy(before_move_pos)*before_move_pos.unit
        print('movePos', movePos[self.atom_indices])
        print('center of mass', prot_com)
        print('Water coord', self.atom_indices)
        water_dist = movePos[self.atom_indices[0]] - prot_com
        print('water_dist._value', np.linalg.norm(water_dist._value))
        print('self.radius._value', self.radius._value)
        if np.linalg.norm(water_dist._value) <= self.radius._value:
            for index, resnum in enumerate(self.atom_indices):
                movePos[resnum] = movePos[resnum] - water_dist + sphere_displacement
                print('before', before_move_pos[resnum])
                print('after', movePos[resnum])
            #TODO check units, rotate water molecule
            #TODO make sure
            #movePos[:] = movePos*unit.nanometers
            context.setPositions(movePos)
        return context



class CombinationMove(Move):
    """Move object that allows Move object moves to be performed according to.
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

