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
import math
import copy
import random
import os
from openeye.oechem import *


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

def pDB2OEMol(pdbfile):
    '''This function takes in a pdbfile as a string (e.g. 'protein.pdb') and reads it into and OEGraphMol'''

    # check if file exists
    if os.path.isfile(pdbfile):
        # read file into an input stream
        ifs = oemolistream(pdbfile)
        # set the format of the input stream to pdb (other wise SMI default)
        ifs.SetFormat(OEFormat_PDB)
        # create OEMol destination
        pdb_OEMol = OEGraphMol()
        # assign input stream to OEMol
        OEReadMolecule(ifs, pdb_OEMol)
        return pdb_OEMol
    else:
        print('PDB filename not found.')


class SideChainMove(object):
    """Move that provides methods for:
        1. calculating the properties needed to rotate a sidechain residue
        of a structure in the NCMC simulation
        2. Executing a rotation of a random 'rotatable' bond in the designated sidechain
        by a random angle of rotation: 'theta'

        Calculated properties include: backbone atom indicies, atom pointers and indicies
        of the residue sidechain, bond pointers and indices for rotatable heavy bonds in
        the sidechain, and atom indices upstream of selected bond

        The class contains functions to randomly select a bond and angle to be rotated
        and applies a rotation matrix to the target atoms to update their coordinates"""

    def __init__(self, parmed_struct, residue_list):
        self.parmed_struct = parmed_struct
        self.residue_list = residue_list
        self.all_atoms = self.getAllAtoms()
        self.rot_bond_atoms, self.rot_bonds, self.qry_atoms, self.oestructure = self.getRotBondAtoms()

    def getBackboneAtoms(self, molecule):
        '''This function takes a OEGraphMol PDB structure and returns a list of backbone atoms'''

        backbone_atoms = []
        # Call this function to find atoms and bonds
        OEFindRingAtomsAndBonds(molecule)

        if not OEHasResidues(molecule):
                OEPerceiveResidues(molecule, OEPreserveResInfo_All)
        aiter = molecule.GetAtoms(OEIsBackboneAtom())
        for atom in aiter:
            bb_atom_idx = atom.GetIdx()
            backbone_atoms.append(bb_atom_idx)

        return backbone_atoms

    def getAllAtoms(self):

        atom_indices = []
        struct = self.parmed_struct
        topology = struct.topology
        for atom in topology.atoms():
            atom_indices.append(atom.index)
        return atom_indices


    def getTargetAtoms(self, molecule, residue_list):
        '''This function takes a OEGraphMol PDB structure and a list of residue numbers and
            generates a dictionary containing all the atom pointers and indicies for the
            non-backbone, atoms of those target residues, as well as a list of backbone atoms.
            Note: The atom indicies start at 0 and are thus -1 from the PDB file indicies'''

        # Call this function to find atoms and bonds
        OEFindRingAtomsAndBonds(molecule)
        backbone_atoms = self.getBackboneAtoms(molecule)

        # create and clear dictionary to store atoms that make up residue list
        qry_atoms = {}
        qry_atoms.clear()

        reslib = []

        print('Searching residue list for atoms...')
        # loop through all the atoms in the PDB OEGraphMol structure
        for atom in molecule.GetAtoms():
            # check if the atom is in backbone
            if atom.GetIdx() not in backbone_atoms:
                # if heavy, find what residue it is associated with
                myres = OEAtomGetResidue(atom)
                # check if the residue number is amongst the list of residues
                if myres.GetResidueNumber() in residue_list:
                    # store the atom location in a query atom dict keyed by its atom index
                    qry_atoms.update({atom : atom.GetIdx()})
                    print('Found atom %s in residue number %i %s'%(atom,myres.GetResidueNumber(),myres.GetName()))
                    if myres not in reslib:
                        reslib.append(myres)
        print('\n')
        return qry_atoms, backbone_atoms

    def findHeavyRotBonds(self, pdb_OEMol, qry_atoms):
        '''This function takes in an OEGraphMol PDB structure as well as a dictionary of atom locations (keys)
            and atom indicies.  It loops over the query atoms and identifies any heavy bonds associated with each atom.
            It stores and returns the bond indicies (keys) and the two atom indicies for each bond in a dictionary
            **Note: atom indicies start at 0, so are offset by 1 compared to pdb)'''

        # Call this function to find atoms and bonds
        OEFindRingAtomsAndBonds(pdb_OEMol)

        # create and clear dictionary to store bond and atom indicies that are rotatable + heavy
        rot_bonds = {}
        rot_bonds.clear()

        for atom in qry_atoms.keys():
            myres = OEAtomGetResidue(atom)
            for bond in atom.GetBonds():
                # retrieve the begnning and ending atoms
                begatom = bond.GetBgn()
                endatom = bond.GetEnd()
                # if begnnning and ending atoms are not Hydrogen, and the bond is rotatable
                if endatom.GetAtomicNum() >1 and begatom.GetAtomicNum() >1 and bond.IsRotor():
                    # if the bond has not been added to dictionary already..
                    # (as would happen if one of the atom pairs was previously looped over)
                    if bond not in rot_bonds:
                        # print the bond index
                        print('Bond number',bond, 'is rotatable, non-terminal, and contains only heavy atoms')
                        # store bond pointer (key) and atom indicies in dictionary if not already there
                        #rot_bonds.update({bond : {'AtomIdx_1' : bond.GetBgnIdx(), 'AtomIdx_2': bond.GetEndIdx()}})
                        rot_bonds.update({bond : myres.GetResidueNumber()})

        # Return dictionary with bond atom indicies keyed by bond index

        return rot_bonds

    #gets the atoms that are connected "upstream" of each rotbond

    def getRotAtoms(self, rotbonds, molecule, backbone_atoms):
        '''This function identifies and stores neighboring, upstream atoms for a given sidechain bond'''
        backbone = backbone_atoms
        query_list = []
        idx_list = []
        rot_atom_dict = {}
        rot_atom_dict.clear()

        for bond in rotbonds.keys():
            idx_list.clear()
            query_list.clear()
            resnum = (rotbonds[bond])
            thisbond = bond
            ax1 = bond.GetBgn()
            ax2 = bond.GetEnd()

            if resnum in rot_atom_dict.keys():
                rot_atom_dict[resnum].update({thisbond : []})
            else:
                rot_atom_dict.update({resnum : {thisbond : []}})

            idx_list.append(ax1.GetIdx())
            idx_list.append(ax2.GetIdx())

            # add axis atoms to query atom_list
            #if ax1 not in query_list and ax1.GetIdx() not in backbone:
            #    query_list.append(ax1)
            #if ax2 not in query_list and ax2.GetIdx() not in backbone:
            #    query_list.append(ax2)

            if ax1 not in query_list and ax1.GetIdx() not in backbone_atoms:
                query_list.append(ax1)
            if ax2 not in query_list and ax2.GetIdx() not in backbone_atoms:
                query_list.append(ax2)

            for atom in query_list:
                checklist = atom.GetAtoms()
                for candidate in checklist:
                    if candidate not in query_list and candidate.GetIdx() not in backbone and candidate != ax2:
                        query_list.append(candidate)
                        if candidate.GetAtomicNum() >1:
                            can_nbors = candidate.GetAtoms()
                            for can_nbor in can_nbors:
                                if can_nbor not in query_list and candidate.GetIdx() not in backbone and candidate != ax2:
                                    query_list.append(can_nbor)

            for atm in query_list:
                y = atm.GetIdx()
                if y not in idx_list:
                    idx_list.append(y)

            rot_atom_dict[resnum].update({thisbond : list(idx_list)})
            print("Moving these atoms:", idx_list)

        return rot_atom_dict


    def getRotBondAtoms(self):
        '''This function takes in a PDB filename (as a string) and list of residue numbers.  It returns
            a nested dictionary of rotatable bonds (containing only heavy atoms), that are keyed by residue number,
            then keyed by bond pointer, containing values of atom indicies [axis1, axis2, atoms to be rotated]
            **Note: The atom indicies start at 0, and are offset by -1 from the PDB file indicies'''

        # read .pdb file into OEGraphMol
        pdbfile = self.parmed_struct.save('protein.pdb', overwrite = True)
        structure = pDB2OEMol('protein.pdb')
        print('\nPDB file opened into OEGraphMol\n')
        # Generate dictionary containing locations and indicies of heavy residue atoms
        print('Dictionary of all query atoms generated from residue list\n')
        qry_atoms, backbone_atoms = self.getTargetAtoms(structure, self.residue_list)
        # Identify bonds containing query atoms and return dictionary of indicies
        rot_bonds = self.findHeavyRotBonds(structure, qry_atoms)
        # Generate dictionary of residues, bonds and atoms to be rotated
        rot_atoms = self.getRotAtoms(rot_bonds, structure, backbone_atoms)
        return rot_atoms, rot_bonds, qry_atoms, structure

    def chooseBondandTheta(self):
        '''This function takes a dictionary containing nested dictionary, keyed by res#,
        then keyed by bond_ptrs, containing a list of atoms to move, randomly selects a bond,
        and generates a random angle (radians).  It returns the atoms associated with the
        the selected bond, the pointer for the selected bond and the randomly generated angle'''

        struct = self.parmed_struct
        my_rot_atoms = self.rot_bond_atoms

        res_choice = random.choice(list(my_rot_atoms.keys()))

        bond_choice = random.choice(list(my_rot_atoms[res_choice].keys()))

        targetatoms = my_rot_atoms[res_choice][bond_choice]

        theta_ran = random.random()*2*math.pi
        #theta_ran = 0.0

        return theta_ran, targetatoms, res_choice, bond_choice

    def rotation_matrix(self, axis, theta):
        ''' This function returns the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians. '''
        axis = np.asarray(axis)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


    def move(self, nc_context, verbose = False):
        ''' This rotates the target atoms around a selected bond by angle theta and updates
        the atom coordinates in the parmed structure as well as the ncmc context object '''


        # determine the axis, theta, residue, and bond + atoms to be rotated
        my_theta, my_target_atoms, my_res, my_bond = self.chooseBondandTheta()
        print('\nRotating %s in %s by %.2f radians' %(my_bond, my_res, my_theta))

        #retrieve the current positions
        initial_positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
        nc_positions = copy.deepcopy(initial_positions)

        model = copy.copy(self.parmed_struct)

        # set the parmed model to the same coordinates as the context
        for idx, atom in enumerate(self.all_atoms):
            if verbose:
                print('Before:')
                print(atom, idx)
                print(nc_positions[atom], model.positions[atom])

            model.atoms[atom].xx = nc_positions[atom][0]/nc_positions.unit*10
            model.atoms[atom].xy = nc_positions[atom][1]/nc_positions.unit*10
            model.atoms[atom].xz = nc_positions[atom][2]/nc_positions.unit*10

            if verbose:
                print('After:')
                print(nc_positions[atom], model.positions[atom])

        positions = model.positions

        # find the rotation axis using the updated positions
        axis1 = my_target_atoms[0]
        axis2 = my_target_atoms[1]
        rot_axis = (positions[axis1] - positions[axis2])/positions.unit

        #calculate the rotation matrix
        my_rot_matrix = self.rotation_matrix(rot_axis, my_theta)

        # apply the rotation matrix to the target atoms
        for idx, atom in enumerate (my_target_atoms):

            my_position = positions[atom]

            if verbose: print('The current position for %i is: %s'%(atom, my_position))

            # find the reduced position (substract out axis)
            red_position = (my_position - model.positions[axis2])/positions.unit
            # find the new positions by multiplying by rot matrix
            new_position = np.dot(my_rot_matrix, red_position)*positions.unit + positions[axis2]

            if verbose: print("The new position should be:",new_position)

            positions[atom] = new_position
            # Update the parmed model with the new positions
            model.atoms[atom].xx = new_position[0]/positions.unit
            model.atoms[atom].xy = new_position[1]/positions.unit
            model.atoms[atom].xz = new_position[2]/positions.unit

            #update the copied ncmc context array with the new positions
            nc_positions[atom][0] = model.atoms[atom].xx*nc_positions.unit/10
            nc_positions[atom][1] = model.atoms[atom].xy*nc_positions.unit/10
            nc_positions[atom][2] = model.atoms[atom].xz*nc_positions.unit/10

            if verbose: print('The updated position for this atom is:', model.positions[atom])

        # update the actual ncmc context object with the new positions
        nc_context.setPositions(nc_positions/nc_positions.unit)

        # update the class parmed_struct positions
        self.parmed_struct.positions = model.positions

        if verbose:
            filename = 'sc_move_%s_%s_%s.pdb' % (my_res, axis1, axis2)
            mod_prot = model.save(filename, overwrite = True)

        print("\nMove completed")
        return nc_context

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
