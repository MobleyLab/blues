"""
moves.py: Provides the main Move class which allows definition of moves
which alter the positions of subsets of atoms in a context during a BLUES
simulation, in order to increase sampling.
Also provides functionality for CombinationMove definitions which consist of
a combination of other pre-defined moves such as via instances of Move.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, Kalistyn Burley, David L. Mobley
"""

import parmed
from simtk import unit, openmm
import mdtraj
import numpy as np
import sys
import math
import copy
import random
#from openeye.oechem import *


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
        """
        self.acceptance_ratio = 1.0

    def reset_iter(self):
        """Resets relevent attributes between iterations
        """
        self.acceptance_ratio = 1.0
    def move(self, context):
        """Function that can change the positions of a context.
        Base class `move()` just returns the same context.

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context..

        """
        return context

    def initializeSystem(self, system, integrator):
        """If the system or integrator needs to be modified to perform the move
        ex. adding a force this method is called during the start
        of the simulation to change the system.

        Parameters
        ----------
        system : simtk.openmm.System object
            System to be modified.
        integrator : simtk.openmm.Integrator object
            Integrator to be modified.
        Returns
        -------
        system : simtk.openmm.System object
            The modified System object.
        integrator : simtk.openmm.Integrator object
            The modified Integrator object.

        """
        new_sys = system
        new_int = integrator
        return new_sys, new_int

    def beforeMove(self, context):
        return context

    def afterMove(self, context):
        return context
    def _error(self, context):
        return context

    def move(self, context):
        return context



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
        Move.__init__(self)
        self.structure = structure
        self.resname = resname
        self.atom_indices = self.getAtomIndices(structure, self.resname)
        self.topology = structure[self.atom_indices].topology
        self.totalmass = 0
        self.masses = []

        self.center_of_mass = None
        self.positions = structure[self.atom_indices].positions
        self.calculateProperties()


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
        #coordinates = np.asarray(positions._value, np.float32)
        coordinates = np.array(positions._value, np.float32)

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

    def __init__(self, structure, residue_list):
        self.structure = structure
        self.molecule = self._pmdStructureToOEMol()
        self.residue_list = residue_list
        self.all_atoms = [atom.index for atom in self.structure.topology.atoms()]
        self.rot_atoms, self.rot_bonds, self.qry_atoms = self.getRotBondAtoms()
        self.atom_indices = self.rot_atoms

    def _pmdStructureToOEMol(self):

        from oeommtools.utils import openmmTop_to_oemol
        top = self.structure.topology
        pos = self.structure.positions
        molecule = openmmTop_to_oemol(top, pos, verbose=False)
        OEPerceiveResidues(molecule, OEPreserveResInfo_All)
        OEPerceiveResidues(molecule)
        OEFindRingAtomsAndBonds(molecule)

        return molecule

    def getBackboneAtoms(self, molecule):
        """This function takes a OEGraphMol and returns a list of backbone atoms"""

        backbone_atoms = []
        pred = OEIsBackboneAtom()
        for atom in molecule.GetAtoms(pred):
            bb_atom_idx = atom.GetIdx()
            backbone_atoms.append(bb_atom_idx)

        return backbone_atoms

    def getTargetAtoms(self, molecule, backbone_atoms, residue_list):
        """This function takes a OEGraphMol PDB structure and a list of residue numbers and
            generates a dictionary containing all the atom pointers and indicies for the
            non-backbone, atoms of those target residues, as well as a list of backbone atoms.
            Note: The atom indicies start at 0 and are thus -1 from the PDB file indicies"""

        # create and clear dictionary to store atoms that make up residue list
        qry_atoms = {}
        qry_atoms.clear()

        reslib = []

        #print('Searching residue list for atoms...')
        # loop through all the atoms in the PDB OEGraphMol structure
        for atom in molecule.GetAtoms():
            # check if the atom is in backbone
            if atom.GetIdx() not in backbone_atoms:
                # if heavy, find what residue it is associated with
                myres = OEAtomGetResidue(atom)
                # check if the residue number is amongst the list of residues
                if myres.GetResidueNumber() in residue_list and myres.GetName() != "HOH":
                    # store the atom location in a query atom dict keyed by its atom index
                    qry_atoms.update({atom : atom.GetIdx()})
                    #print('Found atom %s in residue number %i %s'%(atom,myres.GetResidueNumber(),myres.GetName()))
                    if myres not in reslib:
                        reslib.append(myres)

        return qry_atoms, backbone_atoms

    def findHeavyRotBonds(self, pdb_OEMol, qry_atoms):
        '''This function takes in an OEGraphMol PDB structure as well as a dictionary of atom locations (keys)
            and atom indicies.  It loops over the query atoms and identifies any heavy bonds associated with each atom.
            It stores and returns the bond indicies (keys) and the two atom indicies for each bond in a dictionary
            **Note: atom indicies start at 0, so are offset by 1 compared to pdb)'''

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
                        #print('Bond number',bond, 'is rotatable, non-terminal, and contains only heavy atoms')
                        # store bond pointer (key) and atom indicies in dictionary if not already there
                        #rot_bonds.update({bond : {'AtomIdx_1' : bond.GetBgnIdx(), 'AtomIdx_2': bond.GetEndIdx()}})
                        rot_bonds.update({bond : myres.GetResidueNumber()})

        return rot_bonds

    def getRotAtoms(self, rotbonds, molecule, backbone_atoms):
        """This function identifies and stores neighboring, upstream atoms for a given sidechain bond"""
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
            #print("Moving these atoms:", idx_list)

        return rot_atom_dict

    def getRotBondAtoms(self):
        """This function takes in a PDB filename (as a string) and list of residue numbers.  It returns
            a nested dictionary of rotatable bonds (containing only heavy atoms), that are keyed by residue number,
            then keyed by bond pointer, containing values of atom indicies [axis1, axis2, atoms to be rotated]
            **Note: The atom indicies start at 0, and are offset by -1 from the PDB file indicies"""
        backbone_atoms = self.getBackboneAtoms(self.molecule)

        # Generate dictionary containing locations and indicies of heavy residue atoms
        #print('Dictionary of all query atoms generated from residue list\n')
        qry_atoms, backbone_atoms = self.getTargetAtoms(self.molecule, backbone_atoms, self.residue_list)

        # Identify bonds containing query atoms and return dictionary of indicies
        rot_bonds = self.findHeavyRotBonds(self.molecule, qry_atoms)

        # Generate dictionary of residues, bonds and atoms to be rotated
        rot_atoms = self.getRotAtoms(rot_bonds, self.molecule, backbone_atoms)
        return rot_atoms, rot_bonds, qry_atoms

    def chooseBondandTheta(self):
        """This function takes a dictionary containing nested dictionary, keyed by res#,
        then keyed by bond_ptrs, containing a list of atoms to move, randomly selects a bond,
        and generates a random angle (radians).  It returns the atoms associated with the
        the selected bond, the pointer for the selected bond and the randomly generated angle"""

        res_choice = random.choice(list(self.rot_atoms.keys()))
        bond_choice = random.choice(list(self.rot_atoms[res_choice].keys()))
        targetatoms = self.rot_atoms[res_choice][bond_choice]
        theta_ran = random.random()*2*math.pi

        return theta_ran, targetatoms, res_choice, bond_choice

    def rotation_matrix(self, axis, theta):
        """This function returns the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        axis = np.asarray(axis)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


    def move(self, nc_context, verbose=False):
        """This rotates the target atoms around a selected bond by angle theta and updates
        the atom coordinates in the parmed structure as well as the ncmc context object"""


        # determine the axis, theta, residue, and bond + atoms to be rotated
        theta, target_atoms, res, bond = self.chooseBondandTheta()
        print('Rotating bond: %s in resnum: %s by %.2f radians' %(bond, res, theta))

        #retrieve the current positions
        initial_positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
        nc_positions = copy.deepcopy(initial_positions)

        model = copy.copy(self.structure)

        # set the parmed model to the same coordinates as the context
        for idx, atom in enumerate(self.all_atoms):
            if verbose:
                print('Before:')
                print(atom, idx)
                print(nc_positions[atom], model.positions[atom])

            model.atoms[atom].xx = nc_positions[atom][0].value_in_unit(unit.angstroms)
            model.atoms[atom].xy = nc_positions[atom][1].value_in_unit(unit.angstroms)
            model.atoms[atom].xz = nc_positions[atom][2].value_in_unit(unit.angstroms)

            if verbose:
                print('After:')
                print(nc_positions[atom], model.positions[atom])

        positions = model.positions

        # find the rotation axis using the updated positions
        axis1 = target_atoms[0]
        axis2 = target_atoms[1]
        rot_axis = (positions[axis1] - positions[axis2])/positions.unit

        #calculate the rotation matrix
        rot_matrix = self.rotation_matrix(rot_axis, theta)

        # apply the rotation matrix to the target atoms
        for idx, atom in enumerate(target_atoms):

            my_position = positions[atom]

            if verbose: print('The current position for %i is: %s'%(atom, my_position))

            # find the reduced position (substract out axis)
            red_position = (my_position - model.positions[axis2])._value
            # find the new positions by multiplying by rot matrix
            new_position = np.dot(rot_matrix, red_position)*positions.unit + positions[axis2]

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
        nc_context.setPositions(nc_positions)

        # update the class structure positions
        self.structure.positions = model.positions

        if verbose:
            filename = 'sc_move_%s_%s_%s.pdb' % (res, axis1, axis2)
            mod_prot = model.save(filename, overwrite = True)
        return nc_context

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

        if len(coord_files) < 2:
            raise ValueError('You should include at least two files in coord_files '+
                             'in order to benefit from smart darting')
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
        dartindex = list(range(len(self.dartboard)))
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
