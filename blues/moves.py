"""
Provides the main Move class which allows definition of moves
which alter the positions of subsets of atoms in a context during a BLUES
simulation, in order to increase sampling.
Also provides functionality for CombinationMove definitions which consist of
a combination of other pre-defined moves such as via instances of Move.

Authors: Samuel C. Gill

Contributors: Nathan M. Lim, Kalistyn Burley, David L. Mobley
"""

import copy
import math
import random
import sys
import traceback

import pint
import mdtraj
import numpy
import parmed
from openmm import unit
import tempfile
import numpy as np 
import openmm
from io import StringIO
from rdkit import Chem
from rdkit.Chem import AllChem, SanitizeFlags
from rdkit.Chem.rdMolTransforms import GetDihedralRad,GetDihedralDeg
from scipy.stats import norm

from openff.pablo import topology_from_pdb, ResidueDefinition

from blues.integrators import AlchemicalExternalLangevinIntegrator, AlchemicalExternalRestrainedLangevinIntegrator
from blues import utils
import blues.globalvar
#from blues.restraints import  add_boresch_restraints


try:
    import openeye.oechem as oechem
    if not oechem.OEChemIsLicensed():
        print('ImportError: Need License for OEChem! SideChainMove class will be unavailable.')
    try:
        import oeommtools.utils as oeommtools
    except ImportError:
        print('ImportError: Could not import oeommtools. SideChainMove class will be unavailable.')
except ImportError:
    print('ImportError: Could not import openeye-toolkits. SideChainMove class will be unavailable.')


class Move(object):
    """This is the base Move class. Move provides methods for calculating properties
    and applying the move on the set of atoms being perturbed in the NCMC simulation.
    """

    def __init__(self):
        """Initialize the Move object
        Currently empy.
        """

    def initializeSystem(self, system, integrator):
        """If the system or integrator needs to be modified to perform the move
        (ex. adding a force) this method is called during the start
        of the simulation to change the system or integrator to accomodate that.

        Parameters
        ----------
        system : openmm.System
            System to be modified.
        integrator : openmm.Integrator
            Integrator to be modified.

        Returns
        -------
        system : openmm.System
            The modified System object.
        integrator : openmm.Integrator
            The modified Integrator object.

        """
        new_sys = system
        new_int = integrator
        return new_sys, new_int

    def beforeMove(self, context):
        """This method is called at the start of the NCMC portion if the
        context needs to be checked or modified before performing the move
        at the halfway point.

        Parameters
        ----------
        context : openmm.Context
            Context containing the positions to be moved.

        Returns
        -------
        context : openmm.Context
            The same input context, but whose context were changed by this function.

        """
        return context

    def afterMove(self, context):
        """This method is called at the end of the NCMC portion if the
        context needs to be checked or modified before performing the move
        at the halfway point.

        Parameters
        ----------
        context : openmm.Context
            Context containing the positions to be moved.

        Returns
        -------
        context : openmm.Context
            The same input context, but whose context were changed by this function.

        """

        return context

    def _error(self, context):
        """This method is called if running during NCMC portion results
        in an error. This allows portions of the context, such as the
        context parameters that would not be fixed by just reverting the
        positions/velocities of the context.

        Parameters
        ----------
        context : openmm.Context
            Context containing the positions to be moved.

        Returns
        -------
        context : openmm.Context
            The same input context, but whose context were changed by this function.

        """

        return context

    def move(self, context):
        """This method is called at the end of the NCMC portion if the
        context needs to be checked or modified before performing the move
        at the halfway point.

        Parameters
        ----------
        context : openmm.Context
            Context containing the positions to be moved.

        Returns
        -------
        context : openmm.Context
            The same input context, but whose context were changed by this function.
        """
        return context


class RandomLigandRotationMove(Move):
    """optdomLightRotationMove that provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.
    Current methods calculate the object's atomic masses and center of masss.
    Calculating the object's center of mass will get the positions
    and total mass.

    Parameters
    ----------
    resname : str
        String specifying the residue name of the ligand.
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    random_state : integer or numpy.RandomState, optional
        The generator used for random numbers. If an integer is given, it fixes the seed. Defaults to the global numpy random number generator.
    ligand_indices : list of int, optional
        List of atom indices identifying the ligand to be rotated. If provided, overrides `resname` selection.
    Attributes
    ----------
    structure : parmed.Structure
        The structure of the ligand or selected atoms to be rotated.
    resname : str, default='LIG'
        The residue name of the ligand or selected atoms to be rotated.
    topology : openmm.Topology
        The topology of the ligand or selected atoms to be rotated.
    atom_indices : list
        Atom indicies of the ligand.
    masses : list
        Particle masses of the ligand with units.
    totalmass : int
        Total mass of the ligand.
    center_of_mass : numpy.array
        Calculated center of mass of the ligand in XYZ coordinates. This should
        be updated every iteration.
    positions : numpy.array
        Ligands positions in XYZ coordinates. This should be updated
        every iteration.

    Examples
    --------
    >>> from blues.move import RandomLigandRotationMove
    >>> ligand = RandomLigandRotationMove(structure, 'LIG')
    >>> ligand.resname
        'LIG'
    """

    def __init__(self, 
        structure, 
        resname='LIG', 
        ligand_indices=None, 
        random_state=None, ligand_pdb_file=None,
        restraints=None, K_r=None, K_angle=None,
        restrained_receptor_atoms=None, restrained_ligand_atoms=None,
        lambda_restraints = "max(0, 1-(1/0.10)*abs(lambda-0.5))",
        no_move=False,
        ):
        self.structure = structure
        self.resname = resname
        self.random_state = random_state
        self.atom_indices = self.getAtomIndices(structure, self.resname)
        self.ligand_indices = ligand_indices
        self.topology = structure[self.atom_indices].topology
        self.totalmass = 0
        self.restraints = restraints
        self.masses = []
        self.center_of_mass = None
        self.K_r = K_r
        self.K_angle = K_angle
        self.lambda_restraints = lambda_restraints
        self.restrained_receptor_atoms = restrained_receptor_atoms
        self.restrained_ligand_atoms = restrained_ligand_atoms
        self.positions = structure[self.atom_indices].positions
        self.no_move = no_move
        if self.ligand_indices:
            self.topology = structure[self.ligand_indices].topology
            self.positions = structure[self.ligand_indices].positions
        
        if ligand_pdb_file is not None:
            self._loadBindingModeTraj(ligand_pdb_file)
        self._calculateProperties()

    
    def _loadBindingModeTraj(self, ligand_pdb_file):
        """
        Load the binding mode trajectory from the file.
        """
        import mdtraj as md
        binding_mode_traj = []
        binding_mode_traj.append(md.load(ligand_pdb_file))
        self.binding_mode_traj = binding_mode_traj
    
    def initializeSystem(self, system, integrator, config):
        """
        Changes the system by adding forces corresponding to restraints (if specified)
        and freeze protein and/or waters, if specified in __init__()


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
        
        def find_force_group(system, force_type):
            """Returns the force group number for the first force of the given type."""
            for force in system.getForces():
                if isinstance(force, force_type):
                    return force.getForceGroup()
            raise ValueError(f"No force of type {force_type.__name__} found in system.")

        steric_group = find_force_group(new_sys, openmm.NonbondedForce)
        self.steric_group = steric_group

        if self.restraints:
            return self.initializeRestraints(new_sys, integrator, config)

        return new_sys, integrator


    def initializeRestraints(self, system: openmm.System, integrator: openmm.Integrator, config:dict):
        """
        Initialize the restraint forces for the system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM system to be modified
        integrator : openmm.Integrator
            The current integrator to be replaced with a restrained version

        Returns
        -------
        system : openmm.System
            The modified System object.
        integrator : openmm.Integrator
            The modified Integrator object.
        """
        # if self.restrained_receptor_atoms is None:
        #     self.restrained_receptor_atoms = self.basis_particles

        new_sys = system
        old_int = integrator

        # Get available force group
        force_list = new_sys.getForces()
        group_list = list(set([force.getForceGroup() for force in force_list]))
        group_avail = [j for j in list(range(32)) if j not in group_list]
        
        if len(group_avail) < len(self.restraints):
            raise ValueError("Not enough available force groups for all requested restraints.")
        
        self.restraint_groups = {}  # Dict to store group for each restraint type

        for i, restraint_type in enumerate(sorted(self.restraints)):
            self.restraint_groups[restraint_type] = group_avail[i]

        
        # Verify the old integrator has the required attributes
        if not hasattr(old_int, '_alchemical_functions'):
            raise AttributeError("Old integrator missing _alchemical_functions")

        # Get system parameters from old integrator
        old_int._system_parameters = {system_parameter for system_parameter in old_int._alchemical_functions.keys()}
        # Extract kwargs for the new integrator
        integrator_kwargs = config or {}

        # Get integrator kwargs if available, otherwise use defaults
        new_int = AlchemicalExternalLangevinIntegrator(
            restraint_group=set(self.restraint_groups.values()),
            lambda_restraints=self.lambda_restraints, 
            alchemical_functions = old_int._alchemical_functions,
            nsteps_neq=integrator_kwargs['nstepsNC'],
            nprop=integrator_kwargs['nprop'],
            prop_lambda=integrator_kwargs['propLambda'],
            splitting=integrator_kwargs['splitting'])
            #**old_int.int_kwargs)

        #new_int.reset()

        # Verify we have the required trajectory data
        if not hasattr(self, 'binding_mode_traj') or len(self.binding_mode_traj) == 0:
            raise ValueError("No binding mode trajectory available for restraints")
        
        for index, pose in enumerate(self.binding_mode_traj):

            # Cache the positions once
            pose_nm = numpy.array(pose.openmm_positions(0).value_in_unit(unit.nanometers))

            pose_pos = pose_nm[self.atom_indices]
            pose_allpos = pose_nm * unit.nanometers

            # Optional: update just the ligand portion of a shared template
            new_pos = pose_nm.copy()
            new_pos[self.atom_indices] = pose_pos
            new_pos = new_pos * unit.nanometers

            if 'boresch' in self.restraints:
                # Only validate force constants, allow atoms to be None
                if not all(x is not None for x in [self.K_r, self.K_angle]):
                    raise ValueError("Missing required force constants for Boresch restraints (K_r, K_angle)")
                

                new_sys = add_boresch_restraints(sys=new_sys, struct=self.structure, pos=pose_allpos, ligand_atoms=self.atom_indices, 
                                                 pose_num=index, force_group=self.restraint_groups['boresch'],
                                            restrained_receptor_atoms=self.restrained_receptor_atoms, restrained_ligand_atoms=self.restrained_ligand_atoms,
                                            K_r=self.K_r, K_angle=self.K_angle)
                
            if 'boresch' not in self.restraints and 'rmsd' not in self.restraints:
                raise ValueError(f'Invalid restraint type: {self.restraints}')
        
        return new_sys, new_int

    def getAtomIndices(self, structure, resname):
        """
        Get atom indices of a ligand from ParmEd Structure.

        Parameters
        ----------
        resname : str
            String specifying the residue name of the ligand.
        structure: parmed.Structure
            ParmEd Structure object of the atoms to be moved.

        Returns
        -------
        atom_indices : list of ints
            List of atoms in the coordinate file matching lig_resname.
        """        
        # TODO: Add option for resnum to better select residue names
        atom_indices = []
        for i, atom in enumerate(structure.atoms):
            if str(resname) in atom.residue.name:
                atom_indices.append(i) 

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
        masses: 1xn numpy.array * openmm.unit.dalton
            array of masses of len(self.atom_indices), denoting
            the masses of the atoms in self.atom_indices
        totalmass: float * openmm.unit.dalton
            The sum of the mass found in masses
        """

        expected_atoms = topology.getNumAtoms()

        masses = unit.Quantity(numpy.zeros([expected_atoms, 1], numpy.float32), unit.dalton)

        atom_count = 0  # Track actual number of atoms processed
        for idx, atom in enumerate(topology.atoms()):
            if atom.element:
                masses[idx] = atom.element._mass
                atom_count += 1
            else:
                masses[idx] = 1.008 * unit.dalton  # Assign a default hydrogen mass
                atom_count += 1

        totalmass = masses.sum()


        if atom_count != expected_atoms:
            raise ValueError("Mismatch in topology atom count!")

        return masses, totalmass

    def getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a numpy.array

        Parameters
        ----------
        positions: nx3 numpy array * openmm.unit compatible with openmm.unit.nanometers
            ParmEd positions of the atoms to be moved.
        masses : numpy.array
            numpy.array of particle masses

        Returns
        -------
        center_of_mass: numpy array * openmm.unit compatible with openmm.unit.nanometers
            1x3 numpy.array of the center of mass of the given positions
        """

        coordinates = numpy.asarray(positions._value, numpy.float32)
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit


        # Ensure masses shape matches coordinates shape
        if masses.shape[0] != coordinates.shape[0]:
            print(f"Warning: Mass count ({masses.shape[0]}) does not match coordinate count ({coordinates.shape[0]}).")
        
        return center_of_mass

    def _calculateProperties(self):
        """Calculate the masses and center of mass for the object. This function
        is called upon initailization of the class."""
        self.masses, self.totalmass = self.getMasses(self.topology)
        if len(self.masses) != len(self.atom_indices):
            print(f"🚨 WARNING: Mass count ({len(self.masses)}) does not match atom count ({len(self.atom_indices)})!")

        self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)
    def move(self, context):
        """Function that performs a random rotation about the
        center of mass of the ligand.

        Parameters
        ----------
        context: openmm.openmm.Context object
            Context containing the positions to be moved.

        Returns
        -------
        context: openmm.openmm.Context object
            The same input context, but whose positions were changed by this function.
        """
        #if self.no_move: 
        #    return context
        #positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        #
        #self.positions = positions[self.atom_indices]
        #self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)
        #reduced_pos = self.positions - self.center_of_mass

        ## Define random rotational move on the ligand
        #rand_quat = mdtraj.utils.uniform_quaternion(size=None, random_state=self.random_state)
        #rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
        ##multiply lig coordinates by rot matrix and add back COM translation from origin
        #rot_move = numpy.dot(reduced_pos, rand_rotation_matrix) * positions.unit + self.center_of_mass

        ## Update ligand positions in nc_sim
        #for index, atomidx in enumerate(self.atom_indices):
        #    positions[atomidx] = rot_move[index]
        #context.setPositions(positions)
        #positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        #self.positions = positions[self.atom_indices]
        return context


class MoveEngine(object):
    """MoveEngine provides perturbation functions for the context during the NCMC
    simulation.

    Parameters
    ----------
    moves : blues.move object or list of N blues.move objects
        Specifies the possible moves to be performed.

    probabilities: list of floats, optional, default=None
        A list of N probabilities, where probabilities[i] corresponds to the
        probaility of moves[i] being selected to perform its associated
        move() method. If None, uniform probabilities are assigned.

    Attributes
    ----------
    moves : blues.move or list of N blues.move objects
        Possible moves to be performed.
    probabilities : list of floats
        Normalized probabilities for each move.
    selected_move : blues.move
        Selected move to be performed.
    move_name : str
        Name of the selected move to be performed

    Examples
    --------
    Load a parmed.Structure, list of moves with probabilities, initialize
    the MoveEngine class, and select a move from our list.

    >>> import parmed
    >>> from blues.moves import *
    >>> structure = parmed.load_file('tests/data/eqToluene.prmtop', xyz='tests/data/eqToluene.inpcrd')
    >>> probabilities = [0.25, 0.75]
    >>> moves = [SideChainMove(structure, [111]),RandomLigandRotationMove(structure, 'LIG')]
    >>> mover = MoveEngine(moves, probabilities)
    >>> mover.moves
    [<blues.moves.SideChainMove at 0x7f2eaa168470>,
     <blues.moves.RandomLigandRotationMove at 0x7f2eaaaa51d0>]
    >>> mover.selectMove()
    >>> mover.selected_move
    <blues.moves.RandomLigandRotationMove at 0x7f2eaaaa51d0>
    """

    def __init__(self, moves, probabilities=None):
        #make a list from moves if not a list
        if isinstance(moves, list):
            self.moves = moves
        else:
            self.moves = [moves]
        #normalize probabilities
        if probabilities is None:
            single_prob = 1. / len(self.moves)
            self.probabilities = [single_prob for x in (self.moves)]
        else:
            prob_sum = float(sum(probabilities))
            self.probabilities = [x / prob_sum for x in probabilities]
        #if move and probabilitiy lists are different lengths throw error
        if len(self.moves) != len(self.probabilities):
            print('moves and probability list lengths need to match')
            raise IndexError
        #use index in selecting move
        self.selected_move = None

    def selectMove(self):
        """Chooses the move which will be selected for a given NCMC
        iteration
        """
        rand_num = numpy.random.choice(len(self.probabilities), p=self.probabilities)
        self.selected_move = self.moves[rand_num]
        self.move_name = self.selected_move.__class__.__name__

    def runEngine(self, context):
        """Selects a random Move object based on its
        assigned probability and and performs its move() function
        on a context.

        Parameters
        ----------
        context : openmm.Context
            OpenMM context whose positions should be moved.

        Returns
        -------
        context : openmm.Context
            OpenMM context whose positions have been moved.
        """
        
        try:
            # print(f"🔍 Selected Move: {self.selected_move}")
            # print(f"🔍 Move type: {type(self.selected_move)}")
            new_context = self.selected_move.move(context)
        except Exception as e:
            #In case the move isn't properly implemented, print out useful info
            print('Error: move not implemented correctly, printing traceback:')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(e)
            raise SystemExit

        return new_context


########################
## UNDER DEVELOPMENT ###
########################


class SideChainMove(Move):
    """**NOTE:** Usage of this class requires a valid OpenEye license.

    SideChainMove provides methods for calculating properties needed to
    rotate a sidechain residue given a parmed.Structure. Calculated properties
    include: backbone atom indicies, atom pointers and indicies of the residue
    sidechain, bond pointers and indices for rotatable heavy bonds in
    the sidechain, and atom indices upstream of selected bond.

    The class contains functions to randomly select a bond and angle to be rotated
    and applies a rotation matrix to the target atoms to update their coordinates on the
    object 'model' (i.e sidechain) being perturbed in the NCMC simulation.

    Parameters
    ----------
    structure : parmed.Structure
        The structure of the entire system to be simulated.
    residue_list : list of int
        List of the residue numbers of the sidechains to be rotated.
    verbose : bool, default=False
        Enable verbosity to print out detailed information of the rotation.
    write_move : bool, default=False
        If True, writes a PDB of the system after rotation.

    Attributes
    ----------
    structure : parmed.Structure
        The structure of the entire system to be simulated.
    molecule : oechem.OEMolecule
        The OEMolecule containing the sidechain(s) to be rotated.
    residue_list : list of int
        List containing the residue numbers of the sidechains to be rotated.
    all_atoms : list of int
        List containing the atom indicies of the sidechains to be rotated.
    rot_atoms : dict
        Dictionary of residues, bonds and atoms to be rotated
    rot_bonds : dict of oechem.OEBondBase
        Dictionary containing the bond pointers of the rotatable bonds.
    qry_atoms : dict of oechem.OEAtomBase
        Dictionary containing all the atom pointers (as OpenEye objects) that
        make up the given residues.


    Examples
    --------
    >>> from blues.move import SideChainMove
    >>> sidechain = SideChainMove(structure, [1])

    """

    def __init__(self, structure, residue_list, verbose=False, write_move=False):
        self.structure = structure
        self.molecule = self._pmdStructureToOEMol()
        self.residue_list = residue_list
        self.all_atoms = [atom.index for atom in self.structure.topology.atoms()]
        self.rot_atoms, self.rot_bonds, self.qry_atoms = self.getRotBondAtoms()
        self.atom_indices = self.rot_atoms
        self.verbose = verbose
        self.write_move = write_move

    def _pmdStructureToOEMol(self, prmtop, inpcrd, resname):
        """Helper function for converting the parmed structure into an OEMolecule."""
        structure_LIG = parmed.load_file(prmtop, xyz = inpcrd)
        mask = "!(:%s)" %resname
        structure_LIG.strip(mask)
        top = structure_LIG.topology
        pos = structure_LIG.positions
        molecule = oeommtools.openmmTop_to_oemol(top, pos, verbose=False)
        oechem.OEPerceiveBondOrders(molecule)
        oechem.OEAssignAromaticFlags(molecule)
        oechem.OEFindRingAtomsAndBonds(molecule)

        return molecule

     #def _pmdStructureToOEMol(self):
     #    """Helper function for converting the parmed structure into an OEMolecule."""
     #    top = self.structure.topology
     #    pos = self.structure.positions
     #    molecule = oeommtools.openmmTop_to_oemol(top, pos)
     #    oechem.OEPerceiveResidues(molecule)
     #    oechem.OEFindRingAtomsAndBonds(molecule)
     #    return molecule

    def getBackboneAtoms(self, molecule):
        """Takes an OpenEye Molecule, finds the backbone atoms and
        returns the indicies of the backbone atoms.

        Parameters
        ----------
        molecule : oechem.OEMolecule
            The OEmolecule of the simulated system.

        Returns
        -------
        backbone_atoms : list of int
            List containing the atom indices of the backbone atoms.

        """

        backbone_atoms = []
        pred = oechem.OEIsBackboneAtom()
        for atom in molecule.GetAtoms(pred):
            bb_atom_idx = atom.GetIdx()
            backbone_atoms.append(bb_atom_idx)

        return backbone_atoms

    def getTargetAtoms(self, molecule, backbone_atoms, residue_list):
        """Takes an OpenEye molecule and a list of residue numbers then
        generates a dictionary containing all the atom pointers and indicies for the
        non-backbone, atoms of those target residues, as well as a list of backbone atoms.
        Note: The atom indicies start at 0 and are thus -1 from the PDB file indicies

        Parameters
        ----------
        molecule : oechem.OEMolecule
            The OEmolecule of the simulated system.
        backbone_atoms : list of int
            List containing the atom indices of the backbone atoms.
        residue_list : list of int
            List containing the residue numbers of the sidechains to be rotated.

        Returns
        -------
        backbone_atoms : list of int
            List containing the atom indices of the backbone atoms to be rotated.
        qry_atoms : dict of oechem.OEAtomBase
            Dictionary containing all the atom pointers (as OpenEye objects) that
            make up the given residues.

        """

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
                myres = oechem.OEAtomGetResidue(atom)
                # check if the residue number is amongst the list of residues
                if myres.GetResidueNumber() in residue_list and myres.GetName() != "HOH":
                    # store the atom location in a query atom dict keyed by its atom index
                    qry_atoms.update({atom: atom.GetIdx()})
                    #print('Found atom %s in residue number %i %s'%(atom,myres.GetResidueNumber(),myres.GetName()))
                    if myres not in reslib:
                        reslib.append(myres)

        return qry_atoms, backbone_atoms

    def findHeavyRotBonds(self, pdb_OEMol, qry_atoms):
        """Takes in an OpenEye molecule as well as a dictionary of atom locations (keys)
        and atom indicies.  It loops over the query atoms and identifies any heavy bonds associated with each atom.
        It stores and returns the bond indicies (keys) and the two atom indicies for each bond in a dictionary
        Note: atom indicies start at 0, so are offset by 1 compared to pdb)

        Parameters
        ----------
        pdb_OEMol : oechem.OEMolecule
            The OEmolecule of the simulated system generated from a PDB file.
        qry_atoms : dict of oechem.OEAtomBase
            Dictionary containing all the atom pointers (as OpenEye objects) that
            make up the given residues.

        Returns
        -------
        rot_bonds : dict of oechem.OEBondBase
            Dictionary containing the bond pointers of the rotatable bonds.


        """
        # create and clear dictionary to store bond and atom indicies that are rotatable + heavy
        rot_bonds = {}
        rot_bonds.clear()

        for atom in qry_atoms.keys():
            myres = oechem.OEAtomGetResidue(atom)
            for bond in atom.GetBonds():
                # retrieve the begnning and ending atoms
                begatom = bond.GetBgn()
                endatom = bond.GetEnd()
                # if begnnning and ending atoms are not Hydrogen, and the bond is rotatable
                if endatom.GetAtomicNum() > 1 and begatom.GetAtomicNum() > 1 and bond.IsRotor():
                    # if the bond has not been added to dictionary already..
                    # (as would happen if one of the atom pairs was previously looped over)
                    if bond not in rot_bonds:
                        #print('Bond number',bond, 'is rotatable, non-terminal, and contains only heavy atoms')
                        # store bond pointer (key) and atom indicies in dictionary if not already there
                        #rot_bonds.update({bond : {'AtomIdx_1' : bond.GetBgnIdx(), 'AtomIdx_2': bond.GetEndIdx()}})
                        rot_bonds.update({bond: myres.GetResidueNumber()})

        return rot_bonds

    def getRotAtoms(self, rotbonds, molecule, backbone_atoms):
        """Function identifies and stores neighboring, upstream atoms for a given sidechain bond.

        Parameters
        ----------
        rot_bonds : dict of oechem.OEBondBase
            Dictionary containing the bond pointers of the rotatable bonds.
        molecule : oechem.OEMolecule
            The OEmolecule of the simulated system.
        backbone_atoms : list of int
            List containing the atom indices of the backbone atoms.


        Returns
        -------
        rot_atom_dict : dict of oechem.OEAtomBase
            Dictionary containing the atom pointers for a given sidechain bond.

        """
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
                rot_atom_dict[resnum].update({thisbond: []})
            else:
                rot_atom_dict.update({resnum: {thisbond: []}})

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
                        if candidate.GetAtomicNum() > 1:
                            can_nbors = candidate.GetAtoms()
                            for can_nbor in can_nbors:
                                if can_nbor not in query_list and candidate.GetIdx(
                                ) not in backbone and candidate != ax2:
                                    query_list.append(can_nbor)

            for atm in query_list:
                y = atm.GetIdx()
                if y not in idx_list:
                    idx_list.append(y)

            rot_atom_dict[resnum].update({thisbond: list(idx_list)})
            #print("Moving these atoms:", idx_list)

        return rot_atom_dict

    def getRotBondAtoms(self):
        """This function is called on class initialization.

        Takes in a PDB filename (as a string) and list of residue numbers.  Returns
        a nested dictionary of rotatable bonds (containing only heavy atoms), that are keyed by residue number,
        then keyed by bond pointer, containing values of atom indicies [axis1, axis2, atoms to be rotated]
        Note: The atom indicies start at 0, and are offset by -1 from the PDB file indicies

        Returns
        -------
        rot_atoms : dict
            Dictionary of residues, bonds and atoms to be rotated
        rot_bonds : dict of oechem.OEBondBase
            Dictionary containing the bond pointers of the rotatable bonds.
        qry_atoms : dict of oechem.OEAtomBase
            Dictionary containing all the atom pointers (as OpenEye objects) that
            make up the given residues.

        """
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
        """This function is called on class initialization.

        Takes a dictionary containing nested dictionary, keyed by res#,
        then keyed by bond_ptrs, containing a list of atoms to move, randomly selects a bond,
        and generates a random angle (radians).  It returns the atoms associated with the
        the selected bond, the pointer for the selected bond and the randomly generated angle


        Returns
        -------
        theta_ran :

        targetatoms :

        res_choice :

        bond_choice :

        """

        res_choice = random.choice(list(self.rot_atoms.keys()))
        bond_choice = random.choice(list(self.rot_atoms[res_choice].keys()))
        targetatoms = self.rot_atoms[res_choice][bond_choice]
        theta_ran = random.random() * 2 * math.pi

        return theta_ran, targetatoms, res_choice, bond_choice

    def rotationMatrix(self, axis, theta):
        """Function returns the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians.

        Parameters
        ----------
        axis :

        theta : float
            The angle of rotation in radians.
        """
        axis = numpy.asarray(axis)
        axis = axis / math.sqrt(numpy.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return numpy.array([[aa + bb - cc - dd, 2 * (bc + ad),
                             2 * (bd - ac)], [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def move(self, context, verbose=False):
        """Rotates the target atoms around a selected bond by angle theta and updates
        the atom coordinates in the parmed structure as well as the ncmc context object


        Parameters
        ----------
        context: openmm.openmm.Context object
            Context containing the positions to be moved.
        verbose : bool, default=False
            Enable verbosity to print out detailed information of the rotation.

        Returns
        -------
        context: openmm.openmm.Context object
            The same input context, but whose positions were changed by this function.

        """

        # determine the axis, theta, residue, and bond + atoms to be rotated
        theta, target_atoms, res, bond = self.chooseBondandTheta()
        print('Rotating bond: %s in resnum: %s by %.2f radians' % (bond, res, theta))

        #retrieve the current positions
        initial_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        nc_positions = copy.deepcopy(initial_positions)

        model = copy.copy(self.structure)

        # set the parmed model to the same coordinates as the context
        for idx, atom in enumerate(self.all_atoms):
            if self.verbose:
                print('Before:')
                print(atom, idx)
                print(nc_positions[atom], model.positions[atom])

            model.atoms[atom].xx = nc_positions[atom][0].value_in_unit(unit.angstroms)
            model.atoms[atom].xy = nc_positions[atom][1].value_in_unit(unit.angstroms)
            model.atoms[atom].xz = nc_positions[atom][2].value_in_unit(unit.angstroms)

            if self.verbose:
                print('After:')
                print(nc_positions[atom], model.positions[atom])

        positions = model.positions

        # find the rotation axis using the updated positions
        axis1 = target_atoms[0]
        axis2 = target_atoms[1]
        rot_axis = (positions[axis1] - positions[axis2]) / positions.unit

        #calculate the rotation matrix
        rot_matrix = self.rotationMatrix(rot_axis, theta)

        # apply the rotation matrix to the target atoms
        for idx, atom in enumerate(target_atoms):

            my_position = positions[atom]

            if self.verbose:
                print('The current position for %i is: %s' % (atom, my_position))

            # find the reduced position (substract out axis)
            red_position = (my_position - model.positions[axis2])._value
            # find the new positions by multiplying by rot matrix
            new_position = numpy.dot(rot_matrix, red_position) * positions.unit + positions[axis2]

            if self.verbose: print("The new position should be:", new_position)

            positions[atom] = new_position
            # Update the parmed model with the new positions
            model.atoms[atom].xx = new_position[0] / positions.unit
            model.atoms[atom].xy = new_position[1] / positions.unit
            model.atoms[atom].xz = new_position[2] / positions.unit

            #update the copied ncmc context array with the new positions
            nc_positions[atom][0] = model.atoms[atom].xx * nc_positions.unit / 10
            nc_positions[atom][1] = model.atoms[atom].xy * nc_positions.unit / 10
            nc_positions[atom][2] = model.atoms[atom].xz * nc_positions.unit / 10

            if self.verbose:
                print('The updated position for this atom is:', model.positions[atom])

        # update the actual ncmc context object with the new positions
        context.setPositions(nc_positions)

        # update the class structure positions
        self.structure.positions = model.positions

        if self.write_move:
            filename = 'sc_move_%s_%s_%s.pdb' % (res, axis1, axis2)
            mod_prot = model.save(filename, overwrite=True)
        return context

class WaterTranslationMove(Move):
    """ Move that translates a random water within a specified radius of the protein's
    center of mass to another point within that radius
    Parameters
    ----------
    structure:
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        water_name: str, optional, default='WAT'
            Residue name of the waters in the system.
        protein_selection: str, option, default='protein'
            Expression in the MDTraj atom selection DSL to specify the desired protein atoms.
            The center of mass of this is used to translate the water molecules.
        radius: float*unit compatible with openmm.unit.nanometers, optional, default=2.0*unit.nanometers
            Defines the radius within the protein center of mass to choose a water
            and the radius in which to randomly translate that water.
    """

    def __init__(self, structure, water_name=['WAT','HOH'], protein_selection='protein', radius=2.3*unit.nanometers):
        #initialize self attributes
        self.radius = radius #
        self.water_name = water_name
        self.water_residues = [] #contains indices of the atoms of the waters
        self.before_ncmc_check = True
        self.structure = structure
        #we want to get a mdtraj trajectory representation of the structure
        #so save structure to a temp file to be read with mdtraj
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='r+') as t:
            fname = t.name
            structure.save(t,format='pdb')
            t.flush()
            self.traj = mdtraj.load(fname)

        #go through the topology and identify water and protein residues
        residues = structure.topology.residues()
        #looks for residues with water_name in them
        for res in residues:
            if res.name in self.water_name: #checks if the name of the residue is 'WAT' or 'HOH'
                water_mol = [] #list of each waters atom indices
                for atom in res.atoms():
                   water_mol.append(atom.index) #append the index of each of the atoms of the water residue
                self.water_residues.append(water_mol)#append the water atom indices as a self attribute (above)

        self.atom_indices = self.water_residues[0] #the atom indices of the first water, this is the alchemical water
        #get the protein atoms of the selection and the masses
        self.protein_atoms = self.traj.topology.select(protein_selection)
        self.protein_masses = self._getMasses(self.structure.topology)[self.protein_atoms]
        #go keeps track if water is inside specified radius
        self.go = True



    def _random_sphere_point(self, radius, origin):
        """function to generate a uniform random point
        in a sphere of a specified radius.
        Used to randomly translate the water molecule
        Parameters
        ----------
        radius: float
            Defines the radius of the sphere in which a point
            will be uniformly randomly generated.
        """
        #print("Radius of _random_sphere_point", radius)
        r = radius * ( numpy.random.random()**(1./3.) )  #r (radius) = specified radius * cubed root of a random number between 0.00 and 0.99999
        phi = numpy.random.uniform(0,2*numpy.pi) #restriction of phi (or azimuth angle) is set from 0 to 2pi. random.uniform allows the values to be chosen w/ an equal probability
        costheta = numpy.random.uniform(-1,1) #restriction set from -1 to 1
        theta = numpy.arccos(costheta) #calculate theta, the angle between r and Z axis
        x = numpy.sin(theta) * numpy.cos(phi) #x,y,and z are cartesian coordinates
        y = numpy.sin(theta) * numpy.sin(phi)
        z = numpy.cos(theta)

        sphere_point = numpy.array([x, y, z]) * r + origin
        return sphere_point #sphere_point = a random point with an even distribution

    def _getMasses(self, topology):
        """Returns a list of masses of the specified ligand atoms.
        Parameters
        ----------
        topology: parmed.Topology
            ParmEd topology object containing atoms of the system.
        """
        masses = unit.Quantity(numpy.zeros([int(topology.getNumAtoms()),1],numpy.float32), unit.dalton)
        for idx,atom in enumerate(topology.atoms()):
            masses[idx] = atom.element._mass #gets the mass of the atom, adds to list (along with index)
        return masses

    def _getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a np.array
        Parameters
        ----------
        positions: nx3 np.array
            Positions of the atoms to be moved.
        masses : nx1 numpy.array
            numpy.array of particle masses
        """
        try: #try with units, if not do without units in except clause
            coordinates = numpy.asarray(positions._value, numpy.float32) #gives the value of atomic positions as an array
            center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit

        except:
            coordinates = numpy.asarray(positions, numpy.float32) #gives the value of atomic positions as an array
            center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) 

        return center_of_mass

    def beforeMove(self, context):
        """
        Temporary fterHop/unction (until multiple alchemical regions are supported),
        which is performed at the beginning of a ncmc iteration. Selects
        a random water within self.radius of the protein's center of mass
        and switches the positions and velocities with the alchemical water
        defined by self.atom_indices, effecitvely duplicating mulitple
        alchemical region support.
        Parameters
        ----------
        context: openmm.openmm Context object
            The context which corresponds to the NCMC simulation.
        """
        start_state = context.getState(getPositions=True, getVelocities=True)
        start_pos = start_state.getPositions(asNumpy=True) #gets starting positions
        start_vel = start_state.getVelocities(asNumpy=True) #gets starting velocities
        switch_pos = numpy.copy(start_pos)*start_pos.unit #starting position (a shallow copy) is * by start_pos.unit to retain units
        switch_vel = numpy.copy(start_vel)*start_vel.unit #starting vel (a shallow copy) is * by start_pos.unit to retain units

        #look for a random water inside the radius
        is_inside_sphere = False
        #use random.shuffle to pick random particles (limits upper bound)
        water_residues_shuffle = copy.deepcopy(self.water_residues)
        numpy.random.shuffle(water_residues_shuffle)
        #we want to use mdtraj distance function, but needs to act on trajectory object
        #get around this by selecting the first protein atom to act as a dummy atom for the com
        self.traj.xyz[0,:,:] = start_pos
        #get the current center of mass of the protein selection
        current_com = self._getCenterOfMass(self.traj.xyz[0,:,:][self.protein_atoms], self.protein_masses)
        #set the dummy com atom position
        self.traj.xyz[0,self.protein_atoms[0],:] = current_com
        self.traj.unitcell_vectors[0] = context.getState(getPositions=True).getPeriodicBoxVectors(asNumpy=True)
        #find a water within radius range of the center of mass
        while ((not is_inside_sphere) or len(water_residues_shuffle)==0):
            if len(water_residues_shuffle)==0:
                break
            #pop out the first water in the list and see if it's within the sphere radius
            water_choice = water_residues_shuffle.pop(0)
            pairs = self.traj.topology.select_pairs(numpy.array(water_choice[0]).flatten(), numpy.array(self.protein_atoms[0]).flatten())
            water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)
            water_dist = numpy.linalg.norm(water_distance)
            if water_dist <= (self.radius.value_in_unit(unit.nanometers)):
                is_inside_sphere = True
        #replace chosen water's positions/velocities with alchemical water
        if is_inside_sphere:
            switch_pos[self.atom_indices] = start_pos[water_choice]
            switch_vel[self.atom_indices] = start_vel[water_choice]
            switch_pos[water_choice] = start_pos[self.atom_indices]
            switch_vel[water_choice] = start_vel[self.atom_indices]
            context.setPositions(switch_pos)
            context.setVelocities(switch_vel)
            #keep track if switch occurs or not
            self.go = True
        else:
            self.go = False
        return context

    def move(self, context):
        """
        This function is called by the blues.MoveEngine object during a simulation.
        Translates the alchemical water randomly within a sphere of self.radius.
        """
        #If there was no water selected in beforeMove skip the move
        if self.go == False:
            return context
        #get the position of the system from the context
        before_move_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        movePos = numpy.copy(before_move_pos)*before_move_pos.unit #makes a copy of the position of the system from the context
        movePos_end = numpy.copy(before_move_pos)*before_move_pos.unit #use this to get around switching dummy atom
        #get the current center of mass of the protein selection
        current_com = self._getCenterOfMass(self.traj.xyz[0,:,:][self.protein_atoms], self.protein_masses)
        #set the current center of mass as a dummy atom in self.traj
        self.traj.xyz[0,:,:] = movePos
        self.traj.xyz[0][self.protein_atoms[0]] = current_com
        self.traj.unitcell_vectors[0] = context.getState(getPositions=True).getPeriodicBoxVectors(asNumpy=True)
        origin = self.traj.xyz[0][self.protein_atoms[0]]*movePos.unit

        #Generate uniform random point in a sphere of a specified radius
        sphere_displacement = self._random_sphere_point(self.radius, origin)
        pairs = self.traj.topology.select_pairs(numpy.array(self.atom_indices[0]).flatten(), numpy.array(self.protein_atoms[0]).flatten())
        water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)
        #if the water molecule ends up outside the defined region we can't perform the move
        if water_distance >= (self.radius.value_in_unit(unit.nanometers)):
            #return the starting context
            return context

        #move the water molecule
        displacement_vector = movePos_end[self.atom_indices[0]] - sphere_displacement
        movePos_end[self.atom_indices] = movePos_end[self.atom_indices] - displacement_vector
        if 1: #debug statement
            self.traj.xyz[0][self.atom_indices] = movePos_end[self.atom_indices]
            pairs = self.traj.topology.select_pairs(numpy.array(self.atom_indices[0]).flatten(), numpy.array(self.protein_atoms[0]).flatten())
            water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)

        #set the new positions
        context.setPositions(movePos_end)

        return context

    def afterMove(self, context):
        """This method is called at the end of the NCMC portion.
        This checks if the water has ended up outside the defined
        translational region. If so, it rejects the move
        (by manually setting protocol work to a high value).

        Parameters
        ----------
        context: openmm.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: openmm.openmm.Context object
            The same inumpyut context, but whose context were changed by this function.
        """
        before_final_move_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        #Update positions for distance calculation
        movePos_a = numpy.copy(before_final_move_pos)*before_final_move_pos.unit

        self.traj.xyz[0,:,:] = movePos_a;
        #find the center of mass of the specified protein residues
        current_com = self._getCenterOfMass(self.traj.xyz[0,:,:][self.protein_atoms], self.protein_masses)
        self.traj.xyz[0,self.protein_atoms[0],:] = current_com
        self.traj.unitcell_vectors[0] = context.getState(getPositions=True).getPeriodicBoxVectors(asNumpy=True)

        #find the distance between the chosen water and the protein com
        pairs = self.traj.topology.select_pairs(numpy.array(self.atom_indices[0]).flatten(), numpy.array(self.protein_atoms[0]).flatten())
        water_distance = mdtraj.compute_distances(self.traj, pairs, periodic=True)
        #We reject if the water molecule ends up outside our defined radius value
        if water_distance > numpy.linalg.norm(self.radius._value) and self.go == True:
            #we can update this if we implement acceptance_ratios into the acceptance to be handle this more 'correctly'
            #essentially this will result in rejected though
            context._integrator.setGlobalVariableByName("protocol_work", 999999)
        return context


class SmartDartMove(RandomLigandRotationMove):
    """**WARNING:** This class has not been completely tested. Use at your own risk.

    Move object that allows center of mass smart darting moves to be performed on a ligand,
    allowing translations of a ligand between pre-defined regions in space. The
    `SmartDartMove.move()` method translates the ligand to the locations of the ligand
    found in the coord_files. These locations are defined in terms of the basis_particles.
    These locations are picked with a uniform probability. Based on Smart Darting Monte Carlo [smart-dart]_

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
    dart_radius: openmm.unit float object compatible with openmm.unit.nanometers unit,
        optional, default=0.2*openmm.unit.nanometers
        The radius of the darting region around each dart.
    self_dart: boolean, optional, default='False'
        When performing the center of mass darting in `SmartDartMove.move()`,this
        specifies whether or not to include the darting region where the center
        of mass currently resides as an option to dart to.
    resname : str, optional, default='LIG'
        String specifying the residue name of the ligand.

    References
    ----------
    .. [smart-dart] I. Andricioaei, J. E. Straub, and A. F. Voter, J. Chem. Phys. 114, 6994 (2001).
        https://doi.org/10.1063/1.1358861

    """

    def __init__(self,
                 structure,
                 basis_particles,
                 coord_files,
                 topology=None,
                 dart_radius=0.2 * unit.nanometers,
                 self_dart=False,
                 resname='LIG'):

        super(SmartDartMove, self).__init__(structure, resname=resname)

        if len(coord_files) < 2:
            raise ValueError('You should include at least two files in coord_files ' +
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
        ----------
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
            lig_pos = numpy.asarray(context_pos._value)[self.atom_indices] * unit.nanometers
            particle_pos = numpy.asarray(context_pos._value)[self.basis_particles] * unit.nanometers
            #calculate center of mass of ligand
            self.calculateProperties()
            center_of_mass = self.getCenterOfMass(lig_pos, self.masses)
            #get particle positions
            new_coord = self._findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], center_of_mass)
            #old_coord should be equal to com
            old_coord = self._findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
            numpy.testing.assert_almost_equal(old_coord._value, center_of_mass._value, decimal=1)
            #add the center of mass in euclidian and new basis set (defined by the basis_particles)
            n_dartboard.append(new_coord)
            dartboard.append(old_coord)
        self.n_dartboard = n_dartboard
        self.dartboard = dartboard

    def move(self, context):
        """
        Function for performing smart darting move with darts that
        depend on particle positions in the system.

        Parameters
        ----------
        context: openmm.openmm.Context object
            Context containing the positions to be moved.

        Returns
        -------
        context: openmm.openmm.Context object
            The same input context, but whose positions were changed by this function.

        """

        atom_indices = self.atom_indices
        if len(self.n_dartboard) == 0:
            raise ValueError('No darts are specified. Make sure you use ' +
                             'SmartDartMove.dartsFromParmed() before using the move() function')

        #get state info from context
        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        #get the ligand positions
        lig_pos = numpy.asarray(oldDartPos._value)[self.atom_indices] * unit.nanometers
        #updates the darting regions based on the current position of the basis particles
        self._findDart(context)
        #find the ligand's current center of mass position
        center = self.getCenterOfMass(lig_pos, self.masses)
        #calculate the distance of the center of mass to the center of each darting region
        selected_dart, changevec = self._calc_from_center(com=center)
        #selected_dart is the selected darting region

        #if the center of mass was within one darting region, move the ligand to another region
        if selected_dart != None:
            newDartPos = numpy.copy(oldDartPos)
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
        com: 1x3 numpy.array*openmm.unit.nanometers
            Current center of mass coordinates of the ligand.

        Returns
        -------
        selected_dart: openmm.unit.nanometers, or None
            The distance of a dart to a center. Returns
            None if the distance is greater than the darting region.
        changevec: 1x3 numpy.array*openmm.unit.nanometers,
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
            dist = numpy.sqrt(numpy.sum((diff) * (diff))) * unit.nanometers
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

    def _findDart(self, context):
        """
        Helper function to dynamically update dart positions based on the current positions
        of the basis particles.

        Parameters
        ---------
        context: Context object from openmm.openmm
            Context from the ncmc simulation.

        Returns
        -------
        dart_list list of 1x3 numpy.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights.

        """

        basis_particles = self.basis_particles
        #make sure there's an equal number of particle pair lists
        #and particle weight lists
        dart_list = []
        state_info = context.getState(True, True, False, True, True, False)
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
        selected_dart :
        changevec: 1x3 numpy.array * openmm.unit.nanometers
            The vector difference of the ligand center of mass
            to the closest dart center (if within the dart region).


        Returns
        -------
        dart_switch: 1x3 numpy.array * openmm.unit.nanometers

        """
        dartindex = list(range(len(self.dartboard)))
        if self.self_dart == False:
            dartindex.pop(selected_dart)
        dartindex = numpy.random.choice(dartindex)
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
        a: 3x3 numpy.array
            Defines vectors that will create the new basis.
        b: 1x3 numpy.array
            Defines position of particle to be transformed into
            new basis set.

        Returns
        -------
        changed_coord: 1x3 numpy.array
            Coordinates of b in new basis.

        """

        ainv = numpy.linalg.inv(a.T)
        changed_coord = numpy.dot(ainv, b.T) * unit.nanometers
        return changed_coord

    def _undoBasis(self, a, b):
        """
        Transforms positions in a transformed basis (b) to the regular
        basis set. Used to transform the dart positions in the local
        coordinate basis set to the cartesian basis set.

        Parameters
        ----------
        a: 3x3 numpy.array
            Defines vectors that defined the new basis.
        b: 1x3 numpy.array
            Defines position of particle to be transformed into
            regular basis set.

        Returns
        -------
        changed_coord: 1x3 numpy.array
            Coordinates of b in new basis.
        """

        a = a.T
        changed_coord = numpy.dot(a, b.T) * unit.nanometers
        return changed_coord

    def _normalize(self, vector):
        """Normalize a given vector

        Parameters
        ----------
        vector: 1xn numpy.array
            Vector to be normalized.

        Returns
        -------
        unit_vec: 1xn numpy.array
            Normalized vector.

        """

        magnitude = numpy.sqrt(numpy.sum(vector * vector))
        unit_vec = vector / magnitude
        return unit_vec

    def _localCoord(self, particle1, particle2, particle3):
        """
        Defines a new coordinate system using 3 particles
        returning the new basis set vectors

        Parameters
        ----------
        particle1, particle2, particle3: 1x3 numpy.array
            numpy.array corresponding to a given particle's positions

        Returns
        -------
        vec1, vec2, vec3: 1x3 numpy.array
            Basis vectors of the coordinate system defined
            by particles1-3.

        """

        part2 = particle2 - particle1
        part3 = particle3 - particle1
        vec1 = part2
        vec2 = part3
        vec3 = numpy.cross(vec1, vec2) * unit.nanometers
        return vec1, vec2, vec3

    def _findNewCoord(self, particle1, particle2, particle3, center):
        """
        Finds the coordinates of a given center in the standard basis
            in terms of a new basis defined by particles1-3

        Parameters
        ----------
        particle1, particle2, particle3: 1x3 numpy.array
            numpy.array corresponding to a given particle's positions
        center: 1x3 numpy.array * openmm.unit compatible with openmm.unit.nanometers
            Coordinate of the center of mass in the standard basis set.

        Returns
        -------
        new_coord : numpy.array
            Updated coordinates in terms of new basis.
        """

        #calculate new basis set
        vec1, vec2, vec3 = self._localCoord(particle1, particle2, particle3)
        basis_set = numpy.zeros((3, 3)) * unit.nanometers
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
        particle1, particle2, particle3: 1x3 numpy.array
            numpy.array corresponding to a given particle's positions
        center: 1x3 numpy.array * openmm.unit compatible with openmm.unit.nanometers
            Coordinate of the center of mass in the non-standard basis set.

        Returns
        -------
        adjusted_center : numpy.array
            Corrected coordinates of new center in euclidian coordinates.

        """

        vec1, vec2, vec3 = self._localCoord(particle1, particle2, particle3)
        basis_set = numpy.zeros((3, 3)) * unit.nanometers
        basis_set[0] = vec1
        basis_set[1] = vec2
        basis_set[2] = vec3
        #since the origin is centered at particle1 by convention
        #subtract to account for this
        old_coord = self._undoBasis(basis_set, center)
        adjusted_center = old_coord + particle1
        return adjusted_center


class CombinationMove(Move):
    """**WARNING:** This class has not been completely tested. Use at your own risk.

    Move object that allows Move object moves to be performed according to
    the order in move_list. To ensure detailed balance, the moves have an equal
    chance to be performed in listed or reverse order.

    Parameters
    ----------
    moves : list of blues.move.Move

    """

    def __init__(self, moves):
        self.moves = moves

    def move(self, context):
        """Performs the move() functions of the Moves in move_list on
        a context.

        Parameters
        ----------
        context: openmm.openmm.Context object
            Context containing the positions to be moved.

        Returns
        -------
        context: openmm.openmm.Context object
            The same input context, but whose positions were changed by this function.

        """
        rand = numpy.random.random()
        #to maintain detailed balance this executes both
        #the forward and reverse order moves with equal probability
        if rand > 0.5:
            for single_move in self.move_list:
                single_move.move(context)
        else:
            for single_move in reverse(self.move_list):
                single_move.move(context)

def sanitize_rdmol(rdmol) :
    """
    Checks and corrects the chemical structure of molecule. (Checks valence, aromaticity, kekulization, and stereochemistry)
    
    Args:
        mol: Chem.Mol; rdkit molecule

    Returns:
        Chem.Mol: santized rdkit molecule
    """
    # From Jeffrey Wagner, sanitize RDKIT mol
    Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL)
    Chem.AssignStereochemistryFrom3D(rdmol)
    Chem.Kekulize(rdmol, clearAromaticFlags=True)
    Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
    return rdmol


def assignBondOrderFromSmiles(smiles: str, rdmol):
    """
    Assigns the bond order of a rdkit molecule lacking proper bond information based on its smiles
    
    Args:
        smiles: str; smiles str representing the correct bond orders
        molfile: str; pdb, sdf, or mol2 filename of molecule that lacks correct bond order information

    Returns:
        Chem.Mol: RDKit Mol with 
    """

    # the smiles will contain the bond order information
    # the molfile contains the positions and index order
    # the returned mol will contain the bond order from the 
    # smiles but the index order from the molfile

    #Chem.SanitizeMol(rdmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)

    smi_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    smi_mol = Chem.RemoveHs(smi_mol)

    smi_mol= Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), allHsExplicit=False), sanitize=False)

    #if smi_mol.GetNumAtoms() < rdmol.GetNumAtoms():
        #AllChem.EmbedMolecule(smi_mol)
        #AllChem.UFFOptimizeMolecule(smi_mol)
        #Chem.MolToMolFile(smi_mol, "smi_mol.mol")
        #sanitize_rdmol(smi_mol)

        #Chem.MolToMolFile(lig_mol_wo_bond_orders, "pdb_mol.mol")
        #sanitize_rdmol(rdmol)


    lig_mol = AllChem.AssignBondOrdersFromTemplate(
        smi_mol, 
        rdmol, 
    )
    #Chem.MolToMolFile(lig_mol_wo_bond_orders, "bondorder_mol.mol")
    return lig_mol


def pmdStructureToRDMol(struct, resname, smiles):
    """Helper function for converting the parmed structure into an RDMol."""
    structure_LIG = copy.deepcopy(struct)
    structure_LIG.symmetry = None
    mask = f":{resname}"
    structure_LIG.strip(f"!({mask})")
    structure_LIG.symmetry = None
    top = structure_LIG.topology
    pos = structure_LIG.positions

    buf = StringIO()
    structure_LIG.write_pdb(buf)
    pdb_block = buf.getvalue()
    print(pdb_block)

    pdb_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    print('PDBMOL:', pdb_mol.GetNumAtoms())
    bondorder_mol = assignBondOrderFromSmiles(smiles, pdb_mol)

    return bondorder_mol


def getRotatingAtoms(rdmol, fixed_atom, rotating_atom):
    """Return all atoms on the 'rotating_atom' side of the fixed_atom–rotating_atom bond."""
    visited = set()
    stack = [rotating_atom]
    group = []
    while stack != []:
        a = stack.pop()
        if a in visited:
            continue
        visited.add(a)
        group.append(a)
        for nbr in rdmol.GetAtomWithIdx(a).GetNeighbors():
            if nbr.GetIdx() != fixed_atom and nbr.GetIdx() not in visited:
                stack.append(nbr.GetIdx())
    return group


def normalize(v): 
    return v / np.linalg.norm(v)
    
def rotationMatrix(axis, theta):
    axis = normalize(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d),     2*(b*d+a*c)],
        [2*(b*c+a*d),     a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c),     2*(c*d+a*b),     a*a+d*d-b*b-c*c]
    ])

def rotateAtom(mol, mol_coords, aid, origin, axis, theta):
    
    R = rotationMatrix(axis, theta)
    
    atom_coords = mol_coords[aid]
    
    v = np.array(atom_coords) - origin
    
    new_v = R @ v + origin
    
    mol_coords[aid] = new_v * unit.nanometer
    
    return mol_coords 

def setTorsion(mol, mol_coords, a1, a2, a3, a4, target_angle_deg):
    current = GetDihedralRad(mol.GetConformer(0), a1, a2, a3, a4)
    target = np.deg2rad(target_angle_deg)
    theta = (target - current + np.pi) % (2 * np.pi) - np.pi
        
    p2 = np.array(mol_coords[a2])
    p3 = np.array(mol_coords[a3])
    axis = p3 - p2
    atoms_to_move = getRotatingAtoms(mol, a2, a3)

    for atom in atoms_to_move:
        mol_coords = rotateAtom(mol, mol_coords, atom, p2, axis, theta)
    return mol_coords


class RandomRotatableBondMove(Move):
    """RandomRotatableBondMove that provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.
    Current methods calculates the properties needed to randomly rotate a rotatable
    bond of a structure in the NCMC simulation and then executes a rotation of
    a user-specified 'rotatable' bond in the designated small molecule by a random
    angle of rotation 'theta'.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    prmtop: str
        String specifying the name of the parameter file.
    inpcrd: str
        String specifying the name of the restart file.
    smiles: str
        smiles of the ligand to be enhanced
    dihedral_atoms: list
        List containing the four atomnames describing the rotatable bond of interest.
    alch_list: list
        List containing atomnames corresponding to the atoms considered to be
        in the alchemical region during NCMC move.
    resname : str
        String specifying the residue name of the ligand.

    Attributes
    ----------
    structure : parmed.Structure
        The structure of the ligand or selected atoms to be rotated.
    atom_indices : list
        Atom indicies of atoms present in the alchemical region of the ligand.
    atom_indices_ligand :
        Atom indicies of all atoms present in the ligand.
    dihedral_atoms : list
        Atomnames corresponding to the atoms describing the rotatable bond
    positions : numpy.array
        Ligands positions in XYZ coordinates. This should be updated
        every iteration.
    molecule : OEMol
        OEChem Molecule describing the ligand

    Examples
    --------
    >>> from blues.move import RandomRotatableBondMove
    >>> ligand = RandomRotatableBondMove(structure, prmtopFileName, inpcrdFileName, smiles, dihedral_atoms, alch_list, 'LIG')
    """

    def __init__(self, structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname='LIG', null=False):
        self.structure = structure
        self.resname = resname
        self.atom_indices, self.atom_indices_ligand = self.getAtomIndices(structure, resname, alch_list)
        self.smiles = smiles
        self.dihedral_atoms = dihedral_atoms
        self.dihedral_indices = self.getDihedralIndices(structure, resname, dihedral_atoms)
        self.adj_dihedral_indices = [i - self.getFirstAtomIndex(structure, resname) for i in self.dihedral_indices] 
        self.positions = structure[self.atom_indices_ligand].positions
        self.molecule = self.getMoleculeFromPDB(pdb,smiles,resname)
        self.null = null
        self.before_angle = None

        conf = self.molecule.GetConformer()


    def beforeMove(self, context):
        self.before_angle = self.getAngle(context)
        return context


    def getMoleculeFromPDB(self, pdb, smiles, resname):
        topology = topology_from_pdb(
            pdb,
            additional_definitions=[
                ResidueDefinition.anon_from_smiles(
                    smiles
                )
            ]
        )
        for m in topology.molecules:
            for a in m.atoms:
                if a.metadata['residue_name'] == resname:
                    if m.conformers[0].units == 'angstrom':
                        m.conformers[0] = pint.Quantity(m.conformers[0]/10, 'nanometer')
                        
                    return m.to_rdkit()

    def getAngle(self, context):
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices_ligand].value_in_unit(unit.nanometer)
        conf = self.molecule.GetConformer()
        conf.SetPositions(np.array(self.positions))
        a1, a2, a3, a4 = tuple(self.adj_dihedral_indices)
        theta =  GetDihedralDeg(self.molecule.GetConformer(), a1, a2, a3, a4)
        return theta

    def getAtomIndices(self, structure, resname, alch_list):
        """
        Get atom indices of a ligand from ParmEd Structure.
        Arguments
        ---------
        structure: parmed.Structure
            ParmEd Structure object of the atoms to be moved.
        resname : str
            String specifying the resiue name of the ligand.
        alch_list: list
            List containing atomnames corresponding to the atoms considered to be
            in the alchemical region during NCMC move.
        Returns
        -------
        atom_indices : list of ints
            list of atoms in the coordinate file matching alch_list
        atom_indices_ligand :
            Atom indicies of all atoms present in the ligand.
        """
        atom_indices = []
        atom_indices_ligand = []
        topology = structure.topology
        for atom in topology.atoms():
           if str(resname) in atom.residue.name:
              atom_indices_ligand.append(atom.index)
              print("LIG:", resname, atom.name, atom.index)
              if atom.name in alch_list:
                  atom_indices.append(atom.index)
                  print("ALCH:", resname, atom.name, atom.index)

        return atom_indices, atom_indices_ligand

    def getDihedralIndices(self, structure, resname, dihedral_atoms):
        atom_indices = []
        topology = structure.topology
        for atom_name in self.dihedral_atoms:
            for atom in topology.atoms():   
                if str(resname) in atom.residue.name and atom.name == atom_name:
                    atom_indices.append(atom.index)
        return atom_indices

    def getFirstAtomIndex(self, structure, resname):
        for res in structure.residues:
            if res.name == resname: 
                return res.atoms[0].idx
                
    def move(self, context):
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices_ligand].value_in_unit(unit.nanometer)

        conf = self.molecule.GetConformer()
        conf.SetPositions(np.array(self.positions))

        a1, a2, a3, a4 = tuple(self.adj_dihedral_indices)

        theta1 = random.uniform( - math.pi, math.pi ) * 180 / math.pi
        theta0 = self.before_angle
        print('PREVANGLE:', theta1)
        print('NEWANGLE:', theta0)
        
        if self.null:
            print('ANGLEDIFF:', 0)
            return context

        angle_diff = ((theta0 - theta1 + 180) % 360 - 180)
        print('ANGLEDIFF:', angle_diff)

        mol_coords = positions[self.atom_indices_ligand]

        mol_coords = setTorsion(self.molecule, mol_coords, a1, a2, a3, a4, theta1)


        for index, atomidx in enumerate(self.atom_indices_ligand):
            positions[atomidx] = numpy.array(mol_coords[index])*unit.nanometers


        context.setPositions(positions)
        self.positions = positions[self.atom_indices_ligand]
        return context

class StateCutoffRotatableBondMove(RandomRotatableBondMove):
    """RandomRotatableBondMove that provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.
    Current methods calculates the properties needed to randomly rotate a rotatable
    bond of a structure in the NCMC simulation and then executes a rotation of
    a user-specified 'rotatable' bond in the designated small molecule by a random
    angle of rotation 'theta'.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    prmtop: str
        String specifying the name of the parameter file.
    inpcrd: str
        String specifying the name of the restart file.
    smiles: str
        smiles of the ligand to be enhanced
    dihedral_atoms: list
        List containing the four atomnames describing the rotatable bond of interest.
    alch_list: list
        List containing atomnames corresponding to the atoms considered to be
        in the alchemical region during NCMC move.
    resname : str
        String specifying the residue name of the ligand.

    Attributes
    ----------
    structure : parmed.Structure
        The structure of the ligand or selected atoms to be rotated.
    atom_indices : list
        Atom indicies of atoms present in the alchemical region of the ligand.
    atom_indices_ligand :
        Atom indicies of all atoms present in the ligand.
    dihedral_atoms : list
        Atomnames corresponding to the atoms describing the rotatable bond
    positions : numpy.array
        Ligands positions in XYZ coordinates. This should be updated
        every iteration.
    molecule : OEMol
        OEChem Molecule describing the ligand

    Examples
    --------
    >>> from blues.move import RandomRotatableBondMove
    >>> ligand = RandomRotatableBondMove(structure, prmtopFileName, inpcrdFileName, smiles, dihedral_atoms, alch_list, 'LIG')
    """

    def __init__(self, structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname='LIG', nstates=1, theta=0, null=False):
        super().__init__(structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname, null)

        self.nstates=nstates
        self.theta=theta
        self.peak_locations = self.getPeakLocations(self.theta, self.nstates)

        conf = self.molecule.GetConformer()

    def wrap180(self, angle):
        return ((angle + 180) % 360) - 180


    def getPeakLocations(self, theta, N):
        step = 360.0 / N
        return [self.wrap180(theta + k * step) for k in range(N)]

    def getSigma(self, N, alpha=0.12):
        r = 180.0 / N
        return r / np.sqrt(-2 * np.log(alpha))


    def getState(self, theta, peaks):
        peaks = np.asarray(peaks)
        d = np.angle(np.exp(1j*(theta - peaks)))
        return np.argmin(np.abs(d))


    def getNewAngle(self, theta0, context):
        newpeak = random.randint(0,self.nstates-1)
        
        currpeak = self.getState(theta0, self.peak_locations)

        delta_mu = self.peak_locations[newpeak] - self.peak_locations[currpeak]

        theta_new =  theta0 + delta_mu
        return self.wrap180(theta_new)

    def move(self, context):
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices_ligand].value_in_unit(unit.nanometer)

        conf = self.molecule.GetConformer()
        conf.SetPositions(np.array(self.positions))

        a1, a2, a3, a4 = tuple(self.adj_dihedral_indices)

        #prev_angle = GetDihedralDeg(self.molecule.GetConformer(), a1, a2, a3, a4)
        prev_angle = self.before_angle
        print('theta1:', prev_angle)

        if self.null:
            print('1ANGLEDIFF:', 0)
            return context
        else:
            new_angle = self.getNewAngle(prev_angle, self.peak_locations)
            print('theta2:', new_angle)

        angle_diff = self.wrap180(prev_angle - new_angle)
        print('ANGLEDIFF:', angle_diff)

        mol_coords = positions[self.atom_indices_ligand]

        mol_coords = setTorsion(self.molecule, mol_coords, a1, a2, a3, a4, new_angle)


        for index, atomidx in enumerate(self.atom_indices_ligand):
            positions[atomidx] = numpy.array(mol_coords[index])*unit.nanometers

        context.setPositions(positions)
        self.positions = positions[self.atom_indices_ligand]
        return context

class GaussianResponsibilityRotatableBondMove(RandomRotatableBondMove):
    """RandomRotatableBondMove that provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.
    Current methods calculates the properties needed to randomly rotate a rotatable
    bond of a structure in the NCMC simulation and then executes a rotation of
    a user-specified 'rotatable' bond in the designated small molecule by a random
    angle of rotation 'theta'.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    prmtop: str
        String specifying the name of the parameter file.
    inpcrd: str
        String specifying the name of the restart file.
    smiles: str
        smiles of the ligand to be enhanced
    dihedral_atoms: list
        List containing the four atomnames describing the rotatable bond of interest.
    alch_list: list
        List containing atomnames corresponding to the atoms considered to be
        in the alchemical region during NCMC move.
    resname : str
        String specifying the residue name of the ligand.

    Attributes
    ----------
    structure : parmed.Structure
        The structure of the ligand or selected atoms to be rotated.
    atom_indices : list
        Atom indicies of atoms present in the alchemical region of the ligand.
    atom_indices_ligand :
        Atom indicies of all atoms present in the ligand.
    dihedral_atoms : list
        Atomnames corresponding to the atoms describing the rotatable bond
    positions : numpy.array
        Ligands positions in XYZ coordinates. This should be updated
        every iteration.
    molecule : OEMol
        OEChem Molecule describing the ligand

    Examples
    --------
    >>> from blues.move import RandomRotatableBondMove
    >>> ligand = RandomRotatableBondMove(structure, prmtopFileName, inpcrdFileName, smiles, dihedral_atoms, alch_list, 'LIG')
    """

    def __init__(self, structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname='LIG', nstates=1, theta=0, peak_locations=[], sigma=None, null=False):
        super().__init__(structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname, null)

        if peak_locations is None:
            self.nstates=nstates
            self.theta=theta
            self.peak_locations = self.getPeakLocations(self.theta, self.nstates)
        else:
            self.peak_locations = peak_locations
            self.nstates = len(peak_locations)
        if sigma is None:
            self.sigma = self.getSigma(self.nstates)
        else:
            self.sigma = sigma

        conf = self.molecule.GetConformer()


    def getState(self, theta, peaks):
        peaks = np.asarray(peaks)
        d = np.angle(np.exp(1j*(theta - peaks)))
        return np.argmin(np.abs(d))


    def wrapped_gaussian(self, theta, means, sigma):
        means = np.array(means)
        diff = (theta - means + 180) % 360 - 180
        return np.exp(-0.5 * (diff / sigma)**2)


    def mixture_weight(self, theta, stateid, means, sigma):
        densities = self.wrapped_gaussian(theta, means, sigma)
        return densities[stateid] / densities.sum()

    def get_hastings(self, theta1, theta2, stateid1, stateid2, means, sigma):
        forward = self.mixture_weight(theta1, stateid1, means, sigma)
        reverse = self.mixture_weight(theta2, stateid2, means, sigma)
        return reverse/forward

    def move(self, context):
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices_ligand].value_in_unit(unit.nanometer)

        conf = self.molecule.GetConformer()
        conf.SetPositions(np.array(self.positions))

        a1, a2, a3, a4 = tuple(self.adj_dihedral_indices)

        #theta1 = GetDihedralDeg(self.molecule.GetConformer(), a1, a2, a3, a4)
        theta1 = self.before_angle
        print('theta1:', theta1)

        densities = self.wrapped_gaussian(theta1, self.peak_locations, self.sigma)
        weights = densities / densities.sum()

        stateid1 = np.random.choice(len(self.peak_locations), p=weights)

        stateid2 = np.random.choice(len(self.peak_locations))

        theta2 = (theta1 + (self.peak_locations[stateid2] - self.peak_locations[stateid1])) % 360
        print('theta2:', theta2)

        log_hastings = np.log(self.get_hastings(theta1, theta2, stateid1, stateid2, self.peak_locations, self.sigma))
        integrator = context.getIntegrator()
        integrator.setGlobalVariableByName("log_hastings", log_hastings)


        if self.null:
            print('1ANGLEDIFF:', 0)
            return context

        angle_diff = ((theta1 - theta2 + 180) % 360 - 180)
        print('ANGLEDIFF:', angle_diff)

        mol_coords = positions[self.atom_indices_ligand]

        mol_coords = setTorsion(self.molecule, mol_coords, a1, a2, a3, a4, theta2)

        for index, atomidx in enumerate(self.atom_indices_ligand):
            positions[atomidx] = numpy.array(mol_coords[index])*unit.nanometers

        context.setPositions(positions)
        self.positions = positions[self.atom_indices_ligand]
        return context

class MixedGaussianRotatableBondMove(RandomRotatableBondMove):
    """RandomRotatableBondMove that provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.
    Current methods calculates the properties needed to randomly rotate a rotatable
    bond of a structure in the NCMC simulation and then executes a rotation of
    a user-specified 'rotatable' bond in the designated small molecule by a random
    angle of rotation 'theta'.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    prmtop: str
        String specifying the name of the parameter file.
    inpcrd: str
        String specifying the name of the restart file.
    smiles: str
        smiles of the ligand to be enhanced
    dihedral_atoms: list
        List containing the four atomnames describing the rotatable bond of interest.
    alch_list: list
        List containing atomnames corresponding to the atoms considered to be
        in the alchemical region during NCMC move.
    resname : str
        String specifying the residue name of the ligand.

    Attributes
    ----------
    structure : parmed.Structure
        The structure of the ligand or selected atoms to be rotated.
    atom_indices : list
        Atom indicies of atoms present in the alchemical region of the ligand.
    atom_indices_ligand :
        Atom indicies of all atoms present in the ligand.
    dihedral_atoms : list
        Atomnames corresponding to the atoms describing the rotatable bond
    positions : numpy.array
        Ligands positions in XYZ coordinates. This should be updated
        every iteration.
    molecule : OEMol
        OEChem Molecule describing the ligand

    Examples
    --------
    >>> from blues.move import RandomRotatableBondMove
    >>> ligand = RandomRotatableBondMove(structure, prmtopFileName, inpcrdFileName, smiles, dihedral_atoms, alch_list, 'LIG')
    """

    def __init__(self, structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname='LIG', nstates=1, theta=0, peak_locations=None, sigma=None, null=False):

        super().__init__(structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname, null)

        if peak_locations is None:
            self.nstates=nstates
            self.theta=theta
            self.peak_locations = self.getPeakLocations(self.theta, self.nstates)
        else:
            self.peak_locations = peak_locations
            self.nstates = len(peak_locations)
        if sigma is None:
            self.sigma = self.getSigma(self.nstates)
        else:
            self.sigma = sigma

        conf = self.molecule.GetConformer()


    def wrap180(self, angle):
        return ((angle + 180) % 360) - 180

    def getPeakLocations(self, theta, N):
        step = 360.0 / N
        return [self.wrap180(theta + k * step) for k in range(N)]

    def getSigma(self, N, alpha=0.12):
        r = 180.0 / N
        return r / np.sqrt(-2 * np.log(alpha))

    def getState(self, theta, peaks):
        peaks = np.asarray(peaks)
        d = np.angle(np.exp(1j*(theta - peaks)))
        return np.argmin(np.abs(d))
        
    def wrappedGaussian(self, theta, mu, sigma, n_wraps=1):
        total = 0.0
        for k in range(int(-n_wraps), int(n_wraps + 1)):
            total += norm.pdf(theta, mu + 360.0*k, sigma)
        return total

    def wrappedGMMDensity(self, theta,  means, n_wraps=1):
        return sum(1/len(means) * self.wrappedGaussian(theta, mu, self.sigma, n_wraps)
               for mu in means)

    def getAngle(self, context):
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices_ligand].value_in_unit(unit.nanometer)
        conf = self.molecule.GetConformer()
        conf.SetPositions(np.array(self.positions))
        a1, a2, a3, a4 = tuple(self.adj_dihedral_indices)
        theta =  GetDihedralDeg(self.molecule.GetConformer(), a1, a2, a3, a4)
        return theta


    def move(self, context):
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices_ligand].value_in_unit(unit.nanometer)

        conf = self.molecule.GetConformer()
        conf.SetPositions(np.array(self.positions))

        a1, a2, a3, a4 = tuple(self.adj_dihedral_indices)

        theta0 = self.before_angle
        print("theta0:", theta0)

        stateid1 = np.random.randint(len(self.peak_locations))
        theta1 = np.random.normal(self.peak_locations[stateid1], self.sigma)
        print("theta1:", theta1)
        print("stateid1:", stateid1)

        q_old = self.wrappedGMMDensity(theta0, self.peak_locations)
        q_new = self.wrappedGMMDensity(theta1, self.peak_locations)

        log_hastings = np.log(q_old / q_new)
        integrator = context.getIntegrator()
        integrator.setGlobalVariableByName("log_hastings", log_hastings)
        print(integrator.getGlobalVariableByName('log_hastings'))

        if self.null:
            print('ANGLEDIFF:', 0)
            return context

        angle_diff = ((theta0 - theta1 + 180) % 360 - 180)
        print('ANGLEDIFF:', angle_diff)

        mol_coords = positions[self.atom_indices_ligand]

        mol_coords = setTorsion(self.molecule, mol_coords, a1, a2, a3, a4, theta1)

        for index, atomidx in enumerate(self.atom_indices_ligand):
            positions[atomidx] = numpy.array(mol_coords[index])*unit.nanometers

        context.setPositions(positions)
        self.positions = positions[self.atom_indices_ligand]

        return context


class MixedGaussianMeanDisplacementRotatableBondMove(MixedGaussianRotatableBondMove):
    """RandomRotatableBondMove that provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.
    Current methods calculates the properties needed to randomly rotate a rotatable
    bond of a structure in the NCMC simulation and then executes a rotation of
    a user-specified 'rotatable' bond in the designated small molecule by a random
    angle of rotation 'theta'.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    prmtop: str
        String specifying the name of the parameter file.
    inpcrd: str
        String specifying the name of the restart file.
    smiles: str
        smiles of the ligand to be enhanced
    dihedral_atoms: list
        List containing the four atomnames describing the rotatable bond of interest.
    alch_list: list
        List containing atomnames corresponding to the atoms considered to be
        in the alchemical region during NCMC move.
    resname : str
        String specifying the residue name of the ligand.

    Attributes
    ----------
    structure : parmed.Structure
        The structure of the ligand or selected atoms to be rotated.
    atom_indices : list
        Atom indicies of atoms present in the alchemical region of the ligand.
    atom_indices_ligand :
        Atom indicies of all atoms present in the ligand.
    dihedral_atoms : list
        Atomnames corresponding to the atoms describing the rotatable bond
    positions : numpy.array
        Ligands positions in XYZ coordinates. This should be updated
        every iteration.
    molecule : OEMol
        OEChem Molecule describing the ligand

    Examples
    --------
    >>> from blues.move import RandomRotatableBondMove
    >>> ligand = RandomRotatableBondMove(structure, prmtopFileName, inpcrdFileName, smiles, dihedral_atoms, alch_list, 'LIG')
    """

    def __init__(self, structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname='LIG', nstates=1, theta=0, peak_locations=None, sigma=None, null=False):
        super().__init__(structure, xml, pdb, dihedral_atoms, alch_list, smiles, resname, nstates, theta, peak_locations, sigma, null)

    def wrappedGaussianPerPeak(self, theta, means, sigma, n_wraps=1):
        return np.array([self.wrappedGaussian(theta, mu, sigma, n_wraps) for mu in means])

    def move(self, context):
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        self.positions = positions[self.atom_indices_ligand].value_in_unit(unit.nanometer)

        conf = self.molecule.GetConformer()
        conf.SetPositions(np.array(self.positions))

        a1, a2, a3, a4 = tuple(self.adj_dihedral_indices)

        theta0 = self.before_angle

        d_fwd = self.wrappedGaussianPerPeak(theta0, self.peak_locations, self.sigma)
        w_fwd = d_fwd / d_fwd.sum()
        stateid0 = np.random.choice(len(self.peak_locations), p=w_fwd)

        stateid1 = np.random.randint(len(self.peak_locations))

        theta1 = (theta0 + (self.peak_locations[stateid1] - self.peak_locations[stateid0])) % 360
        print("theta0:", theta0)
        print("stateid0:", stateid0)

        print("theta1:", theta1)
        print("stateid1:", stateid1)

        d_rev = self.wrappedGaussianPerPeak(theta1,self.peak_locations, self.sigma)
        w_rev = d_rev / d_rev.sum()
        log_hastings = np.log(w_rev[stateid1] / w_fwd[stateid0])
        integrator = integrator = context.getIntegrator()
        integrator.setGlobalVariableByName("log_hastings", log_hastings)
        print(integrator.getGlobalVariableByName('log_hastings'))

        if self.null:
            return context

        angle_diff = ((theta0 - theta1 + 180) % 360 - 180)
        print('ANGLEDIFF:', angle_diff)

        mol_coords = positions[self.atom_indices_ligand]

        mol_coords = setTorsion(self.molecule, mol_coords, a1, a2, a3, a4, theta1)

        for index, atomidx in enumerate(self.atom_indices_ligand):
            positions[atomidx] = numpy.array(mol_coords[index])*unit.nanometers

        context.setPositions(positions)
        self.positions = positions[self.atom_indices_ligand]
        return context
