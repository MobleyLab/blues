import simtk.unit as unit
import numpy as np
import mdtraj as md
from mdtraj.formats.xyzfile import XYZTrajectoryFile
from mdtraj.utils import in_units_of
from blues.moldart.lin_math import adjust_angle
from blues.moldart.lin_math import getRotTrans
from blues.moves import RandomLigandRotationMove
from blues.moldart.darts import makeDihedralDifferenceDf
import chemcoord as cc
import copy
import tempfile
from blues.moldart.chemcoord import give_cartesian_edit
from blues.moldart.darts import makeDartDict, checkDart, makeDihedralDifferenceDf
from blues.moldart.boresch import add_rmsd_restraints, add_boresch_restraints
import parmed
from blues.integrators import AlchemicalExternalLangevinIntegrator, AlchemicalNonequilibriumLangevinIntegrator
from blues.moldart.rigid import createRigidBodies, resetRigidBodies
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from scipy.integrate import cumtrapz
from openeye import oechem
import random

import logging
logger = logging.getLogger(__name__)


def AtomsInSameRing(atomA, atomB):

    if not atomA.IsInRing() or not atomB.IsInRing():
        return False

    if atomA == atomB:
        return True

    firstpath = [a for a in oechem.OEShortestPath(atomA, atomB, oechem.OEAtomIsInChain())]
    firstpathlength = len(firstpath)

    if firstpathlength == 2:
        return True  # neighbors

    if firstpathlength == 0:
        return False  # not is same ring system

    smallestA = oechem.OEAtomGetSmallestRingSize(atomA)
    smallestB = oechem.OEAtomGetSmallestRingSize(atomB)

    if firstpathlength > smallestA and firstpathlength > smallestB:
        return False  # too far away

    # try to find the second shortest different path
    excludepred = ChainAtomOrAlreadyTraversed(firstpath[1:-1])
    secondpath = [a for a in oechem.OEShortestPath(atomA, atomB, excludepred)]
    secondpathlength = len(secondpath)

    if secondpathlength == 0:
        return False  # can not be in the same ring

    if secondpathlength > smallestA and secondpathlength > smallestB:
        return False  # too far away

    sumringsize = len(firstpath) + len(secondpath) - 2
    if sumringsize > smallestA and sumringsize > smallestB:
        return False

    inringA = oechem.OEAtomIsInRingSize(atomA, sumringsize)
    inringB = oechem.OEAtomIsInRingSize(atomB, sumringsize)
    return inringA and inringB

class ChainAtomOrAlreadyTraversed(oechem.OEUnaryAtomPred):
    def __init__(self, exclude):
        oechem.OEUnaryAtomPred.__init__(self)
        self.exclude = exclude

    def __call__(self, atom):
        if not atom.IsInRing():
            return False
        return (atom in self.exclude)

    def CreateCopy(self):
        return ChainAtomOrAlreadyTraversed(self.exclude).__disown__()

def findCentralAngle(buildlist):
    connection_list = []
    index_list = [0,1,2]
    for i in buildlist.index.values[:3]:
        connection_list.append(buildlist['b'][i])
    #count the number of bonds to the first buildatom
    counts = connection_list.count(buildlist.index.values[0])
    #if 2 then the first atom is the center atom
    if counts == 2:
        center_index = 0
    #otherwise the second atom is the center atom
    else:
        center_index = 1
    index_list.pop(center_index)
    vector_list = []
    for index in index_list:
        vector_list.append([index, center_index])
    return vector_list



class MolDartMove(RandomLigandRotationMove):
    """
    Class for performing molecular darting (moldarting) moves during an NCMC simulation.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    pdb_files: list of str:
        List of paths to pdb files with the same system as the structure,
        whose ligand internal coordinates will be used as the darts for
        internal coordinate darting.
    fit_atoms: list of ints
        A list of ints corresponding to the atoms of the protein to be fitted,
        to remove rotations/translations changes from interfering
        with the darting procedure.
    resname : str, optional, default='LIG'
        String specifying the residue name of the ligand.
    dihedral_cutoff: float, optional, default=0.5
        Minimum cutoff to use for the dihedral dart cutoffs (in radians).
    distance_cutoff: float, optional, default=5.5
        Minimum cutoff to use for the translational cutoffs
    rotation_cutoff: float, optional, default=29.0
        Minimum cutoff to use for the rotation dart cutoffs (in degrees).
    dart_buffer: float, optional, default=0.9
    dart_region_order: list of str, default=['translation', 'dihedral',  'rotation']
        List corresponding to the order the darts separating poses should be
        constructed in.
    transition_matrix: None or nxn numpy.array, optional, default=None
        The transition matrix to define transition probabilities between darts.
        If None, this assumes a uniform transition matrix with the self transition
        probability set to 0 (zeros across the diagonal).
        Otherwise a nxn matrix must be passed, where n == len(pdb_files), where entry
        Mij corresponds to the probability when in the pdb_files[i] dart to dart to
        the pdb_files[j] dart.
    rigid_ring:boolean, optional, default=False:
        If True will rigidfy any dihedrals of molecules that are bonded to
        either a double bonded or ring atom, which are otherwise extremely
        senstive to these dihedral changes.
    rigid_move: boolean, optional, default=False:
        If True, will ignore internal coordinate changes while darting
        and will effectively perform a rigid body rotation between darts
        instead.
    rigid_darts: {None, 'rigid_darts', 'rigid_ring', 'rigid_molecule'}, default=None
        Chooses how to handle the dihedrals of the molecule when darting.
            `rigid_ring`will rigidfy any dihedrals of molecules that are bonded to
            either a double bonded or ring atom, which are otherwise extremely
            senstive to these dihedral changes.
            `rigid_molecule will ignore internal coordinate changes while darting
            and will effectively perform a rigid body rotation between darts
            instead.
            `rigid_darts` will only move bonds that exhibit significant deviations
            that are non-ring atoms.
            If None then all dihedrals will be darted according to the difference
            between the current pose and the selected pose.
    freeze_waters: int, optional, default=0
        The number of waters to set the mass to 0. If this value is non-zero,
        this sets all waters besides the nearest freeze_waters waters from the ligand
        masses to 0, preventing them from moving. These waters are updated
        at the start of the iteration and after the move takes place.
    freeze_protein: False or float, default=False
        If not False, sets the masses of all protein atoms beyond the freeze_protein
        cutoff distance (in angstroms) based on ALL poses in pdb_files. This
        is different from the the option in Simulation as it takes into account the
        distance from the provided poses and the use of these two should be
        mutually exculsive.
    restraints: {'rmsd', 'boresch', None}, optional, default='rmsd'
        Applies restrains so that the ligand remains close to the specified poses
        during the course of the simulation. If this is not used, the probability
        of darting can be greatly diminished, since the ligand can be much more
        mobile in the binding site with it's interactions turned off.
        'rmsd' specifies the use of RMSD restraints on the selected receptor and ligand atoms,
        'boresch' specifies the use of boresch-style restraints, and None causes no restraints
        to be used.
    restrained_receptor_atom: list, optional, default=None
        The three atoms of the receptor to use for boresch style restraints.
        If unspecified uses a random selection through a heuristic process
        via Yank. This is only necessary when `restraints==True`
    K_r: float, optional, default=10
        The value of the bond restraint portion of the boresch restraints
        (given in units of kcal/(mol*angstrom**2)).
        Only used if restraints=True.
    K_angle: float, optional, default=10
        The value of the angle and dihedral restraint portion of the boresh restraints
        (given in units of kcal/(mol*rad**2)).
        Only used if restraints=True.
    lambda_restraints: str, optional, default='max(0, 1-(1/0.10)*abs(lambda-0.5))'
        The Lepton-compatible string specifying how the restraint lambda parameters
        are handled.


    """
    def __init__(self, structure, pdb_files, fit_atoms, resname='LIG',
        transition_matrix=None,
         dihedral_cutoff=0.5, distance_cutoff=5.5, rotation_cutoff=29.0, dart_buffer=0.9,
        dart_region_order = ['translation', 'dihedral',  'rotation'],
        rigid_darts='rigid_darts',
        rigid_ring=False, rigid_move=False, freeze_waters=0, freeze_protein=False,
        restraints=None, restrained_receptor_atoms=None,
        receptor_cutoff=0.5,
        K_r=10, K_angle=10, K_RMSD=0.6, RMSD0=2,
        rigid_body=False,
        centroid_darting=True,
        lambda_restraints='max(0, 1-(1/0.10)*abs(lambda-0.5))'
        ):
        super(MolDartMove, self).__init__(structure, resname)
        #md trajectory representation of only the ligand atoms
        self.pdb_files = pdb_files
        self.trajs = [md.load(traj) for traj in pdb_files]
        self.restrained_receptor_atoms = []
        if restrained_receptor_atoms is None:
            for traj in self.trajs:
                ca_atoms = traj.top.select('name CA and protein')
                receptor_atoms = md.compute_neighbors(traj, cutoff=receptor_cutoff, query_indices=self.atom_indices, haystack_indices=ca_atoms)
                self.restrained_receptor_atoms.append(receptor_atoms)
        elif all(isinstance(item, int) for item in restrained_receptor_atoms):
            self.restrained_receptor_atoms = [restrained_receptor_atoms for i in range(len(pdb_files))]

        elif all(isinstance(item, list) for item in restrained_receptor_atoms) and len(restrained_receptor_atoms) == len(pdb_files):
            self.restrained_receptor_atoms = restrained_receptor_atoms
            #exit()
        self.binding_mode_traj = []
        #positions of only the ligand atoms
        self.binding_mode_pos = []
        #fit atoms are the atom indices which should be fit to to remove rot/trans changes
        self.fit_atoms = fit_atoms
        if 1:
            if fit_atoms is None:
                for traj in self.trajs:
                    ca_atoms = traj.top.select('name CA and protein')
                    receptor_atoms = md.compute_neighbors(traj, cutoff=receptor_cutoff, query_indices=self.atom_indices, haystack_indices=ca_atoms)
                    self.fit_atoms.append(receptor_atoms)
            elif all(isinstance(item, int) for item in fit_atoms):
                self.fit_atoms = [fit_atoms for i in range(len(pdb_files))]

            elif all(isinstance(item, list) for item in fit_atoms) and len(fit_atoms) == len(pdb_files):
                self.fit_atoms = fit_atoms

        #chemcoord cartesian xyz of the ligand atoms
        self.internal_xyz = []
        #chemcoord internal zmatrix representation of the ligand atoms
        self.internal_zmat = []
        self.buildlist = None
        self.rigid_darts = rigid_darts
        #self.rigid_move = bool(rigid_move)
        #self.rigid_ring = bool(rigid_ring)
        #ref traj is the reference md trajectory used for superposition
        self.ref_traj = None
        #sim ref corresponds to the simulation positions
        self.sim_ref = None
        #dictionary of how to break up darting regions
        self.darts = None
        #tracks how many times moves are attempted and when ligand is within darting regions
        self.moves_attempted = 0
        self.times_within_dart = 0
        #if a subset of waters are to be frozen the number is specified here
        self.freeze_waters = freeze_waters
        #if restraints are used to keep ligand within darting regions specified here
        self.restraints = restraints
        self.freeze_protein = freeze_protein
        self.K_r = K_r
        self.K_angle = K_angle
        self.K_RMSD = K_RMSD
        self.RMSD0 = RMSD0
        self.lambda_restraints = lambda_restraints
        self.rigid_body = rigid_body
        self.centroid_darting = centroid_darting

        #find pdb inputs inputs
        if len(pdb_files) <= 1:
            raise ValueError('Should specify at least two pdbs in pdb_files for darting to be beneficial')
        self.dart_groups = list(range(len(pdb_files)))
        #chemcoords reads in xyz files only, so we need to use mdtraj
        #to get the ligand coordinates in an xyz file
        if 0:
            with tempfile.NamedTemporaryFile(suffix='.xyz') as t:
                fname = t.name
                traj = md.load(pdb_files[0]).atom_slice(self.atom_indices)
                xtraj = XYZTrajectoryFile(filename=fname, mode='w')
                xtraj.write(xyz=in_units_of(traj.xyz, traj._distance_unit, xtraj.distance_unit),
                            types=[i.element.symbol for i in traj.top.atoms] )
                xtraj.close()
                xyz = cc.Cartesian.read_xyz(fname)
                self.buildlist = xyz.get_construction_table()
                if self.rigid_darts is not None:
                    ring_atoms = []
                    from openeye import oechem
                    ifs = oechem.oemolistream()
                    ifs.open(fname)
                    #double_bonds = []
                    h_list = []
                    for mol in ifs.GetOEGraphMols():
                        oechem.OEFindRingAtomsAndBonds(mol)
                        for atom in mol.GetAtoms():
                            if atom.IsInRing():
                            #if not atom.IsRotor():
                                ring_atoms.append(atom.GetIdx())
                            if atom.IsHydrogen():
                                h_list.append(atom.GetIdx())
                        #bgn_idx = [bond.GetBgnIdx() for bond in mol.GetBonds() if bond.GetEndIdx() in ring_atoms]
                        #end_idx = [bond.GetEndIdx() for bond in mol.GetBonds() if bond.GetBgnIdx() in ring_atoms]
                    rigid_atoms = ring_atoms
                    #select all atoms that are bonded to a ring/double bond atom
                    angle_ring_atoms = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in rigid_atoms]
                    for mol in ifs.GetOEGraphMols():
                        for atom in mol.GetAtoms():
                            if atom.IsHydrogen():
                                h_list.append(atom.GetIdx())
                    #self.dihedral_ring_atoms = list(set(angle_ring_atoms + h_list))
                    self.dihedral_ring_atoms = list(set(rigid_atoms + h_list))
                    #self.dihedral_ring_atoms = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in rigid_atoms]
        else:
            self.buildlist = self._createBuildlist(pdb_files, self.atom_indices)
            self.traj_dart_dict = self._findDihedralRingAtoms(pdb_files, atom_indices=self.atom_indices, rigid_darts=self.rigid_darts)
            print('self.traj_dart_dict keys1', self.traj_dart_dict)
            print('self.traj_dart_dict keys', list(self.traj_dart_dict.keys()))
            self.dihedral_ring_atoms = self.traj_dart_dict['dihedral_ring_atoms']
            #get the construction table so internal coordinates are consistent between poses


        #use the positions from the structure to be used as a reference for
        #superposition of the rest of poses
        with tempfile.NamedTemporaryFile(suffix='.pdb') as t:
            fname = t.name
            self.structure.save(fname, overwrite=True)
            struct_traj = md.load(fname)
        ref_traj = md.load(pdb_files[0])[0]
        num_atoms = ref_traj.n_atoms
        self.ref_traj = copy.deepcopy(struct_traj)
        #add the trajectory and xyz coordinates to a list
        if 0:
            for j, pdb_file in enumerate(pdb_files):
                traj = copy.deepcopy(self.ref_traj)
                pdb_traj = md.load(pdb_file)[0]
                num_atoms_traj = traj.n_atoms
                num_atoms_pdb = pdb_traj.n_atoms
                #this assumes the ligand follows immeditely after the protein
                #to handle the case when no solvent is present
                num_atoms = min(num_atoms_traj, num_atoms_pdb)
                traj.xyz[0][:num_atoms] = pdb_traj.xyz[0][:num_atoms]
                traj.superpose(reference=ref_traj, atom_indices=fit_atoms,
                    ref_atom_indices=fit_atoms
                    )
                self.binding_mode_traj.append(copy.deepcopy(traj))
                #get internal representation
                self.internal_xyz.append(copy.deepcopy(xyz))
                #take the xyz coordinates from the poses and update
                #the chemcoords cartesian xyz class to match
                for index, entry in enumerate(['x', 'y', 'z']):
                    for i in range(len(self.atom_indices)):
                        sel_atom = self.atom_indices[i]
                        self.internal_xyz[j]._frame.at[i, entry] = self.binding_mode_traj[j].xyz[0][:,index][sel_atom]*10
                self.internal_zmat.append(self.internal_xyz[j].get_zmat(construction_table=self.buildlist))
            #set the binding_mode_pos by taking the self.atom_indices indices
            self.binding_mode_pos = [np.asarray(atraj.xyz[0])[self.atom_indices]*10.0 for atraj in self.binding_mode_traj]
        else:
            self.internal_xyz, self.internal_zmat, self.binding_mode_pos, self.binding_mode_traj = self._createZmat(pdb_files, atom_indices=self.atom_indices, reference_traj=ref_traj)

        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])
        self.sim_ref = copy.deepcopy(self.binding_mode_traj[0])

        self.darts = makeDartDict(self.internal_zmat, self.binding_mode_pos, self.buildlist,
                                dihedral_cutoff=dihedral_cutoff, distance_cutoff=distance_cutoff,
                                rotation_cutoff=rotation_cutoff, dart_buffer=dart_buffer, order=dart_region_order)
        if transition_matrix is None:
            self.transition_matrix = np.ones((len(pdb_files), len(pdb_files)))
            np.fill_diagonal(self.transition_matrix, 0)
        else:
            self.transition_matrix = transition_matrix
        self.transition_matrix = self._checkTransitionMatrix(self.transition_matrix, self.dart_groups)
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
        if np.shape(self.transition_matrix) != (len(pdb_files), len(pdb_files)):
            raise ValueError('Transition matrix should be an nxn matrix, where n is the length of pdb_files')
        #get values that differ significantly
        dihedral_diff_df = makeDihedralDifferenceDf(self.internal_zmat)
        #find which atoms experience signficant changes and allow those to change while darting, other parts of the molecule stay the same
        if self.rigid_darts is not None:
            #find the bonded atoms that are not part of the dihedral ri
#            core = list(set([self.buildlist.at[i, 'b'] for i in dihedral_diff_df['atomnum'].values if i not in self.dihedral_ring_atoms]))
            core = list(set([self.buildlist.at[i, 'b'] for i in dihedral_diff_df['atomnum'].values]))

            #self.only_darts_dihedrals = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in core]
            #self.only_darts_dihedrals = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in core]

            self.only_darts_dihedrals = [i for i in self.darts['dihedral'].keys()]


    def refitPoses(self, current_pose, trajs, fit_atoms, atom_indices):
        #current_pose current trajectory traj

        #binding_mode_pos = [np.asarray(atraj.xyz[0])[self.atom_indices]*10.0 for atraj in self.binding_mode_traj]
        #logger.info("1current_pose {}".format(current_pose))
        #logger.info("1trajs {}".format(trajs))


        binding_mode_traj = [traj.superpose(current_pose, atom_indices=fit_atoms[index], ref_atom_indices=fit_atoms[index]).atom_slice(self.atom_indices) for index, traj in enumerate(trajs)]
        #logger.info("binding_mode_traj1 {}".format(binding_mode_traj))
        #binding_mode_pos = [np.asarray(traj.superpose(current_pose, atom_indices=fit_atoms[index], ref_atom_indices=fit_atoms[index]).xyz[0]) for index, traj in enumerate(self.trajs)]
        binding_mode_pos = [np.asarray(traj.xyz[0])*10.0 for index, traj in enumerate(self.trajs)]
        #logger.info("binding_mode_pos1 {}".format(binding_mode_pos))

        binding_mode_pos = [traj[self.atom_indices] for traj in binding_mode_pos]
        #logger.info("binding_mode_pos2 {}".format(binding_mode_pos))

        return binding_mode_traj, binding_mode_pos


    @classmethod
    def _loadfiles(cls, structure_files, topology):
        try:
            traj = md.load(structure_files, top=topology)
        except TypeError:
            traj = md.load(structure_files)

        return traj
    @classmethod
    def _createZmat(cls, structure_files, atom_indices, topology=None, reference_traj=None, fit_atoms=None):
        """
        Takes a list of structure files and creates xyz and Zmat representations for each of those
        structures using chemcoord and mdtraj

        Parameters
        ----------
        structure_files: list of str
            List corresponding to the path of the structures to create representations of.
        atom_indices: list of ints
            The atom indices of the ligand in the structure files.
        topology: str, optional, default=None
            Path of topology file, if structure_files doesn't contain topology information.
        reference_traj: mdtraj.Trajectory or str, optional, default=None
            Trajectory object, or path to file, containing the reference system to superpose to. If None then
            no fitting occurs.
        fit_atoms: list, optional, default=None
            List of atom indices to be used in fitting the structure_files positions to the
            reference trajectory (if reference_traj is not None)

        Returns
        -------
        internal_xyz: list of pandas.Dataframe
        internal_zmat: list of pandas.Dataframe
        binding_mode_pos: list of np.arrays
            np.arrays corresponding to the positions of the ligand
        binding_mode_traj: list of md.Trajectory
            List of md.Trajectory objects corresponding to the whole system
        """
        #portion to get xyz
        buildlist = cls._createBuildlist(structure_files, atom_indices, topology=topology)
        traj = cls._loadfiles(structure_files[0], topology).atom_slice(atom_indices)

        with tempfile.NamedTemporaryFile(suffix='.xyz') as t:
            fname = t.name
            xtraj = XYZTrajectoryFile(filename=fname, mode='w')
            xtraj.write(xyz=in_units_of(traj.xyz[0], traj._distance_unit, xtraj.distance_unit),
                        types=[i.element.symbol for i in traj.top.atoms] )
            xtraj.close()
            xyz = cc.Cartesian.read_xyz(fname)
        internal_xyz = []
        internal_zmat = []
        binding_mode_traj = []
        #add the trajectory and xyz coordinates to a list
        if isinstance(reference_traj, str):
            reference_traj = cls._loadfiles(reference_traj, topology)
        for j, pdb_file in enumerate(structure_files):
            if reference_traj:
                num_atoms = reference_traj.n_atoms
                traj = cls._loadfiles(pdb_file, topology=topology)[0]
                #traj = md.load(pdb_file)[0]
                num_atoms_pdb = traj.n_atoms
                num_atoms_traj = reference_traj.n_atoms
                num_atoms = min(num_atoms_traj, num_atoms_pdb)
                traj.atom_slice(range(num_atoms), inplace=True)
                if fit_atoms == None:
                    traj.superpose(reference=reference_traj, atom_indices=fit_atoms,
                        ref_atom_indices=fit_atoms
                        )
                else:
                    traj.superpose(reference=reference_traj, atom_indices=fit_atoms[j],
                        ref_atom_indices=fit_atoms[j]
                        )

            else:
                traj = md.load(pdb_file)

            binding_mode_traj.append(copy.deepcopy(traj))
            #get internal representation
            internal_xyz.append(copy.deepcopy(xyz))
            #take the xyz coordinates from the poses and update
            #the chemcoords cartesian xyz class to match
            for index, entry in enumerate(['x', 'y', 'z']):
                for i in range(len(atom_indices)):
                    sel_atom = atom_indices[i]
                    internal_xyz[j]._frame.at[i, entry] = binding_mode_traj[j].xyz[0][:,index][sel_atom]*10
            internal_zmat.append(internal_xyz[j].get_zmat(construction_table=buildlist))
        #set the binding_mode_pos by taking the self.atom_indices indices
        #convert nanometers into meters (to be compatible with chemcoord, which uses angstroms)
        binding_mode_pos = [np.asarray(atraj.xyz[0])[atom_indices]*10.0 for atraj in binding_mode_traj]
        return internal_xyz, internal_zmat, binding_mode_pos, binding_mode_traj

    @classmethod
    def _createBuildlist(cls, structure_files, atom_indices, topology=None):

        with tempfile.NamedTemporaryFile(suffix='.xyz') as t:
            fname = t.name
            traj = cls._loadfiles(structure_files[0], topology).atom_slice(atom_indices)
            xtraj = XYZTrajectoryFile(filename=fname, mode='w')
            xtraj.write(xyz=in_units_of(traj.xyz[0], traj._distance_unit, xtraj.distance_unit),
                        types=[i.element.symbol for i in traj.top.atoms] )
            xtraj.close()
            xyz = cc.Cartesian.read_xyz(fname)
            buildlist = xyz.get_construction_table()
            return buildlist

    @classmethod
    def _getDarts(cls, structure_files, atom_indices, topology=None, reference_traj=None, fit_atoms=None, dihedral_cutoff=0.5, distance_cutoff=5.5, rotation_cutoff=29.0, dart_buffer=0.9, order=['translation', 'dihedral',  'rotation']):
        """
        Parameters
        ----------
        structure_files: list of str
            List corresponding to the path of the structures to create representations of.
        atom_indices: list of ints
            The atom indices of the ligand in the structure files.
        topology: str, optional, default=None
            Path of topology file, if structure_files doesn't contain topology information.
        reference_traj: mdtraj.Trajectory or str, optional, default=None
            Trajectory object, or path to file, containing the reference system to superpose to. If None then
            trajectories are fitted to the first structure in structure_files.
        fit_atoms: list, optional, default=None
            List of atom indices to be used in fitting the structure_files positions to the
            reference trajectory (if reference_traj is not None)
        dihedral_cutoff: float, optional, default=0.5
            Minimum cutoff to use for the dihedral dart cutoffs (in radians).
        distance_cutoff: float, optional, default=5.5
            Minimum cutoff to use for the translational cutoffs
        rotation_cutoff: float, optional, default=29.0
            Minimum cutoff to use for the rotation dart cutoffs (in degrees).
        dart_buffer: float, optional, default=0.9
            Specifies how much further to reduce the translational and rotational darting regions so that the chance of overlap is reduced.
        order: list of strs, optional, default=['translation', 'dihedral', 'rotation']
            The order in which to construct the darting regions. Darting regions will be made sequentially.the
            If all the poses are separated by the darting regions at any point in this process, then no additional
            regions will be made (so order matters).

        Returns
        -------
        dart_storage: dict
            Dict containing the darts associated with `rotation`, `translation` and `dihedral`
            keys that refer to the size of the given dart, if not empty

        """
        if reference_traj is None:
            reference_traj = cls._loadfiles(structure_files[0], topology)
        internal_xyz, internal_zmat, binding_mode_pos, binding_mode_traj = cls._createZmat(structure_files=structure_files,
                    atom_indices=atom_indices,
                    topology=topology,
                    reference_traj=reference_traj,
                    fit_atoms=fit_atoms)
        buildlist = MolDartMove._createBuildlist(structure_files, atom_indices, topology=topology)
        darts = makeDartDict(internal_zmat, binding_mode_pos, buildlist, dihedral_cutoff=dihedral_cutoff, distance_cutoff=distance_cutoff, rotation_cutoff=rotation_cutoff,
                            dart_buffer=dart_buffer, order=order)
        return darts

    @classmethod
    def getMaxRange(cls, dihedral, pose_value=None, density_percent=0.9):
        import matplotlib.pyplot as plt
        #make the range larger than the periodic boundries
        dihedral = np.concatenate((dihedral-2*np.pi, dihedral, dihedral+2*np.pi))

        if density_percent > 1.0:
            raise ValueError('density_percent must be less than 1!')
        pi_range = np.linspace(-2*np.pi, 2*np.pi+np.pi/50.0, num=360, endpoint=True).reshape(-1,1)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(dihedral)
        log_dens = np.exp(kde.score_samples(pi_range))
        #plt.plot(pi_range, (log_dens))
        #plt.xlim(-np.pi,np.pi)
        #plt.show()
        maxes =  argrelextrema(log_dens, np.greater)
        mins = argrelextrema(log_dens, np.less)
        min_value = [pi_range[i] for i in mins]
        max_value = np.array([pi_range[i] for i in maxes]).reshape(-1)
        if 0:
            max_value_check = max_value.tolist()
            remove_index_dict = {}
            for index, i in enumerate(max_value_check):
                remove_index_dict[i] = []
                for index2,j in enumerate(max_value_check):
                    print('mod pi', j%np.pi)
                    if (abs(i-j) < 0.25) or (abs(i-j+np.pi) < 0.25) or (abs(i-j-np.pi) < 0.25):
                        remove_index_dict[i].append(index2)
            keep_values = []
            for key, value in remove_index_dict.items():
                if value:
                    keep_values.append(value[0])
            print('keep_values', keep_values)
            new_values = [max_value_check[i] for i in keep_values]
            max_value = np.array(list(set(new_values))).reshape(-1)
            print('max_value')

        if np.size(min_value) == 0:
            #that means that there is only a maximum peak
            probability = 0
            i_range = 0
            dx = pi_range[1] - pi_range[0]

            for i in range(1,100):
                if probability < density_percent:
                    #find total area to cover density_percent of the density
                    target_range = log_dens[maxes[0][0]-i:maxes[0][0]+i]
                    probability = cumtrapz(target_range, dx=dx)[-1]
                    i_range = i
                else:
                    break

            min_value = np.array([[pi_range[maxes[0][0]-i_range], pi_range[maxes[0][0]+i_range]]])

            region_space = i*dx[0]
            max_return = max_value[0]
        else:
            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots()
            #ax.plot(pi_range, np.exp(log_dens), '-')

            #there are multiple minima
            for index, i in enumerate(max_value):
                #find where the minima lie to figure out where the peak densities are
                #min_range = np.array(min_value).reshape(-1).tolist()
                min_range = [-20]+np.array(min_value).reshape(-1).tolist()+[20]

                for j in range(len(min_range)-1):
                    if np.less(i, min_range[j+1]) and  np.greater(i, min_range[j]):
                        #pick range with smallest region to prevent overlaps
                        min_region = min(abs(i-min_range[j+1]), abs(i-min_range[j]))
                        break
                debug_counter = 0
                if np.greater(pose_value, i-min_region) and np.less(pose_value, i+min_region) or np.greater(pose_value+np.pi, i-min_region) and np.less(pose_value+np.pi, i+min_region):
                #if np.greater(pose_value, i-min_region) and np.less(pose_value, i+min_region):
                    debug_counter =1
                    max_return = i
                    #ax.vlines(max_return, 0, 1, colors='red')

                    pi_range = np.linspace(-2*np.pi, 2*np.pi+np.pi/50.0, num=360, endpoint=True).reshape(-1).tolist()
                    dx = pi_range[1]-pi_range[0]
                    max_index = pi_range.index(i)
                    space = int(min_region / dx)
                    max_space_sub, max_space_add = (max_index-space, max_index+space)
                    #might have issues with the ends bleeding off, so address those
                    if max_space_sub < 0:
                        max_space_sub = 0
                    if max_space_add > 360:
                        max_space_add = 359
                    #target_range = log_dens.reshape(-1)[max_index-space:max_index+space]
                    target_range = log_dens.reshape(-1)[max_space_sub:max_space_add]

                    print('maxes', maxes, 'i', i)
                    print('max_index-space', max_index-space)
                    print('max_index+space', max_index+space)

                    region_probability = cumtrapz(target_range, dx=dx)[-1]

                    if density_percent == 1.00:
                        region_space = space*dx
                    else:
                        probability = 0.0
                        for spacing in range(1,100):
                            probability = cumtrapz((log_dens.reshape(-1)[max_index-spacing:max_index+spacing]), dx=dx)[-1]
                            if probability/float(region_probability) >= density_percent:
                                region_space = spacing*dx
                                break
                    print('this is the break')
                    break
                else:
                    #ADDED
                    print('pvalue', pose_value)
                    print('doing none', max_value)
                    print('min_range', min_range, 'pose_value', pose_value, 'min_region', min_region)
                    print('min', [i%np.pi for i in min_range[1:-1]], 'max', [i%np.pi for i in max_value])
                    #ax.plot(pi_range, np.exp(log_dens), '-')
                    #ax.vlines(pose_value, 0,1, colors='green')
                    #ADDED
                    pass
        if debug_counter == 0:
            print('pose_value', pose_value)
            plt.plot(pi_range, (log_dens))
            #plt.xlim(-np.pi,np.pi)
            plt.vlines(pose_value, 0,0.75, color='blue')
            #plt.vlines(min_range, 0,0.75, color='red')

            plt.show()

        #half region space because we look within +- the max_return value
        try:
            region_space = region_space/2.0
        except:
            plt.plot(pi_range, (log_dens))
            #plt.xlim(-np.pi,np.pi)
            plt.vlines(max_value, 0,0.75)
            plt.vlines(min_range, 0,0.75, color='red')

            plt.show()
            #exit()
        #enforce periodicity within region
        if max_return > np.pi:
            max_return = max_return - 2*np.pi
        elif max_return < -np.pi:
            max_return = max_return + 2*np.pi
        return max_return, region_space




    def getDartsFromTrajs(self, traj_files, structure_files=None,  atom_indices=None,  topology=None, dihedral_select='pose', same_range=True, set_self=True):

        if structure_files == None:
            structure_files=self.pdb_files
        if topology==None:
            topology = structure_files[0]
        if atom_indices == None:
            atom_indices = self.atom_indices


        traj_files = [self._loadfiles(i, topology) for i in traj_files]

        internal_xyz, internal_zmat, binding_mode_pos, binding_mode_traj = self._createZmat(structure_files, atom_indices, topology, reference_traj=None, fit_atoms=None)
        traj_storage = []
        dihedral_angles = []
        for i in internal_zmat[0].index.values[3:]:
            value_list = [i, internal_zmat[0].loc[i]['b'], internal_zmat[0].loc[i]['a'], internal_zmat[0].loc[i]['d']]
            dihedral_angles.append(value_list)
        for index, traj in enumerate(traj_files):
            dihedrals = md.compute_dihedrals(traj, dihedral_angles)
            #dihedrals in radians
            print('dihedrals', dihedrals, np.shape(dihedrals))
            if dihedral_select=='first':
                pose_mins = [dihedrals[0,i] for i in range(len(internal_zmat[index].index.values[3:]))]
            elif dihedral_select=='last':
#                pose_mins = [dihedrals[-1:i] for i in range(len(internal_zmat[index].index.values[3:]))]
                pose_mins = [dihedrals[-1,i] for i in range(len(internal_zmat[index].index.values[3:]))]

            elif dihedral_select=='pose':
                pose_mins = [np.deg2rad(internal_zmat[index].loc[i,'dihedral']) for i in internal_zmat[index].index.values[3:]]
            #contains all the dictionary lists conrresponding to the min/max ranges
            print('pose_mins', pose_mins, pose_mins[0])
            traj_dict = {value: self.getMaxRange(dihedrals[:,aindex].reshape(-1,1), pose_mins[aindex]) for aindex, value in enumerate(internal_zmat[index].index.values[3:])}
            #print(traj_dict)
            traj_storage.append(traj_dict)
        output_mat = [copy.deepcopy(zmat) for zmat in internal_zmat]
        #print('output before', output_mat[0])
        for zindex, zmat in enumerate(output_mat):
            range_list = [0,0,0]
            dihedral_max = [0,0,0]
            for  i in internal_zmat[zindex].index.values[3:]:
                #zmat._frame.loc[i,'dihedral'] = np.rad2deg(traj_storage[zindex][i][0])
                range_list.append(np.rad2deg(traj_storage[zindex][i][1]))
                dihedral_max.append(np.rad2deg(traj_storage[zindex][i][0]))
            #using mdtraj gives opposite sign compared to chemcoord, so multipy by -1
            #zmat._frame['dihedral_max'] = -1*dihedral_max
            #zmat._frame['dihedral_max'] = dihedral_max.multipy(-1)
            zmat._frame['dihedral_max'] = [-1*di for di in dihedral_max]

            zmat._frame['dart_range'] = range_list
            print('traj_storage', traj_storage)
        if same_range:
            #set dihedral ranges to minimum values
            starting_frame = copy.deepcopy(output_mat[0]._frame['dart_range'])
            for zindex, zmat in enumerate(output_mat):
                starting_frame= pd.concat([starting_frame, zmat._frame['dart_range']]).min(level=0)
            print('minimum', starting_frame )
            for zindex, zmat in enumerate(output_mat):
                zmat._frame['dart_range'] = starting_frame
        #now have to set up darting using darting regions instead
        if set_self==True:
            self.internal_zmat = output_mat
        change_darts = True
        if change_darts ==  True:
            pass
            dihedral_difference = makeDihedralDifferenceDf(self.internal_zmat, dihedral_cutoff=0.01)
            print("self.traj_dart_dict['rotate_list']", self.traj_dart_dict['rotate_list'])
            print('traj_dart_dict', self.traj_dart_dict)
            #exit()
            for rotate in self.traj_dart_dict['rotate_list']:
                output_atoms = []
                #filter out H atoms
                if any(elem in self.darts['dihedral'] for elem in self.traj_dart_dict['bond_groups'][rotate]) == False:
                    #check which atoms\\
                    for next_atom in  self.traj_dart_dict['bond_groups'][rotate]:
                        print('is it in there', next_atom, dihedral_difference.atomnum.values)
                        if next_atom not in self.darts['dihedral'] and (next_atom in dihedral_difference.atomnum.values):
                            output_atoms.append(next_atom)
                            print('next atom', next_atom, 'not in dihedral')
                    print('test0', output_atoms)
                    print('test', output_mat[0].index[output_atoms].tolist())
                    #selected_df = output_mat[0].loc[output_mat[0].index[output_atoms].tolist(), 'dart_range']
                    print('debug', output_mat[0].loc[output_atoms])
                    selected_df = output_mat[0].loc[output_atoms, 'dart_range']

                    print('selected_df\n', selected_df)
                    selected_df=selected_df[selected_df == selected_df.max()]
                    print('max\n', selected_df)
                    print('next', selected_df.iloc[0])
                    max_value = selected_df.iloc[0]
                    selected_index = selected_df.index.tolist()[0]
                    print('max_value', max_value, 'selected_index', selected_index)
                    dihedral_difference = makeDihedralDifferenceDf(self.internal_zmat, dihedral_cutoff=0.01)
                    print('dihedral difference', dihedral_difference)
                    print('atom_num.loc', selected_index)
                    print("dihedral_difference['atomnum'].loc[selected_index]", dihedral_difference['atomnum'].loc[selected_index])
                    #print("self.darts['dihedral'][selected_index]", self.darts['dihedral'][selected_index])
                    self.darts['dihedral'][selected_index] = dihedral_difference['atomnum'].loc[selected_index]
                    print('debug1', dihedral_difference[dihedral_difference['atomnum']==selected_index]['diff'])
                    #self.darts['dihedral'][selected_index] = dihedral_difference[dihedral_difference['atomnum']==selected_index]
                    self.darts['dihedral'][selected_index] = dihedral_difference[dihedral_difference['atomnum']==selected_index]['diff'].iloc[0] / 2.0

                    print('darts', self.darts)
                #if rotate not in self.darts['dihedral']:
                #    print('rotate', rotate, 'not in dihedral')
        print('exiting')

        print('darts', self.darts)
        print('exiting')

        #exit()
        def removeDartOverlaps():
            pass
        return output_mat


    @classmethod
    def _checkTrajectoryDarts(cls, structure_files, atom_indices, traj_files, darts, topology=None, reference_traj=None, fit_atoms=None):
        """
        Parameters
        ----------
        structure_files: list of str
            List corresponding to the path of the structures to create representations of.
        atom_indices: list of ints
            The atom indices of the ligand in the structure files.
        topology: str, optional, default=None
            Path of topology file, if structure_files doesn't contain topology information.
        reference_traj: mdtraj.Trajectory or str, optional, default=None
            Trajectory object, or path to file, containing the reference system to superpose to. If None then
            trajectories are fitted to the first structure in structure_files.
        fit_atoms: list, optional, default=None
            List of atom indices to be used in fitting the structure_files positions to the
            reference trajectory (if reference_traj is not None)

        Returns
        -------
        all_darts: list of lists
            List containing a list of ints for each traj_files item. Each int corresponds to a frame of that trajectory if it matches
            a pose from the poses specified in structure_files

        """
        if not isinstance(traj_files, list):
            traj_files = [traj_files]
        if reference_traj is None:
            reference_traj = cls._loadfiles(structure_files[0], topology)
        else:
            if isinstance(reference_traj, str):
                reference_traj = cls._loadfiles(reference_traj, topology)

        internal_xyz, internal_zmat, binding_mode_pos, binding_mode_traj = cls._createZmat(structure_files=structure_files,
                    atom_indices=atom_indices,
                    topology=topology,
                    reference_traj=reference_traj,
                    fit_atoms=fit_atoms)
        buildlist = MolDartMove._createBuildlist(structure_files, atom_indices, topology=topology)
        temp_xyz = copy.deepcopy(internal_xyz[0])
        all_darts = []
        for traj in traj_files:
            print('traj', traj)
            traj = cls._loadfiles(traj, topology)
            traj.superpose(reference=reference_traj,
                atom_indices=fit_atoms)
            traj_frames = []
            for frame in range(traj.n_frames):
                temp_xyz._frame.loc[:, ['x', 'y', 'z']] = traj.xyz[frame][atom_indices]*10
                temp_zmat = temp_xyz.get_zmat(construction_table=buildlist)
                poses = checkDart(internal_zmat, current_pos=(np.array(traj.openmm_positions(frame)._value))[atom_indices]*10,
                    current_zmat=temp_zmat, pos_list=binding_mode_pos,
                    construction_table=buildlist,
                    dart_storage=darts
                    )
                traj_frames.append(poses)
            all_darts.append(traj_frames)

        return all_darts



    @classmethod
    def _findDihedralRingAtoms(cls, structure_files, atom_indices, rigid_darts=False):
        buildlist = cls._createBuildlist(structure_files, atom_indices)
        #central
        central_atom = buildlist.index.tolist()[findCentralAngle(buildlist)[0][1]]
        #if rigid_darts is not None:
        if True:
            with tempfile.NamedTemporaryFile(suffix='.xyz', mode='w') as t:
                fname = t.name
                traj = md.load(structure_files[0]).atom_slice(atom_indices)
                xtraj = XYZTrajectoryFile(filename=fname, mode='w')
                xtraj.write(xyz=in_units_of(traj.xyz, traj._distance_unit, xtraj.distance_unit),
                            types=[i.element.symbol for i in traj.top.atoms] )
                xtraj.close()
                ring_atoms = []
                from openeye import oechem
                ifs = oechem.oemolistream()
                ifs.open(fname)
                #double_bonds = []
                h_list = []
                ring_atom_list = []
                rotatable_atom_bonds = {}
                rigid_atom_bonds = {}
                all_rotatable_bonds = []
                all_rotatable_bonds_dict = {}
                first_buildlist = buildlist.index.tolist()[:3:] + ['origin', 'e_x', 'e_y', 'e_z']

                for mol in ifs.GetOEGraphMols():
                    oechem.OEFindRingAtomsAndBonds(mol)
                    for atom in mol.GetAtoms():
                        atom_list = []
                        for atom1 in mol.GetAtoms():
                            if AtomsInSameRing(atom,atom1):
                                atom_list.append(atom1.GetIdx())
                        #find out if bond is rotatable
                        if any([bond.IsRotor() for bond in atom.GetBonds()]):
                            all_rotatable_bonds.append(atom.GetIdx())

                        if atom.IsInRing():
                        #if not atom.IsRotor():
                        #find if connected to a nonring rotatable atom
                        #keep track of this for darting later in rotatable_atom_bonds
                            for bond in atom.GetBonds():
                                parent_atom = atom.GetIdx()
                                neighbor_atom = bond.GetNbr(atom).GetIdx()
                                if bond.IsRotor():
                                    #try accept to add dictionary entry if not present
                                    try:
                                        rotatable_atom_bonds[parent_atom].append(neighbor_atom)
                                    except:
                                        rotatable_atom_bonds[parent_atom] = []
                                        rotatable_atom_bonds[parent_atom].append(neighbor_atom)
                                else:
                                    try:
                                        rigid_atom_bonds[parent_atom].append(neighbor_atom)
                                    except:
                                        rigid_atom_bonds[parent_atom] = []
                                        rigid_atom_bonds[parent_atom].append(neighbor_atom)
                            ring_atoms.append(atom.GetIdx())
                        if atom.IsHydrogen():
                            h_list.append(atom.GetIdx())
                        if len(atom_list) > 0:
                            ring_atom_list.append(atom_list)
                for dict_type in [rotatable_atom_bonds, rigid_atom_bonds]:
                    for key in dict_type:
                        dict_type[key] = list(set(dict_type[key]))
                all_rotatable_bonds = list(set(all_rotatable_bonds))
                all_rotatable_bonds_dict = {}
                for item in all_rotatable_bonds:
                    all_rotatable_bonds_dict[item] = []
                print('all_rotatable_bonds', all_rotatable_bonds)
                for rotatable_atom_key, rotatable_atom_value in rotatable_atom_bonds.items():
                    print('rotatable_atom', rotatable_atom_key, rotatable_atom_value)
                    #find all the atoms bonded to the atoms
                    #bonded_atoms = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] ==  rotatable_atom and buildlist.at[i, 'b'] !=  central_atom and i in all_rotatable_bonds]
                    #bonded_atoms_new = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] ==  rotatable_atom and i not in ring_atoms and i in all_rotatable_bonds]
                    #bonded_atoms_newest = [i for i in all_rotatable_bonds if buildlist.at[i, 'b'] == rotatable_atom and i in all_rotatable_bonds]
                    bonded_atoms = 'placeholder'
                    for atom_value in rotatable_atom_value:
                        atom_list = []
                        if buildlist.at[atom_value, 'b'] == rotatable_atom_key:
                            all_rotatable_bonds_dict[rotatable_atom_key].append(atom_value)
                        else:
                            all_rotatable_bonds_dict[atom_value].append(rotatable_atom_key)
                for key in list(all_rotatable_bonds_dict.keys()):
                    if not all_rotatable_bonds_dict[key]:
                        del all_rotatable_bonds_dict[key]
                #rotatable atoms are the atoms that are at the end of the bond that should be treated for rotation
                rotatable_atoms = []
                for atom_key, atom_value in all_rotatable_bonds_dict.items():
                    for atom in atom_value:
                        rotatable_atoms.append(atom)
                    #all_rotatable_bonds_dict[rotatable_atom_key] = atom_value
                #have clause that checks if two of the bonded atoms are in the origin, if so then don't rotate that bond
                    #check that i is not a origin atom, that it is bound to a rotatable bond
                    #find atoms bonded to atom but not in
                #exit()
                print('all_rotatable_bonds_dict', all_rotatable_bonds_dict)
                print('rotatable atom bonds', rotatable_atom_bonds, 'ring_atoms', set(ring_atoms))
                print('rigid_atom_bonds', rigid_atom_bonds)
                print('all_rotatable_bonds', list(set(all_rotatable_bonds)))
                print('all_rotatable_atom_dict', all_rotatable_bonds_dict)
                print(buildlist)
                #check which rotatale bonds are attached to

            #check if an atom (atom1) is bonded to a rotatable atom (atom2) if all if so, check that all bonds bonded to atom1 are not part of the initial build list
            #if they aren't add that as a rotatable group (so the dihedral can change)
                #bgn_idx = [bond.GetBgnIdx() for bond in mol.GetBonds() if bond.GetEndIdx() in ring_atoms]
                #end_idx = [bond.GetEndIdx() for bond in mol.GetBonds() if bond.GetBgnIdx() in ring_atoms]
            rigid_atoms = ring_atoms
            #select all atoms that are bonded to a ring/double bond atom
            angle_ring_atoms = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] in rigid_atoms]
            #print('mols', ifs.GetOEGraphMols())
            print('atom_list', atom_list)
            new_atoms = [list(x) for x in set(tuple(x) for x in ring_atom_list)]
            unique_list = []
            for entry in new_atoms:
                for entry2 in new_atoms:
                    if entry != entry2:
                        if all(elem in entry for elem in entry2):
                            unique_list.append(entry)
            print('unique_list', unique_list)
            set_1 = set()
            if 0:
                for item in unique_list:
                    print('item', type(item), item, set(item))
                    print(set(item))
                    new_item = set(item)
                    set_1.add(new_item)

                print('set', {set(x) for x in unique_list})
            unique_set = set({frozenset(sorted(x)) for x in unique_list})
            print('unique_set', unique_set)
            unique_list = list(set({frozenset(sorted(x)) for x in unique_list}))
            #find end atoms (atoms that are bonded to atoms)

            #self.dihedral_ring_atoms = list(set(angle_ring_atoms + h_list))
            #self.dihedral_ring_atoms = list(set(rigid_atoms + h_list))
            dihedral_ring_atoms = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] in rigid_atoms]
            #find the order of atoms and the indices that should be treated the same (if any)
            rotate_keys = list(rotatable_atom_bonds.keys())
            #want to sort based on build order in
            rotate_keys_sort = list((buildlist.index.tolist().index(i) for i in rotate_keys))
            print('first', rotate_keys_sort)
            print('np.argsort(rotate_keys_sort)', np.argsort(rotate_keys_sort))

            rotate_keys = [rotate_keys[np.argsort(rotate_keys_sort)[i]] for i in range(len(rotate_keys))]
            #if the first 3 atoms are part of the build list then ignore them since their rotations are already accounted for
            print('test build', buildlist['b'].loc[3:])
            first_buildlist = buildlist.index.tolist()[:3:] + ['origin', 'e_x', 'e_y', 'e_z']
            #rotate_keys = [i for i in rotate_keys if i not in buildlist['b']]
            #rotate_keys = [i for i in rotate_keys if i not in buildlist.index.tolist()[:3]]
            print(buildlist)
            print('rotate_keys', rotate_keys)
            bond_groups = {}
            for i in buildlist.index:
                #print('test', i, buildlist.loc[i, ['b', 'a', 'd']])
                #print('test', i, buildlist[buildlist['b'] == i], buildlist[buildlist['b'] == i].index.tolist())
                testa = buildlist[buildlist['b'] == i]
                bond_groups[i] = buildlist[buildlist['b'] == i].index.tolist()

            #find the atoms bonded to these rotate_key atoms
            rotate_list = []
            ring_tracker = copy.deepcopy(unique_list)
            print('rotate_keys1', rotate_keys)
            #for i in rotate_keys:
            #    for bonded_atoms in bond_groups[i]:
            for atoms in rotate_keys:
                #check bonded atoms of atoms. If bonded atoms are part of first atoms, skip them
                if atoms in first_buildlist:
                    #if rotatable_atom_bonds[atoms][0] in first_buildlist:
                    if rotatable_atom_bonds[atoms][0] == central_atom:
                        pass
                    elif rotatable_atom_bonds[atoms][0] not in rotate_list:
                        rotate_list.append(rotatable_atom_bonds[atoms][0])
                elif atoms not in rotate_list:
                     rotate_list.append(atoms)
            rotate_list = []
            for atom in rotatable_atoms:
                count = 0
                if atom in first_buildlist:
                    count += 1
                for bond in bond_groups:
                    if bond in first_buildlist:
                        count +=1
                if count >= 2:
                    pass
                else:
                    rotate_list.append(atom)
                    #if rotated bonds atoms in this
                    #check rigid_atom_bonds for rotate_keys
                    #if rigid_atoms_bonds in the first three buildlist, don't rotate
                    #
            #once we know what atoms are the ones to be rotated then we can isolate them for darting
            print('rotate_list0', rotate_list)
            for value in all_rotatable_bonds_dict.values():
                print('value', value)
                for target_atom in value:
                    print('target_atom', target_atom)
                    if target_atom not in rotate_list:
                        rotate_list.append(target_atom)
            print('rotate_list1', rotate_list)
            #NEW THING: check if the first 3/4 atoms take care of a ring. If that's the case then skip that one
            #print(buildlist.loc[rotate_keys])
            print('bond groups', bond_groups)
            output_dict = {}
            output_dict['rotate_list'] = rotate_list
            output_dict['bond_groups'] = bond_groups
            output_dict['dihedral_ring_atoms'] = dihedral_ring_atoms
            output_dict['all_rotatable_bonds_dict'] = all_rotatable_bonds_dict
            print('returning output_dict', output_dict)
            #exit()
            return output_dict

    @staticmethod
    def _checkTransitionMatrix(transition_matrix, dart_groups):
        """Checks if transition matrix obeys the proper
        properties if used in a Monte Carlo scheme

        Parameters
        ----------
        transition_matrix: nxn np.array
            transition matrix where each entry corresponds to the probability of jumping to another state
        dart_groups: list
            list containing the number of darts used for darting
        Returns
        -------
        transition_matrix: nxn np.array
            transitino matrix with the row probabilities normalized.
        """
        #check if values are non-negative

        ltz = np.where(transition_matrix<0)[0]
        if len(ltz)>0:
            raise ValueError('transition_matrix values should all be non-negative')
        #check reversiblility
        indices = np.triu_indices(np.shape(transition_matrix)[0])
        error_indices = []
        for i,j in zip(indices[0],indices[1]):
                if transition_matrix[i,j]==0 or transition_matrix[j,i]==0:
                    if transition_matrix[j,i]>0 or transition_matrix[i,j]>0:
                        missing_indices="[{},{}][{},{}]".format(i,j,j,i)
                        error_indices.append(missing_indices)
        if len(error_indices)>0:
            raise ValueError('transition_matrix needs to be reversible to maintain detailed balance. '+
            'The following pairs should either both be zero or non-zero {}'.format(error_indices))
        #check if size of transition matrix is correct
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        if np.shape(transition_matrix) != (len(dart_groups), len(dart_groups)):
            raise ValueError('Transition matrix should be an nxn matrix, where n is the length of pdb_files')

        return transition_matrix



    def _poseDart(self, context, atom_indices):
        """Check whether molecule is within a dart, and
        if it is, return the dart vectors for it's atoms.

        Parameters
        ----------
        context: simtk.openmm.context object
            Context containing the position array to dart
        atom_indices: list of ints
            List of atom indices of the atoms that are to be
            darted.
        Returns
        -------
        binding_index: int or None
            The index of the selected dart, or None if
            no dart was selected.
        diff_list: list of n 1x3 np.arrays
            List of vector differences between the simulation positions
            and the dart centers
        symm_list: List of list floats (not implemented)
            List of symmetric atoms

        """

        nc_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        #update sim_traj positions for superposing binding modes
        self.sim_traj.xyz[0] = nc_pos._value
        self.sim_ref.xyz[0] = nc_pos._value
        #make a temp_pos to specify dart centers to compare
        #distances between each dart and binding mode reference
        temp_pos = []
        num_lig_atoms = len(self.atom_indices)
        temp_pos = np.zeros((num_lig_atoms, 3))

        for index, atom in enumerate(atom_indices):
            #keep track of units
            temp_pos[index] = nc_pos[atom]._value

        #fit different binding modes to current protein
        #to remove rotational changes
        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        traj_positions = self.sim_traj.xyz[0]
        xyz_ref._frame.loc[:, ['x', 'y', 'z']] = traj_positions[self.atom_indices]*10

        current_zmat = xyz_ref.get_zmat(construction_table=self.buildlist)
        #logger.info("Freezing selection '{}' ({} atoms) on {}".format(freeze_selection, len(mask_idx), system))
        logger.info("sim_traj {}".format(self.sim_traj))
        logger.info("self.trajs {}".format(self.trajs))


        print('sim_traj', self.sim_traj)
        print('self.trajs', self.trajs)
        #logger.info("before self.binding_mode_traj {} {}".format(len(self.binding_mode_traj), self.binding_mode_traj))
        #logger.info("before self.binding_mode_pos {} {}".format(len(self.binding_mode_pos), self.binding_mode_pos))
        #logger.info("before self.binding_mode_thetrajs {} {}".format(len(self.trajs), self.trajs))
        self.binding_mode_traj, self.binding_mode_pos = self.refitPoses(self.sim_traj, self.trajs, self.fit_atoms, self.atom_indices)
        #logger.info("self.binding_mode_traj {} {}".format(len(self.binding_mode_traj), self.binding_mode_traj))
        #logger.info("self.binding_mode_pos {} {}".format(len(self.binding_mode_pos), self.binding_mode_pos))
        print('type', type(current_zmat))
        print('type internal_zmat', type(self.internal_zmat))
        selected = checkDart(self.internal_zmat, current_pos=(np.array(self.sim_traj.openmm_positions(0)._value))[self.atom_indices]*10,

                    current_zmat=current_zmat, pos_list=self.binding_mode_pos,
                    construction_table=self.buildlist,
                    dart_storage=self.darts
                    )
        if len(selected) >= 1:
            #returns binding mode
            #diff_list will be used to dart
            return selected
        elif len(selected) == 0:
            return []

    def _dart_selection(self, binding_mode_index, transition_matrix):
        """
        Picks a new dart based on an inital binding mode index and transition matrix.
        Returns the randomly selected dart and acceptance criteria factoring in the probabilities
        from the transition matrix

        Parameters
        ----------
        binding_mode_index: int
            The binding mode index of a dart
        transition_matrix: nxn np.array
            The transition matrix to determine the transition probabilities
            from a given dart

        Returns
        -------
        rand_index: int
            The randomly chosen binding mode index selected using the transition matrix
            probabilities.
        acceptance_ratio:
            The probability ratio that needs to be factored into the acceptance criterion
            for using the transition matrix.
        """
        rand_index = np.random.choice(self.dart_groups, p=transition_matrix[binding_mode_index])
        prob_forward = transition_matrix[binding_mode_index][rand_index]
        prob_reverse = transition_matrix[rand_index][binding_mode_index]
        acceptance_ratio = float(prob_reverse)/prob_forward
        return rand_index, acceptance_ratio

    def _moldRedart(self, atom_indices, binding_mode_pos, binding_mode_index, nc_pos, rigid_darts):
        """
        Helper function to choose a random pose and determine the vector
        that would translate the current particles to that dart center
        -Gets the cartesian coordinates of the simulation them into internal coordinates
        -calculates the differences present with those internal coordinates and
            the given dart
        -selects a new internal coordinate dart and darts to it, taking into account
            the differences with the original dart
        -transforms the internal coordinates back into a cartesian representation


        Parameters
        ----------
        atom_indices: list

        binding_mode_pos: list of nx3 np.arrays
            The list that contains the coordinates of the various binding modes
        binding_mode_index: int
            integer given by poseRedart that specifes which binding mode
            out of the list it matches with

        Returns
        -------
        nc_pos: nx3 np.array * unit.nanometers
            Positions of the system after the darting procedue.

        """
        #put new function to handle degree wrapping
        def wrapDegrees(a):
            if a > 180.0:
                return a - 360
            if a < -180.0:
                return a + 360
            else:
                return a

        if atom_indices == None:
            atom_indices = self.atom_indices
        #choose a random binding pose
        #rand_index = np.random.choice(self.dart_groups, self.transition_matrix[binding_mode_index])

        rand_index, self.dart_ratio = self._dart_selection(binding_mode_index, self.transition_matrix)
        dart_ratio = self.dart_ratio
        self.acceptance_ratio = self.acceptance_ratio * dart_ratio


        #get matching binding mode pose and get rotation/translation to that pose
        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        traj_positions = self.sim_traj.xyz[0]
        xyz_ref._frame.loc[:, ['x', 'y', 'z']] = traj_positions[self.atom_indices]*10
        zmat_new = copy.deepcopy(self.internal_zmat[rand_index])
        zmat_diff = xyz_ref.get_zmat(construction_table=self.buildlist)
        zmat_traj = copy.deepcopy(xyz_ref.get_zmat(construction_table=self.buildlist))
        #get appropriate comparision zmat
        zmat_compare = self.internal_zmat[binding_mode_index]
        #we don't need to change the bonds/dihedrals since they are fast to sample
        #if the molecule is treated as rigid, we won't change the intrnal coordinates
        #otherwise we find the differences in the dihedral angles between the simulation
        #and reference poses and take that into account when darting to the new pose
        change_list = ['dihedral']
        old_list = ['bond', 'angle']
        #TODO: play around with this
        print('ring_atoms', self.dihedral_ring_atoms)
        print('rotate_list', self.traj_dart_dict['rotate_list'])
        print('bond_groups', self.traj_dart_dict['bond_groups'])
        #exit()
        if rigid_darts == 'rigid_ring':
            rigid_dihedrals_atoms = [i for i in self.dihedral_ring_atoms if i in zmat_new._frame.index[3:]]
            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]
        elif rigid_darts == 'rigid_darts':
            #rigid_dihedrals_atoms = [i for i in self.only_darts_dihedrals if i in zmat_new._frame.index[3:]]
            move_atoms = []
            for anchor_atoms in self.traj_dart_dict['rotate_list']:
                for rotate_atoms in self.traj_dart_dict['bond_groups'][anchor_atoms]:
                    move_atoms.append(rotate_atoms)
            rigid_dihedrals_atoms = [i for i in zmat_new._frame.index[3:] if i not in move_atoms]
            print('rigid_dihedrals_atoms', sorted(rigid_dihedrals_atoms), 'move_atoms', sorted(move_atoms))
            rigid_dihedrals_atoms = [i for i in zmat_new._frame.index if i not in self.only_darts_dihedrals]
            print('test0', zmat_new)

            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]
            print('test1a', zmat_new)
            print('zmat_traj', zmat_traj)

        if rigid_darts == 'rigid_molecule':
            old_list =  old_list + change_list

        else:
            if 'dart_range' in self.internal_zmat[binding_mode_index]._frame:
                #test
                if 1:
                    new_change =  old_list + change_list

                    zmat_new._frame.loc[rigid_dihedrals_atoms, new_change] = zmat_traj._frame.loc[rigid_dihedrals_atoms, new_change]

                print('doing the new thing')

                for center_atom in self.traj_dart_dict['rotate_list']:
                    #pick one atom at random to use the dart_range for to displace everything uniformly
                    chosen_atom = random.choice(self.traj_dart_dict['bond_groups'][center_atom])
                    chosen_atom = self.traj_dart_dict['bond_groups'][center_atom][1]
                    print('chosen_atom', chosen_atom)
                    #chosen_atom = center_atom


                    #find the random displacement based on what the chosen atom is
                    #displacement = -1*zmat_new._frame['dart_range'].loc[chosen_atom]
#                    displacement = zmat_new._frame['dart_range'].loc[chosen_atom]*(2*(np.random.random() - 0.5))
                    displacement = zmat_new._frame.loc[chosen_atom,'dart_range']*(2*(np.random.random() - 0.5))
                    #displacement = zmat_new._frame.loc[chosen_atom,'dart_range']*(2*(1 - 0.5))
                    displacement = 0.0




                    ##zmat_new._frame['dihedral'].loc[chosen_atom] =  zmat_new._frame['dihedral_max'].loc[chosen_atom] + displacement
                    #choose the new starting point and figure out the displacement
                    #use that displacement for all the atoms bonded to that group
                    for rotate_atom in self.traj_dart_dict['bond_groups'][center_atom]:
#                        rotate_displacement = zmat_traj._frame['dihedral'].loc[chosen_atom] - zmat_traj._frame['dihedral'].loc[rotate_atom]
                        #rotate_displacement = wrapDegrees(zmat_traj._frame.loc[chosen_atom, 'dihedral'] - zmat_traj._frame.loc[rotate_atom,'dihedral'])
                        rotate_displacement = zmat_traj._frame.loc[chosen_atom, 'dihedral'] - zmat_traj._frame.loc[rotate_atom,'dihedral']

                        #previous commented out
                        #zmat_new._frame['dihedral'].loc[rotate_atom] =  zmat_new._frame['dihedral_max'].loc[rotate_atom] + rotate_displacement

#                       zmat_new._frame['dihedral'].loc[rotate_atom] =  zmat_new._frame['dihedral_max'].loc[chosen_atom] + rotate_displacement + displacement
                        #add_distance = rotate_displacement + displacement
                        #add_distance = wrapDegrees(wrapDegrees(rotate_displacement) + wrapDegrees(displacement)) is right

                        add_distance = wrapDegrees(wrapDegrees(rotate_displacement) + wrapDegrees(displacement)) #default
                        #add_distance = wrapDegrees(wrapDegrees(-rotate_displacement) - wrapDegrees(displacement)) #kinda works? but not
                        #add_distance = wrapDegrees(wrapDegrees(-rotate_displacement) + wrapDegrees(displacement)) #kinda works? but not right


                        print('add_distance for', rotate_atom, add_distance, rotate_displacement, displacement)
                        zmat_new._frame.loc[rotate_atom,'dihedral'] =  zmat_new._frame.loc[chosen_atom,'dihedral_max'] - add_distance
                        #debug here
                        #zmat_new._frame.loc[rotate_atom,'dihedral'] = zmat_new._frame.loc[rotate_atom,'dihedral_max']
                        #zmat_new._frame.loc[rotate_atom,'dihedral'] =  zmat_new._frame.loc[chosen_atom,'dihedral_max']

                        #zmat_new._frame.loc[rotate_atom,'dihedral'] =  zmat_new._frame.loc[rotate_atom,'dihedral_max']



                    #TODO Continue here
                print('zmat after random\n', zmat_new)
                if 0:
                    print('doing the right part')
                    #zmat_new._frame['dart_range'] * 2*np.random.random() - zmat_new._frame['dart_range']
                    zmat_size = len(self.internal_zmat[binding_mode_index].index)
                    print((2*np.random.random((zmat_size, 1)) - 1))
                    #diff = zmat_new._frame['dart_range']*(2*np.random.random((zmat_size, 1)) - 1)
                    #diff = zmat_new._frame['dart_range'].multiply(2*np.random.random((zmat_size, 1)) - 1)
                    diff = zmat_new._frame['dart_range'].multiply(2*(np.random.random((zmat_size)) - 0.5))
                    print('diff', diff)
                    print('zmat_new before', zmat_new)


                    zmat_new._frame['dihedral'] = zmat_new._frame['dihedral_max'] + diff
                    print('zmat_new after', zmat_new)

                #figure out the range to alter the dihedrals
                #TODO might need to do something with the bond/angle for the first 3 bonds/angles


            else:
                print('doing the wrong part')
                zmat_indices = zmat_traj.index.values
                changed = (zmat_diff._frame.loc[:, change_list] - zmat_compare._frame.loc[:, change_list]).reindex(zmat_indices)
                abs_bond_diff = zmat_diff._frame.loc[:, 'bond'].iloc[0] - zmat_compare._frame.loc[:, 'bond'].iloc[0]
                abs_angle_diff = zmat_diff._frame.loc[:, 'angle'].iloc[:2] - zmat_compare._frame.loc[:, 'angle'].iloc[:2]

                zmat_diff._frame.loc[:, change_list] = changed
                zmat_diff._frame.loc[(zmat_diff._frame.index.isin([zmat_diff._frame.index[0]])), 'bond'] = abs_bond_diff
                zmat_diff._frame.loc[(zmat_diff._frame.index.isin(zmat_diff._frame.index[:2])), 'angle'] = abs_angle_diff

                #Then add back those changes to the darted pose
                zmat_new._frame.loc[:, change_list] = zmat_new._frame.loc[zmat_new._frame.index[2:], 'dihedral'] + zmat_diff._frame.loc[zmat_diff._frame.index[2:], 'dihedral']
                zmat_new._frame.loc[zmat_new._frame.index[0], 'bond'] = zmat_new._frame.loc[zmat_new._frame.index[0], 'bond'] + zmat_diff._frame.loc[zmat_new._frame.index[0], 'bond']
                zmat_new._frame.loc[zmat_new._frame.index[:2], 'angle'] = zmat_new._frame.loc[zmat_new._frame.index[:2], 'angle'] + zmat_diff._frame.loc[zmat_diff._frame.index[:2], 'angle']

        #We want to keep the bonds and angles the same between jumps, since they don't really vary
        zmat_new._frame.loc[:, old_list] = zmat_traj._frame.loc[:, old_list]
        #added
        if rigid_darts == 'rigid_ring':
            rigid_dihedrals_atoms = [i for i in self.dihedral_ring_atoms if i in zmat_new._frame.index[3:]]
            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]
        elif rigid_darts == 'rigid_darts':
            #if 0 for debugging
            if 0:
                rigid_dihedrals_atoms = [i for i in zmat_new._frame.index if i not in self.only_darts_dihedrals]

                zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]
        zmat_new._frame.loc[:, 'angle'] = zmat_traj._frame.loc[:, 'angle']
        zmat_new._frame.loc[:, 'bond'] = zmat_traj._frame.loc[:, 'bond']


        zmat_new._frame['dihedral'] = zmat_new._frame['dihedral'].apply(np.vectorize(wrapDegrees))
        #find translation differences in positions of first two atoms to reference structure
        #find the appropriate rotation to transform the structure back
        #repeat for second bond
        def normalize_vectors(dart_array, ref_array, vector_list):
            ref1 = ref_array[vector_list[0][0]] - ref_array[vector_list[0][1]]
            ref2 = ref_array[vector_list[1][0]] - ref_array[vector_list[1][1]]
            dart1 = dart_array[vector_list[0][0]] - dart_array[vector_list[0][1]]
            dart2 = dart_array[vector_list[1][0]] - dart_array[vector_list[1][1]]
            normal1 = dart1/np.linalg.norm(dart1) * np.linalg.norm(ref1)
            normal2 = dart2/np.linalg.norm(dart2) * np.linalg.norm(ref2)
            centered_dart = np.tile(dart_array[vector_list[0][1]], (3,1))
            centered_dart[vector_list[0][0]] = normal1 + centered_dart[vector_list[0][0]]
            centered_dart[vector_list[1][0]] = normal2 + centered_dart[vector_list[1][0]]
            return centered_dart
        def test_angle(dart_three, vector_list):
            angle1 = dart_three[vector_list[0][0]] - dart_three[vector_list[0][1]]
            angle2 = dart_three[vector_list[1][0]] - dart_three[vector_list[1][1]]
            dart_angle = angle1.dot(angle2) / (np.linalg.norm(angle1) * np.linalg.norm(angle2))
            return np.degrees(np.arccos(dart_angle))
        def angle_calc(angle1, angle2):
            angle = np.arccos(angle1.dot(angle2) / ( np.linalg.norm(angle1) * np.linalg.norm(angle2) ) )
            degrees = np.degrees(angle)
            return degrees
        def calc_angle(vec1, vec2):
            angle = np.arccos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return angle



        #the third atom listed isn't guaranteed to be the center atom (to calculate angles)
        #so we first have to check the build list to see the atom order
        vector_list = findCentralAngle(self.buildlist)
        #find translation differences in positions of first two atoms to reference structure
        #find the appropriate rotation to transform the structure back
        #repeat for second bond
        #get first 3 new moldart positions, apply same series of rotation/translations
        #start_indices = atom_indices[self.buildlist.index.get_values()[:3]]
        start_indices = [atom_indices[i] for i in self.buildlist.index.values[:3]]

        sim_three = self.sim_traj.xyz[0][start_indices]
        ref_three  = binding_mode_pos[binding_mode_index].xyz[0][start_indices]
        dart_three = binding_mode_pos[rand_index].xyz[0][start_indices]
        dart_ref = binding_mode_pos[rand_index].xyz[0][start_indices]

        change_three = np.copy(sim_three)
        vec1_sim = sim_three[vector_list[0][0]] - sim_three[vector_list[0][1]]
        vec2_sim = sim_three[vector_list[1][0]] - sim_three[vector_list[1][1]]

        #calculate rotation from ref pos to sim pos
        #change angle of one vector
        ref_angle = self.internal_zmat[binding_mode_index]._frame['angle'][self.buildlist.index.values[2]]
        ad_vec = adjust_angle(vec1_sim, vec2_sim, np.radians(ref_angle), maintain_magnitude=True)
        ad_vec = ad_vec / np.linalg.norm(ad_vec) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.values[2]]/10.
        #apply changed vector to center coordinate to get new position of first particle

        nvec2_sim = vec2_sim / np.linalg.norm(vec2_sim) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.values[2]]/10.
        change_three[vector_list[0][0]] = sim_three[vector_list[0][1]] + ad_vec
        change_three[vector_list[1][0]] = sim_three[vector_list[0][1]] + nvec2_sim
        try:
            rot_mat, centroid = getRotTrans(change_three, ref_three, center=vector_list[0][1])
        except:
            rot_mat = np.identity((3,3))
            centroid = copy.deepcopy(dart_three)
            self.acceptance_ratio = 0
        #perform the same angle change on new coordinate
        centroid_orig = dart_three[vector_list[0][1]]
        #perform rotation
        if self.centroid_darting:
            dart_three = (dart_three -  np.tile(centroid_orig, (3,1))).dot(rot_mat) + np.tile(centroid_orig, (3,1)) - np.tile(centroid, (3,1))
        vec1_dart = dart_three[vector_list[0][0]] - dart_three[vector_list[0][1]]
        vec2_dart = dart_three[vector_list[1][0]] - dart_three[vector_list[1][1]]

        #adjust the angle manually because the first three atom positions are directly
        #translated from the reference without angle adjustments
        if 1:
            new_angle = zmat_new['angle'][self.buildlist.index[2]]
            ad_dartvec = adjust_angle(vec1_dart, vec2_dart, np.radians(new_angle), maintain_magnitude=True)
            ###
            ad_dartvec = ad_dartvec / np.linalg.norm(ad_dartvec) * zmat_new._frame['bond'][self.buildlist.index.values[1]]/10.
            nvec2_dart = vec2_dart / np.linalg.norm(vec2_dart) * zmat_new._frame['bond'][self.buildlist.index.values[2]]/10.
            dart_three[vector_list[0][0]] = dart_three[vector_list[0][1]] + ad_dartvec
            dart_three[vector_list[1][0]] = dart_three[vector_list[0][1]] + nvec2_dart

        #get xyz from internal coordinates
        zmat_new.give_cartesian_edit = give_cartesian_edit.__get__(zmat_new)
        #MAKE SURE THIS IS ON dart_three NOT sim three
        #xyz_new = (zmat_new.give_cartesian_edit(start_coord=dart_three*10.)).sort_index()
        xyz_new = (zmat_new.give_cartesian_edit(start_coord=sim_three*10.)).sort_index()
        #print('subtract', zmat_new._frame.loc[2:,['bond', 'angle', 'dihedral']] - zmat_traj._frame.loc[2:,['bond', 'angle', 'dihedral']])
        new_df = zmat_new._frame['dihedral'].copy()
        print('zmat after move\n', zmat_new)
       #print('dir', dir(xyz_new))
        convert = xyz_new.to_zmat(construction_table=self.buildlist)
        print('after conversion\n', xyz_new.to_zmat(construction_table=self.buildlist))
        print('bond_diff\n', zmat_new['bond']- convert['bond'])
        print('angle_diff\n', zmat_new['angle']- convert['angle'])
        print('dihedral_diff\n', zmat_new['dihedral']- convert['dihedral'])

        new_df.rename(columns={'dihedral':'zmat_new'})
        new_df['zmat_traj'] = zmat_traj._frame['dihedral']
        new_df = pd.DataFrame({'zmat_new': zmat_new._frame['dihedral'], 'zmat_traj':zmat_traj._frame['dihedral']})
        alist = []
        def dihedralDifference(input_df, construction_table=None):
            """Computes the difference in dihedral angles
            between the pairs present in zmatrices a and b
            with the cartesian distance in cos, sin
            """
            a_di, b_di = input_df['zmat_new'], input_df['zmat_traj']
            a_dir, b_dir = np.deg2rad(a_di), np.deg2rad(b_di)
            b_cos, b_sin = np.cos(b_dir), np.sin(b_dir)
            a_cos, a_sin = np.cos(a_dir), np.sin(a_dir)
            cos_diff = np.square(b_cos - a_cos)
            sin_diff = np.square(b_sin - a_sin)
            dist = np.sqrt(cos_diff + sin_diff)
            return dist
        if 0:
            for i in zmat_new._frame.index:
                from blues.moldart.darts import dihedralDifference
                alist.append(dihedralDifference(new_df.loc[i, 'zmat_new'], new_df.loc[i, 'zmat_traj']))
            new_df['dihedral_difference'] = alist

        #self.sim_traj.xyz[0][self.atom_indices] = xyz_new._frame.loc[:, ['x', 'y', 'z']].get_values() / 10.
        self.sim_traj.xyz[0][self.atom_indices] = xyz_new._frame.loc[:, ['x', 'y', 'z']].values / 10.

        self.sim_traj.superpose(reference=self.sim_ref, atom_indices=self.fit_atoms[rand_index],
                ref_atom_indices=self.fit_atoms[rand_index]
                )
        nc_pos = self.sim_traj.xyz[0] * unit.nanometers
        self.sim_traj.save('last_output.pdb')

        return nc_pos, rand_index

    def initializeSystem(self, system, integrator):
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

        structure = self.structure
        new_sys = system
        old_int = integrator
        # added water portion to freeze waters further than the freeze_waters closest waters
        if self.freeze_waters > 0:
            residues = structure.topology.residues()
            water_name = ['WAT', 'HOH']
            self.water_residues = []
            for res in residues:
                if res.name in water_name:
                    water_mol = []
                    for atom in res.atoms():
                        water_mol.append(atom.index)
                    self.water_residues.append(water_mol)
            water_oxygens = [i[0] for i in self.water_residues]
            #portion to calculate ligand com
            self._calculateProperties()
            self.water_oxygens = [i[0] for i in self.water_residues]
            positions = np.array(structure.positions.value_in_unit(unit.nanometers))*unit.nanometers
            lig_pos = positions[self.atom_indices]
            center_of_mass = self.getCenterOfMass(lig_pos, self.masses)
            water_pos = positions.take(water_oxygens, axis=0)*unit.nanometers
            #find athe distance of all waters to the ligand center of mass
            water_dist = np.linalg.norm(np.subtract(water_pos, center_of_mass), axis=1)
            #find the x waters that are the closest to the ligand
            water_sort = np.argsort(water_dist)[:self.freeze_waters]
            self.waters_within_distance = sorted([self.water_residues[i] for i in water_sort])
            flat_waters_all = [item for sublist in self.water_residues for item in sublist]
            flat_waters_within_distance = [item for sublist in self.waters_within_distance for item in sublist]
            num_atoms = system.getNumParticles()
            for atom in range(num_atoms):
                if atom in flat_waters_all and atom not in flat_waters_within_distance:
                    system.setParticleMass(atom, 0*unit.daltons)
                else:
                    pass

        if self.freeze_protein:
            #if active freezes protein residues beyond a certain cutoff
            res_list = []
            for traj in self.binding_mode_traj:
                lig_pos = traj.openmm_positions(0)
                structure.positions = lig_pos

                mask = parmed.amber.AmberMask(self.structure,"((:HOH)|(:%s<:%f))&!(:%s)" % ('LIG', self.freeze_protein, 'NA,CL'))
                site_idx = [i for i in mask.Selected()]
                res_list = res_list + site_idx
            res_list = list(set(res_list))
            num_atoms = system.getNumParticles()
            for atom in range(num_atoms):
                if self.freeze_waters > 0:
                    if atom in res_list or atom in flat_waters_within_distance:
                        pass
                    else:
                        system.setParticleMass(atom, 0*unit.daltons)
                else:
                    if atom in res_list:
                        pass
                    else:
                        system.setParticleMass(atom, 0*unit.daltons)

        ###
        if self.rigid_body:
            new_sys, self.real_particles, self.vsiteParticles, self.constraint_list = createRigidBodies(new_sys,  self.sim_traj.openmm_positions(0), [self.atom_indices])
            #exit()
        if self.restraints:
            force_list = new_sys.getForces()
            group_list = list(set([force.getForceGroup() for force in force_list]))
            group_avail = [j for j in list(range(32)) if j not in group_list]
            self.restraint_group = group_avail[0]

            old_int._system_parameters = {system_parameter for system_parameter in old_int._alchemical_functions.keys()}


            new_int = AlchemicalExternalRestrainedLangevinIntegrator(restraint_group=self.restraint_group,
                                               lambda_restraints=self.lambda_restraints, **old_int.int_kwargs)

            new_int.reset()
            initial_traj = self.binding_mode_traj[0].openmm_positions(0).value_in_unit(unit.nanometers)
            for index, pose in enumerate(self.binding_mode_traj):
                #pose_pos is the positions of the given pose
                #the ligand positions in this will be used to replace the ligand positions of self.binding_mode_traj[0]
                #in new_pos to add new restraints

                pose_pos = np.array(pose.openmm_positions(0).value_in_unit(unit.nanometers))[self.atom_indices]
                pose_allpos = np.array(pose.openmm_positions(0).value_in_unit(unit.nanometers))*unit.nanometers
                new_pos = np.copy(initial_traj)
                new_pos = np.array(pose.openmm_positions(0).value_in_unit(unit.nanometers))
                new_pos[self.atom_indices] = pose_pos
                new_pos= new_pos * unit.nanometers
                #restraint_lig = [self.atom_indices[i] for i in self.buildlist.index.get_values()[:3]]
                restraint_lig = [self.atom_indices[i] for i in self.buildlist.index.values[:3]]

                #choose restraint type based on specified parameter
                restraint_style = {'boresch':add_boresch_restraints, 'rmsd':add_rmsd_restraints}
                #check which force groups aren't being used and set restraint forces to that
                if self.restraints == 'boresch':
                    new_sys = restraint_style[self.restraints](new_sys, structure, pose_allpos, self.atom_indices, index, self.restraint_group,
                                            self.restrained_receptor_atoms[index], restraint_lig,
                                            K_r=self.K_r, K_angle=self.K_angle, K_RMSD=self.K_RMSD, RMSD0=self.RMSD0)

                elif self.restraints == 'rmsd':
                    #restrain only the heavy atoms
                    nonheavy_atoms = self.internal_xyz[0]._frame['atom'].str.contains('H')
                    heavy_atoms = [self.atom_indices[i] for i in range(len(self.atom_indices)) if nonheavy_atoms.iloc[i] == False]

                    new_sys = restraint_style[self.restraints](new_sys, structure, pose_allpos, heavy_atoms, index, self.restraint_group,
                                            self.restrained_receptor_atoms[index], restraint_lig,
                                            K_r=self.K_r, K_angle=self.K_angle, K_RMSD=self.K_RMSD, RMSD0=self.RMSD0)


        else:
            new_int = old_int
        return new_sys, new_int


    def beforeMove(self, context):
        """
        This method is called at the start of the NCMC portion if the
        context needs to be checked or modified before performing the move
        at the halfway point.
        If restraints are being used, Check if simlation positions are
        currently in a pose. If so turn on `restraint_pose` for that pose.

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose context were changed by this function.
        """
        self.acceptance_ratio = 1

        if self.freeze_waters > 0:
            #if freezing a subset of waters switch positions of unfrozen waters with frozen waters
            #within a cutoff of the ligand
            start_state = context.getState(getPositions=True, getVelocities=True)
            start_pos = start_state.getPositions(asNumpy=True)
            start_vel = start_state.getVelocities(asNumpy=True)
            switch_pos = np.copy(start_pos)*start_pos.unit
            switch_vel = np.copy(start_vel)*start_vel.unit
            lig_com = self.getCenterOfMass(switch_pos[self.atom_indices], masses=self.masses)
            water_pos = switch_pos.take(self.water_oxygens, axis=0)*unit.nanometers
            water_dist = np.linalg.norm(np.subtract(water_pos, lig_com), axis=1)
            water_sort = np.argsort(water_dist)[:self.freeze_waters]
            waters_within_distance_new = sorted([self.water_residues[i] for i in water_sort])
            culled_new = [i for i in waters_within_distance_new if i not in self.waters_within_distance]
            culled_old = [i for i in self.waters_within_distance if i not in waters_within_distance_new]
            for new_water, old_water in list(zip(culled_new, culled_old)):
                    for j in range(3):
                        switch_pos[old_water[j]] = copy.deepcopy(start_pos[new_water[j]])
                        switch_pos[new_water[j]] = copy.deepcopy(start_pos[old_water[j]])
                        switch_vel[old_water[j]] = start_vel[new_water[j]]
                        switch_vel[new_water[j]] = start_vel[old_water[j]]

            context.setPositions(switch_pos)
            context.setVelocities(switch_vel)


        if self.restraints:
            #if using restraints
            selected_list = self._poseDart(context, self.atom_indices)
            if len(selected_list) >= 1:
                self.selected_pose = np.random.choice(selected_list, replace=False)
                self.dart_begin = self.selected_pose
                self.num_poses_begin_restraints = len(selected_list)

                for i in range(len(self.binding_mode_traj)):
                    context.setParameter('restraint_pose_'+str(i), 0)
                context.setParameter('restraint_pose_'+str(self.selected_pose), 1)

            else:
                #TODO handle the selected_pose when not in a pose
                #probably can set to an arbitrary pose when the acceptance_ratio is treated properly
                self.selected_pose = 0
                self.acceptance_ratio = 0
        else:
            pass
        if self.rigid_body:
            after_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
            resetRigidBodies(context.getSystem(), after_pos, self.real_particles, self.vsiteParticles, self.constraint_list, self.atom_indices)
            context.reinitialize(preserveState=True)

        return context

    def afterMove(self, context):
        """
        If restraints were specified,Check if current positions are in
        the same pose as the specified restraint.
        If not, reject the move (to maintain detailed balance).

        This method is called at the end of the NCMC portion if the
        context needs to be checked or modified before performing the move
        at the halfway point.

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose context were changed by this function.

        """
        if self.restraints:
            selected_list = self._poseDart(context, self.atom_indices)
            if self.selected_pose not in selected_list:
                self.acceptance_ratio = 0
            else:
                self.num_poses_end_restraints = len(selected_list)
                for i in range(len(self.binding_mode_traj)):
                    context.setParameter('restraint_pose_'+str(i), 0)
                self.acceptance_ratio = self.acceptance_ratio*(float(self.num_poses_end_restraints)/self.num_poses_begin_restraints)

        #take into account the number of possible states at the start/end of this proceudre
        #and factor that into the acceptance criterion
        else:
            self.acceptance_ratio = self.acceptance_ratio*(float(self.num_poses_end)/float(self.num_poses_begin))


        return context

    def _error(self, context):
        """
        This method is called if running during NCMC portion results
        in an error. This allows portions of the context, such as the
        context parameters that would not be fixed by just reverting the
        positions/velocities of the context.
        In case a nan occurs we want to make sure restraints are turned off
        for subsequent NCMC iterations.

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.
        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose context were changed by this function.

        """
        if self.restraints:
            for i in range(len(self.binding_mode_traj)):
                context.setParameter('restraint_pose_'+str(i), 0)
        return context

    def move(self, context):
        """
        Function for performing internal coordinate darting moves
        that jumps between different internal coordinates specified
        when the object was initialized.

        Parameters
        ----------
        context: simtk.openmm.Context object
            Context containing the positions to be moved.

        Returns
        -------
        context: simtk.openmm.Context object
            The same input context, but whose positions were changed by this function.

        """
        self.moves_attempted += 1
        state = context.getState(getPositions=True, getEnergy=True)
        oldDartPos = state.getPositions(asNumpy=True)
        selected_list = self._poseDart(context, self.atom_indices)

        if self.restraints:
            context.setParameter('restraint_pose_'+str(self.selected_pose), 0)
        else:
            #the move is instantaneous without restraints, so find overlap of darting regions
            #to incorporate later into the acceptance criterion
            self.num_poses_begin = len(selected_list)

        if len(selected_list) == 0:
            #this means that the current ligand positions are outside the defined darts
            #therefore we don't perform the move
            pass
        else:
            #now self.binding_mode_pos should be fitted to structure at this point
            self.selected_pose = np.random.choice(selected_list, replace=False)
            self.times_within_dart += 1

            #calculate changes in angle/dihedral compared to reference
            #apply angle/dihedral changes to new pose
            #translate new pose to center of first molecule
            #find rotation that matches atom1 and atom2s of the build list
            #apply that rotation using atom1 as the origin
            new_pos, darted_pose = self._moldRedart(atom_indices=self.atom_indices,
                                            binding_mode_pos=self.binding_mode_traj,
                                            binding_mode_index=self.selected_pose,
                                            nc_pos=oldDartPos,
                                            rigid_darts=self.rigid_darts)

            self.selected_pose = darted_pose
            context.setPositions(new_pos)
            overlap_after = self._poseDart(context, self.atom_indices)
            #the acceptance depends on the instantaenous move
            #therefore find the ratio of number of poses before and after
            self.num_poses_end = len(overlap_after)

            # to maintain detailed balance, check to see the overlap of the start and end darting regions
            # if there is no overlap after the move, acceptance ratio will be 0

            #check if new positions overlap when moving
            if self.restraints:
                for i in range(len(self.binding_mode_traj)):
                    context.setParameter('restraint_pose_'+str(i), 0)
                context.setParameter('restraint_pose_'+str(self.selected_pose), 1)

        if self.freeze_waters > 0:
            start_state = context.getState(getPositions=True, getVelocities=True)
            start_pos = start_state.getPositions(asNumpy=True)
            start_vel = start_state.getVelocities(asNumpy=True)
            switch_pos = np.copy(start_pos)*start_pos.unit
            switch_vel = np.copy(start_vel)*start_vel.unit
            lig_com = self.getCenterOfMass(switch_pos[self.atom_indices], masses=self.masses)
            water_pos = switch_pos.take(self.water_oxygens, axis=0)*unit.nanometers
            water_dist = np.linalg.norm(np.subtract(water_pos, lig_com), axis=1)
            water_sort = np.argsort(water_dist)[:self.freeze_waters]
            waters_within_distance_new = sorted([self.water_residues[i] for i in water_sort])
            culled_new = [i for i in waters_within_distance_new if i not in self.waters_within_distance]
            culled_old = [i for i in self.waters_within_distance if i not in waters_within_distance_new]
            for new_water, old_water in list(zip(culled_new, culled_old)):
                    for j in range(3):
                        switch_pos[old_water[j]] = copy.deepcopy(start_pos[new_water[j]])
                        switch_pos[new_water[j]] = copy.deepcopy(start_pos[old_water[j]])
                        switch_vel[old_water[j]] = start_vel[new_water[j]]
                        switch_vel[new_water[j]] = start_vel[old_water[j]]
            context.setPositions(switch_pos)
            context.setVelocities(switch_vel)
        if self.rigid_body:
            after_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
            resetRigidBodies(context.getSystem(), after_pos, self.real_particles, self.vsiteParticles, self.constraint_list, self.atom_indices)
            context.reinitialize(preserveState=True)
        return context


class AlchemicalExternalRestrainedLangevinIntegrator(AlchemicalExternalLangevinIntegrator):
    def __init__(self,
                 alchemical_functions,
                 restraint_group,
                 splitting="R V O H O V R",
                 temperature=298.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True,
                 nsteps_neq=100,
                 nprop=1,
                 prop_lambda=0.3,
                 lambda_restraints='max(0, 1-(1/0.10)*abs(lambda-0.5))',
                 *args, **kwargs):
        self.lambda_restraints = lambda_restraints
        self.restraint_energy = "energy"+str(restraint_group)

        super(AlchemicalExternalRestrainedLangevinIntegrator, self).__init__(
                     alchemical_functions,
                     splitting,
                     temperature,
                     collision_rate,
                     timestep,
                     constraint_tolerance,
                     measure_shadow_work,
                     measure_heat,
                     nsteps_neq,
                     nprop,
                     prop_lambda,
                     *args, **kwargs)
        try:
            self.addGlobalVariable("restraint_energy", 0)
        except:
            pass


    def updateRestraints(self):
        self.addComputeGlobal('lambda_restraints', self.lambda_restraints)

    def _add_integrator_steps(self):
        """
        Override the base class to insert reset steps around the integrator.
        """

        # First step: Constrain positions and velocities and reset work accumulators and alchemical integrators
        self.beginIfBlock('step = 0')
        self.addComputeGlobal("restraint_energy", self.restraint_energy)
        self.addComputeGlobal("perturbed_pe", "energy-restraint_energy")
        self.addComputeGlobal("unperturbed_pe", "energy-restraint_energy")
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self._add_reset_protocol_work_step()
        self._add_alchemical_reset_step()
        self.endBlock()

        # Main body

        if self._n_steps_neq == 0:
            # If nsteps = 0, we need to force execution on the first step only.
            self.beginIfBlock('step = 0')
            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            self.addComputeGlobal("step", "step + 1")
            self.endBlock()
        else:
            #call the superclass function to insert the appropriate steps, provided the step number is less than n_steps
            self.beginIfBlock("step < nsteps")#
            self.addComputeGlobal("restraint_energy", self.restraint_energy)
            self.addComputeGlobal("perturbed_pe", "energy-restraint_energy")
            self.beginIfBlock("first_step < 1")##
            #TODO write better test that checks that the initial work isn't gigantic
            self.addComputeGlobal("first_step", "1")
            self.addComputeGlobal("restraint_energy", self.restraint_energy)
            self.addComputeGlobal("unperturbed_pe", "energy-restraint_energy")
            self.endBlock()##
            #initial iteration
            self.addComputeGlobal("protocol_work", "protocol_work + (perturbed_pe - unperturbed_pe)")
            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            #if more propogation steps are requested
            self.beginIfBlock("lambda > prop_lambda_min")###
            self.beginIfBlock("lambda <= prop_lambda_max")####

            self.beginWhileBlock("prop < nprop")#####
            self.addComputeGlobal("prop", "prop + 1")

            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            self.endBlock()#####
            self.endBlock()####
            self.endBlock()###
            #ending variables to reset
            self.updateRestraints()
            self.addComputeGlobal("restraint_energy", self.restraint_energy)
            self.addComputeGlobal("unperturbed_pe", "energy-restraint_energy")
            self.addComputeGlobal("step", "step + 1")
            self.addComputeGlobal("prop", "1")

            self.endBlock()#

