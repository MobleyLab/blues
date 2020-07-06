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
        restrained_ligand_atoms=None,
        receptor_cutoff=0.8,
        K_r=10, K_angle=10, K_RMSD=0.6, RMSD0=2,
        rigid_body=False,
        centroid_darting=True,
        lambda_restraints='max(0, 1-(1/0.10)*abs(lambda-0.5))',
        darting_sampling='uniform',
        buildlist=None,
        ignore_warning=False,
        old_restraint=True
        ):
        super(MolDartMove, self).__init__(structure, resname)
        self.old_restraint = old_restraint
        #md trajectory representation of only the ligand atoms
        self.pdb_files = pdb_files
        self.trajs = [md.load(traj) for traj in pdb_files]
        self.min_atoms = min([traj.n_atoms for traj in self.trajs])
        self.trajs = [traj.atom_slice(range(self.min_atoms), inplace=True) for traj in self.trajs]
        self.restrained_receptor_atoms = []
        print('restrained_receptor_atoms', restrained_receptor_atoms)

        if restrained_receptor_atoms is None:
            for traj in self.trajs:
                ca_atoms = traj.top.select('name CA and protein')
                receptor_atoms = md.compute_neighbors(traj, cutoff=receptor_cutoff, query_indices=self.atom_indices, haystack_indices=ca_atoms)
                self.restrained_receptor_atoms.append(receptor_atoms[0])
        elif all(isinstance(item, int) for item in restrained_receptor_atoms):
            self.restrained_receptor_atoms = [restrained_receptor_atoms for i in range(len(pdb_files))]

        elif all(isinstance(item, list) for item in restrained_receptor_atoms) and len(restrained_receptor_atoms) == len(pdb_files):
            self.restrained_receptor_atoms = restrained_receptor_atoms
            #exit()
        #for atom in self.restrained_receptor_atoms:
        if 1:
            self.indiv_atoms = []
            self.indiv_sidechain_atoms = []
            self.indiv_residue_atoms = []
            for receptor_atoms in self.restrained_receptor_atoms:
                if 0:
                    indiv_atoms = [atom.index for atom in self.trajs[0].topology.atoms if atom.index in receptor_atoms]
                    self.indiv_atoms.append(indiv_atoms)
                    indiv_residues = list(set(atom.residue for atom in self.trajs[0].topology.atoms if atom.index in receptor_atoms))
                    indiv_residue_atoms = [atom.index for atom in self.trajs[0].topology.atoms if atom.residue in indiv_residues]
                    indiv_sidechain_atoms = [atom.index for atom in self.trajs[0].topology.atoms if (atom.residue in indiv_residues) and atom.is_sidechain]
                else:
                    ring_sidechain = ['HIS', 'PRO', 'TYR', 'TRP']
                    indiv_residues = list(set([atom.residue for atom in self.trajs[0].topology.atoms if atom.index in receptor_atoms and atom.residue.name not in ring_sidechain]))
                    indiv_residue_atoms = [atom.index for atom in self.trajs[0].topology.atoms if atom.residue in indiv_residues and atom.residue.name not in ring_sidechain]
                    indiv_sidechain_atoms = [atom.index for atom in self.trajs[0].topology.atoms if (atom.residue in indiv_residues) and atom.is_sidechain]

                self.indiv_sidechain_atoms.append(indiv_sidechain_atoms)
                self.indiv_residue_atoms.append(indiv_residue_atoms)

            self.all_sidechain_atoms = list(set([item for sublist in self.indiv_sidechain_atoms for item in sublist]))
            self.all_residue_atoms = list(set([item for sublist in self.indiv_residue_atoms for item in sublist]))
            print('allsidechain', len(self.all_sidechain_atoms))
            print('allress', len(self.all_residue_atoms))
            #exit()

            if 1:
                import oeommtools
                from openeye import oechem

                top = self.structure.topology
                pos = self.structure.positions
                molecule = oeommtools.utils.openmmTop_to_oemol(top, pos, verbose=False)
                oechem.OEPerceiveResidues(molecule)
                oechem.OEFindRingAtomsAndBonds(molecule)


        print('self.restrained_receptor_atoms', self.restrained_receptor_atoms)

        self.binding_mode_traj = []
        #positions of only the ligand atoms
        self.binding_mode_pos = []
        #fit atoms are the atom indices which should be fit to to remove rot/trans changes
        self.fit_atoms = fit_atoms
        if 1:
            if fit_atoms is None:
                self.fit_atoms = []
                for index, traj in enumerate(self.trajs):
                    ca_atoms = traj.top.select('name CA and protein')
                    receptor_atoms = md.compute_neighbors(traj, cutoff=receptor_cutoff, query_indices=self.atom_indices, haystack_indices=ca_atoms)
                    self.fit_atoms.append(receptor_atoms[0])
            elif all(isinstance(item, int) for item in fit_atoms):
                self.fit_atoms = [fit_atoms for i in range(len(pdb_files))]

            elif all(isinstance(item, list) for item in fit_atoms) and len(fit_atoms) == len(pdb_files):
                self.fit_atoms = fit_atoms
            print('fit_atoms', fit_atoms)
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
        self.same_range = True #for trajectory darting
        self.restrained_ligand_atoms = restrained_ligand_atoms
        if darting_sampling not in ['uniform', 'gaussian']:
            raise ValueError('darting_sampling must be either gaussian or uniform')
        self.darting_sampling = darting_sampling
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
            if self.buildlist == None:
                self.buildlist = self._createBuildlist(pdb_files, self.atom_indices)
            self.traj_dart_dict = self._findDihedralRingAtoms(pdb_files, atom_indices=self.atom_indices, rigid_darts=self.rigid_darts, buildlist=self.buildlist)
            #print('self.traj_dart_dict keys1', self.traj_dart_dict)
            #print('self.traj_dart_dict keys', list(self.traj_dart_dict.keys()))
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
            self.internal_xyz, self.internal_zmat, self.binding_mode_pos, self.binding_mode_traj = self._createZmat(pdb_files, atom_indices=self.atom_indices, reference_traj=ref_traj, buildlist=self.buildlist)

        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])
        self.sim_ref = copy.deepcopy(self.binding_mode_traj[0])

        self.darts = makeDartDict(self.internal_zmat, self.binding_mode_pos, self.buildlist,
                                dihedral_cutoff=dihedral_cutoff, distance_cutoff=distance_cutoff,
                                rotation_cutoff=rotation_cutoff, dart_buffer=dart_buffer, order=dart_region_order,
                                ignore_warning=ignore_warning)
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
        if self.rigid_darts is not None and dihedral_diff_df is not None:
            #find the bonded atoms that are not part of the dihedral ri
#            core = list(set([self.buildlist.at[i, 'b'] for i in dihedral_diff_df['atomnum'].values if i not in self.dihedral_ring_atoms]))
            print('self buildlist', self.buildlist)
            print("dihedral_diff_df['atomnum']", dihedral_diff_df)
            core = list(set([self.buildlist.at[i, 'b'] for i in dihedral_diff_df['atomnum'].values]))

            #self.only_darts_dihedrals = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in core]
            #self.only_darts_dihedrals = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in core]

            self.only_darts_dihedrals = [i for i in self.darts['dihedral'].keys()]
        self.zmat_inital_atoms = np.array(self.atom_indices)[self.internal_zmat[0].index.values[:3]].tolist()


    def refitPoses(self, current_pose, trajs, fit_atoms, atom_indices):
        #fits trajectory onto reference poses
        #current_pose current trajectory traj



        binding_mode_traj = [traj.superpose(current_pose, atom_indices=fit_atoms[index], ref_atom_indices=fit_atoms[index]).atom_slice(self.atom_indices) for index, traj in enumerate(trajs)]
        #logger.info("binding_mode_traj1 {}".format(binding_mode_traj))
        #binding_mode_pos = [np.asarray(traj.superpose(current_pose, atom_indices=fit_atoms[index], ref_atom_indices=fit_atoms[index]).xyz[0]) for index, traj in enumerate(self.trajs)]
        binding_mode_pos = [np.asarray(traj.xyz[0])*10.0 for index, traj in enumerate(self.trajs)]
        #logger.info("binding_mode_pos1 {}".format(binding_mode_pos))

        binding_mode_pos = [traj[self.atom_indices] for traj in binding_mode_pos]
        #logger.info("binding_mode_pos2 {}".format(binding_mode_pos))

        return binding_mode_traj, binding_mode_pos


    @classmethod
    def _loadfiles(cls, structure_files, topology, stride=None):
        try:
            traj = md.load(structure_files, top=topology, stride=stride)
        except TypeError:
            traj = md.load(structure_files)

        return traj
    @classmethod
    def _createZmat(cls, structure_files, atom_indices, topology=None, reference_traj=None, fit_atoms=None, buildlist=None):
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
        if buildlist is None:
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
                num_atoms = min(num_atoms_traj, num_atoms_pdb, reference_traj.n_atoms)
                ref_traj_copy = copy.deepcopy(reference_traj)
                traj.atom_slice(range(num_atoms), inplace=True)
                ref_traj_copy.atom_slice(range(num_atoms), inplace=True)
                if fit_atoms == None:
                    traj.superpose(reference=ref_traj_copy, atom_indices=fit_atoms,
                        ref_atom_indices=fit_atoms
                        )
                else:
                    traj.superpose(reference=ref_traj_copy, atom_indices=fit_atoms[j],
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
    def _getDarts(cls, structure_files, atom_indices, topology=None, reference_traj=None, fit_atoms=None,
        dihedral_cutoff=0.5, distance_cutoff=5.5, rotation_cutoff=29.0, dart_buffer=0.9, order=['translation', 'dihedral',  'rotation'],
        buildlist=None):
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
                    fit_atoms=fit_atoms,
                    buildlist=buildlist)
        if buildlist is None:
            buildlist = MolDartMove._createBuildlist(structure_files, atom_indices, topology=topology)
        darts = makeDartDict(internal_zmat, binding_mode_pos, buildlist, dihedral_cutoff=dihedral_cutoff, distance_cutoff=distance_cutoff, rotation_cutoff=rotation_cutoff,
                            dart_buffer=dart_buffer, order=order)
        return darts

    @classmethod
    def getMaxRange(cls, dihedral, pose_value=None, density_percent=0.9, bandwidth=0.15, debug=False):
        """
        Parameters
        ----------
        dihedral: array
            Dihedral array of dihedral values (in radians).
        pose_value: float
            Value that corresponds to some maximum of the dihedral (in radians)
        deinsity_percent: float, optional, default=0.9
            Value that determines how large darting region should be to take up that percent
            of the probability density.
        bandwidth: float, optional, default=0.15
            How wide the bandwidth is for kernal density estimation
        Returns
        -------
        dart_storage: dict
            Dict containing the darts associated with `rotation`, `translation` and `dihedral`
            keys that refer to the size of the given dart, if not empty

        """

        import matplotlib
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt
        orig_dihedral = dihedral
        #make the range larger than the periodic boundries
        dihedral = np.concatenate((dihedral-2*np.pi, dihedral, dihedral+2*np.pi))
        #print('len', np.size(dihedral))
        if density_percent > 1.0:
            raise ValueError('density_percent must be less than 1!')
        pi_range = np.linspace(-2*np.pi, 2*np.pi+np.pi/50.0, num=360, endpoint=True).reshape(-1,1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dihedral)

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
            print('doing min')
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
            print('doing else')
            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots()
            #ax.plot(pi_range, np.exp(log_dens), '-')

            #there are multiple minima
            print('max_value', max_value)
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
                print('min_region', min_region)
                print('pose_value', pose_value)
                print('pose_value, i-min_region', pose_value, i-min_region)
                print('pose_value, i+min_region', pose_value, i+min_region)
                if np.greater(pose_value, i-min_region) and np.less(pose_value, i+min_region) or np.greater(pose_value+np.pi, i-min_region) and np.less(pose_value+np.pi, i+min_region):
                #if np.greater(pose_value, i-min_region) and np.less(pose_value, i+min_region):
                    debug_counter =1
                    max_return = i
                    #ax.vlines(max_return, 0, 1, colors='red')

                    pi_range = np.linspace(-2*np.pi, 2*np.pi+np.pi/50.0, num=360, endpoint=True).reshape(-1).tolist()
                    dx = pi_range[1]-pi_range[0]
                    print('index', index, 'i', i)
                    max_index = pi_range.index(i)
                    print('max_return', i, 'max_index', max_index)
                    if 1:
                        if i < -np.pi:
                            max_index = max_index + 179
                        elif i > np.pi:
                            max_index = max_index - 179
                    space = int(min_region / dx)
                    max_space_sub, max_space_add = (max_index-space, max_index+space)
                    #might have issues with the ends bleeding off, so address those
                    if max_space_sub < 0:
                        max_space_sub = 0
                    if max_space_add > 360:
                        max_space_add = 359
                    #target_range = log_dens.reshape(-1)[max_index-space:max_index+space]
                    target_range = log_dens.reshape(-1)[max_space_sub:max_space_add]

                    region_probability = cumtrapz(target_range, dx=dx)[-1]
                    #max region is all the max range of all the points in that dihedral region
                    max_region = space*dx
                    if density_percent == 1.00:
                        region_space = space*dx
                    else:
                        try:
                            probability = 0.0
                            for spacing in range(1,100):
                                probability = cumtrapz((log_dens.reshape(-1)[max_index-spacing:max_index+spacing]), dx=dx)[-1]
                                #print('probability', probability, probability/float(region_probability))
                                print('max_index-spacing, max_index+spacing', max_index-spacing, max_index+spacing, dx, np.pi/dx)
                                print('pi value', pi_range[max_index])
                                if probability/float(region_probability) >= density_percent:
                                    region_space = spacing*dx
                                    break
                        except Exception as e:
                            plt.clf()
                            plt.plot(pi_range, (log_dens))
                            #plt.xlim(-np.pi,np.pi)
                            plt.vlines(pose_value, 0,0.75, color='blue')
                            #plt.vlines(min_range, 0,0.75, color='red')
                            for j in max_value:
                                print('j', j)
                                plt.vlines(j, 0,0.75, color='red', linestyles='dashed')
                            for j in max_value:
                                plt.vlines(j, 0,0.75, color='red', linestyles='dashed')
                            #print('pi value', pi_range[max_index+180-1] )
                            plt.vlines(pi_range[max_index], 0,0.75, color='green')

                            plt.savefig('debug1.png')
                            print('exception', e)
                            exit()

                    break
                else:
                    plt.clf()
                    plt.plot(pi_range, (log_dens))
                    #plt.xlim(-np.pi,np.pi)
                    plt.vlines(pose_value, 0,0.75, color='blue')
                    #plt.vlines(min_range, 0,0.75, color='red')
                    for j in max_value:
                        plt.vlines(j, 0,0.75, color='red')

                    plt.savefig('debug.png')

                    #ADDED
                    #ax.plot(pi_range, np.exp(log_dens), '-')
                    #ax.vlines(pose_value, 0,1, colors='green')
                    #ADDED
                    pass
        if debug_counter == 0:
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
        bandwidth_filter = region_space/7.
        #new_dihedral = dihedral[(dihedral > max_return-region_space) & (dihedral < max_return+region_space)].reshape(-1,1)
        new_dihedral = dihedral[(dihedral > max_return-(region_space-bandwidth_filter)) & (dihedral < max_return+(region_space-bandwidth_filter))].reshape(-1,1)

        pi_range = np.linspace(-2*np.pi, 2*np.pi+np.pi/50.0, num=360, endpoint=True).reshape(-1,1)

        if 0:
            #plotting debugging
            new_kde = KernelDensity(kernel='tophat', bandwidth=0.025).fit(new_dihedral)
            new_log_dens = np.exp(new_kde.score_samples(pi_range))
            new_sample = new_kde.sample(1000)
            #print('new_sample', new_sample)
            f, axes = plt.subplots(2,1,sharex=True)
            axes[0].plot(pi_range, log_dens, color='orange', alpha=0.5)
            axes[0].plot(pi_range, new_log_dens, color='blue', ms=5)
            axes[0].axvline(max_return)
            axes[0].axvline(max_return-region_space)
            axes[0].axvline(max_return+region_space)
            #axes[0].axvline(new_sample, color='red')

            axes[0].set_xlabel('dihedral')
            axes[0].set_ylabel('probability')
            #axes[1].hist(new_sample,bins=10)
            #axes[1].axvline(max_return-region_space,color='red')
            #axes[1].axvline(max_return+region_space,color='red')
            from scipy.stats import norm
            mu, std = norm.fit(new_dihedral)
            #print('mu', mu, 'std', std)
            mu, std = norm.fit(new_dihedral)
            #might want to change this to use the original data mean (mu)
            #print('new_mu', mu, 'std', std)

            #print('max', norm(mu,std).cdf(max_return), norm(mu,std).cdf(max_return-region_space), norm(mu,std).cdf(max_return+region_space))
            #print('max', norm(max_return,std).cdf(max_return), norm(max_return,std).cdf(max_return-region_space), norm(max_return,std).cdf(max_return+region_space))
            percent_outside_region = norm(max_return,std).cdf(max_return-region_space)*2
            percent_increase = 1./(1.-percent_outside_region)
            #print('inverse', norm(max_return,std).ppf(0.5), percent_outside_region, percent_increase)
            #print('pdf', norm(max_return,std).pdf(max_return), 'pdf-region_space', norm(max_return,std).pdf(max_return-region_space))

            xmin, xmax = plt.xlim()
            x = np.linspace(-2*np.pi, 2*np.pi, 1000)
            p = norm.pdf(x, mu, std)
            axes[1].plot(x,p, linewidth=2,color='black')
            #check_sample = new_sample[(new_sample < max_return-region_space) or (new_sample > max_return+region_space)].reshape(-1,1)
            check_sample = new_sample[np.logical_or((new_sample < max_return-region_space), (new_sample > max_return+region_space))].reshape(-1,1)
            all_distribution = np.linspace(-2*np.pi,2*np.pi,36000)
            #scored_distribution = new_kde.score_samples(all_distribution.reshape(-1,1))
            #print('scored_distribution', scored_distribution)
            #exit()
            #print('check_sample', np.size(check_sample))
            #axes[1].hist(dihedral,bins=25,color='green',alpha=0.5)
            plt.show()
            exit()

        if debug:
            #plotting debugging
            new_kde = KernelDensity(kernel='tophat', bandwidth=0.025).fit(new_dihedral)
            new_log_dens = np.exp(new_kde.score_samples(pi_range))
            new_sample = new_kde.sample(1000)
            #print('new_sample', new_sample)
            f, axes = plt.subplots(2,1,sharex=False)
            if 0:
                axes[0].plot(pi_range, log_dens, color='orange', alpha=0.5)
                axes[0].plot(pi_range, new_log_dens, color='blue', ms=5)
                axes[0].axvline(max_return, color='green')
                axes[0].axvline(max_return-region_space, color='green')
                axes[0].axvline(max_return+region_space, color='green')
                #axes[0].axvline(new_sample, color='red')

            else:
                axes[0].plot(np.rad2deg(pi_range), log_dens, color='orange', alpha=0.5)
                axes[0].plot(np.rad2deg(pi_range), new_log_dens, color='blue', ms=5)
                axes[0].axvline(np.rad2deg(max_return))
                axes[0].axvline(np.rad2deg(max_return-region_space,), color='green')
                axes[0].axvline(np.rad2deg(max_return+region_space), color='green')

            axes[0].set_xlabel('dihedral')
            axes[0].set_ylabel('probability')
            #axes[1].hist(new_sample,bins=10)
            #axes[1].axvline(max_return-region_space,color='red')
            #axes[1].axvline(max_return+region_space,color='red')
            from scipy.stats import norm
            mu, std = norm.fit(new_dihedral)
            #print('mu', mu, 'std', std)
            mu, std = norm.fit(new_dihedral)
            #might want to change this to use the original data mean (mu)
            #print('new_mu', mu, 'std', std)

            #print('max', norm(mu,std).cdf(max_return), norm(mu,std).cdf(max_return-region_space), norm(mu,std).cdf(max_return+region_space))
            #print('max', norm(max_return,std).cdf(max_return), norm(max_return,std).cdf(max_return-region_space), norm(max_return,std).cdf(max_return+region_space))
            percent_outside_region = norm(max_return,std).cdf(max_return-region_space)*2
            percent_increase = 1./(1.-percent_outside_region)
            #print('inverse', norm(max_return,std).ppf(0.5), percent_outside_region, percent_increase)
            #print('pdf', norm(max_return,std).pdf(max_return), 'pdf-region_space', norm(max_return,std).pdf(max_return-region_space))

            xmin, xmax = plt.xlim()
            x = np.linspace(-2*np.pi, 2*np.pi, 1000)
            p = norm.pdf(x, mu, std)
            axes[1].plot(np.rad2deg(x),p, linewidth=2,color='black')
            #check_sample = new_sample[(new_sample < max_return-region_space) or (new_sample > max_return+region_space)].reshape(-1,1)
            check_sample = new_sample[np.logical_or((new_sample < max_return-region_space), (new_sample > max_return+region_space))].reshape(-1,1)
            all_distribution = np.linspace(-2*np.pi,2*np.pi,36000)
            #scored_distribution = new_kde.score_samples(all_distribution.reshape(-1,1))
            #print('scored_distribution', scored_distribution)
            #exit()
            #print('check_sample', np.size(check_sample))
            #axes[1].hist(dihedral,bins=25,color='green',alpha=0.5)
            axes[0].legend(('region_space:',np.rad2deg(region_space), ' max_return:',np.rad2deg(max_return)))
            output_name = 'plot_'+str(debug)+'.png'
            f.set_size_inches(10,4)
            plt.savefig(output_name)
            plt.close()
            #plt.show()


        if 1:
            #multiply by -1 since chemcoords dihedrals are opposite
            #region_space = (space*dx)/2.
            #new_dihedral = -1*dihedral[(dihedral > (max_return-region_space)) & (dihedral < (max_return+region_space))].reshape(-1,1)
            #new_dihedral = -1*dihedral[(dihedral > (max_return-region_space)) & (dihedral < (max_return+region_space))].reshape(-1,1)
            new_dihedral = dihedral[(dihedral > (max_return-region_space)) & (dihedral < (max_return+region_space))].reshape(-1,1)

            #new_dihedral = -1*dihedral

            from scipy.stats import norm
            #multiply by -1 since chemcoords dihedrals are opposite
#            max_return_deg = np.rad2deg(-1*max_return)
            max_return_deg = np.rad2deg(max_return)

            mu, std = norm.fit(np.rad2deg(new_dihedral))
            gauss = norm(max_return_deg,std)
            percent_outside_region = gauss.cdf(max_return_deg-np.rad2deg(region_space))*2
            percent_increase = 1./(1.-percent_outside_region)
            #print('region_space', region_space, 'max_return', max_return)
            #print('percent_outside_region', percent_outside_region, percent_increase)
            #print('random_samples', gauss.rvs(size=10, random_state=2),gauss.pdf(gauss.rvs(size=10, random_state=2)))
            pi_space = np.linspace(-2*180,2*180,720)
            if 0:
                plt.plot(pi_space,gauss.pdf(pi_space))
                plt.hist(np.rad2deg(new_dihedral),bins=100,density=True)

                plt.axvline(max_return_deg-np.rad2deg(region_space))
                plt.axvline(max_return_deg+np.rad2deg(region_space))

                plt.show()
                exit()

        if 0:
            #plotting debugging
            f, axes = plt.subplots(4,1,sharex=False)
            axes[0].scatter(range(np.size(orig_dihedral)), orig_dihedral)
            axes[0].set_xlabel('frame')
            axes[0].set_ylabel('dihedral')
            axes[1].hist(dihedral, bins=100)
            axes[1].set_xlim(-2*np.pi,2*np.pi)
            axes[1].set_xlabel('dihedral (radians)')
            axes[1].set_ylabel('counts')

            axes[2].plot(pi_range, log_dens, color='orange', alpha=0.5)
            #axes[1].plot(pi_range, new_log_dens, color='blue', ms=5)
            #axes[0].axvline(new_sample, color='red')
            axes[2].set_xlabel('dihedral (radians)')
            axes[2].set_ylabel('probability')

            axes[3].plot(pi_range, log_dens, color='orange', alpha=0.5)
            #axes[1].plot(pi_range, new_log_dens, color='blue', ms=5)
            axes[3].axvline(max_return)
            axes[3].axvline(max_return-region_space)
            axes[3].axvline(max_return+region_space)
            #axes[0].axvline(new_sample, color='red')

            axes[3].set_xlabel('dihedral (radians)')
            axes[3].set_ylabel('probability')
            axes[3].axvline(max_return-region_space,color='red')
            axes[3].axvline(max_return+region_space,color='red')

            plt.show()


        print('max_return, region_space, gauss, percent_increase', max_return, region_space, gauss, percent_increase)
        return max_return, region_space, gauss, percent_increase




    def getDartsFromTrajs(self, traj_files, structure_files=None,  atom_indices=None,  topology=None, dihedral_select='pose',
                          density_percent=0.9, stride=None, same_range=True, set_self=True, bandwidth=0.15, debug=False, only_darts=False,
                          buildlist=None, fit_atoms=None, ref_traj=None):
        """
        Function to determine the dart ranges and the maximums if using dart ranges dart instead of determinisitcally.

        Parameters
        ----------
        List corresponding to the path of the trajectories to use to define the dart ranges for the dihedrals.
        structure_files: list of str
            List corresponding to the path of the structures to create representations of.
        atom_indices: list of ints
            The atom indices of the ligand in the structure files.
        topology: str, optional, default=None
            Path of topology file, if structure_files doesn't contain topology information.
        dihedral_select: 'pose', 'first', or 'last', default='pose'
            The pose of the dihedral to define the maximum of the dihedral range for the darts. 'pose' indicates
            to use the dihedral value of the poses provided from pdb_files.
            'first' specifies using the first frame of the trajectory from traj_files to specify the maximum,
            and similarly 'last' specifies using the last frame of the trajectory.
        density_percent: float, default=0.9
            How much density should be included in the range of the darts when determining how wide the darts should be.
            Ranges from 0-1.0.
        stride: int, default=None
            How much to stride the trajectories to determine the dart sizes, to make this function run faster.
        same_range: boolean, default=True
            Set all the ranges to be the same for each matching dihedral across each pose. *THE FALSE SETTING IS NOT
            VALIDATED YET, SO BE CAREFUL*
        set_self: boolean, default=True
            Changes self.internal_zmat to the new internal_zmat with the darting ranges.
        bandwidth: float, default=0.15
            The bandwidth used for kernal density estimation smoothing out the percentage densities.
        debug: boolean, default=False:
            Whether to output debug info for if this function doesn't work properly.
        only_darts, boolean, default=True:
            If true only tries to find the darting regions for dihedral darts that are already defined self.darts,
            else find darting ranges for all the dihedrals.
        buildlist, boolean, default=False:
            Specifies a buildlist to use when creating the zmatrices. If None, then uses the buildlist from self.buildlist.
        Returns
        -------
        dart_storage: dict
            Dict containing the darts associated with `rotation`, `translation` and `dihedral`
            keys that refer to the size of the given dart, if not empty

        """

        if structure_files == None:
            structure_files=self.pdb_files
        if topology==None:
            topology = structure_files[0]
        if atom_indices == None:
            atom_indices = self.atom_indices
        if buildlist == None:
            buildlist = self.buildlist
        if fit_atoms == None:
            fit_atoms = self.fit_atoms
        if ref_traj == None:
            ref_traj = self.ref_traj
        if topology == str:
            topology = [topology for i in len(structure_files)]
        
        traj_files = [self._loadfiles(i, topology[index], stride=stride) for index, i in enumerate(traj_files)]
        traj_files = [traj.atom_slice(range(self.min_atoms), inplace=True) for traj in traj_files]
        #for i,j in enumerate(structure_files):
        #    md.load(j).save(str(i)+'.pdb')
        internal_xyz, internal_zmat, binding_mode_pos, binding_mode_traj = self._createZmat(structure_files, atom_indices, topology, reference_traj=None, fit_atoms=None, buildlist=self.buildlist)
        traj_storage = []
        dihedral_angles = []
        trans_storage = {}
        rot_storage = []
        first_atom, second_atom, third_atom = self.zmat_inital_atoms
        print('first_atom', first_atom)
        for index, traj in enumerate(traj_files):
            print('ref_traj', ref_traj)
            print('traj', traj)
            print('fit_atoms[index]', fit_atoms[index])
            #traj.superpose(ref_traj, atom_indices=fit_atoms[index])
            traj.superpose(ref_traj, atom_indices=fit_atoms[index])

            lig_frames = traj.xyz[:,[first_atom,second_atom,third_atom],:]
            print(np.shape(lig_frames), lig_frames)
            print('first atom', lig_frames[:,0,:])
            print(np.mean(lig_frames[:,0,:],axis=0), np.std(lig_frames[:,0,:], axis=0))
            trans_storage[index] = {}
            trans_storage[index]['mean'] = np.mean(lig_frames[:,0,:],axis=0)
            trans_storage[index]['std'] = np.std(lig_frames[:,0,:], axis=0)
            traj[-1].save(str(index)+'_.pdb')
            ref_traj.save('ref.pdb')
        trans_storage[0]['std'] = trans_storage[0]['std'] * 2
        for index, traj in enumerate(traj_files):
            #should make it dependent on each trajectory
            trans_storage[index]['std'] = trans_storage[0]['std']

            #continue here
        self.trans_dart = trans_storage
        #return 0
        print(self.trans_dart)

        for i in internal_zmat[0].index.values[3:]:
            value_list = [i, internal_zmat[0].loc[i]['b'], internal_zmat[0].loc[i]['a'], internal_zmat[0].loc[i]['d']]
            dihedral_angles.append(value_list)

        for index, traj in enumerate(traj_files):
            dihedrals = md.compute_dihedrals(traj, dihedral_angles)
            #dihedrals in radians
            #print('dihedrals', dihedrals, np.shape(dihedrals))

            # only set values for darts if True
            if only_darts:
                selected_key = list(sorted(self.darts['dihedral'].keys()))
                print('1', selected_key)
                #selected_di = [list(internal_zmat[index].index.values[3:]).index(i) for i in selected_key]
                selected_di = [list(internal_zmat[index].index.values[3:]).index(i) for i in selected_key]

                print('1a', selected_di)

                print('2', internal_zmat[index].index.values[3:])
                #exit()
            else:
                selected_key = internal_zmat[index].index.values[3:]
                selected_di = range(len(internal_zmat[index].index.values[3:]))

            if dihedral_select=='first':
                pose_mins = [dihedrals[0,i] for i in range(len(internal_zmat[index].index.values[3:]))]
                #pose_mins = [dihedrals[0,i] for i in range(len(selected_di))]

            elif dihedral_select=='last':
#                pose_mins = [dihedrals[-1:i] for i in range(len(internal_zmat[index].index.values[3:]))]
                #pose_mins = [dihedrals[-1,i] for i in range(len(internal_zmat[index].index.values[3:]))]

                #pose_mins = [dihedrals[-1,i] for i in range(len(selected_di))]
                print('di shape', selected_di, np.shape(selected_di))
                print('len atoms', len(list(internal_zmat[index].index.values)))
                pose_mins = [dihedrals[-1,i] for i in selected_di]
                print('pose_mins', pose_mins)
                #exit()

            elif dihedral_select=='pose':
            #contains all the dictionary lists conrresponding to the min/max ranges
            #print('pose_mins', pose_mins, pose_mins[0])
                #pose_mins = [np.deg2rad(internal_zmat[index].loc[i,'dihedral']) for i in internal_zmat[index].index.values[3:]]

                pose_mins = [np.deg2rad(internal_zmat[index].loc[i,'dihedral']) for i in selected_di]
                print('pose_mins', pose_mins)
            #print('len_pose_mins', len(pose_mins))
            #exit()
            if debug != False:
                traj_dict = {value: self.getMaxRange(dihedrals[:,aindex].reshape(-1,1),
                #traj_dict = {value: self.getMaxRange(dihedrals[:,value].reshape(-1,1),

                    pose_mins[aindex], density_percent=density_percent, bandwidth=bandwidth,
                    debug=str(index)+'_'+str(value)) for aindex, value in enumerate(internal_zmat[index].index.values[3:])}
                    #debug=str(index)+'_'+str(value)) for aindex, value in enumerate(selected_di)}

            else:
                traj_dict = {selected_key[aindex]: self.getMaxRange(dihedrals[:,aindex].reshape(-1,1),
                #traj_dict = {value: self.getMaxRange(dihedrals[:,value].reshape(-1,1),
                    pose_mins[aindex], bandwidth=bandwidth,
                    #density_percent=density_percent) for aindex, value in enumerate(internal_zmat[index].index.values[3:])}
                    #density_percent=density_percent) for aindex, value in enumerate(selected_di)}
                    density_percent=density_percent) for aindex, value in enumerate(selected_di)}



            test = [(i,j,k,l) for i,j,k,l in dihedral_angles if i == 8]
            #print(traj_dict)
            traj_storage.append(traj_dict)
            print('traj_storage', traj_storage)
            #exit()
        #exit()

        output_mat = [copy.deepcopy(zmat) for zmat in internal_zmat]
        #print('output before', output_mat[0])
        for zindex, zmat in enumerate(output_mat):
            #the first three atoms are automatically placed, no there isn't a need for more info
            range_list = [0,0,0]
            dihedral_max = [0,0,0]
            gauss_list = [0,0,0]
            percent_list = [0,0,0]

            for  j in internal_zmat[zindex].index.values[3:]:
            #for  i in selected_di:
                #if j in selected_di:
                if j in selected_key:

                    print('j', j)
                    #i = selected_di.index(j)
                    #print('i', i)
                    print('j', j)
                    print('traj_storage[zindex]', traj_storage[zindex])
                    #zmat._frame.loc[i,'dihedral'] = np.rad2deg(traj_storage[zindex][i][0])
                    range_list.append(np.rad2deg(traj_storage[zindex][j][1]))
                    dihedral_max.append(np.rad2deg(traj_storage[zindex][j][0]))
                    gauss_list.append(traj_storage[zindex][j][2])
                    percent_list.append(traj_storage[zindex][j][3])
                else:
                    range_list.append(0)
                    dihedral_max.append(0)
                    gauss_list.append(0)
                    percent_list.append(0)

            #using mdtraj gives opposite sign compared to chemcoord, so multipy by -1
            zmat._frame['dihedral_max'] = [-1*di for di in dihedral_max] #TEMP
            #TEMP
            #zmat._frame['dihedral_max'] = [di for di in dihedral_max]

            zmat._frame['dart_range'] = range_list
            zmat._frame['gauss'] =  gauss_list
            zmat._frame['ratio'] = percent_list
            #print('traj_storage', traj_storage)
        if same_range:
            self.same_range = True
            old_zmat = copy.deepcopy(output_mat)
            #print('output_mat old', old_zmat)
            #for zmat1, zmat2 in zip(output_mat, old_zmat):
            #    print('diff', zmat1._frame['ratio'] - zmat2._frame['ratio'])

            #set dihedral ranges to minimum values
            starting_frame = copy.deepcopy(output_mat[0]._frame['dart_range'])
            for zindex, zmat in enumerate(output_mat):
                starting_frame= pd.concat([starting_frame, zmat._frame['dart_range']]).min(level=0)
            for zindex, zmat in enumerate(output_mat):
                zmat._frame['dart_range'] = starting_frame
            #print('minimum', starting_frame )
            #TODO: put section here for adjusting gaussian distribution
            #print('zmat before all')
            #print('output_mat', output_mat)
            #old_zmat = copy.deepcopy(output_mat)
            if 1:
                for zindex, zmat in enumerate(output_mat):
                    #adjust ratios based on new regions
                    for i in zmat._frame.index:
                        #skip for the first three entries (since those are invariant)
                        if zmat._frame.loc[i,'gauss'] == 0:
                            pass
                        else:
                            #TODO: make test for ratios
                            #find the percent density that lies outside the regions
                            #print('zmat._frame.loc[i,"gauss"])', zmat._frame.loc[i,'gauss'])
                            percent_outside_region = (zmat._frame.loc[i,'gauss']).cdf(zmat._frame.loc[i,'gauss'].mean()-zmat._frame.loc[i,'dart_range'])*2
                            #print('ratio before', zmat._frame.loc[i,'ratio'])
                            zmat._frame.loc[i,'ratio'] = 1./(1.-percent_outside_region)
                            #print('ratio after', zmat._frame.loc[i,'ratio'])
                #print('output_mat new', output_mat)
                #print('output_mat old', old_zmat)
                #for index, (zmat1, zmat2) in enumerate(zip(output_mat, old_zmat)):
                #    print('diff', index, zmat1._frame['ratio'] - zmat2._frame['ratio'])
            #print('output mat', output_mat)
            #print('one', output_mat[0]._frame.index)
            #exit()
        #if not same range we have to find the volume associated with each
        else:
            self.same_range = False
        #now have to set up darting using darting regions instead
        if set_self==True:
            self.internal_zmat = output_mat
        change_darts = True
        if change_darts ==  True:
            #change
            pass
            dihedral_difference = makeDihedralDifferenceDf(self.internal_zmat, dihedral_cutoff=0.01)
            #print("self.traj_dart_dict['rotate_list']", self.traj_dart_dict['rotate_list'])
            #print('traj_dart_dict', self.traj_dart_dict)
            #exit()
            for rotate in self.traj_dart_dict['rotate_list']:
                if 0:
                    #TODO check if this is necessary
                    output_atoms = []
                    #filter out H atoms
                    #for any of the rotatable bonds check if any entry in the dihedrals is in the bond groups
                    if any(elem in self.darts['dihedral'] for elem in self.traj_dart_dict['bond_groups'][rotate]) == False:
                        #check which atoms\\
                        for next_atom in  self.traj_dart_dict['bond_groups'][rotate]:
                            print('is it in there', next_atom, dihedral_difference.atomnum.values)
                            if next_atom not in self.darts['dihedral'] and (next_atom in dihedral_difference.atomnum.values):
                                output_atoms.append(next_atom)
                                print('next atom', next_atom, 'not in dihedral')
                    #print('test0', output_atoms)
                    #print('test', output_mat[0].index[output_atoms].tolist())
                    #selected_df = output_mat[0].loc[output_mat[0].index[output_atoms].tolist(), 'dart_range']
                    #print('debug', output_mat[0].loc[output_atoms])
                    selected_df = output_mat[0].loc[output_atoms, 'dart_range']

                    #print('selected_df\n', selected_df)
                    selected_df=selected_df[selected_df == selected_df.max()]
                    #print('max\n', selected_df)
                    #print('next', selected_df.iloc[0])
                    print('selected_df', selected_df, 'rotate', rotate)
                    try:
                        max_value = selected_df.iloc[0]
                        selected_index = selected_df.index.tolist()[0]
                        #print('max_value', max_value, 'selected_index', selected_index)
                        dihedral_difference = makeDihedralDifferenceDf(self.internal_zmat, dihedral_cutoff=0.01)
                        #print('dihedral difference', dihedral_difference)
                        #print('atom_num.loc', selected_index)
                        #print("dihedral_difference['atomnum'].loc[selected_index]", dihedral_difference['atomnum'].loc[selected_index])
                        #print("self.darts['dihedral'][selected_index]", self.darts['dihedral'][selected_index])
                        self.darts['dihedral'][selected_index] = dihedral_difference['atomnum'].loc[selected_index]
                        #print('debug1', dihedral_difference[dihedral_difference['atomnum']==selected_index]['diff'])
                        #self.darts['dihedral'][selected_index] = dihedral_difference[dihedral_difference['atomnum']==selected_index]
                        self.darts['dihedral'][selected_index] = dihedral_difference[dihedral_difference['atomnum']==selected_index]['diff'].iloc[0] / 2.0

                        #print('darts', self.darts)
                    except IndexError:
                        pass
                #if rotate not in self.darts['dihedral']:
                #    print('rotate', rotate, 'not in dihedral')

        def removeDartOverlaps():
            #temporary for now Might want to consider if darts are the same across all poses then to exclude it
            pass
        #print('darts', self.darts)

        return output_mat


    @classmethod
    def _checkTrajectoryDarts(cls, structure_files, atom_indices, traj_files, darts,
                             topology=None, reference_traj=None, fit_atoms=None, buildlist=None, internal_zmat=None):
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
        buildlist: boolean, default=False:
            Specifies a buildlist to use when creating the zmatrices. If None, then uses the buildlist from self.buildlist.


        Returns
        -------
        all_darts: list of lists
            List containing a list of ints for each traj_files item. Each int corresponds to a frame of that trajectory if it matches
            a pose from the poses specified in structure_files

        """
        if not isinstance(traj_files, list):
            traj_files = [traj_files]
        if not isinstance(structure_files, list):
            structure_files = [structure_files]

        print('sfiles', structure_files[0], topology)
        if reference_traj is None:
            reference_traj = cls._loadfiles(structure_files[0], topology)
        else:
            if isinstance(reference_traj, str):
                reference_traj = cls._loadfiles(reference_traj, topology)

        internal_xyz, internal_zmat_fromcreate, binding_mode_pos, binding_mode_traj = cls._createZmat(structure_files=structure_files,
                    atom_indices=atom_indices,
                    topology=topology,
                    reference_traj=reference_traj,
                    fit_atoms=fit_atoms,
                    buildlist=buildlist)
        if buildlist is None:
            buildlist = MolDartMove._createBuildlist(structure_files, atom_indices, topology=topology)
        if internal_zmat is None:
            internal_zmat = internal_zmat_fromcreate
        temp_xyz = copy.deepcopy(internal_xyz[0])
        all_darts = []
        for traj in traj_files:
            #print('traj', traj)
            traj = cls._loadfiles(traj, topology)
            print('traj', traj)
            print('reference', reference_traj)
            traj.superpose(reference=reference_traj, atom_indices=fit_atoms[0])
            traj_frames = []
            print('dart_storage',darts)
            #exit()
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
    def _checkTrajectoryDarts2(cls, structure_files, atom_indices, traj_files, darts,
                             topology=None, reference_traj=None, fit_atoms=None, buildlist=None, internal_zmat=None):
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
        buildlist: boolean, default=False:
            Specifies a buildlist to use when creating the zmatrices. If None, then uses the buildlist from self.buildlist.

        Returns
        -------
        all_darts: list of lists
            List containing a list of ints for each traj_files item. Each int corresponds to a frame of that trajectory if it matches
            a pose from the poses specified in structure_files

        """
        if not isinstance(traj_files, list):
            traj_files = [traj_files]
        if not isinstance(structure_files, list):
            structure_files = [structure_files]

        print('sfiles', structure_files[0], topology)
        if reference_traj is None:
            reference_traj = cls._loadfiles(structure_files[0], topology)
        else:
            if isinstance(reference_traj, str):
                reference_traj = cls._loadfiles(reference_traj, topology)

        internal_xyz, internal_zmat_fromcreate, binding_mode_pos, binding_mode_traj = cls._createZmat(structure_files=structure_files,
                    atom_indices=atom_indices,
                    topology=topology,
                    reference_traj=reference_traj,
                    fit_atoms=fit_atoms,
                    buildlist=buildlist)
        if buildlist is None:
            buildlist = MolDartMove._createBuildlist(structure_files, atom_indices, topology=topology)
        if internal_zmat is None:
            internal_zmat = internal_zmat_fromcreate
        temp_xyz = copy.deepcopy(internal_xyz[0])
        all_darts = []
        for traj in traj_files:
            #print('traj', traj)
            traj = cls._loadfiles(traj, topology)
            print('traj', traj)
            print('reference', reference_traj)
            traj.superpose(reference=reference_traj, atom_indices=fit_atoms[0])
            traj_frames = []
            print('dart_storage',darts)
            #exit()
            for frame in range(traj.n_frames):
                temp_xyz._frame.loc[:, ['x', 'y', 'z']] = traj.xyz[frame][atom_indices]*10
                temp_zmat = temp_xyz.get_zmat(construction_table=buildlist)
                traj_frames.append(temp_zmat._frame)
            all_darts.append(traj_frames)

        return all_darts


    @classmethod
    def _findDihedralRingAtoms(cls, structure_files, atom_indices, rigid_darts=False, buildlist=None):
        if buildlist is None:
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
                #first_buildlist = ['origin', 'e_x', 'e_y', 'e_z']

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
                    if 1:
                        #debugging
                        for atom in mol.GetAtoms():
                            for bond in atom.GetBonds():
                                #print('test')
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
                for dict_type in [rotatable_atom_bonds, rigid_atom_bonds]:
                    for key in dict_type:
                        dict_type[key] = list(set(dict_type[key]))
                #print('rotatable_atom_bonds', rotatable_atom_bonds)
                all_rotatable_bonds = list(set(all_rotatable_bonds))
                all_rotatable_bonds_dict = {}
                for item in all_rotatable_bonds:
                    all_rotatable_bonds_dict[item] = []
                #print('all_rotatable_bonds', all_rotatable_bonds)
                #print('debug0', rotatable_atom_bonds)
                for rotatable_atom_key, rotatable_atom_value in rotatable_atom_bonds.items():
                    #print('debug1')
                    #print('rotatable_atom', rotatable_atom_key, rotatable_atom_value)
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
                for key, value in all_rotatable_bonds_dict.items():
                    all_rotatable_bonds_dict[key] = list(set(value))
                #print('rotatable_atom_bonds new', all_rotatable_bonds_dict)
                #exit()
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
                #print('all_rotatable_bonds_dict', all_rotatable_bonds_dict)
                #print('rotatable atom bonds', rotatable_atom_bonds, 'ring_atoms', set(ring_atoms))
                #print('rigid_atom_bonds', rigid_atom_bonds)
                #print('all_rotatable_bonds', list(set(all_rotatable_bonds)))
                #print('all_rotatable_atom_dict', all_rotatable_bonds_dict)
                #print(buildlist)
                #check which rotatale bonds are attached to

            #check if an atom (atom1) is bonded to a rotatable atom (atom2) if all if so, check that all bonds bonded to atom1 are not part of the initial build list
            #if they aren't add that as a rotatable group (so the dihedral can change)
                #bgn_idx = [bond.GetBgnIdx() for bond in mol.GetBonds() if bond.GetEndIdx() in ring_atoms]
                #end_idx = [bond.GetEndIdx() for bond in mol.GetBonds() if bond.GetBgnIdx() in ring_atoms]
            rigid_atoms = ring_atoms
            #select all atoms that are bonded to a ring/double bond atom
            angle_ring_atoms = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] in rigid_atoms]
            #print('mols', ifs.GetOEGraphMols())
            #print('atom_list', atom_list)
            new_atoms = [list(x) for x in set(tuple(x) for x in ring_atom_list)]
            unique_list = []
            for entry in new_atoms:
                for entry2 in new_atoms:
                    if entry != entry2:
                        if all(elem in entry for elem in entry2):
                            unique_list.append(entry)
            #print('unique_list', unique_list)
            set_1 = set()
            if 0:
                for item in unique_list:
                    #print('item', type(item), item, set(item))
                    print(set(item))
                    new_item = set(item)
                    set_1.add(new_item)

                #print('set', {set(x) for x in unique_list})
            unique_set = set({frozenset(sorted(x)) for x in unique_list})
            #print('unique_set', unique_set)
            unique_list = list(set({frozenset(sorted(x)) for x in unique_list}))
            #find end atoms (atoms that are bonded to atoms)

            #self.dihedral_ring_atoms = list(set(angle_ring_atoms + h_list))
            #self.dihedral_ring_atoms = list(set(rigid_atoms + h_list))
            dihedral_ring_atoms = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] in rigid_atoms]
            #find the order of atoms and the indices that should be treated the same (if any)
            rotate_keys = list(rotatable_atom_bonds.keys())
            #want to sort based on build order in
            rotate_keys_sort = list((buildlist.index.tolist().index(i) for i in rotate_keys))
            #print('first', rotate_keys_sort)
            #print('np.argsort(rotate_keys_sort)', np.argsort(rotate_keys_sort))

            rotate_keys = [rotate_keys[np.argsort(rotate_keys_sort)[i]] for i in range(len(rotate_keys))]
            #if the first 3 atoms are part of the build list then ignore them since their rotations are already accounted for
            #print('test build', buildlist['b'].loc[3:])
            first_buildlist = buildlist.index.tolist()[:3:] + ['origin', 'e_x', 'e_y', 'e_z']
            #first_buildlist = ['origin', 'e_x', 'e_y', 'e_z']

            #rotate_keys = [i for i in rotate_keys if i not in buildlist['b']]
            #rotate_keys = [i for i in rotate_keys if i not in buildlist.index.tolist()[:3]]
            #print(buildlist)
            #print('rotate_keys', rotate_keys)
            bond_groups = {}
            for i in buildlist.index:
                #print('test', i, buildlist.loc[i, ['b', 'a', 'd']])
                #print('test', i, buildlist[buildlist['b'] == i], buildlist[buildlist['b'] == i].index.tolist())
                testa = buildlist[buildlist['b'] == i]
                bond_groups[i] = buildlist[buildlist['b'] == i].index.tolist()

            #find the atoms bonded to these rotate_key atoms
            rotate_list = []
            ring_tracker = copy.deepcopy(unique_list)
            #print('rotate_keys1', rotate_keys)
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
            #print('rotate_list0', rotate_list)
            for value in all_rotatable_bonds_dict.values():
                print('value', value)
                for target_atom in value:
                    print('target_atom', target_atom)
                    if target_atom not in rotate_list:
                        rotate_list.append(target_atom)
            #print('rotate_list1', rotate_list)
            #NEW THING: check if the first 3/4 atoms take care of a ring. If that's the case then skip that one
            #print(buildlist.loc[rotate_keys])
            #print('bond groups', bond_groups)
            output_dict = {}
            output_dict['rotate_list'] = rotate_list
            #print('rotate_list', rotate_list)
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
        #print('current_zmat', current_zmat)
        #logger.info("Freezing selection '{}' ({} atoms) on {}".format(freeze_selection, len(mask_idx), system))
        logger.info("sim_traj {}".format(self.sim_traj))
        logger.info("self.trajs {}".format(self.trajs))


        #print('sim_traj', self.sim_traj)
        #print('self.trajs', self.trajs)
        #logger.info("before self.binding_mode_traj {} {}".format(len(self.binding_mode_traj), self.binding_mode_traj))
        #logger.info("before self.binding_mode_pos {} {}".format(len(self.binding_mode_pos), self.binding_mode_pos))
        #logger.info("before self.binding_mode_thetrajs {} {}".format(len(self.trajs), self.trajs))
        self.binding_mode_traj, self.binding_mode_pos = self.refitPoses(self.sim_traj, self.trajs, self.fit_atoms, self.atom_indices)
        #logger.info("self.binding_mode_traj {} {}".format(len(self.binding_mode_traj), self.binding_mode_traj))
        #logger.info("self.binding_mode_pos {} {}".format(len(self.binding_mode_pos), self.binding_mode_pos))
        #print('type', type(current_zmat))
        #print('type internal_zmat', type(self.internal_zmat))
        selected = checkDart(self.internal_zmat, current_pos=(np.array(self.sim_traj.openmm_positions(0)._value))[self.atom_indices]*10,

                    current_zmat=current_zmat, pos_list=self.binding_mode_pos,
                    construction_table=self.buildlist,
                    dart_storage=self.darts
                    )
        print('selected_dart', selected)
        if len(selected) >= 1:
            #returns binding mode
            #diff_list will be used to dart
            if 0:
                trans_region = []
                print('sim_traj', self.sim_traj)
                copy_traj = copy.deepcopy(self.sim_traj)
                print('self.trans_dart', self.trans_dart)
                print('self.zmat_inital_atoms[0]', self.zmat_inital_atoms[0])
                #if 'dart_range' in self.internal_zmat[0]._frame:
                for key, value in self.trans_dart.items():
                    copy_traj.superpose(self.ref_traj, atom_indices=self.fit_atoms[key])

                    #print('self.sim_traj.xyz[0][ligand_atoms] - value["mean"]', self.sim_traj.xyz[0][self.zmat_inital_atoms[0]] - value['mean'])
                    #print('self.sim_traj.xyz[0][self.zmat_inital_atoms[0]]', self.sim_traj.xyz[0][self.zmat_inital_atoms[0]], value['mean'], value['std'])
                    print('self.sim_traj.xyz[0][ligand_atoms] - value["mean"]', copy_traj.xyz[0][self.zmat_inital_atoms[0]] - value['mean'])
                    print('self.sim_traj.xyz[0][self.zmat_inital_atoms[0]]', copy_traj.xyz[0][self.zmat_inital_atoms[0]], value['mean'], value['std'])

                    #if np.all(np.less_equal(self.sim_traj.xyz[0][self.zmat_inital_atoms[0]] - value['mean'], value['std'])):
                    if np.all(np.less_equal(np.abs(copy_traj.xyz[0][self.zmat_inital_atoms[0]] - value['mean']), value['std'])):

                            trans_region.append(key)
                output = []
                for selected_value in selected:
                    if selected_value in trans_region:
                        output.append(selected_value)

                print('trans overlap', trans_region)
                print('output', output)
                return output
            return selected
        elif len(selected) == 0:
            return []

    def _dart_selection(self, binding_mode_index, transition_matrix, same_range=True, ):
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
        if same_range==True:
            rand_index = np.random.choice(self.dart_groups, p=transition_matrix[binding_mode_index])
            prob_forward = transition_matrix[binding_mode_index][rand_index]
            prob_reverse = transition_matrix[rand_index][binding_mode_index]
            acceptance_ratio = float(prob_reverse)/prob_forward
        elif same_range==False:
            selected_dihedral = list(self.darts['dihedral'].keys())
            density_list = []
            for index, zmat in enumerate(self.internal_zmat):
                if index != binding_mode_index:
                    print('selected_dihedral', selected_dihedral)
                    print('index', index, zmat['dart_range'])
                    sum_total = zmat.loc[selected_dihedral, 'dart_range'].sum()
                    print(sum_total)
                    density = sum_total
                    density_list.append(density)
                else:
                    print('index', index, zmat['dart_range'])

                    #ignore density for the current binding mode because it isn't chosen
                    pass
                    density_list.append(0.0)

            #print('density_list', density_list, 'binding_mode_index', binding_mode_index )
            #density_list.pop(binding_mode_index)
            print('density_list', density_list, 'binding_mode_index', binding_mode_index )

            density_array = np.array(density_list)/np.sum(density_list)
            print('density_array', density_array)
            print('dart groups', self.dart_groups)
            print(self.internal_zmat)
            rand_index = np.random.choice(self.dart_groups, p=transition_matrix[binding_mode_index])
            #change the acceptance ratio here
            acceptance_ratio = 1.0
            #exit()
        return rand_index, acceptance_ratio

    def _dart_selection_edit(self, binding_mode_index, transition_matrix):
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

        rand_index, self.dart_ratio = self._dart_selection(binding_mode_index, self.transition_matrix, same_range=self.same_range)
        #print('rand_index', rand_index)
        dart_ratio = self.dart_ratio
        #print('dart_ratio', dart_ratio)
        self.acceptance_ratio = self.acceptance_ratio * dart_ratio


        #get matching binding mode pose and get rotation/translation to that pose
        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        traj_positions = self.sim_traj.xyz[0]
        xyz_ref._frame.loc[:, ['x', 'y', 'z']] = traj_positions[self.atom_indices]*10
        zmat_new = copy.deepcopy(self.internal_zmat[rand_index])
        zmat_diff = xyz_ref.get_zmat(construction_table=self.buildlist)
        print('zmat_from_traj', zmat_diff)
        zmat_traj = copy.deepcopy(xyz_ref.get_zmat(construction_table=self.buildlist))
        #get appropriate comparision zmat
        zmat_compare = self.internal_zmat[binding_mode_index]
        #we don't need to change the bonds/dihedrals since they are fast to sample
        #if the molecule is treated as rigid, we won't change the internal coordinates
        #otherwise we find the differences in the dihedral angles between the simulation
        #and reference poses and take that into account when darting to the new pose
        change_list = ['dihedral']
        old_list = ['bond', 'angle']
        #TODO: play around with this
        #print('ring_atoms', self.dihedral_ring_atoms)
        #print('rotate_list', self.traj_dart_dict['rotate_list'])
        #print('bond_groups', self.traj_dart_dict['bond_groups'])
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
            #print('rigid_dihedrals_atoms', sorted(rigid_dihedrals_atoms), 'move_atoms', sorted(move_atoms))
            rigid_dihedrals_atoms = [i for i in zmat_new._frame.index if i not in self.only_darts_dihedrals]
            #print('test0', zmat_new)

            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]
            #print('test1a', zmat_new)
            #print('zmat_traj', zmat_traj)
        elif rigid_darts == 'flexible_darts':
            #rigid_dihedrals_atoms = [i for i in self.only_darts_dihedrals if i in zmat_new._frame.index[3:]]
            move_atoms = []
            for anchor_atoms in self.traj_dart_dict['rotate_list']:
                for rotate_atoms in self.traj_dart_dict['bond_groups'][anchor_atoms]:
                    move_atoms.append(rotate_atoms)
            #rigid_dihedrals_atoms = [i for i in zmat_new._frame.index[3:] if i not in move_atoms]
            #print('rigid_dihedrals_atoms', sorted(rigid_dihedrals_atoms), 'move_atoms', sorted(move_atoms))
            rigid_dihedrals_atoms = [i for i in zmat_new._frame.index if i not in self.only_darts_dihedrals]
            print('rigid_dihedrals_atoms', rigid_dihedrals_atoms)
            #print('test0', zmat_new)

            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]
            #print('test1a', zmat_new)
            #print('zmat_traj', zmat_traj)

        if rigid_darts == 'rigid_molecule':
            old_list =  old_list + change_list
            rigid_dihedrals_atoms = [i for i in zmat_new._frame.index[3:]]

        else:
            if 'dart_range' in self.internal_zmat[binding_mode_index]._frame:
                if rigid_darts == None:
                    rigid_dihedrals_atoms = []
                    move_atoms = []
                    for center_atom in self.traj_dart_dict['rotate_list']:
                        #if any([i in self.traj_dart_dict['bond_groups'][center_atom] for i in self.darts['dihedral'].keys()]):
                        if any([i in self.traj_dart_dict['bond_groups'][center_atom] for i in zmat_new._frame.index[3:]]):

                            move_atoms.append(center_atom)

                #test
                if 1:
                    new_change =  old_list + change_list

                    zmat_new._frame.loc[rigid_dihedrals_atoms, new_change] = zmat_traj._frame.loc[rigid_dihedrals_atoms, new_change]

                #print('doing the new thing')
                if rigid_darts == 'rigid_darts':
                    move_atoms = []
                    for center_atom in self.traj_dart_dict['rotate_list']:
                        if any([i in self.traj_dart_dict['bond_groups'][center_atom] for i in self.darts['dihedral'].keys()]):
                            move_atoms.append(center_atom)
                if rigid_darts == 'flexible_darts':
                    move_atoms = []
                    for center_atom in self.traj_dart_dict['bond_groups']:
                        print('center atoms', center_atom)

                        if any([i in self.traj_dart_dict['bond_groups'][center_atom] for i in self.darts['dihedral'].keys()]):
                            move_atoms.append(center_atom)
                #print('move_atoms', move_atoms)
                #exit()
                #print('move_atoms', move_atoms)
                #exit()
#                for center_atom in self.traj_dart_dict['rotate_list']:
                for center_atom in move_atoms:

                    #pick one atom at random to use the dart_range for to displace everything uniformly

                    #all_bonds = [(i, self.internal_zmat[binding_mode_index]._frame.loc[i,'dart_range']) for i in self.traj_dart_dict['bond_groups'][center_atom]]
                    #print('all_bonds', all_bonds)
                    #exit()
                    print('all possible choices', self.traj_dart_dict['bond_groups'][center_atom])
                    possible_atoms = [i for i in self.traj_dart_dict['bond_groups'][center_atom] if i in self.darts['dihedral'].keys()]
                    print('possible_atoms', possible_atoms)
                    #chosen_atom = random.choice(self.traj_dart_dict['bond_groups'][center_atom])
                    chosen_atom = random.choice(possible_atoms)
                    #chosen_atom = self.traj_dart_dict['bond_groups'][center_atom][0]
                    #chosen_atom = self.traj_dart_dict['bond_groups'][center_atom][2]
                    #chosen_atom = 8


                    #print('chosen_atom', chosen_atom)
                    #chosen_atom = center_atom


                    #find the random displacement based on what the chosen atom is
                    #displacement = -1*zmat_new._frame['dart_range'].loc[chosen_atom]
#                    displacement = zmat_new._frame['dart_range'].loc[chosen_atom]*(2*(np.random.random() - 0.5))
                    #print('zmat_new._frame.loc[chosen_atom,"dihedral_max"]', zmat_new._frame.loc[chosen_atom,'dihedral_max'])
                    #print('zmat_traj.current', zmat_traj._frame.loc[chosen_atom,'dihedral'])

                    if self.darting_sampling == 'uniform':
                        print('doing uniform sampling')
                        displacement = zmat_new._frame.loc[chosen_atom,'dart_range']*(2*(np.random.random() - 0.5))
                        if 0:
                            ratio_before = self.internal_zmat[binding_mode_index].loc[chosen_atom,'dart_range']
                            ratio_after = self.internal_zmat[rand_index].loc[chosen_atom,'dart_range']
                            self.acceptance_ratio = ratio_before/ratio_after
                        #print('displacement', displacement)
                    elif self.darting_sampling == 'gaussian':
                        zmat_new._frame.loc[chosen_atom,'gauss'].random_state = np.random.RandomState()
                        #random_number = zmat_new._frame.loc[chosen_atom,'gauss'].rvs(size=10000)
                        random_number = zmat_new._frame.loc[chosen_atom,'gauss'].rvs()
                        displacement = zmat_new._frame.loc[chosen_atom,'dihedral_max'] - random_number
                        #print('displacement', displacement)
                        while displacement > zmat_new._frame.loc[chosen_atom,'dart_range']:
                            #print('displacement', displacement, zmat_new._frame.loc[chosen_atom,'dart_range'])

                            random_number = zmat_new._frame.loc[chosen_atom,'gauss'].rvs()
                            displacement = zmat_new._frame.loc[chosen_atom,'dihedral_max'] - random_number

                        import matplotlib.pyplot as plt
                        #need to do two things to maintain detailed balance
                        #evaluate current probability of having the dihedral before and after
                        #print('random_number', random_number, 'pdf', zmat_new._frame.loc[chosen_atom,'gauss'].pdf(random_number))
                        #print('zmat_new', zmat_new)
                        #print('max', zmat_new._frame.loc[chosen_atom,'gauss'].pdf(zmat_new._frame.loc[chosen_atom,'dihedral_max']))
                        #find the probability of proposing the move based on the gaussian probability of the initial and darted dihedral position
                        gauss_prob_before = self.internal_zmat[binding_mode_index]._frame.loc[chosen_atom,'gauss'].pdf(zmat_traj._frame.loc[chosen_atom,'dihedral'])
                        gauss_prob_after = self.internal_zmat[rand_index]._frame.loc[chosen_atom,'gauss'].pdf(random_number)
                        #adjust the probability based on the ratio
                        gauss_ratio_before = self.internal_zmat[binding_mode_index].loc[chosen_atom,'ratio']
                        gauss_ratio_after = self.internal_zmat[rand_index].loc[chosen_atom,'ratio']
                        #divide the ratios of the probability densities of the darted_pose/current pose to maintain detailed balance
                        #(since the total probabilites could be different because we adjust the darting ranges)

                        #print('gauss before', gauss_prob_before)
                        #print('gauss after', gauss_prob_after)
                        #print('modificiation', (gauss_prob_after*gauss_ratio_after)/(gauss_prob_before*gauss_ratio_before))
                        #self.acceptance_ratio = (gauss_prob_after*gauss_ratio_after)/(gauss_prob_before*gauss_ratio_before)

                        #self.acceptance_ratio = self.acceptance_ratio*(gauss_prob_after*gauss_ratio_after)/(gauss_prob_before*gauss_ratio_before)
                        print('gauss_prob_before*gauss_ratio_before', gauss_prob_before*gauss_ratio_before)
                        print('gauss_prob_after*gauss_ratio_after', gauss_prob_after*gauss_ratio_after)
                        print('gauss_prob_before', gauss_prob_before, 'gauss_ratio_before', gauss_ratio_before)
                        print('gauss_prob_after', gauss_prob_after, 'gauss_ratio_after', gauss_ratio_after)
                        self.acceptance_ratio = self.acceptance_ratio*(gauss_prob_before*gauss_ratio_before)/(gauss_prob_after*gauss_ratio_after)

                        #print('acceptance_ratio', self.acceptance_ratio)
                        #exit()
                        #self.acceptance_ratio = self.acceptance_ratio * (self.internal_zmat[rand_index].loc[chosen_atom,'ratio']/self.internal_zmat[self.selected_pose].loc[chosen_atom,'ratio'])
                        #displacement = zmat_new._frame.loc[chosen_atom,'dihedral_max'] - zmat_new._frame.loc[chosen_atom,'gauss'].rvs(size=1)[0]
                        if 0:
                            random_number = zmat_new._frame.loc[chosen_atom,'gauss'].rvs(size=10000)
                            num_list = []
                            output = zmat_new._frame.loc[chosen_atom,'dihedral_max'] - random_number
                            plt.hist(output, bins=50)
                            plt.savefig('test.png')
                            #displacement = zmat_new._frame.loc[chosen_atom,'dihedral_max'] - random_number
                            #print('displacement', displacement, 'random_number', random_number, 'dihedral_max', zmat_new._frame.loc[chosen_atom,'dihedral_max'])
                            exit()
                    #displacement = zmat_new._frame.loc[chosen_atom,'dart_range']*(2*(1 - 0.5))
                    #displacement = 0.0




                    ##zmat_new._frame['dihedral'].loc[chosen_atom] =  zmat_new._frame['dihedral_max'].loc[chosen_atom] + displacement
                    #choose the new starting point and figure out the displacement
                    #use that displacement for all the atoms bonded to that group
                    counter = 0
                    for rotate_atom in sorted(self.traj_dart_dict['bond_groups'][center_atom]):
                        print('rotate atom', rotate_atom)
#                        rotate_displacement = zmat_traj._frame['dihedral'].loc[chosen_atom] - zmat_traj._frame['dihedral'].loc[rotate_atom]
                        #rotate_displacement = wrapDegrees(zmat_traj._frame.loc[chosen_atom, 'dihedral'] - zmat_traj._frame.loc[rotate_atom,'dihedral'])
                        rotate_displacement = zmat_traj._frame.loc[chosen_atom, 'dihedral'] - zmat_traj._frame.loc[rotate_atom,'dihedral']

                        #previous commented out
                        #zmat_new._frame['dihedral'].loc[rotate_atom] =  zmat_new._frame['dihedral_max'].loc[rotate_atom] + rotate_displacement

#                       zmat_new._frame['dihedral'].loc[rotate_atom] =  zmat_new._frame['dihedral_max'].loc[chosen_atom] + rotate_displacement + displacement
                        #add_distance = rotate_displacement + displacement
                        #add_distance = wrapDegrees(wrapDegrees(rotate_displacement) + wrapDegrees(displacement)) is right

                        add_distance = wrapDegrees(wrapDegrees(rotate_displacement) + wrapDegrees(displacement)) #default
                        edit_add_distance = rotate_displacement + displacement
                        edit_result = zmat_new._frame.loc[chosen_atom,'dihedral_max'] - edit_add_distance
                        #add_distance = wrapDegrees(wrapDegrees(-rotate_displacement) - wrapDegrees(displacement)) #kinda works? but not
                        #add_distance = wrapDegrees(wrapDegrees(-rotate_displacement) + wrapDegrees(displacement)) #kinda works? but not right


                        #print('add_distance for', rotate_atom, add_distance, rotate_displacement, displacement)
                        zmat_di_before = zmat_new._frame.loc[rotate_atom,'dihedral']
                        #zmat_new._frame.loc[rotate_atom,'dihedral'] =  zmat_new._frame.loc[chosen_atom,'dihedral_max'] - add_distance #right
                        #changed this in most recent
                        zmat_new._frame.loc[rotate_atom,'dihedral'] =  zmat_new._frame.loc[chosen_atom,'dihedral_max'] - add_distance
                        #zmat_new._frame.loc[rotate_atom,'dihedral'] =  wrapDegrees(zmat_new._frame.loc[chosen_atom,'dihedral_max'] - add_distance)
                        #print(rotate_atom, 'before', zmat_di_before, 'after', zmat_new._frame.loc[rotate_atom,'dihedral'], 'rotate_displacement', rotate_displacement, 'displacement', displacement, 'add_distance', add_distance )
                        #print("zmat_new._frame.loc[chosen_atom,'dihedral_max']", zmat_new._frame.loc[chosen_atom,'dihedral_max'], 'edit_add_distance', edit_add_distance, 'edit_result', edit_result)
                        #debug here
                        #zmat_new._frame.loc[rotate_atom,'dihedral'] = zmat_new._frame.loc[rotate_atom,'dihedral_max']
                        #zmat_new._frame.loc[rotate_atom,'dihedral'] =  zmat_new._frame.loc[chosen_atom,'dihedral_max']

                        #zmat_new._frame.loc[rotate_atom,'dihedral'] =  zmat_new._frame.loc[rotate_atom,'dihedral_max']



                    #TODO Continue here
                #print('zmat after random\n', zmat_new)
                if 0:
                    #print('doing the right part')
                    #zmat_new._frame['dart_range'] * 2*np.random.random() - zmat_new._frame['dart_range']
                    zmat_size = len(self.internal_zmat[binding_mode_index].index)
                    print((2*np.random.random((zmat_size, 1)) - 1))
                    #diff = zmat_new._frame['dart_range']*(2*np.random.random((zmat_size, 1)) - 1)
                    #diff = zmat_new._frame['dart_range'].multiply(2*np.random.random((zmat_size, 1)) - 1)
                    diff = zmat_new._frame['dart_range'].multiply(2*(np.random.random((zmat_size)) - 0.5))
                    #print('diff', diff)
                    #print('zmat_new before', zmat_new)


                    zmat_new._frame['dihedral'] = zmat_new._frame['dihedral_max'] + diff
                    #print('zmat_new after', zmat_new)

                #figure out the range to alter the dihedrals
                #TODO might need to do something with the bond/angle for the first 3 bonds/angles


            else:
                #print('doing the wrong part')
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

        ##start_indices = [atom_indices[i] for i in self.buildlist.index.values[:3]]
        start_indices = [i for i in self.buildlist.index.values[:3]]

        sim_three = self.sim_traj.xyz[0][start_indices]
        print('binding_mode_pos[binding_mode_index]', binding_mode_pos[binding_mode_index])
        print('start_indices', start_indices)
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
        if self.centroid_darting == True:
            dart_three = (dart_three -  np.tile(centroid_orig, (3,1))).dot(rot_mat) + np.tile(centroid_orig, (3,1)) - np.tile(centroid, (3,1))
        elif self.centroid_darting == 'random':
            def offset_superpose(old_traj, old_reference, frame=0, atom_indices=None,
                          ref_atom_indices=None, parallel=True):
                """Su perpose each conformation in this trajectory upon a reference
                Parameters
                ----------
                reference : md.Trajectory
                    Align self to a particular frame in `reference`
                frame : int
                    The index of the conformation in `reference` to align to.
                atom_indices : array_like, or None
                    The indices of the atoms to superpose. If not
                    supplied, all atoms will be used.
                ref_atom_indices : array_like, or None
                    Use these atoms on the reference structure. If not supplied,
                    the same atom indices will be used for this trajectory and the
                    reference one.
                parallel : bool
                    Use OpenMP to run the superposition in parallel over multiple cores
                Returns
                -------
                self
                """
                traj = copy.deepcopy(old_traj)
                reference = copy.deepcopy(old_reference)
                if atom_indices is None:
                    atom_indices = slice(None)

                if ref_atom_indices is None:
                    ref_atom_indices = atom_indices

                if not isinstance(ref_atom_indices, slice) and (
                    len(ref_atom_indices) != len(atom_indices)):
                    raise ValueError("Number of atoms must be consistent!")

                n_frames = traj.xyz.shape[0]
                self_align_xyz = np.asarray(traj.xyz[:, atom_indices, :], order='c')
                self_displace_xyz = np.asarray(traj.xyz, order='c')
                ref_align_xyz = np.array(reference.xyz[frame, ref_atom_indices, :],
                                         copy=True, order='c').reshape(1, -1, 3)

                offset = np.mean(self_align_xyz, axis=1, dtype=np.float64).reshape(n_frames, 1, 3)
                self_align_xyz -= offset
                if self_align_xyz.ctypes.data != self_displace_xyz.ctypes.data:
                    # when atom_indices is None, these two arrays alias the same memory
                    # so we only need to do the centering once
                    self_displace_xyz -= offset

                ref_offset = ref_align_xyz[0].astype('float64').mean(0)
                print('ref_offset', ref_offset)
                return ref_offset
            #take offset and dart region and translate it to the new one
            if 0:
                offset = offset_superpose(self.sim_ref, self.ref_traj, atom_indices=self.fit_atoms[rand_index])
                #offset = [0,0,0]
                first_atom_offset = (offset + self.trans_dart[rand_index]['mean'] + 2*(np.random.rand(1,3) - 0.5)*self.trans_dart[rand_index]['std'])
                trans_difference = first_atom_offset - dart_three[0]
                print('trans_difference', trans_difference)
            #dart_three = dart_three + trans_difference

            #dart_three = dart_three - trans_difference
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
        xyz_new = (zmat_new.give_cartesian_edit(start_coord=dart_three*10.)).sort_index()
        #xyz_new = (zmat_new.give_cartesian_edit(start_coord=sim_three*10.)).sort_index()
        #print('subtract', zmat_new._frame.loc[2:,['bond', 'angle', 'dihedral']] - zmat_traj._frame.loc[2:,['bond', 'angle', 'dihedral']])
        new_df = zmat_new._frame['dihedral'].copy()
        #print('zmat after move\n', zmat_new)
       #print('dir', dir(xyz_new))
        convert = xyz_new.to_zmat(construction_table=self.buildlist)
        #print('after conversion\n', xyz_new.to_zmat(construction_table=self.buildlist))
        #print('bond_diff\n', zmat_new['bond']- convert['bond'])
        #print('angle_diff\n', zmat_new['angle']- convert['angle'])
        #print('dihedral_diff\n', zmat_new['dihedral']- convert['dihedral'])

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
        print('zmat_traj', zmat_traj)
        #for i,j in enumerate(self.internal_zmat):
        #    print('internal_zmat',i,'\n',j)
        print('zmat_new', zmat_new)
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
            print(old_int._system_parameters)
            print(old_int.int_kwargs)
            if self.old_restraint:
                new_int = AlchemicalExternalRestrainedLangevinIntegrator(restraint_group=self.restraint_group,
                                                   lambda_restraints=self.lambda_restraints, **old_int.int_kwargs)
            else:
                old_int.int_kwargs['alchemical_functions']['lambda_restraints'] = self.lambda_restraints
                print(old_int.int_kwargs)
                new_int = AlchemicalExternalLangevinIntegrator(restraint_group=self.restraint_group,
                                                   lambda_restraints=self.lambda_restraints, **old_int.int_kwargs)

            for number in range(old_int.getNumTabulatedFunctions()):
                name = old_int.getTabulatedFunctionName(number)
                new_int.addTabulatedFunction(name, copy.deepcopy(old_int.getTabulatedFunction(number)))
            #exit()
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
                restraint_lig = self.restrained_ligand_atoms
                if restraint_lig is None:
                    for atom in range(len(self.atom_indices)):
                        indices = self.buildlist.index.values[0+atom:3+atom]
                        elements = self.internal_zmat[0]._frame.loc[indices,'atom']
                        h_element = elements[elements=='H']
                        print('elements', elements)
                        print('h_element', h_element.index.values, h_element)
                        restraint_lig = [self.atom_indices[i] for i in indices]

                        neighbors = md.compute_neighbors(pose, 0.7, restraint_lig, haystack_indices=self.restrained_receptor_atoms[index])[0]
                        print('neighbors', neighbors)
                        if len(h_element.index.values) == 0 and len(neighbors) > 2:
                            print('breaking')
                            break
                    #exit()
                    #restraint_lig = [self.atom_indices[i] for i in self.buildlist.index.values[:3]]
                    print('indices', indices)
                    restraint_lig = [self.atom_indices[i] for i in indices]

                print('restraint_lig', restraint_lig)
                #choose restraint type based on specified parameter
                restraint_style = {'boresch':add_boresch_restraints, 'rmsd':add_rmsd_restraints}
                #check which force groups aren't being used and set restraint forces to that
                pose.save('debug.pdb')
                if self.restraints == 'boresch':
                    print('restrained_receptor_atoms[index]', self.restrained_receptor_atoms[index])
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
        self.acceptance_ratio = 1.0

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
                print('chosing pose', self.selected_pose)
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
            print('pose after move', selected_list)
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
            if 0:
                if self.num_poses_begin == 0:
                    self.acceptance_ratio = 0
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
        print('themove')
        self.moves_attempted += 1
        state = context.getState(getPositions=True, getEnergy=True)
        oldDartPos = state.getPositions(asNumpy=True)
        selected_list = self._poseDart(context, self.atom_indices)
        print('start of move selected', selected_list)
        if self.restraints:
            context.setParameter('restraint_pose_'+str(self.selected_pose), 0)
        else:
            #the move is instantaneous without restraints, so find overlap of darting regions
            #to incorporate later into the acceptance criterion
            self.num_poses_begin = len(selected_list)

        if len(selected_list) == 0:
            #this means that the current ligand positions are outside the defined darts
            #therefore we don't perform the move
            self.acceptance_ratio = 0
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
            print('selected_pose before redarting', self.selected_pose)
            new_pos, darted_pose = self._moldRedart(atom_indices=self.atom_indices,
                                            binding_mode_pos=self.binding_mode_traj,
                                            binding_mode_index=self.selected_pose,
                                            nc_pos=oldDartPos,
                                            rigid_darts=self.rigid_darts)

            self.selected_pose = darted_pose
            context.setPositions(new_pos)
            print('checking pose after')
            overlap_after = self._poseDart(context, self.atom_indices)
            print('selected_pose after redarting', self.selected_pose)

            #the acceptance depends on the instantaenous move
            #therefore find the ratio of number of poses before and after
            self.num_poses_end = len(overlap_after)
           #print('overlap_after', self.num_poses_end)

            # to maintain detailed balance, check to see the overlap of the start and end darting regions
            # if there is no overlap after the move, acceptance ratio will be 0

            #check if new positions overlap when moving
            if self.restraints:
                for i in range(len(self.binding_mode_traj)):
                    context.setParameter('restraint_pose_'+str(i), 0)
                context.setParameter('restraint_pose_'+str(self.selected_pose), 1)
            else:
                #if restraints aren't being used, then we can find out the overlap here
                #print('num pose begin', self.num_poses_begin, 'num_poses_end', self.num_poses_end)
                if self.num_poses_begin == 0:
                    self.acceptance_ratio = 0
                else:
                    self.acceptance_ratio = self.acceptance_ratio*(float(self.num_poses_end)/float(self.num_poses_begin))


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
        if 0:
            #debug code for populations
            after_pos = context.getState(getPositions=True).getPositions(asNumpy=True)
            non_moved_index = [i for i in range(29) if i not in [ 7, 8, 9, 10, 11, 13, 12, 14, 15]]
            after_pos[non_moved_index] = state.getPositions(asNumpy=True)[non_moved_index]
        return context

from openmmtools.integrators import PrettyPrintableIntegrator
class AlchemicalExternalRestrainedLangevinIntegrator(AlchemicalExternalLangevinIntegrator, PrettyPrintableIntegrator):
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
        self.pretty_print()
        #exit()

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


class AlchemicalExternalRestrainedLangevinIntegrator_new(AlchemicalExternalLangevinIntegrator, PrettyPrintableIntegrator):
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

        super(AlchemicalExternalRestrainedLangevinIntegrator_new, self).__init__(
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
        self.pretty_print()
        #exit()

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

