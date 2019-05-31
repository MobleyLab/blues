import simtk.unit as unit
import numpy as np
import mdtraj as md
from mdtraj.formats.xyzfile import XYZTrajectoryFile
from mdtraj.utils import in_units_of
from blues.moldart.lin_math import adjust_angle
from blues.moldart.lin_math import getRotTrans
from blues.moves import RandomLigandRotationMove
import chemcoord as cc
import copy
import tempfile
from blues.moldart.chemcoord import give_cartesian_edit
from blues.moldart.darts import makeDartDict, checkDart, makeDihedralDifferenceDf
from blues.moldart.boresch import add_rmsd_restraints, add_boresch_restraints
import parmed
from blues.integrators import AlchemicalExternalLangevinIntegrator, AlchemicalNonequilibriumLangevinIntegrator
from blues.moldart.rigid import createRigidBodies, resetRigidBodies

import logging
logger = logging.getLogger(__name__)


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
                    #self.dihedral_ring_atoms = list(set(rigid_atoms + h_list))
                    self.dihedral_ring_atoms = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in rigid_atoms]
        else:
            self.buildlist = self._createBuildlist(pdb_files, self.atom_indices)
            self.dihedral_ring_atoms = self._findDihedralRingAtoms(pdb_files, atom_indices=self.atom_indices, rigid_darts=self.rigid_darts)
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

            self.only_darts_dihedrals = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in core]

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
        if rigid_darts is not None:
            with tempfile.NamedTemporaryFile(suffix='.xyz') as t:
                fname = t.name
                traj = md.load(structure_files[0]).atom_slice(atom_indices)
                xtraj = XYZTrajectoryFile(filename=fname, mode='w')
                xtraj.write(xyz=in_units_of(traj.xyz, traj._distance_unit, xtraj.distance_unit),
                            types=[i.element.symbol for i in traj.top.atoms] )

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
            angle_ring_atoms = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] in rigid_atoms]
            for mol in ifs.GetOEGraphMols():
                for atom in mol.GetAtoms():
                    if atom.IsHydrogen():
                        h_list.append(atom.GetIdx())
            #self.dihedral_ring_atoms = list(set(angle_ring_atoms + h_list))
            #self.dihedral_ring_atoms = list(set(rigid_atoms + h_list))
            dihedral_ring_atoms = [i for i in range(len(atom_indices)) if buildlist.at[i, 'b'] in rigid_atoms]
            return dihedral_ring_atoms

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

        selected = checkDart(self.internal_zmat, current_pos=(np.array(self.sim_traj.openmm_positions(0)._value))[self.atom_indices]*10,

                    current_zmat=current_zmat, pos_list=self.binding_mode_pos,
                    construction_table=self.buildlist,
                    dart_storage=self.darts
                    )
        if len(selected) >= 1:
            #returns binding mode index, and the diff_list
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
        #if the molecule is treated as rigid, we won't change the internal coordinates
        #otherwise we find the differences in the dihedral angles between the simulation
        #and reference poses and take that into account when darting to the new pose
        change_list = ['dihedral']
        old_list = ['bond', 'angle']
        if rigid_darts == 'rigid_ring':
            rigid_dihedrals_atoms = [i for i in self.dihedral_ring_atoms if i in zmat_new._frame.index[3:]]
            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]
        elif rigid_darts == 'rigid_darts':
            #rigid_dihedrals_atoms = [i for i in self.only_darts_dihedrals if i in zmat_new._frame.index[3:]]
            rigid_dihedrals_atoms = [i for i in zmat_new._frame.index if i not in self.only_darts_dihedrals]

            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]

        if rigid_darts == 'rigid_molecule':
            old_list =  old_list + change_list

        else:
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
            rigid_dihedrals_atoms = [i for i in zmat_new._frame.index if i not in self.only_darts_dihedrals]

            zmat_new._frame.loc[rigid_dihedrals_atoms,['dihedral']] = zmat_traj._frame.loc[rigid_dihedrals_atoms,['dihedral']]


        #find translation differences in positions of first two atoms to reference structure
        #find the appropriate rotation to transform the structure back
        #repeat for second bond
        def findCentralAngle(buildlist):
            connection_list = []
            index_list = [0,1,2]
            for i in buildlist.index.get_values()[:3]:
                connection_list.append(buildlist['b'][i])
            #count the number of bonds to the first buildatom
            counts = connection_list.count(self.buildlist.index.get_values()[0])
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
        start_indices = [atom_indices[i] for i in self.buildlist.index.get_values()[:3]]

        sim_three = self.sim_traj.xyz[0][start_indices]
        ref_three  = binding_mode_pos[binding_mode_index].xyz[0][start_indices]
        dart_three = binding_mode_pos[rand_index].xyz[0][start_indices]
        dart_ref = binding_mode_pos[rand_index].xyz[0][start_indices]

        change_three = np.copy(sim_three)
        vec1_sim = sim_three[vector_list[0][0]] - sim_three[vector_list[0][1]]
        vec2_sim = sim_three[vector_list[1][0]] - sim_three[vector_list[1][1]]

        #calculate rotation from ref pos to sim pos
        #change angle of one vector
        ref_angle = self.internal_zmat[binding_mode_index]._frame['angle'][self.buildlist.index.get_values()[2]]
        ad_vec = adjust_angle(vec1_sim, vec2_sim, np.radians(ref_angle), maintain_magnitude=True)
        ad_vec = ad_vec / np.linalg.norm(ad_vec) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.get_values()[2]]/10.
        #apply changed vector to center coordinate to get new position of first particle

        nvec2_sim = vec2_sim / np.linalg.norm(vec2_sim) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.get_values()[2]]/10.
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
        new_angle = zmat_new['angle'][self.buildlist.index[2]]
        ad_dartvec = adjust_angle(vec1_dart, vec2_dart, np.radians(new_angle), maintain_magnitude=False)
        ###
        ad_dartvec = ad_dartvec / np.linalg.norm(ad_dartvec) * zmat_new._frame['bond'][self.buildlist.index.get_values()[1]]/10.
        nvec2_dart = vec2_dart / np.linalg.norm(vec2_dart) * zmat_new._frame['bond'][self.buildlist.index.get_values()[2]]/10.
        dart_three[vector_list[0][0]] = dart_three[vector_list[0][1]] + ad_dartvec
        dart_three[vector_list[1][0]] = dart_three[vector_list[0][1]] + nvec2_dart

        #get xyz from internal coordinates
        zmat_new.give_cartesian_edit = give_cartesian_edit.__get__(zmat_new)

        xyz_new = (zmat_new.give_cartesian_edit(start_coord=dart_three*10.)).sort_index()

        self.sim_traj.xyz[0][self.atom_indices] = xyz_new._frame.loc[:, ['x', 'y', 'z']].get_values() / 10.
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
                restraint_lig = [self.atom_indices[i] for i in self.buildlist.index.get_values()[:3]]
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

