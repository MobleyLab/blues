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
import types
from blues.moldart.chemcoord import give_cartesian_edit
from blues.moldart.darts import makeDartDict, checkDart
from blues.moldart.boresch import add_rmsd_restraints, add_boresch_restraints
import parmed
from blues.integrators import AlchemicalExternalLangevinIntegrator, AlchemicalNonequilibriumLangevinIntegrator
import time

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
        rigid_ring=False, rigid_move=False, freeze_waters=0, freeze_protein=False,
        restraints='rmsd', restrained_receptor_atoms=None,
        K_r=10, K_angle=10, K_RMSD=0.6, RMSD0=2, lambda_restraints='max(0, 1-(1/0.10)*abs(lambda-0.5))'
        ):
        super(MolDartMove, self).__init__(structure, resname)
        #md trajectory representation of only the ligand atoms
        self.binding_mode_traj = []
        #positions of only the ligand atoms
        self.binding_mode_pos = []
        #fit atoms are the atom indices which should be fit to to remove rot/trans changes
        self.fit_atoms = fit_atoms
        #chemcoord cartesian xyz of the ligand atoms
        self.internal_xyz = []
        #chemcoord internal zmatrix representation of the ligand atoms
        self.internal_zmat = []
        self.buildlist = None
        self.rigid_move = bool(rigid_move)
        self.rigid_ring = bool(rigid_ring)
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
        self.restrained_receptor_atoms = restrained_receptor_atoms
        self.freeze_protein = freeze_protein
        self.K_r = K_r
        self.K_angle = K_angle
        self.K_RMSD = K_RMSD
        self.RMSD0 = RMSD0
        self.lambda_restraints = lambda_restraints

        #find pdb inputs inputs
        if len(pdb_files) <= 1:
            raise ValueError('Should specify at least two pdbs in pdb_files for darting to be beneficial')
        self.dart_groups = list(range(len(pdb_files)))
        #chemcoords reads in xyz files only, so we need to use mdtraj
        #to get the ligand coordinates in an xyz file
        with tempfile.NamedTemporaryFile(suffix='.xyz') as t:
            fname = t.name
            traj = md.load(pdb_files[0]).atom_slice(self.atom_indices)
            xtraj = XYZTrajectoryFile(filename=fname, mode='w')
            xtraj.write(xyz=in_units_of(traj.xyz, traj._distance_unit, xtraj.distance_unit),
                        types=[i.element.symbol for i in traj.top.atoms] )
            xtraj.close()
            xyz = cc.Cartesian.read_xyz(fname)
            self.buildlist = xyz.get_construction_table()
            if self.rigid_ring:
                ring_atoms = []
                from openeye import oechem
                ifs = oechem.oemolistream()
                ifs.open(fname)
                double_bonds = []
                for mol in ifs.GetOEGraphMols():
                    for atom in mol.GetAtoms():
                        if atom.IsInRing():
                            ring_atoms.append(atom.GetIdx())
                    #for bond in mol.GetBonds():
                    #    print('bond', bond.GetOrder())
                    bgn_idx = [bond.GetBgnIdx() for bond in mol.GetBonds() if bond.GetOrder() > 1]
                    end_idx = [bond.GetEndIdx() for bond in mol.GetBonds() if bond.GetOrder() > 1]
                    double_bonds = bgn_idx + end_idx
                #    for bond in mol.GetBonds():
                #        if bond.GetOrder() > 1:
                #            [bond.GetBgnIdx() for bond in mol.GetBonds() if bond.GetOrder > 1]
                #        print('bai', bond.GetBgnIdx(), 'eai', bond.GetEndIdx())
                #        print('order', bond.GetOrder())
                rigid_atoms = set(ring_atoms + double_bonds)
                angle_ring_atoms = [i for i in range(len(self.atom_indices)) if self.buildlist.at[i, 'b'] in rigid_atoms]
                self.dihedral_ring_atoms = angle_ring_atoms

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
        for j, pdb_file in enumerate(pdb_files):
            traj = copy.deepcopy(self.ref_traj)
            pdb_traj = md.load(pdb_file)[0]
            traj.xyz[0][:num_atoms] = pdb_traj.xyz[0]
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
        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])
        self.sim_ref = copy.deepcopy(self.binding_mode_traj[0])
        self.darts = makeDartDict(self.internal_zmat, self.binding_mode_pos, self.buildlist)
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

    @staticmethod
    def _checkTransitionMatrix(transition_matrix, dart_groups):
        """Checks if transition matrix obeys the proper
        properties if used in a Monte Carlo scheme"""
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
        self.sim_traj.superpose(reference=self.ref_traj,
                            atom_indices=self.fit_atoms,
                            ref_atom_indices=self.fit_atoms
                            )


        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        for index, entry in enumerate(['x', 'y', 'z']):
            for i in range(len(self.atom_indices)):
                sel_atom = self.atom_indices[i]
                #set the pandas series with the appropriate data
                #multiply by 10 since openmm works in nm and cc works in angstroms
                xyz_ref._frame.at[i, entry] = self.sim_traj.openmm_positions(0)[sel_atom][index]._value*10

        current_zmat = xyz_ref.get_zmat(construction_table=self.buildlist)
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

    def _moldRedart(self, atom_indices, binding_mode_pos, binding_mode_index, nc_pos, rigid_ring=False, rigid_move=False):
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
        start = time.clock()

        self.sim_traj.superpose(reference=self.ref_traj,
                            atom_indices=self.fit_atoms,
                            ref_atom_indices=self.fit_atoms
                            )
        end = time.clock()
        print('time1', end - start)
        #rand_index = np.random.choice(self.dart_groups, self.transition_matrix[binding_mode_index])
        start = time.clock()

        rand_index, self.dart_ratio = self._dart_selection(binding_mode_index, self.transition_matrix)
        dart_ratio = self.dart_ratio
        self.acceptance_ratio = self.acceptance_ratio * dart_ratio
        end = time.clock()
        print('time2', end - start)


        #get matching binding mode pose and get rotation/translation to that pose
        #TODO decide on making a copy or always point to same object
        start = time.clock()
        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        end = time.clock()
        print('time3', end-start)
        start = time.clock()
        for index, entry in enumerate(['x', 'y', 'z']):
            for i in range(len(self.atom_indices)):
                sel_atom = self.atom_indices[i]
                #set the pandas series with the appropriate data
                #multiply by 10 since openmm works in nm and cc works in angstroms
                xyz_ref._frame.at[i, entry] = self.sim_traj.openmm_positions(0)[sel_atom][index]._value*10
        end = time.clock()
        print('timea', end-start)
        start = time.clock()
        zmat_new = copy.deepcopy(self.internal_zmat[rand_index])
        end = time.clock()
        print('timeb', end-start)
        start = time.clock()
        zmat_diff = xyz_ref.get_zmat(construction_table=self.buildlist)
        end = time.clock()
        print('timec', end-start)
        start = time.clock()
        zmat_traj = copy.deepcopy(xyz_ref.get_zmat(construction_table=self.buildlist))
        end = time.clock()
        print('timed', end-start)
        print('zmat_new', zmat_new)
        print('zmat_traj', zmat_traj)
        print('xyz_ref', xyz_ref)

        #get appropriate comparision zmat
        zmat_compare = self.internal_zmat[binding_mode_index]
        #we don't need to change the bonds/dihedrals since they are fast to sample
        #if the molecule is treated as rigid, we won't change the internal coordinates
        #otherwise we find the differences in the dihedral angles between the simulation
        #and reference poses and take that into account when darting to the new pose
        change_list = ['dihedral']
        old_list = ['bond', 'angle']
        start = time.clock()
        if rigid_move == True:
            old_list = change_list + old_list
        else:

            zmat_indices = zmat_traj.index.values
            #get the difference of the trajectory to the dart
 #           zmat_diff._frame.loc[:, change_list].iloc[3:] = zmat_diff._frame.loc[:, change_list].iloc[3:].subtract(zmat_compare._frame.loc[:, change_list].iloc[3:])
            #zmat_diff._frame.loc[:, change_list].iloc[3:] = zmat_diff._frame.loc[:, change_list].iloc[3:].subtract(zmat_compare._frame.loc[:, change_list].iloc[3:])
            #zmat_diff._frame.loc[:, change_list] = zmat_diff._frame.loc[:, change_list].subtract(zmat_compare._frame.loc[:, change_list].reindex(self.buildlist))
            changed = (zmat_diff._frame.loc[:, change_list] - zmat_compare._frame.loc[:, change_list]).reindex(zmat_indices)
            abs_bond_diff = zmat_diff._frame.loc[:, 'bond'].iloc[0] - zmat_compare._frame.loc[:, 'bond'].iloc[0]
            abs_angle_diff = zmat_diff._frame.loc[:, 'angle'].iloc[:2] - zmat_compare._frame.loc[:, 'angle'].iloc[:2]
            print('bond diff', abs_bond_diff)
            print('angle diff', abs_angle_diff)


            print('zmat_compare', zmat_compare)
            print('changed', changed)
#            zmat_diff._frame.loc[:, change_list].iloc[3:] = (zmat_diff._frame.loc[:, change_list] - zmat_compare._frame.loc[:, change_list].iloc[3:]).reindex(zmat_indices)
            zmat_diff._frame.loc[:, change_list] = changed
            zmat_diff._frame.loc[(zmat_diff._frame.index.isin([zmat_diff._frame.index[0]])), 'bond'] = abs_bond_diff
            zmat_diff._frame.loc[(zmat_diff._frame.index.isin(zmat_diff._frame.index[:2])), 'angle'] = abs_angle_diff

            print('changed zmat', zmat_diff)



            #Then add back those changes to the darted pose
            print('zmat_diff', zmat_diff)

            print('zmat_new', zmat_new)

#            zmat_new._frame.loc[:, change_list].iloc[3:] = zmat_new._frame.loc[:, change_list].iloc[3:].add(zmat_diff._frame.loc[:, change_list].iloc[3:])
            zmat_new._frame.loc[:, change_list] = zmat_new._frame.loc[:, change_list] + zmat_diff._frame.loc[:, change_list]
            zmat_new._frame.loc[(zmat_new._frame.index.isin([zmat_new._frame.index[0]])), 'bond'] = zmat_new._frame.loc[(zmat_new._frame.index.isin([zmat_new._frame.index[0]])), 'bond'] + zmat_diff._frame.loc[(zmat_diff._frame.index.isin([zmat_diff._frame.index[0]])), 'bond']
            #zmat_new._frame.loc[(zmat_new._frame.index.isin([zmat_new._frame.index[:2]])), 'angle'] = zmat_new._frame.loc[(zmat_new._frame.index.isin([zmat_new._frame.index[:2]])), 'angle'] + zmat_diff._frame.loc[(zmat_diff._frame.index.isin([zmat_diff._frame.index[:2]])), 'angle']
            #print('test1', zmat_new._frame.loc[(zmat_new._frame.index.isin([zmat_new._frame.index[:2]])), 'angle'])
            print('sel index', zmat_new._frame.index[:2])
            print('test', zmat_new._frame.loc[zmat_new._frame.index[:2], 'angle'])
            zmat_new._frame.loc[zmat_new._frame.index[:2], 'angle'] = zmat_new._frame.loc[zmat_new._frame.index[:2], 'angle'] + zmat_diff._frame.loc[zmat_diff._frame.index[:2], 'angle']

            print('zmat_new new', zmat_new)

#            for i in change_list:
#            #add changes from zmat_diff to the darted pose
#                zmat_new._frame[i] = zmat_diff._frame[i] + zmat_new._frame[i]
        end = time.clock()
        print('time5', end-start)
        print('loc', zmat_new._frame.loc[:, old_list].iloc[3:])
        zmat_new._frame.loc[:, old_list].iloc[3:] = zmat_traj._frame.loc[:, old_list].iloc[3:]
        #for param in old_list:
            #We want to keep the bonds and angles the same between jumps
#            zmat_new._frame[param] = zmat_traj._frame[param]
        #    zmat_new._frame.loc[param].iloc[3:] = zmat_traj._frame[param].iloc[3:]



        if rigid_ring:
            zmat_new._frame.loc[self.dihedral_ring_atoms,['dihedral']].iloc[3:] = zmat_traj._frame.loc[self.dihedral_ring_atoms,['dihedral']].iloc[3:]

 #           for i in self.dihedral_ring_atoms:
 #               #zmat_new._frame.loc[i,'dihedral'] = zmat_traj._frame.loc[i,'dihedral']
 #               zmat_new._frame.loc[self.dihedral_ring_atoms,['dihedral']] = zmat_traj._frame.loc[self.dihedral_ring_atoms,['dihedral']]

        #find translation differences in positions of first two atoms to reference structure
        #find the appropriate rotation to transform the structure back
        #repeat for second bond
        #get xyz from internal coordinates
        #zmat_new.give_cartesian_edit = types.MethodType(give_cartesian_edit, zmat_new)
        xyz_new = (zmat_new.get_cartesian()).sort_index()
        print('xyz_new', xyz_new)
        xyz_ref.to_xyz('aref.xyz')
        xyz_new.to_xyz('anew.xyz')

        for i in range(len(self.atom_indices)):
            for index, entry in enumerate(['x', 'y', 'z']):
                sel_atom = self.atom_indices[i]
                self.sim_traj.xyz[0][:,index][sel_atom] = (xyz_new._frame[entry][i] / 10.)
        #self.sim_traj.superpose(reference=self.sim_ref, atom_indices=self.fit_atoms,
        #        ref_atom_indices=self.fit_atoms
        #        )
        nc_pos = self.sim_traj.xyz[0] * unit.nanometers
        print('return')
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
            self.calculateProperties()
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

        if self.restraints:
            force_list = new_sys.getForces()
            group_list = list(set([force.getForceGroup() for force in force_list]))
            group_avail = [j for j in list(range(32)) if j not in group_list]
            self.restraint_group = group_avail[0]

            old_int._system_parameters = {system_parameter for system_parameter in old_int._alchemical_functions.keys()}
            print('variables', old_int._system_parameters)
#            print('kwargs', old_int.kwargs)
            print('kwargs', old_int.int_kwargs)


            afunction = dict({'lambda_sterics':'min(1, (1/0.3)*abs(lambda-0.5))',
                'lambda_electrostatics':'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            })
            print('alchemical functions', type(afunction))
            print('alchemical functions', type(afunction.keys()))

            #new_int = AlchemicalExternalRestrainedLangevinIntegrator(restraint_group=self.restraint_group,
            #                                   lambda_restraints=self.lambda_restraints, **old_int.kwargs)
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

                new_sys = restraint_style[self.restraints](new_sys, structure, pose_allpos, self.atom_indices, index, self.restraint_group,
                                        self.restrained_receptor_atoms, restraint_lig,
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
            self.restraint_correction = 0
            total_pe_restraint1_on = state.getPotentialEnergy()

            context.setParameter('restraint_pose_'+str(self.selected_pose), 0)
            state_restraint1_off = context.getState(getPositions=True, getEnergy=True)
            total_pe_restraint1_off = state_restraint1_off.getPotentialEnergy()
            restraint1_energy = total_pe_restraint1_on - total_pe_restraint1_off
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

            #use _moldRedart instead
            #calculate changes in angle/dihedral compared to reference
            #apply angle/dihedral changes to new pose
            #translate new pose to center of first molecule
            #find rotation that matches atom1 and atom2s of the build list
            #apply that rotation using atom1 as the origin
            new_pos, darted_pose = self._moldRedart(atom_indices=self.atom_indices,
                                            binding_mode_pos=self.binding_mode_traj,
                                            binding_mode_index=self.selected_pose,
                                            nc_pos=oldDartPos,
                                            rigid_ring=self.rigid_ring, rigid_move=self.rigid_move)

            self.selected_pose = darted_pose
            context.setPositions(new_pos)
            overlap_after = self._poseDart(context, self.atom_indices)
            #the acceptance depends on the instantaenous move
            #therefore find the ratio of number of poses before and after
            #TODO: Check if probability order is right
            #self.num_poses_end = len(overlap_after)
            #self.acceptance_ratio = self.acceptance_ratio*(float(self.num_poses_end)/self.num_poses_begin)
            #self.acceptance_ratio = self.acceptance_ratio*(float(self.num_poses_end_restraints)/self.num_poses_begin_restraints)

            # to maintain detailed balance, check to see the overlap of the start and end darting regions
            # if there is no overlap after the move, acceptance ratio will be 0

            #check if new positions overlap when moving
            if self.restraints:

                state_restraint2_off = context.getState(getEnergy=True)
                total_pe_restraint2_off = state_restraint2_off.getPotentialEnergy()
                for i in range(len(self.binding_mode_traj)):
                    context.setParameter('restraint_pose_'+str(i), 0)

                context.setParameter('restraint_pose_'+str(self.selected_pose), 1)
                state_restraint2_on = context.getState(getEnergy=True)
                total_pe_restraint2_on = state_restraint2_on.getPotentialEnergy()
                restraint2_energy = total_pe_restraint2_on - total_pe_restraint2_off
                restraint_correction = -(restraint2_energy - restraint1_energy)
                #work = context.getIntegrator().getGlobalVariableByName('protocol_work')
                self.restraint_correction = restraint_correction

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

        #try:
        #    self.addGlobalVariable("lambda_restraints", 0)
        #except:
        #    pass


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
