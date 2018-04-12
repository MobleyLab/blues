#from simtk.openmm.app import *
#from simtk.openmm import *
#from simtk.unit import *
import simtk.unit as unit
import numpy as np
import mdtraj as md
from mdtraj.formats.xyzfile import XYZTrajectoryFile
from mdtraj.utils import in_units_of
from blues.lin_math import adjust_angle
from blues.lin_math import getRotTrans
from blues.moves import RandomLigandRotationMove
import chemcoord as cc
import copy
import tempfile
import types
from blues.mold_helper import give_cartesian_edit
from blues.icdart.dartnew import makeDartDict, checkDart
from blues.icdart.bor import add_restraints
import parmed
from blues.integrators import AlchemicalExternalLangevinIntegrator, AlchemicalNonequilibriumLangevinIntegrator

class MolDart(RandomLigandRotationMove):
    """
    Class for performing smart darting moves during an NCMC simulation.

    Parameters
    ----------
    structure: parmed.Structure
        ParmEd Structure object of the relevant system to be moved.
    pdb_files: list of str:
        List of paths to pdb files with the same system as the structure,
        whose ligand internal coordinates will be used as the darts for
        internal coordinate darting.
    fit_atoms:
        The atoms of the protein to be fitted, to remove rotations/
        translations changes from interfering the darting procedure.
    resname : str, optional, default='LIG'
        String specifying the residue name of the ligand.
    rigid_move: boolean, default=False:
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
    restraints: bool, optional, default=True
        Applies restrains so that the ligand remains close to the specified poses
        during the course of the simulation. If this is not used, the probability
        of darting can be greatly diminished, since the ligand can be much more
        mobile in the binding site with it's interactions turned off.
    restrained_receptor_atom: list, optional, default=None
        The three atoms of the receptor to use for boresch style restraints.
        If unspecified uses a random selection through a heuristic process
        via Yank. This is only necessary when `restraints==True`
    K_r: float
        The value of the bond restraint portion of the boresch restraints
        (given in units of kcal/(mol*angstrom**2)).
        Only used if restraints=True.
    K_angle: flaot
        The value of the angle and dihedral restraint portion of the boresh restraints
        (given in units of kcal/(mol*rad**2)).
        Only used if restraints=True.


    """
    def __init__(self, structure, pdb_files, fit_atoms, resname='LIG',
        rigid_move=False, freeze_waters=0, freeze_protein=False,
        restraints=True, restrained_receptor_atoms=None,
        K_r=10, K_angle=10):
        super(MolDart, self).__init__(structure, resname)
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
        self.restraints = bool(restraints)
        self.restrained_receptor_atoms=restrained_receptor_atoms
        self.freeze_protein = freeze_protein
        self.K_r = K_r
        self.K_angle = K_angle

        if restraints != True and restraints != False:
            raise ValueError('restraints argument should be a boolean')
        #flattens pdb files in the case input is list of lists
        #currently set up so that poses in a list don't jump to the same poses in that list
        pdb_files = [[i] if isinstance(i, str) else i for i in pdb_files]
        flat_pdb = list(set([item for sublist in pdb_files for item in sublist]))
        #find the unique pdb inputs inputs and group them according to their lists
        self.pdb_dict = {}
        for index, key in enumerate(flat_pdb):
            self.pdb_dict[key] = index
        if len(self.pdb_dict) <= 1:
            raise ValueError('Should specify at least two pdbs for darting to be beneficial')
        self.dart_groups = []
        for group in pdb_files:
            glist = [self.pdb_dict[key] for key in group]
            self.dart_groups.append(glist)
        print('dart_groups', self.dart_groups)
        #chemcoords reads in xyz files only, so we need to use mdtraj
        #to get the ligand coordinates in an xyz file
        with tempfile.NamedTemporaryFile(suffix='.xyz') as t:
            fname = t.name
            traj = md.load(flat_pdb[0]).atom_slice(self.atom_indices)
            xtraj = XYZTrajectoryFile(filename=fname, mode='w')
            xtraj.write(xyz=in_units_of(traj.xyz, traj._distance_unit, xtraj.distance_unit),
                        types=[i.element.symbol for i in traj.top.atoms] )
            xtraj.close()
            xyz = cc.Cartesian.read_xyz(fname)
        #get the construction table so internal coordinates are consistent between poses
        self.buildlist = xyz.get_construction_table()
        #use the positions from the structure to be used as a reference for
        #superposition of the rest of poses
        with tempfile.NamedTemporaryFile(suffix='.pdb') as t:
            fname = t.name
            self.structure.save(fname, overwrite=True)
            struct_traj = md.load(fname)
        ref_traj = md.load(flat_pdb[0])[0]
        num_atoms = ref_traj.n_atoms
        self.ref_traj = copy.deepcopy(struct_traj)
        #add the trajectory and xyz coordinates to a list
        for j, pdb_file in enumerate(flat_pdb):
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

    def _dart_selection(self, binding_mode_index):
        possible_groups = []
        for group_index, group_list in enumerate(self.dart_groups):
            if binding_mode_index in group_list:
                possible_groups.append(group_index)
        group_choice = np.random.choice(possible_groups)
        dart_groups_removed = [j for i, j in enumerate(self.dart_groups) if i != group_choice]
        #included to correct detailed balance
        #checks the probability of chosing the chosen dart in the forward direction of the move
        #the ratio will be taken of this probability and the probability of choosing the original dart
        #in the reverse direction and will be used in the acceptance criteria
        num_groups = float(len(possible_groups))
        num_group_choice = float(len(dart_groups_removed))
        probability_selection_before = (1./num_groups)*(1./num_group_choice)

        rand_index = np.random.choice(dart_groups_removed[np.random.choice(len(dart_groups_removed))])
        return rand_index, probability_selection_before

    def _moldRedart(self, atom_indices, binding_mode_pos, binding_mode_index, nc_pos, rigid_move=False):
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
        self.sim_traj.superpose(reference=self.ref_traj,
                            atom_indices=self.fit_atoms,
                            ref_atom_indices=self.fit_atoms
                            )
        rand_index, probability_selection_before = self._dart_selection(binding_mode_index)
        #get matching binding mode pose and get rotation/translation to that pose
        #TODO decide on making a copy or always point to same object
        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        for index, entry in enumerate(['x', 'y', 'z']):
            for i in range(len(self.atom_indices)):
                sel_atom = self.atom_indices[i]
                #set the pandas series with the appropriate data
                #multiply by 10 since openmm works in nm and cc works in angstroms
                xyz_ref._frame.at[i, entry] = self.sim_traj.openmm_positions(0)[sel_atom][index]._value*10
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

        if rigid_move == False:
            for i in change_list:
                zmat_diff._frame[i] = zmat_diff._frame[i] - zmat_compare._frame[i]

            for i in change_list:
            #add changes from zmat_diff to the darted pose
                zmat_new._frame[i] = zmat_diff._frame[i] + zmat_new._frame[i]
        else:
            old_list = change_list + old_list

        for param in old_list:
            #We want to keep the bonds and angles the same between jumps
            zmat_new._frame[param] = zmat_traj._frame[param]
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
        sim_three = np.zeros((3,3))
        ref_three = np.zeros((3,3))
        dart_three = np.zeros((3,3))
        dart_ref = np.zeros((3,3))

        for i in range(3):
            sim_three[i] = self.sim_traj.xyz[0][atom_indices[self.buildlist.index.get_values()[i]]]
            self.buildlist.index.get_values()[i]
            ref_three[i] = binding_mode_pos[binding_mode_index].xyz[0][atom_indices[self.buildlist.index.get_values()[i]]]
            dart_three[i] = binding_mode_pos[rand_index].xyz[0][atom_indices[self.buildlist.index.get_values()[i]]]
            dart_ref[i] = binding_mode_pos[rand_index].xyz[0][atom_indices[self.buildlist.index.get_values()[i]]]

        change_three = np.copy(sim_three)
        vec1_sim = sim_three[vector_list[0][0]] - sim_three[vector_list[0][1]]
        vec2_sim = sim_three[vector_list[1][0]] - sim_three[vector_list[1][1]]
        vec1_ref = ref_three[vector_list[0][0]] - ref_three[vector_list[0][1]]

        #calculate rotation from ref pos to sim pos
        #change angle of one vector
        ref_angle = self.internal_zmat[binding_mode_index]._frame['angle'][self.buildlist.index.get_values()[2]]
        angle_diff = ref_angle - np.degrees(calc_angle(vec1_sim, vec2_sim))
        ad_vec = adjust_angle(vec1_sim, vec2_sim, np.radians(ref_angle), maintain_magnitude=True)
        ad_vec = ad_vec / np.linalg.norm(ad_vec) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.get_values()[2]]/10.
        #apply changed vector to center coordinate to get new position of first particle

        nvec2_sim = vec2_sim / np.linalg.norm(vec2_sim) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.get_values()[2]]/10.
        change_three[vector_list[0][0]] = sim_three[vector_list[0][1]] + ad_vec
        change_three[vector_list[1][0]] = sim_three[vector_list[0][1]] + nvec2_sim
        rot_mat, centroid = getRotTrans(change_three, ref_three, center=vector_list[0][1])
        #perform the same angle change on new coordinate
        centroid_orig = dart_three[vector_list[0][1]]
        #perform rotation
        dart_three = (dart_three -  np.tile(centroid_orig, (3,1))).dot(rot_mat) + np.tile(centroid_orig, (3,1)) - np.tile(centroid, (3,1))
        vec1_dart = dart_three[vector_list[0][0]] - dart_three[vector_list[0][1]]
        vec2_dart = dart_three[vector_list[1][0]] - dart_three[vector_list[1][1]]
        dart_angle = self.internal_zmat[rand_index]._frame['angle'][self.buildlist.index.get_values()[2]]
        angle_change = dart_angle - angle_diff

        if 1:
            ###THIS IS CHANGED FOR RIGID
            new_angle = zmat_new['angle'][self.buildlist.index[2]]
            ad_dartvec = adjust_angle(vec1_dart, vec2_dart, np.radians(new_angle), maintain_magnitude=False)
            ###
            ad_dartvec = ad_dartvec / np.linalg.norm(ad_dartvec) * zmat_new._frame['bond'][self.buildlist.index.get_values()[1]]/10.
            nvec2_dart = vec2_dart / np.linalg.norm(vec2_dart) * zmat_new._frame['bond'][self.buildlist.index.get_values()[2]]/10.
            dart_three[vector_list[0][0]] = dart_three[vector_list[0][1]] + ad_dartvec
            dart_three[vector_list[1][0]] = dart_three[vector_list[0][1]] + nvec2_dart

        #get xyz from internal coordinates
        zmat_new.give_cartesian_edit = types.MethodType(give_cartesian_edit, zmat_new)
        xyz_new = (zmat_new.give_cartesian_edit(start_coord=dart_three*10.)).sort_index()

        for i in range(len(self.atom_indices)):
            for index, entry in enumerate(['x', 'y', 'z']):
                sel_atom = self.atom_indices[i]
                self.sim_traj.xyz[0][:,index][sel_atom] = (xyz_new._frame[entry][i] / 10.)
        self.sim_traj.superpose(reference=self.sim_ref, atom_indices=self.fit_atoms,
                ref_atom_indices=self.fit_atoms
                )
        nc_pos = self.sim_traj.xyz[0] * unit.nanometers
        return nc_pos, rand_index, probability_selection_before

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

        if self.restraints == True:
            force_list = new_sys.getForces()
            group_list = list(set([force.getForceGroup() for force in force_list]))
            group_avail = [j for j in list(range(32)) if j not in group_list]
            self.restraint_group = group_avail[0]

            old_int._system_parameters = {system_parameter for system_parameter in old_int._alchemical_functions.keys()}
            new_int = AlchemicalExternalRestrainedLangevinIntegrator(restraint_group=self.restraint_group, **old_int.kwargs)
            new_int.reset()
            initial_traj = self.binding_mode_traj[0].openmm_positions(0).value_in_unit(unit.nanometers)
            self.atom_indices
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
                #check which force groups aren't being used and set restraint forces to that
                new_sys = add_restraints(new_sys, structure, pose_allpos, self.atom_indices, index, self.restraint_group,
                                        self.restrained_receptor_atoms, restraint_lig, self.K_r, self.K_angle)



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


        if self.restraints == True:
            #if using restraints
            selected_list = self._poseDart(context, self.atom_indices)
            if len(selected_list) >= 1:
                self.selected_pose = np.random.choice(selected_list, replace=False)
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
        if self.restraints == True:
            selected_list = self._poseDart(context, self.atom_indices)
            if self.selected_pose not in selected_list:
                self.acceptance_ratio = 0
            else:
                pass
            for i in range(len(self.binding_mode_traj)):
                context.setParameter('restraint_pose_'+str(i), 0)

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
        if self.restraints == True:
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

        if self.restraints == True:
            self.restraint_correction = 0
            total_pe_restraint1_on = state.getPotentialEnergy()

            context.setParameter('restraint_pose_'+str(self.selected_pose), 0)
            state_restraint1_off = context.getState(getPositions=True, getEnergy=True)
            total_pe_restraint1_off = state_restraint1_off.getPotentialEnergy()
            restraint1_energy = total_pe_restraint1_on - total_pe_restraint1_off


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
            new_pos, darted_pose, prob_before = self._moldRedart(atom_indices=self.atom_indices,
                                            binding_mode_pos=self.binding_mode_traj,
                                            binding_mode_index=self.selected_pose,
                                            nc_pos=oldDartPos,
                                            rigid_move=self.rigid_move)

            self.selected_pose = darted_pose
            context.setPositions(new_pos)
            overlap_after = self._poseDart(context, self.atom_indices)
            dummy_index, prob_after = self._dart_selection(self.selected_pose)

            # to maintain detailed balance, check to see the overlap of the start and end darting regions
            # if there is no overlap after the move, acceptance ratio will be 0
            #TODO: Check if probability order is right
            group_ratio = prob_after/prob_before
            if len(overlap_after) == 0:
                self.acceptance_ratio = 0
            else:
                self.acceptance_ratio = self.acceptance_ratio*float(len(selected_list))/float(len(overlap_after)*(group_ratio))

            #check if new positions overlap when moving
            if self.restraints == True:

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
                 #lambda_restraints='max(0, 1-(1/0.30)*abs(lambda-0.5))',
                 lambda_restraints='max(0, 1-(1/0.10)*abs(lambda-0.5))',
                 *args, **kwargs):
        print('lambda_restraints', lambda_restraints)
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
        #self.addGlobalVariable("restraint_energy", 0)

        try:
            pass
            #self.addGlobalVariable("lambda_restraints", 0)
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
        #try:
        #    self.getGlobalVariableByName("lambda_restraints")
        #except:
        #    self.addGlobalVariable("lambda_restraints", 0)

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




