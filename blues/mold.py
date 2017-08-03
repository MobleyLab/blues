from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.unit as unit
import numpy as np
import mdtraj as md
from mdtraj.formats.xyzfile import XYZTrajectoryFile
from mdtraj.utils import in_units_of
from blues.lin_math import calc_rotation_matrix, adjust_angle, kabsch
from blues.lin_math import getRotTrans
from blues.moves import RandomLigandRotationMove
import itertools
import chemcoord as cc
import copy
import tempfile
import types
from blues.mold_helper import give_cartesian_edit

def checkDifference(a, b, radius):
    def getDihedralDifference(dihedral):
        if dihedral < 0:
            mag_diff = -(dihedral % 180)
        else:
            mag_diff = dihedral % 180
        return mag_diff

    adiff = getDihedralDifference(a)
    bdiff = getDihedralDifference(b)
    absdiff = abs(adiff - bdiff)
    if absdiff < radius:
        return True
    else:
        return False

def rigidDart(apos, bpos, rot, centroid_difference, atom_indices):
    '''
    Get rotation and translation of rigid pose
    Arguments
    ---------
    apos: nx3 np.array
        simulation positions
    bpos: nx3 np.array
        comparison positoins
    rot: 3x3 np.array
        Rotation to be applied from other dart.
    centroid_difference: 1x3 np.array
        Vector difference between other dart and simulation centroids.
    atom_indices
    '''
    a_new = apos[:]
    num_res = len(atom_indices)
    a_res = np.zeros((len(atom_indices),3))
    b_res = np.zeros((len(atom_indices),3))
    for index, i in enumerate(atom_indices):
        a_res[index] = apos[i]
        b_res[index] = bpos[i]
    holder_rot, trans, centa, centb, holder_centroid_difference = rigid_transform_3D(a_res, b_res)
    b_removed_centroid = b_res - (np.tile(centb, (num_res, 1)))
    b_new = (np.tile(centroid_difference, (num_res, 1))) + b_res
    b_rot = (np.dot(rot, b_removed_centroid.T)).T
    #changed
    b_new = b_rot + (np.tile(centroid_difference, (num_res, 1))) + (np.tile(centb, (num_res, 1)))
    b_new = b_new * unit.nanometer
    for index, i in enumerate(atom_indices):
        a_new[atom_indices[index]] = b_new[index]
    return a_new

def calc_rotation_matrix(vec_ref, vec_target):
    '''calculate the rotation matrix that will rotate vec_ref to vec_target
    Arguments
    ---------
    vec_ref: np.array
        Vector to calculate rotation matrix to vec_target
    vec_target: np.array
        Target vector to rotate to.
    '''
    #get normalized vectors
    a = np.array(vec_target) / np.linalg.norm(vec_target)
    b = np.array(vec_ref) / np.linalg.norm(vec_ref)
    #get cross product to get norm of plane these vectors are in
    v = np.cross(a,b)
    vec_cos = np.inner(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #create skew symmetric matrix
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[1]],
                    [-v[1], v[0], 0]
                    ])
    I = np.identity(3)
    #actually calculate rotation matrix
    R = I + vx + vx.dot(vx)*(1/(1+vec_cos))
    return R

def apply_rotation(array, rot_matrix, rotation_center):
    print('before rotation', array)
    n_rows = np.shape(array)[0]
    sub_vec = np.tile(array[rotation_center], (n_rows,1))
    rotated_array = array - sub_vec
    rotated_array = rotated_array.dot(rot_matrix)
    rotated_array = rotated_array + sub_vec
    print('rotated_array', rotated_array)
    return rotated_array




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
    dart_size: simtk.unit.nanometers, or list of simtk.unit.nanometers:
        The size of the nth dart around each nth atom in the atom_indices,
        given by the resname parameter.
         If no list is given the darts are assumed to be of uniform size.
    resname : str, optional, default='LIG'
        String specifying the residue name of the ligand.
    symmetric_atoms: default=None
        Not currently implemented
    rigid_move: boolean, default=False:
        If True, will ignore internal coordinate changes while darting
        and will effectively perform a rigid body rotation between darts
        instead.

    """
    def __init__(self, structure, pdb_files, fit_atoms, dart_size, resname='LIG', symmetric_atoms=None, rigid_move=False):
        super(MolDart, self).__init__(structure, resname)
        self.dartboard = []
        self.dart_size = []
        if type(dart_size) == list:
            if len(dart_size) != len(self.atom_indices):
                raise ValueError('mismatch between length of dart_size (%i) and atom_indices (%i)' % (len(dart_size), len(atom_indices)) )
            self.dart_size = dart_size
        elif type(dart_size._value) == int or type(dart_size._value) == float:
            for entry in self.atom_indices:
                self.dart_size.append(dart_size.value_in_unit(unit.nanometers))
            self.dart_size = self.dart_size*unit.nanometers

        self.binding_mode_traj = []
        self.fit_atoms = fit_atoms
        self.ligand_pos = None
        self.symmetric_atoms = symmetric_atoms
        self.internal_xyz = []
        self.internal_zmat = []
        self.buildlist = None
        self.rigid_move = rigid_move
        self.ref_traj = None
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
        #get the construction table so internal coordinates are consistent
        self.buildlist = xyz.get_construction_table()
        ref_traj = md.load(pdb_files[0])[0]
        self.ref_traj = ref_traj
        for j, pdb_file in enumerate(pdb_files):
            traj = md.load(pdb_file)[0]
 #           traj.superpose(reference=ref_traj, atom_indices=fit_atoms, ref_atom_indices=fit_atoms)
            self.binding_mode_traj.append(copy.deepcopy(traj))
            #get internal representation
            self.internal_xyz.append(copy.deepcopy(xyz))

            for index, entry in enumerate(['x', 'y', 'z']):
                for i in range(len(self.atom_indices)):
                    sel_atom = self.atom_indices[i]
                    self.internal_xyz[j]._frame.set_value(i, entry, self.binding_mode_traj[j].xyz[0][:,index][sel_atom]*10)
            self.internal_zmat.append(self.internal_xyz[j].give_zmat(construction_table=self.buildlist))


        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])

    def dist_from_dart_center(self, sim_atom_pos, binding_mode_atom_pos, symmetric_atoms=None):
        """Function to calculate the distances from the dart centers from all given poses.
        Parameters
        ----------
        sim_atom_pos: nx3 np.array
            simulation positions of the ligand atoms
        binding_mode_atom_pos: nx3 np.array
            positions of the ligand atoms from a given poses
        symmetric_atoms: list of lists
            list of symmetric atoms (not yet implemented)

        Returns
        -------
        sim_atom_pos: nx3 np.array
            Simulation positions of the ligand atoms
        diff_list: list of n floats
            List of distances between each simulation dart center and
            the corresponding reference dart center.

        diff_list: list of n 1x3 np.arrays
            List of the vectors between each simulation dart center
            and its corresponding reference dart center.
        """
        if symmetric_atoms == None:
            symmetric_atoms = self.symmetric_atoms

        num_lig_atoms = len(self.atom_indices)

        dist_list = np.zeros((num_lig_atoms, 1))
        diff_list = np.zeros((num_lig_atoms, 3))
        #Find the distances of the center to each dart, appending
        #the results to dist_list
        #TODO change to handle np.arrays instead

        for index, dart in enumerate(binding_mode_atom_pos):
            diff = sim_atom_pos[index] - dart
            dist = np.sqrt(np.sum((diff)*(diff)))
            diff_list[index] = diff
            dist_list[index] = dist

        if symmetric_atoms != None:
            print('checking for symmetric equivalents')
            #make temporary pos to compare distances with
            #loop over symmetric groups
            for symm_group in symmetric_atoms:
                compare_diff =[]
                compare_dist =[]
                #find the original index, which correspods with the position in temp_sim_pos
                original_index = [self.atom_indices.index(x) for x in symm_group]
                #create permutations of the symmetric atom indices
                iter_symm = itertools.permutations(original_index)
                dist_subset = [dist_list[x] for x in original_index]
                #iterate over the permutations
                xcounter = 0
                for x in iter_symm:
                    print('xcounter', xcounter)
                    xcounter = xcounter+1
                    for i, atom in enumerate(x):
                        #i is the index, atom is the original_atom index
                        #switch original_index with permutation
                        temp_sim_pos = np.copy(sim_atom_pos)
                        temp_sim_pos[original_index[i]] = sim_atom_pos[atom]
                        diff = temp_sim_pos[original_index[i]] - binding_mode_atom_pos[original_index[i]]
                        dist = np.sqrt(np.sum((diff)*(diff)))
                        compare_diff.append(diff)
                        compare_dist.append(dist)
                        print('dist', compare_dist)
                    if np.sum(compare_dist) < np.sum(dist_subset):
                        print('better symmetric equivalent found')
                        #replace changed variables
                        sim_atom_pos[:] = temp_sim_pos
                        #replace diff_list and dist_list with updated values
                        for i, atom in enumerate(x):
                            diff_list[atom] = compare_diff[i]
                            dist_list[atom] = compare_dist[i]
                            #TODO might wanna trade velocities too
                    compare_diff =[]
                    compare_dist =[]



        return sim_atom_pos, dist_list, diff_list

    def poseDart(self, context, atom_indices):
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
        total_diff_list = []
        total_dist_list = []
        symm_pos_list = []

        nc_pos = context.getState(getPositions=True).getPositions()
        #update sim_traj positions for superposing binding modes
        self.sim_traj.xyz = nc_pos._value
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
        print('fit_atoms', self.fit_atoms)
        print('ref_traj', self.ref_traj)
        self.sim_traj.superpose(reference=self.ref_traj,
                            atom_indices=self.fit_atoms,
                            ref_atom_indices=self.fit_atoms)
        for pose in self.binding_mode_traj:
            pose_coord = pose.xyz[0]
            #find the dart vectors and distances to each protein
            #append the list to a storage list
            temp_binding_mode_pos = np.zeros((num_lig_atoms, 3))

            for index, atom in enumerate(atom_indices):
                temp_binding_mode_pos[index] = pose_coord[atom]
            temp_pos, temp_dist, temp_diff = self.dist_from_dart_center(temp_pos, temp_binding_mode_pos)

            #TODO: replace the actual simulation position/velocities with the symmetric equivalents if found!!!!
            total_diff_list.append(temp_diff[:])
            total_dist_list.append(temp_dist[:])
            symm_pos_list.append(np.copy(temp_pos)*unit.nanometers)
            print('making copy')
            print('symm_list', symm_pos_list)
            print('original', temp_pos)

        selected = []
        #check to see which poses fall within the dart size
        for index, single_pose in enumerate(total_dist_list):
            counter = 0
            for atomnumber,dist in enumerate(single_pose):
                if dist <= self.dart_size[atomnumber]._value:
                    counter += 1
            if counter == len(atom_indices):
                selected.append(index)
            print('counter for pose', index, 'is ', counter)
        if len(selected) == 1:
            #returns binding mode index, and the diff_list
            #diff_list will be used to dart
            return selected[0], total_diff_list[selected[0]], symm_pos_list[selected[0]]
        elif len(selected) == 0:
            return None, total_diff_list, None
        elif len(selected) >= 2:
            print('overlapping darts', selected)
            #COM should never be within two different darts
            raise ValueError('sphere size overlap, check darts')

    def moldRedart(self, atom_indices, binding_mode_pos, binding_mode_index, nc_pos, symm_pos, bond_compare=True, rigid_move=False):
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
        changevec: list
            The change in vector that you want to apply,
            typically supplied by poseDart
        binding_mode_pos: list of nx3 np.arrays
            list that contains the coordinates of the various binding modes
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
        #change symmetric atoms
        for i, atom_num in enumerate(atom_indices):
            nc_pos[atom_num] = symm_pos[i]
        rand_index = np.random.randint(len(self.binding_mode_traj))
        ###temp to encourage going to other binding modes
        while rand_index == binding_mode_index:
            rand_index = np.random.randint(len(self.binding_mode_traj))
        ###
        #get matching binding mode pose and get rotation/translation to that pose
        #TODO decide on making a copy or always point to same object
        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        for index, entry in enumerate(['x', 'y', 'z']):
            for i in range(len(self.atom_indices)):
                sel_atom = self.atom_indices[i]
                #set the pandas series with the appropriate data
                #multiply by 10 since openmm works in nm and cc works in angstroms
                xyz_ref._frame.set_value(i, entry, (nc_pos[:,index][sel_atom]._value*10))
        zmat_new = copy.deepcopy(self.internal_zmat[rand_index])
        print('zmat_new', zmat_new)
        if 1:
            zmat_diff = xyz_ref.give_zmat(construction_table=self.buildlist)
            #get appropriate comparision zmat
            zmat_compare = self.internal_zmat[binding_mode_index]
            change_list = ['angle', 'dihedral']
            if bond_compare == True:
                change_list.append('bond')
            if rigid_move == False:
                for i in change_list:
                    zmat_diff._frame[i] = zmat_diff._frame[i] - zmat_compare._frame[i]
                for i in change_list:
                #change form zmat_compare to random index
                    zmat_new._frame[i] = zmat_diff._frame[i] + zmat_new._frame[i]
            else:
                pass

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
            print('dart_angle', np.degrees(np.arccos(dart_angle)))
            return np.degrees(np.arccos(dart_angle))
        def angle_calc(angle1, angle2):
            angle = np.arccos(angle1.dot(angle2) / ( np.linalg.norm(angle1) * np.linalg.norm(angle2) ) )
            degrees = np.degrees(angle)
            return degrees
        def calc_angle(vec1, vec2):
            angle = np.arccos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return angle




        vector_list = findCentralAngle(self.buildlist)
        print('vector_list', vector_list)
        #find translation differences in positions of first two atoms to reference structure
        #find the appropriate rotation to transform the structure back
        #repeat for second bond
        #get first 3 new moldart positions, apply same series of rotation/translations
        sim_three = np.zeros((3,3))
        ref_three = np.zeros((3,3))
        dart_three = np.zeros((3,3))
        dart_ref = np.zeros((3,3))
        for i in range(3):
            sim_three[i] = nc_pos[atom_indices[self.buildlist.index.get_values()[i]]]
            print('using index', [atom_indices[self.buildlist.index.get_values()[i]]])
            self.buildlist.index.get_values()[i]

            ref_three[i] = binding_mode_pos[binding_mode_index].xyz[0][atom_indices[self.buildlist.index.get_values()[i]]]
            dart_three[i] = binding_mode_pos[rand_index].xyz[0][atom_indices[self.buildlist.index.get_values()[i]]]
            dart_ref[i] = binding_mode_pos[rand_index].xyz[0][atom_indices[self.buildlist.index.get_values()[i]]]
            print('dart3 1', dart_three)
        print('debugging')
        print('before sim', sim_three)
        print('before ref', ref_three)
        change_three = np.copy(sim_three)
        vec1_sim = sim_three[vector_list[0][0]] - sim_three[vector_list[0][1]]
        vec2_sim = sim_three[vector_list[1][0]] - sim_three[vector_list[1][1]]
        vec1_ref = ref_three[vector_list[0][0]] - ref_three[vector_list[0][1]]


        #calculate rotation from ref pos to sim pos
        print('vec1_ref', vec1_ref)
        print('vec1_sim', vec1_sim)
        print('before_rotation_sim_angle', test_angle(sim_three, vector_list))
        print('before_rotation_dart_angle', test_angle(dart_three, vector_list))

        #change angle of one vector
        ###edits
        print('d1', self.internal_zmat[binding_mode_index]._frame['angle'])

        ###
        ref_angle = self.internal_zmat[binding_mode_index]._frame['angle'][self.buildlist.index.get_values()[2]]
        print('sim angle', np.degrees(calc_angle(vec1_sim, vec2_sim)))
        print('ref_angle', ref_angle)
        angle_diff = ref_angle - np.degrees(calc_angle(vec1_sim, vec2_sim))
        print('angle_diff', angle_diff)
        ad_vec = adjust_angle(vec1_sim, vec2_sim, np.radians(ref_angle), maintain_magnitude=False)
        print('check that this is a bond length', self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.get_values()[1]])
        ad_vec = ad_vec / np.linalg.norm(ad_vec) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.get_values()[2]]/10.
        print('ad_vec length', np.linalg.norm(ad_vec))
        #apply changed vector to center coordinate to get new position of first particle
        print('change before before rot', change_three)

        nvec2_sim = vec2_sim / np.linalg.norm(vec2_sim) * self.internal_zmat[binding_mode_index]._frame['bond'][self.buildlist.index.get_values()[2]]/10.
        print('new angle1', np.degrees(calc_angle(ad_vec, nvec2_sim)))
        change_three[vector_list[0][0]] = sim_three[vector_list[0][1]] + ad_vec
        change_three[vector_list[1][0]] = sim_three[vector_list[0][1]] + nvec2_sim
        print('change before rot', change_three)
        rot_mat, centroid = getRotTrans(change_three, ref_three, center=vector_list[0][1])
        print('ref_three', ref_three)
        print('sim_three', sim_three)
        print('change_three', change_three)
        print('centroid movement', centroid, np.linalg.norm(centroid))
        #TODO CONTINUE FROM HERE
        #perform the same angle change on new coordinate
        centroid_orig = dart_three[vector_list[0][1]]
        #perform rotation
        print('rot_mat', rot_mat)
        other_rot = kabsch(change_three, ref_three, vector_list[0][1])
        print('other_rot', other_rot)
        dart_three = (dart_three -  np.tile(centroid_orig, (3,1))).dot(rot_mat) + np.tile(centroid_orig, (3,1)) - np.tile(centroid, (3,1))
        vec1_dart = dart_three[vector_list[0][0]] - dart_three[vector_list[0][1]]
        vec2_dart = dart_three[vector_list[1][0]] - dart_three[vector_list[1][1]]
        print('angle after centroid rotation', np.degrees(calc_angle(vec1_dart, vec2_dart)) )
        dart_angle = self.internal_zmat[rand_index]._frame['angle'][self.buildlist.index.get_values()[2]]
        angle_change = dart_angle - angle_diff
        print('angle_change', angle_change)
        if 1:
            ad_dartvec = adjust_angle(vec1_dart, vec2_dart, np.radians(angle_change), maintain_magnitude=False)
            ad_dartvec = ad_dartvec / np.linalg.norm(ad_dartvec) * zmat_new._frame['bond'][self.buildlist.index.get_values()[1]]/10.
            print(self.internal_zmat[rand_index]._frame['bond'][self.buildlist.index.get_values()[1]])
            print('length ad_dartvec', np.linalg.norm(ad_dartvec))
            nvec2_dart = vec2_dart / np.linalg.norm(vec2_dart) * zmat_new._frame['bond'][self.buildlist.index.get_values()[2]]/10.
            print(self.internal_zmat[rand_index]._frame['bond'][self.buildlist.index.get_values()[2]]/10.)
            print('length nvec2_dart', np.linalg.norm(nvec2_dart))
            print('new angle2', np.degrees(calc_angle(ad_dartvec, nvec2_dart)))
            dart_three[vector_list[0][0]] = dart_three[vector_list[0][1]] + ad_dartvec
            dart_three[vector_list[1][0]] = dart_three[vector_list[0][1]] + nvec2_dart

        print('dart_three after', dart_three)

        for i, vectors in enumerate([sim_three, ref_three, dart_three]):
            print(i)
            print(vectors[0],vectors[1],vectors[2])
            test_angle(vectors, vector_list)

        zmat_new.give_cartesian_edit = types.MethodType(give_cartesian_edit, zmat_new)
        #get xyz from internal coordinates
        xyz_new = (zmat_new.give_cartesian_edit(start_coord=dart_three*10.)).sort_index()

        #TODO make sure to sort new xyz

        #overlay new xyz onto the first atom of
        for index, entry in enumerate(['x', 'y', 'z']):
            for i in range(len(self.atom_indices)):
                sel_atom = self.atom_indices[i]
                #TODO from friday: set units for xyz_first_pos and then go about doing the rotation to reorient the molecule after moving
                nc_pos[:,index][sel_atom] = (xyz_new._frame[entry][i] / 10.) * unit.nanometers
            print('putting', nc_pos[sel_atom])

        return nc_pos
        #use rot and translation to dart to another pose

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
        oldDartPos = context.getState(getPositions=True).getPositions(asNumpy=True)
        selected_pose, diff_list, symm_list = self.poseDart(context, self.atom_indices)
        #now self.binding_mode_pos should be fitted to structure at this point
        #use the first entry

        if selected_pose == None:
            print('no pose found')
        else:
            print('yes pose found')
            #use moldRedart instead
            #calculate changes in angle/dihedral compared to reference
            #apply angle/dihedral changes to new pose
            #translate new pose to center of first molecule
            #find rotation that matches atom1 and atom2s of the build list
            #apply that rotation using atom1 as the origin

            new_pos = self.moldRedart(atom_indices=self.atom_indices,
                                            binding_mode_pos=self.binding_mode_traj,
                                            binding_mode_index=selected_pose,
                                            nc_pos=oldDartPos, symm_pos=symm_list,
                                            rigid_move=self.rigid_move)
            context.setPositions(new_pos)
        return context
