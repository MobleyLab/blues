from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as unit
import numpy as np
from blues.ncmc import SimNCMC, get_lig_residues
import mdtraj as md
from blues.lin_math import calc_rotation_matrix, adjust_angle
from blues.rot_mat import getRotTrans
import itertools
import chemcoord as cc
import copy

def rigidDart(apos, bpos, rot, centroid_difference, residueList=None):
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
    residueList
    '''
    if type(residueList) == type(None):
        residueList = self.residueList
    a_new = apos[:]
    num_res = len(residueList)
    a_res = np.zeros((len(residueList),3))
    b_res = np.zeros((len(residueList),3))
    for index, i in enumerate(residueList):
        a_res[index] = apos[i]
        b_res[index] = bpos[i]
    holder_rot, trans, centa, centb, holder_centroid_difference = rigid_transform_3D(a_res, b_res)
    b_removed_centroid = b_res - (np.tile(centb, (num_res, 1)))
    b_new = (np.tile(centroid_difference, (num_res, 1))) + b_res
    b_rot = (np.dot(rot, b_removed_centroid.T)).T
    #changed
    b_new = b_rot + (np.tile(centroid_difference, (num_res, 1))) + (np.tile(centb, (num_res, 1)))
    b_new = b_new * unit.nanometer
    for index, i in enumerate(residueList):
        a_new[residueList[index]] = b_new[index]
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
    n_rows = np.shape(array)[0]
    sub_vec = np.tile(array[rotation_center,:], (n_rows,1))
    rotated_array = array - sub_vec
    rotated_array = rotated_array.dot(rot_matrix)
    rotated_array = rotated_array + sub_vec
    return rotated_array




class MolDart(SimNCMC):
    """
    Class for performing smart darting moves during an NCMC simulation.
    """
    def __init__(self, pdb_files, xyz_file, fit_atoms, dart_size, symmetric_atoms=None, **kwds):
        super(MolDart, self).__init__(**kwds)
        self.dartboard = []
        self.dart_size = []
        print('initizalizing dart', dart_size._value, type(dart_size._value))
        print(self.residueList)
        if type(dart_size._value) == list:
            if len(dart_size) != len(residueList):
                raise ValueError('mismatch between length of dart_size (%i) and residueList (%i)' % (len(dart_size), len(residueList)) )
            self.dart_size = dart_size
        elif type(dart_size._value) == int or type(dart_size._value) == float:
            print('adding the same size darts')
            for entry in self.residueList:
                print('appending dart')
                self.dart_size.append(dart_size.value_in_unit(unit.nanometers))
            self.dart_size = self.dart_size*unit.nanometers

        self.binding_mode_traj = []
        self.fit_atoms = fit_atoms
        self.ligand_pos = None
        self.symmetric_atoms = symmetric_atoms
        self.internal_xyz = []
        self.internal_zmat = []
        self.buildlist = None
        xyz = cc.xyz_functions.read(xyz_file, start_index=0)
        self.buildlist = xyz._get_buildlist()

        for j, pdb_file in enumerate(pdb_files):
            traj = md.load(pdb_file)[0]
            self.binding_mode_traj.append(copy.deepcopy(traj))
            #get internal representation
            self.internal_xyz.append(copy.deepcopy(xyz))

            for index, entry in enumerate(['x', 'y', 'z']):
                for i in range(len(self.residueList)):
                    sel_atom = self.residueList[i]
                    self.internal_xyz[j].frame.set_value(i, entry, self.binding_mode_traj[j].xyz[0][:,index][sel_atom]*10)
                self.internal_zmat.append(self.internal_xyz[j].to_zmat(buildlist=self.buildlist))


        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])

    def add_dart(self, dart):
        self.dartboard.append(dart)

    def dist_from_dart_center(self, sim_atom_pos, binding_mode_atom_pos, symmetric_atoms=None):
        """function to calculate the distances from the dart centers from all given poses
        Arguments
        ---------
        sim_atom_pos: nx3 np.array
            simulation positions of the ligand atoms
        binding_mode_atom_pos: nx3 np.array
            positions of the ligand atoms from a given poses
        symmetric_atoms: list of lists
            list of symmetric atoms
        """
        if symmetric_atoms == None:
            symmetric_atoms = self.symmetric_atoms

        num_lig_atoms = len(self.residueList)

        dist_list = np.zeros((num_lig_atoms, 1))
        diff_list = np.zeros((num_lig_atoms, 3))
        indexList = []
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
                original_index = [self.residueList.index(x) for x in symm_group]
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



        return dist_list, diff_list

    def poseDart(self, context=None, residueList=None):
        """check whether molecule is within a pose, and
        if it is, return the dart vectors for it's atoms
        """
        if context == None:
            context = self.nc_context
        if residueList == None:
            residueList = self.residueList
        total_diff_list = []
        total_dist_list = []
        nc_pos = context.getState(getPositions=True).getPositions()
        #update sim_traj positions for superposing binding modes
        #might need to make self.sim_traj.xyz = nc_pos._value into
        #self.sim_traj.xyz = [nc_pos._value] or np.array
        self.sim_traj.xyz = nc_pos._value
        #make a temp_pos to specify dart centers to compare
        #distances between each dart and binding mode reference
        temp_pos = []
        num_lig_atoms = len(self.residueList)
        temp_pos = np.zeros((num_lig_atoms, 3))

        for index, atom in enumerate(residueList):
            #keep track of units
            temp_pos[index] = nc_pos[atom]._value

        #fit different binding modes to current protein
        #to remove rotational changes
        for pose in self.binding_mode_traj:
            pose = pose.superpose(reference=self.sim_traj,
                            atom_indices=self.fit_atoms,
                            ref_atom_indices=self.fit_atoms)
            pose_coord = pose.xyz[0]
            binding_mode_pos = []
            #find the dart vectors and distances to each protein
            #append the list to a storage list
            temp_binding_mode_pos = np.zeros((num_lig_atoms, 3))
            temp_binding_mode_diff = np.zeros((num_lig_atoms, 3))
            temp_binding_mode_dist = np.zeros((num_lig_atoms, 1))

            for index, atom in enumerate(residueList):
                temp_binding_mode_pos[index] = pose_coord[atom]
            temp_dist, temp_diff = self.dist_from_dart_center(temp_pos, temp_binding_mode_pos)
            #TODO: replace the actual simulation position/velocities with the symmetric equivalents if found!!!!
            total_diff_list.append(temp_diff[:])
            total_dist_list.append(temp_dist[:])

        selected = []
        #check to see which poses fall within the dart size
        for index, single_pose in enumerate(total_dist_list):
            counter = 0
            for atomnumber,dist in enumerate(single_pose):
                if dist <= self.dart_size[atomnumber]._value:
                    counter += 1
            if counter == len(residueList):
                selected.append(index)
            print('counter for pose', index, 'is ', counter)
        if len(selected) == 1:
            #returns binding mode index, and the diff_list
            #diff_list will be used to dart
            return selected[0], total_diff_list[selected[0]]
        elif len(selected) == 0:
            return None, total_diff_list
        elif len(selected) >= 2:
            print('overlapping darts', selected)
            #COM should never be within two different darts
            raise ValueError('sphere size overlap, check darts')

    def poseRedart(self, changevec, binding_mode_pos, binding_mode_index, nc_pos, residueList=None):
        """
        Helper function to choose a random pose and determine the vector
        that would translate the current particles to that dart center
        Arguments
        ----------
        changevec: list
            The change in vector that you want to apply,
            typically supplied by poseDart
        """

        if residueList == None:
            residueList = self.residueList
            changed_pos = copy.deepcopy(nc_pos)
        rand_index = np.random.randint(len(self.binding_mode_traj))
        ###temp to encourage going to other binding modes
        while rand_index == binding_mode_index:
            rand_index = np.random.randint(len(self.binding_mode_traj))
        ###

        print('total residues', residueList)
        for index, atom in enumerate(residueList):
            #index refers to where in list
            #atom refers to atom#
            dartindex = binding_mode_index
            dart_origin = (binding_mode_pos[rand_index].xyz)[0][atom]
            dart_change = dart_origin + changevec[index]
            changed_pos[atom] = dart_change*unit.nanometers

        return changed_pos

    def poseRigidRedart(self, binding_mode_pos, binding_mode_index, nc_pos, residueList=None):
        """
        Helper function to choose a random pose and determine the vector
        that would translate the current particles to that dart center
        Arguments
        ----------
        changevec: list
            The change in vector that you want to apply,
            typically supplied by poseDart
        binding_mode_pos: list of nx3 np.arrays
            list that contains the coordinates of the various binding modes
        binding_mode_index: int
            integer given by poseRedart that specifes which binding mode
            out of the list it matches with
        """

        if residueList == None:
            residueList = self.residueList
        changed_pos = nc_pos[:]
        #choose a random binding pose
        rand_index = np.random.randint(len(self.binding_mode_traj))
        ###temp to encourage going to other binding modes
        while rand_index == binding_mode_index:
            rand_index = np.random.randint(len(self.binding_mode_traj))
        ###
        #get matching binding mode pose and get rotation/translation to that pose

        selected_mode = binding_mode_pos[binding_mode_index].xyz[0]
        random_mode = binding_mode_pos[rand_index].xyz[0]
        rotation, centroid_difference = getRotTrans(nc_pos, selected_mode, residueList)
        return_pos = rigidDart(nc_pos, random_mode, rotation, centroid_difference, residueList)
        return return_pos
        #use rot and translation to dart to another pose

    def moldRedart(self, binding_mode_pos, binding_mode_index, nc_pos, residueList=None):
        """
        Helper function to choose a random pose and determine the vector
        that would translate the current particles to that dart center
        Arguments
        ----------
        changevec: list
            The change in vector that you want to apply,
            typically supplied by poseDart
        binding_mode_pos: list of nx3 np.arrays
            list that contains the coordinates of the various binding modes
        binding_mode_index: int
            integer given by poseRedart that specifes which binding mode
            out of the list it matches with
        """

        if residueList == None:
            residueList = self.residueList
        #choose a random binding pose
        rand_index = np.random.randint(len(self.binding_mode_traj))
        ###temp to encourage going to other binding modes
        while rand_index == binding_mode_index:
            rand_index = np.random.randint(len(self.binding_mode_traj))
        ###
        #get matching binding mode pose and get rotation/translation to that pose
        #TODO decide on making a copy or always point to same object
        xyz_ref = copy.deepcopy(self.internal_xyz[0])
        for index, entry in enumerate(['x', 'y', 'z']):
            for i in range(len(self.residueList)):
                sel_atom = self.residueList[i]
                #set the pandas series with the appropriate data
                #multiply by 10 since openmm works in nm and cc works in angstroms
                xyz_ref.frame.set_value(i, entry, (nc_pos[:,index][sel_atom]._value*10))
        print('xyz_ref', xyz_ref)
        print('buildlist', self.buildlist)
        zmat_diff = xyz_ref.to_zmat(buildlist=self.buildlist)
        #get appropriate comparision zmat
        zmat_compare = self.internal_zmat[binding_mode_index]
        for i in ['angle', 'dihedral']:
            zmat_diff.frame[i] = zmat_diff.frame[i] - zmat_compare.frame[i]
        zmat_new = copy.deepcopy(zmat_diff)
        for i in ['angle', 'dihedral']:
            zmat_new.frame[i] = zmat_diff.frame[i] + zmat_compare.frame[i]

        selected_mode = binding_mode_pos[binding_mode_index].xyz[0]
        #find translation differences in positions of first two atoms to reference structure
        #find the appropriate rotation to transform the structure back
        #repeat for second bond
        #get first 3 new moldart positions, apply same series of rotation/translations
        sim_three = np.zeros((3,3))
        ref_three = np.zeros((3,3))
        dart_three = np.zeros((3,3))
        for i in range(3):
            sim_three[i] = nc_pos[residueList[self.buildlist[i, 0]]]
            print('using index', [residueList[self.buildlist[i, 0]]])
            ref_three[i] = binding_mode_pos[binding_mode_index].xyz[0][residueList[self.buildlist[i, 0]]]
            dart_three[i] = binding_mode_pos[rand_index].xyz[0][residueList[self.buildlist[i, 0]]]
            print('dart3 1', dart_three)
        vec1_sim = sim_three[1,:] - sim_three[0,:]
        vec2_sim = sim_three[2,:] - sim_three[1,:]
        vec1_ref = ref_three[1,:] - ref_three[0,:]
        vec2_ref = ref_three[2,:] - ref_three[1,:]
        #calculate rotation from ref pos to sim pos
        rotation1 = calc_rotation_matrix(vec1_ref, vec1_sim)
        rotation2 = calc_rotation_matrix(vec2_ref, vec2_sim)
        pos_diff = sim_three[0,:] - ref_three[0,:]
        #apply translation, rotations to new positions
        dart_three = dart_three + np.tile(pos_diff, (3,1))
        dart_three = apply_rotation(dart_three, rotation1, 0)
        second_rot = apply_rotation(dart_three[1:], rotation2, 0)
        dart_three[1:] = second_rot
        #added
        vec1_dart = dart_three[0] - dart_three[1]
        vec2_dart = dart_three[2] - dart_three[1]
        print('debug', zmat_new.frame['angle'][self.buildlist[2,0]])
        dart_degrees = zmat_new.frame['angle'][self.buildlist[2,0]]
        if 0:
            adjusted = adjust_angle(vec2_dart, vec1_dart, np.radians(dart_degrees))
            print('adjust', adjusted, np.linalg.norm(adjusted))
            print('old', vec2_dart, np.linalg.norm(vec2_dart))
            dart_three[2] = dart_three[1] + adjusted
        #exit()

        random_mode = self.internal_zmat[rand_index]
        #random_mode = binding_mode_pos[rand_index].xyz[0]
        print('zmat_diff', zmat_diff)
        print('zmat_compare', zmat_compare)
        print('zmat_new', zmat_new)
        print('starting_coord', dart_three)
        xyz_new = (zmat_new.to_xyz(starting_coord=dart_three*10)).sort_index()
        #TODO make sure to sort new xyz
        print('xyz_new.frame unsorted', xyz_new.frame)
        print('xyz_new.frame', xyz_new.frame.sort_index())
        print('darted from', self.internal_xyz[binding_mode_index])

        #overlay new xyz onto the first atom of
        for index, entry in enumerate(['x', 'y', 'z']):
            for i in range(len(self.residueList)):
                sel_atom = self.residueList[i]
                #TODO from friday: set units for xyz_first_pos and then go about doing the rotation to reorient the molecule after moving
                nc_pos[:,index][sel_atom] = (xyz_new.frame[entry][i] / 10) * unit.nanometers
                #self.internal_xyz[j].frame.set_value(i, entry, nc_pos[:,index][sel_atom]*10)
            print('putting', nc_pos[sel_atom])

        return nc_pos
        #use rot and translation to dart to another pose


    def poseRigidMove(self, context=None, residueList=None):
        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context
        stateinfo = context.getState(True, True, False, True, True, False)
        oldEnergy = stateinfo.getPotentialEnergy()
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        selected_pose, diff_list = self.poseDart()
        if selected_pose == None:
            print('no pose found')
        else:
            print('yes pose found')
            print('oldPos', oldDartPos[-3:])

            new_pos = self.poseRigidRedart(binding_mode_pos=self.binding_mode_traj,
                                            binding_mode_index=selected_pose,
                                            nc_pos=oldDartPos)
            context.setPositions(new_pos)
            stateinfo = context.getState(True, True, False, True, True, False)
            newEnergy = stateinfo.getPotentialEnergy()
            print('oldEnergy', oldEnergy)
            print('newEnergy', newEnergy)
            old_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_oldEnergy',old_md_state.getPotentialEnergy())
            self.md_simulation.context.setPositions(new_pos)
            new_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_newEnergy',new_md_state.getPotentialEnergy())


    def moldMove(self, context=None, residueList=None):
        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context
        stateinfo = context.getState(True, True, False, True, True, False)
        oldEnergy = stateinfo.getPotentialEnergy()
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        selected_pose, diff_list = self.poseDart()
        #now self.binding_mode_pos should be fitted to structure at this point
        #use the first entry

        if selected_pose == None:
            print('no pose found')
        else:
            print('yes pose found')
            print('oldPos', oldDartPos[-3:])
            #use moldRedart instead
            #calculate changes in angle/dihedral compared to reference
            #apply angle/dihedral changes to new pose
            #translate new pose to center of first molecule
            #find rotation that matches atom1 and atom2s of the build list
            #apply that rotation using atom1 as the origin

            new_pos = self.moldRedart(binding_mode_pos=self.binding_mode_traj,
                                            binding_mode_index=selected_pose,
                                            nc_pos=oldDartPos)
            context.setPositions(new_pos)
            stateinfo = context.getState(True, True, False, True, True, False)
            newEnergy = stateinfo.getPotentialEnergy()
            print('oldEnergy', oldEnergy)
            print('newEnergy', newEnergy)
            old_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_oldEnergy',old_md_state.getPotentialEnergy())
            self.md_simulation.context.setPositions(new_pos)
            new_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_newEnergy',new_md_state.getPotentialEnergy())

    def poseMove(self, context=None, residueList=None):
        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context
        stateinfo = context.getState(True, True, False, True, True, False)
        oldEnergy = stateinfo.getPotentialEnergy()
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        selected_pose, diff_list = self.poseDart()
        if selected_pose == None:
            print('no pose found')
        else:
            print('yes pose found')
            new_pos = self.poseRedart(changevec=diff_list,
                binding_mode_pos=self.binding_mode_traj,
                binding_mode_index=selected_pose,
                nc_pos=oldDartPos)
            context.setPositions(new_pos)
            stateinfo = context.getState(True, True, False, True, True, False)
            newEnergy = stateinfo.getPotentialEnergy()
            print('oldEnergy', oldEnergy)
            print('newEnergy', newEnergy)
            old_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_oldEnergy',old_md_state.getPotentialEnergy())
            self.md_simulation.context.setPositions(new_pos)
            new_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_newEnergy',new_md_state.getPotentialEnergy())

    def findDart(self, particle_pairs=None, particle_weights=None):
        """
        For dynamically updating dart positions based on positions
        of other particles.
        This takes the weighted average of the specified particles
        and changes the dartboard of the object
        Arguments
        ---------
        particle_pairs: list of list of ints
            each list defines the pairs to define darts
        particle_weights: list of list of floats
            each list defines the weights assigned to each particle positions
        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights
        """
        if particle_pairs == None:
            particle_pairs = self.particle_pairs
        if particle_weights == None:
            particle_weights = self.particle_weights
        #make sure there's an equal number of particle pair lists
        #and particle weight lists
        assert len(particle_pairs) == len(particle_weights)

        dart_list = []
        state_info = self.nc_context.getState(True, True, False, True, True, False)
        temp_pos = state_info.getPositions(asNumpy=True)
        #find particles positions and multiply by weights
        for i, ppair in enumerate(particle_pairs):
            temp_array = np.array([0, 0, 0]) * unit.nanometers
            #weighted average
            temp_wavg = 0
            for j, particle in enumerate(ppair):
                print('temp_pos', particle, temp_pos[particle])
                temp_array += (temp_pos[particle] * float(particle_weights[i][j]))
                temp_wavg += float(particle_weights[i][j])
                print(temp_array)
            #divide by total number of particles in a list and append
            #calculated postion to dart_list
            dart_list.append(temp_array[:] / temp_wavg)
        self.dartboard = dart_list[:]
        return dart_list
