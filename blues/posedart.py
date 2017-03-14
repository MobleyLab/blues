from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as unit
import numpy as np
from blues.ncmc import SimNCMC, get_lig_residues
import mdtraj as md
from blues.rot_mat import getRotTrans, rigidDart
import itertools

class PoseDart(SimNCMC):
    """
    Class for performing smart darting moves during an NCMC simulation.
    """
    def __init__(self, pdb_files, fit_atoms, dart_size, symmetric_atoms=None, **kwds):
        super(PoseDart, self).__init__(**kwds)
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
        for pdb_file in pdb_files:
            traj = md.load(pdb_file)[0]
            self.binding_mode_traj.append(copy.deepcopy(traj))
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
            positions of the ligand atoms from the different poses
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
            for index, dart in enumerate(binding_mode_atom_pos):
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
                    for x in iter_symm:
                        for i, atom in enumerate(x):
                            #i is the index, atom is the original_atom index
                            #switch original_index with permutation
                            temp_sim_pos = sim_atom_pos[:]
                            temp_sim_pos[original_index[i]] = sim_atom_pos[atom]
                            diff = temp_sim_pos[original_index[i]] - binding_mode_atom_pos[original_index[i]]
                            dist = np.sqrt(np.sum((diff)*(diff)))
                            compare_diff.append(diff)
                            compare_dist.append(dist)
                        if np.sum(compare_dist) < np.sum(dist_subset):
                            print('better symmetric equivalent found')
                            #replace changed variables
                            sim_atom_pos = temp_sim_pos[:]
                            #replace diff_list and dist_list with updated values
                            for i, atom in enumerate(x):
                                diff_list[atom] = compare_diff[i]
                                dist_list[atom] = compare_dist[i]
                                #TODO might wanna trade velocities too


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
            total_diff_list.append(temp_diff)
            total_dist_list.append(temp_dist)

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
            print(selected)
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