import numpy as np
from future.utils import iteritems
from itertools import combinations
import pandas as pd
from blues.moldart.lin_math import dartRotTrans
import copy

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


def compare_dihedral_edit(a_dir, b_dir, cutoff=80.0, construction_table=None):
    #originally in radians, convert to degrees
    a_dir, b_dir = np.deg2rad(a_dir), np.deg2rad(b_dir)
    b_cos, b_sin = np.cos(b_dir), np.sin(b_dir)
    a_cos, a_sin = np.cos(a_dir), np.sin(a_dir)
    cos_diff = np.square(b_cos - a_cos)
    sin_diff = np.square(b_sin - a_sin)
    dist = np.sqrt(cos_diff + sin_diff)

    if np.rad2deg(np.arcsin(dist/2.0)*2) <= cutoff:
        return 1
    else:
        #returns 0 if within radius
        return 0

def dihedralDifference(a, b, construction_table=None):
    """Computes the difference in dihedral angles
    between the pairs present in zmatrices a and b
    with the cartesian distance in cos, sin
    """
    a_di, b_di = a['dihedral'], b['dihedral']
    a_dir, b_dir = np.deg2rad(a_di), np.deg2rad(b_di)
    b_cos, b_sin = np.cos(b_dir), np.sin(b_dir)
    a_cos, a_sin = np.cos(a_dir), np.sin(a_dir)
    cos_diff = np.square(b_cos - a_cos)
    sin_diff = np.square(b_sin - a_sin)
    dist = np.sqrt(cos_diff + sin_diff)
    return dist

def makeStorageFrame(dataframe, num_poses):
    """
    Takes the dihedral dihedral and atom columns from a dataframe
    and stores them in a num_poses number of dataframes.

    Parameters
    ----------
    dataframe: pandas.dataframe
        The dataframe to be used as reference.
    num_poses: int
        The number of posed needed to be added.
        Will be joined to the dataframe as `pose_i` where i
        is an index.

    Returns
    -------
    out_frame: pandas.dataframe
        a dataframe labeled with the poses labeled.
    """

    dframe = dataframe['dihedral']
    counter=0
    temp_frame = dframe.copy()
    for i in range(num_poses):
        frame_name = 'pose_'+str(i)
        if counter == 0:
            out_frame = dframe.to_frame(frame_name)
        else:
            temp_frame = dframe.to_frame(frame_name)
            out_frame = out_frame.join(temp_frame)
        counter= counter+1
    for j in out_frame.columns:
        out_frame[:] = -1
    aframe = dataframe['atom']
    out_frame = aframe.to_frame('atom').join(out_frame)


    return out_frame


def makeDihedralDifferenceDf(internal_mat, dihedral_cutoff=0.3):
    print('dihedral_cutoff test', dihedral_cutoff)
    if dihedral_cutoff < 0:
        raise ValueError('Negative dihedral_cutoff distance does not make sense. Please specify a positive cutoff.')
    diff_dict = {}
    output = makeStorageFrame(internal_mat[0], len(internal_mat))
    for index, zmat in enumerate(internal_mat):
        diff_dict[index] = copy.deepcopy(output)
    for zmat in combinations(zip(range(len(internal_mat)), internal_mat), 2):
    #for zmat in combinations(internal_mat, 2):
        dist_series = dihedralDifference(zmat[0][1], zmat[1][1])
        first_index = zmat[0][0]
        second_index = zmat[1][0]
        sort_index = diff_dict[first_index].index
        #joining behaves unintuitively, to make it work properly sort by ascending index first
        diff_dict[first_index] = diff_dict[first_index].sort_index()
        diff_dict[second_index] = diff_dict[second_index].sort_index()
        #join the difference dictionary
        diff_dict[first_index].loc[:,'pose_'+str(second_index)] = dist_series.to_frame('pose_'+str(second_index)).sort_index()
        diff_dict[second_index].loc[:,'pose_'+str(first_index)] = dist_series.to_frame('pose_'+str(first_index)).sort_index()
        #use the old sorting index
        diff_dict[first_index] = diff_dict[first_index].loc[sort_index]
        diff_dict[second_index] = diff_dict[second_index].loc[sort_index]

    dihedral_dict = {}
    for i in internal_mat[0].index:
        dihedral_dict[i] = []
    #loop over entries in poses and find the distances for each
    for i in internal_mat[0].index[3:]:
        for key, df in iteritems(diff_dict):
            for pose in df.columns[1:]:
                if df['atom'].loc[i] != 'H':
                    dihedral_dict[i].append(df[pose].loc[i])

    #remove redundant entries in dict
    for key, di_list in iteritems(dihedral_dict):
        di_list = list(set(di_list))
        #only keep track of the sensible dihedrals (above a small cutoff distance)
        dihedral_dict[key] = [i for i in di_list if i >= dihedral_cutoff]
    #go over all entries and sort by atom#, difference with some cutoff (maybe 40 degrees?)

    #start with one with the largest minimum difference (or maybe just largest difference), above a cutoff
    #need to make dart identification w/ dihedral function
    #using function, identify which poses overlap if any
    #if poses overlap move on to next pose

    ndf = None
    entry_counter = 0
    for key, di_list in iteritems(dihedral_dict):
        for idx, di in enumerate(di_list):
            temp_frame = pd.DataFrame(data={'atomnum':key, 'diff':di}, index=[entry_counter])
            entry_counter = entry_counter +1
            if ndf is None:
                ndf = temp_frame.copy()
            else:
                ndf = ndf.append(temp_frame)

            ndf.append(temp_frame)
    #TODO: figure out how to treat duplicate angles
    if entry_counter == 0:
        return None
    ndf = ndf.sort_values(by='diff', ascending=False)
    #change internal_mat to the internal zmat storage variable
    return ndf

def compareDihedral(internal_mat, atom_index, diff_spread, posedart_dict, inRadians=True):
    posedart_copy = copy.deepcopy(posedart_dict)
    #reset counts for dictionary
    for posenum, zmat in enumerate(internal_mat):
        for other_posenum, other_zmat in enumerate(internal_mat):
            posedart_copy['pose_'+str(posenum)]['dihedral'][atom_index] = []
    #iterate over zmat and get original value of pose
    if inRadians==True:
        diff_spread = np.rad2deg(diff_spread)
    for posenum, zmat in enumerate(internal_mat):
        comparison = zmat['dihedral'].loc[atom_index]

        for other_posenum, other_zmat in enumerate(internal_mat):
            if posenum != other_posenum:
                other_comparison = other_zmat['dihedral'].loc[atom_index]
                result = compare_dihedral_edit(comparison, other_comparison, cutoff=diff_spread)
                if result == 1:
                    posedart_copy['pose_'+str(posenum)]['dihedral'][atom_index].append(other_posenum)
            else:
                pass
    return posedart_copy

def checkDihedralRegion(a, b, atomnum, cutoff=80.0):

    """
    This function checks a given dihedral for a particular atom of the construction table.
    If it is within the `cutoff`, of the dihedral darting region, then this returns 1,
    else returns 0, which is used to count the number of overlapping dihedrals.

    Parameters
    ----------
    a: chemcoord zmatrix
        Zmatrix to compare the dihedral of b (order doesn't matter between a and b)
    b: chemcoord zmatrix
        Zmatrix to compare the dihedral of a
    atomnum: integer
        The atom number of the construction table of a and b to compare to
    cutoff: Checks whether the dihedral distance of a and b is greater than the cutoff ( in degrees)

    Returns
    -------
    number: 1 or 0
        Returns 1 if the dihedral falls within the specified cutoff,
        otherwise returns 0.
    """
    a_di, b_di = a['dihedral'].loc[atomnum], b['dihedral'].loc[atomnum]

    #originally in radians, convert to degrees
    a_dir, b_dir = np.deg2rad(a_di), np.deg2rad(b_di)
    b_cos, b_sin = np.cos(b_dir), np.sin(b_dir)
    a_cos, a_sin = np.cos(a_dir), np.sin(a_dir)
    cos_diff = np.square(b_cos - a_cos)
    sin_diff = np.square(b_sin - a_sin)
    dist = np.sqrt(cos_diff + sin_diff)
    if np.rad2deg(np.arcsin(dist/2.0)*2) <= cutoff:
        return 1
    else:
        #returns 0 if outside radius
        return 0


def createDihedralDarts(internal_mat, dihedral_df, posedart_dict, dart_storage):

    """
    Parameters
    ----------
    internal_mat: Chemcoord.Zmatrix
        list of zmatrices corresponding to the poses used for darting
    dihedral_df: pandas dataframe.
        Output of makeDihedralDifferenceDf(), which contains the dihedral
        differences between different poses.
    posedart_dict: dict
        Dictionary containing all the darting regions to be considered
        where each key corresponds to a pose (`pose_0`, `pose_1`, etc).]
        Each value of the `pose` is another dictionary, whose keys correspond
        to a particular dart (`rotational`, `translational` or dihedral`).
        The values of that dictionary are strings corresponding to the overlap
        of poses with that dart.
    dart_storage: dict
        Dictionary conaining the already defined darts.

    Returns
    -------
    dart_storage, posedart_dict, boolean

    dart_storage: dict
        Dictionary containing the darting regions updated with translational darts
    posedart_dict: dict
        Same as the input posedart_dict except updated with translational darts.
    boolean: bool
        If all the poses are separated with the current darts returns True,
        else if there are still poses that overlap returns False.
    """


    #go through setup phase, checking current darts present and seeing how
    #many poses those darts separate
    if dihedral_df is None:
        for key, pose in iteritems(posedart_dict):
            pose['dihedral'] = None
        return dart_storage, posedart_dict, False
    last_repeat = {}
    unison_dict = {}
    for key, pose in iteritems(posedart_dict):
        if pose['translation'] is None:
            trans_overlap_list = None
        else:
            trans_overlap_list = [set(pose['translation'])]
        if pose['rotation'] is None:
            rot_overlap_list = None
        else:
            rot_overlap_list = [set(pose['rotation'])]
        overlap_list = addSet([trans_overlap_list, rot_overlap_list])
        if overlap_list is None:
            last_repeat[key] = None
        elif len(overlap_list) > 0:
            unison = set.intersection(*overlap_list)
            last_repeat[key] = len(unison)
        else:
            last_repeat[key] = {0}-{0}

    for idx, i in list(zip(dihedral_df.index.tolist(), dihedral_df['atomnum'])):
        #updates posedart_dict with overlaps of poses for each dart
        posedart_copy = compareDihedral(internal_mat, atom_index=i, diff_spread=dihedral_df['diff'].loc[idx]/2.0, posedart_dict=posedart_dict)
        #checks if darts separate all poses from each other if still 0
        dart_check = 0

        #check for duplicates
        for key, pose in iteritems(posedart_copy):
            overlap_list = [set(oi) for oi in list(pose['dihedral'].values()) if len(oi) > 0 ]

            try:
                #if there's no overlaps then this will fail
                unison = set.intersection(*overlap_list)
            except TypeError:
                unison = set([])
            unison_dict[key] = len(unison)

            if len(unison) > 0:
                dart_check += 1
            else:
                pass
                #print('selected internal coordinates separate all poses')
        dboolean = 0
        for key, value in iteritems(unison_dict):
            if last_repeat[key] is None and unison_dict[key] < len(internal_mat)-1:
                dboolean += 1
            try:
                if value < last_repeat[key]:
                    dboolean = dboolean + 1
            except TypeError:
                pass
        if dboolean > 0:
            posedart_dict = copy.deepcopy(posedart_copy)
            dart_storage['dihedral'][i] = dihedral_df['diff'].loc[idx]/2.0
            for key, value in iteritems(unison_dict):
                last_repeat[key] = unison_dict[key]


        #counts all poses and sees if no overlap (-=0)is true for all
        #If so, stop loop
        if dart_check == 0:
            return dart_storage, posedart_dict, True
    return dart_storage, posedart_dict, False

def compareTranslation(trans_mat, trans_spread, posedart_dict, dart_type='translation'):
    """Compares the translational distance of a given translational distance matrix
    and checks which distances are within the trans_spread distance.

    Parameters
    ----------
    trans_mat: np.array
        Array of pairwise comparisions of translational distances between poses.
    trans_spread: float
        Cutoff distance to check if there is translational overlap between
        translational darts.

    posedart_dict: dict
        Same as the input posedart_dict except updated with translational darts.
    dart_type: str
        Uses the given string to be the posedart_dict key to add to

    Returns
    -------
    posedart_copy: dict
        Copy of posedart_dict, but with added translational overlaps.
    """
    posedart_copy = copy.deepcopy(posedart_dict)
    num_poses = np.shape(trans_mat)[0]
    compare_indices = np.triu_indices(num_poses)
    #reset counts for dictionary
    for apose in range(num_poses):
            posedart_copy['pose_'+str(apose)][dart_type] = []

    for pose1, pose2 in zip(compare_indices[0], compare_indices[1]):
        if pose1 != pose2:
            if trans_mat[pose1, pose2] < trans_spread:
                #if less, then the other centroid is within the radius and we need to add
                posedart_copy['pose_'+str(pose1)][dart_type].append(pose2)
                posedart_copy['pose_'+str(pose2)][dart_type].append(pose1)
            else:
                pass


    #iterate over zmat and get original value of pose
    return posedart_copy

def compareRotation(rot_mat, rot_spread, posedart_dict, dart_type='rotation'):
    """
    Compares the rotational distance of a given rotational distance matrix
    and checks which distances are within the trans_spread distance.

    Parameters
    ----------
    trans_mat: np.array
        Array of pairwise comparisions of rotational distances between poses.
    trans_spread: float
        Cutoff distance to check if there is rotational overlap between
        translational darts.

    posedart_dict: dict
        Same as the input posedart_dict except updated with translational darts.
    dart_type: str
        Uses the given string to be the posedart_dict key to add to

    Returns
    -------
    posedart_copy: dict
        Copy of posedart_dict, but with added rotational overlaps.
    """
    posedart_copy = copy.deepcopy(posedart_dict)
    num_poses = np.shape(rot_mat)[0]
    compare_indices = np.triu_indices(num_poses)
    #reset counts for dictionary
    for apose in range(num_poses):
            posedart_copy['pose_'+str(apose)][dart_type] = []

    for pose1, pose2 in zip(compare_indices[0], compare_indices[1]):
        if pose1 != pose2:
            if rot_mat[pose1, pose2] < rot_spread:
                #if less, then the other centroid is within the radius and we need to add
                posedart_copy['pose_'+str(pose1)][dart_type].append(pose2)
                posedart_copy['pose_'+str(pose2)][dart_type].append(pose1)
            else:
                pass


    #iterate over zmat and get original value of pose
    return posedart_copy

def getRotTransMatrices(internal_mat, pos_list, construction_table):
    """
    Creates two arrays comparing the pairwise rotational and translational
    distances between poses.

    Parameters
    ----------
    internal_mat: List of Chemcoord.Zmat objects.
        List of Chemcoord.Zmat objects.
    pos_list:  list of np.arrays
        List of positions of the various ligand poses.
    construction_table: Chemcoord.Zmat.construction_table
        Construction table indicating the build list of atoms to
        create the Z-matrix.

    Returns
    -------
    rot_storage: np.array
        An array that stores the pairwise rotational differences.
    trans_storage: np.array
        An array that stores the pairwise translational differences.
    """
    trans_storage = np.zeros( (len(internal_mat), len(internal_mat)) )
    rot_storage = np.zeros( (len(internal_mat), len(internal_mat)) )
    for zindex in combinations(list(range(len(internal_mat))), 2):
        first_index = zindex[0]
        second_index = zindex[1]
        temp_rot, temp_trans = dartRotTrans(binding_mode_pos=pos_list, internal_zmat=internal_mat,
                                 binding_mode_index=first_index, comparison_index=second_index,
                                 construction_table=construction_table )
        if np.isnan(temp_rot):
        #only be nan if due to rounding error due to no rotation
            rot_storage[zindex[0], zindex[1]] = 0
        else:
            rot_storage[zindex[0], zindex[1]] = temp_rot

        trans_storage[zindex[0], zindex[1]] = temp_trans
        trans_storage = symmetrize(trans_storage)/ 2.0
        rot_storage = symmetrize(rot_storage) / 2.0
    return rot_storage, trans_storage


def findInternalDart(sim_mat, internal_mat, dart_storage):
    """function used to check if a pose from the simulation
    matches a pose from the original poses

    Parameters
    ----------
    sim_mat: Chemcoord Zmatrix object
        The Zmatrix pertaining to the simulation Zmatrix
    internal_mat: Chemcoord Zmatrix object
        The iZmatrix pertaining to a particular pose
    dart_storage: dictionary
        The dictionary containing the darting region information

    """


    dart_list = [0 for p in internal_mat]
    total_counter = sum(len(v) for v in dart_storage.itervalues())
    for index, pose in enumerate(internal_mat):
        dart_counter = 0
        for selection in ['bond', 'angle', 'dihedral' ]:
            if selection == 'dihedral':
                    for atomnum, di in iteritems(dart_storage['dihedral']):
                        inorout = checkDihedralRegion(sim_mat, pose, atomnum, cutoff=di, )
                        dart_counter =+ inorout

        dart_list[index] = dart_counter
    for item in dart_list:
        if item == total_counter:
            print('pose found')

def addSet(set_list):
    """Helper function used to add sets together.
    Used to find if the overlapping regions separate
    all given poses nicely.
    """
    add_return = None
    for i in set_list:
        if add_return is None:
            add_return = i
        elif i != None:
            add_return = add_return + i
    return add_return

def createTranslationDarts(internal_mat, trans_mat, posedart_dict, dart_storage, distance_cutoff=7.5):
    """
    Parameters
    ----------
    internal_mat: Chemcoord.Zmatrix
        list of zmatrices corresponding to the poses used for darting
    trans_mat: nxn np.array
        An nxn array where n is the n poses being used for darting,
        and whose N(i,j) entries are the pairwise translational distances
        between the first atom in the construction table for poses i,j.
    posedart_dict: dict
        Dictionary containing all the darting regions to be considered
        where each key corresponds to a pose (`pose_0`, `pose_1`, etc).]
        Each value of the `pose` is another dictionary, whose keys correspond
        to a particular dart (`rotational`, `translational` or dihedral`).
        The values of that dictionary are strings corresponding to the overlap
        of poses with that dart.

    dart_storage: dict
        Dictionary conaining the already defined darts
    distance_cutoff: float, optional, default=7.5
        Distance cutoff, in angstroms, of the minimum distance to be used to
        define translational darts.
    Returns
    -------
    dart_storage, posedart_dict, boolean

    dart_storage: dict
        Dictionary containing the darting regions updated with translational darts
    posedart_dict: dict
        Same as the input posedart_dict except updated with translational darts.
    boolean: bool
        If all the poses are separated with the current darts returns True,
        else if there are still poses that overlap returns False.
    """

    #go through setup phase, checking current darts present and seeing how
    #many poses those darts separate

    dihedral_present = True
    rotation_present = True
    last_repeat = {}
    unison_dict = {}
    #loop over all poses

    for key, pose in iteritems(posedart_dict):
        if not pose['dihedral']:
            dihedral_present = False
            last_repeat[key] = None

        for key, pose in iteritems(posedart_dict):
            if dihedral_present == True:

                try:
                    di_overlap_list = [set(oi) for oi in list(pose['dihedral'].values()) if len(oi) > 0 ]
                except AttributeError:
                    di_overlap_list = None
                    dihedral_present = False
            else:
                di_overlap_list = None
            if pose['rotation'] is None:
                trans_overlap_list = None
                rotation_present = False

            else:
                trans_overlap_list = [set(pose['translation'])]

            overlap_list = addSet([di_overlap_list, trans_overlap_list])
            if overlap_list is None:
                last_repeat[key] = None
            elif len(overlap_list) > 0:
                unison = set.intersection(*overlap_list)
                last_repeat[key] = len(unison)
            else:
                last_repeat[key] = {0}-{0}
    trans_indices = np.triu_indices(len(internal_mat))
    trans_list = sorted([trans_mat[i,j] for i,j in zip(trans_indices[0], trans_indices[1])], reverse=True)

    #this removes distances less than 1.0 from being used in finding a dart
    #change if really small translational darts are desired
    #without this then dart sizes of 0 can be accepted, which don't make sense

    trans_list = [i for i in trans_list if i > distance_cutoff]
    if len(trans_list) > 0:
        for trans_diff in trans_list:
            #updates posedart_dict with overlaps of poses for each dart
            posedart_copy = compareTranslation(trans_mat=trans_mat, trans_spread=trans_diff, posedart_dict=posedart_dict)
            dart_check = 0

            #check for duplicates
            for key, pose in iteritems(posedart_copy):
                if dihedral_present == True:
                    di_overlap_list = [set(oi) for oi in list(pose['dihedral'].values()) if len(oi) > 0 ]
                    di_overlap_list = None

                else:
                    di_overlap_list = None
                if rotation_present == True:
                    rot_overlap_list = [set(pose['rotation'])]
                else:
                    rot_overlap_list = None
                trans_overlap_list = [set(pose['translation'])]
                overlap_list = addSet([di_overlap_list, trans_overlap_list, rot_overlap_list])
                try:
                    #if there's no overlaps then this will fail
                    unison = set.intersection(*overlap_list)
                except TypeError:
                    unison = set([])

                unison_dict[key] = len(unison)
                if len(unison) > 0 and last_repeat[key] is not None:
                    dart_check += 1
                    last_repeat[key] = len(unison) + 1
                elif last_repeat[key] is None and len(unison) <= len(internal_mat)-1 and len(unison) > 0:
                    dart_check += 1
                    posedart_dict = copy.deepcopy(posedart_copy)
                    #TODO decide if it should be in angles or radians
                    dart_storage['translation'] = [trans_diff]
                    last_repeat[key] = len(unison)

                elif len(unison) == 0:
                    pass
                    #print('selected internal coordinates separate all poses')
                else:
                    pass

            dboolean = 0
            for key, value in iteritems(unison_dict):
                if value < last_repeat[key]:
                    dboolean = dboolean + 1
            if dboolean > 0:
                dart_storage['translation'] = [trans_diff]
                posedart_dict = copy.deepcopy(posedart_copy)
            #if adding a dart doesn't reduce amount of overlap, don't keep that dart
            else:
                pass
            #counts all poses and sees if no overlap (-=0)is true for all
            #If so, stop loop
            if dart_check == 0:
                print('all separated, good to go')
                dart_storage['translation'] = [trans_diff]

                return dart_storage, posedart_dict, True
    else:
        for key, pose in iteritems(posedart_dict):
            pose['translation'] = None


    return dart_storage, posedart_dict, False

def createRotationDarts(internal_mat, rot_mat, posedart_dict, dart_storage, rotation_cutoff=29.0):
    """Finds rotational darts that increase separation between the darts, accounting
    for the darts already present in dart_storage

    Parameters
    ----------
    internal_mat: Chemcoord.Zmatrix
        list of zmatrices corresponding to the poses used for darting
    rot_mat: nxn np.array
        An nxn array where n is the n poses being used for darting,
        and whose N(i,j) entries are the pairwise rotational distances
        between the first atom in the construction table for poses i,j.
    posedart_dict: dict
        Dictionary containing all the darting regions to be considered
        where each key corresponds to a pose (`pose_0`, `pose_1`, etc).]
        Each value of the `pose` is another dictionary, whose keys correspond
        to a particular dart (`rotational`, `translational` or dihedral`).
        The values of that dictionary are strings corresponding to the overlap
        of poses with that dart.

    dart_storage: dict
        Dictionary conaining the already defined darts
    distance_cutoff: float, optional, default=7.5
        Distance cutoff, in angstroms, of the minimum distance to be used to
        define translational darts.
    Returns
    -------
    dart_storage, posedart_dict, boolean

    dart_storage: dict
        Dictionary containing the darting regions updated with rotational darts
    posedart_dict: dict
        Same as the input posedart_dict except updated with rotational darts.
    boolean: bool
        If all the poses are separated with the current darts returns True,
        else if there are still poses that overlap returns False.

    """
    dihedral_present = True
    translation_present = True
    last_repeat = {}
    unison_dict = {}
    di_overlap_list = None
    trans_overlap_list = None
    for key, pose in iteritems(posedart_dict):
        if not pose['dihedral']:
            dihedral_present = False
            last_repeat[key] = None

        for key, pose in iteritems(posedart_dict):
            if dihedral_present == True:

                try:
                    di_overlap_list = [set(oi) for oi in list(pose['dihedral'].values()) if len(oi) > 0 ]
                except AttributeError:
                    di_overlap_list = None
                    dihedral_present = False
            if pose['translation'] is None:
                trans_overlap_list = None
                translation_present = False

            else:
                trans_overlap_list = [set(pose['translation'])]

            overlap_list = addSet([di_overlap_list, trans_overlap_list])
            if overlap_list is None:
                last_repeat[key] = None
            elif len(overlap_list) > 0:
                unison = set.intersection(*overlap_list)
                last_repeat[key] = len(unison)
            else:
                last_repeat[key] = {0}-{0}

    #need to know how many regions are separated to see if adding translational darts improve things
    rot_indices = np.triu_indices(len(internal_mat))
    rot_list = sorted([rot_mat[i,j] for i,j in zip(rot_indices[0], rot_indices[1])], reverse=True)
    #this removes distances less than 0.1 from being used in finding a dart
    #change if really small translational darts are desired
    #without this then dart sizes of 0 can be accepted, which don't make sense
    rot_list = [i for i in rot_list if i >= rotation_cutoff]
    for rot_diff in rot_list:

        #updates posedart_dict with overlaps of poses for each dart
        posedart_copy = compareRotation(rot_mat=rot_mat,rot_spread=rot_diff, posedart_dict=posedart_dict)

        dart_check = 0

        #check for duplicates
        for key, pose in iteritems(posedart_copy):
            if dihedral_present == True:
                di_overlap_list = [set(oi) for oi in list(pose['dihedral'].values()) if len(oi) > 0 ]
            else:
                di_overlap_list = None
            if translation_present == True:
                trans_overlap_list = [set(pose['translation'])]
            else:
                trans_overlap_list = None
            rot_overlap_list = [set(pose['rotation'])]

            overlap_list = addSet([di_overlap_list, trans_overlap_list, rot_overlap_list])

            try:
                #if there's no overlaps then this will fail
                unison = set.intersection(*overlap_list)

            except TypeError:
                unison = set([])
            unison_dict[key] = len(unison)
            #check number of overlaps
            #if number of overlaps is > 0 note that
            #if no overlaps were present because this is the first check
            #then note that and update last overlaps and posedart dict
            if len(unison) > 0 and last_repeat[key] is not None:
                dart_check += 1
                last_repeat[key] = len(unison)

            elif last_repeat[key] is None and len(unison) < len(internal_mat)-1 and len(unison) > 0:
                dart_check += 1
                posedart_dict = copy.deepcopy(posedart_copy)
                #TODO decide if it should be in degrees or radians
                dart_storage['rotation'] = [rot_diff]
                last_repeat[key] = len(unison)

            elif len(unison) == 0:
                pass
                #print('selected internal coordinates separate all poses')
        #check if adding additional regions removes overlaps

        dboolean = 0
        for key, value in iteritems(unison_dict):
            try:
                if value < last_repeat[key]:
                    dboolean = dboolean + 1
            #type error if last_repeat is a set or None
            except TypeError:
                pass

        if dboolean > 0:
            dart_storage['rotation'] = [rot_diff]
            posedart_dict = copy.deepcopy(posedart_copy)
        #if adding a dart doesn't reduce amount of overlap, don't keep that dart
        else:
            pass
        #counts all poses and sees if no overlap (-=0)is true for all
        #If so, stop loop
        if dart_check == 0:
            dart_storage['rotation'] = [rot_diff]
            return dart_storage, posedart_dict, True
    return dart_storage, posedart_dict, False



def makeDartDictOld(internal_mat, pos_list, construction_table, dihedral_cutoff=0.5, distance_cutoff=7.5):
    """
    Makes the dictionary of darting regions used as the basis for darting,
    attempting to make a set of dihedral darts that separate the given poses.
    If all the poses are not separated at this point, translational darts are made,
    and if the remaining poses are not separated, then rotational darts are created.


    Parameters
    ----------
    internal_mat: list of Chemcoord.Zmatrixs
        list of zmats that are to be separated using this function
    pos_list: list
        list of ligand positions
    construction_table: Chemcoord construction_table object
        The construction table used to make the internal_zmat.
    dihedral_cutoff: float
        Minimum cutoff to use for the dihedrals.

    Returns
    -------
    dart_storage: dict
        Dict containing the darts associated with `rotation`, `translation` and `dihedral`
        keys that refer to the size of the given dart, if not empty
    """
    #make diff dict
    dihedral_df = makeDihedralDifferenceDf(internal_mat, dihedral_cutoff=dihedral_cutoff)
    posedart_dict = {'pose_'+str(posnum):{}  for posnum, value in enumerate(internal_mat)}

    for key, value in iteritems(posedart_dict):
        value['bond'] = {}
        value['angle'] = {}
        value['dihedral'] = {}
        value['rotation'] = None
        value['translation'] = None
        try:
            for atomnum in dihedral_df['atomnum']:
                value['dihedral'][atomnum] = []
        except TypeError:
            pass


    dart_storage = {'bond':{}, 'angle':{}, 'dihedral':{}, 'translation':[], 'rotation':[]}
    dart_storage, posedart_dict, dart_boolean = createDihedralDarts(internal_mat, dihedral_df, posedart_dict, dart_storage)
    print(createDihedralDarts(internal_mat, dihedral_df, posedart_dict, dart_storage))
    #if dart_boolean is false, we need to continue looking thru rot/trans for better separation
    if dart_boolean == False:
        rot_mat, trans_mat = getRotTransMatrices(internal_mat, pos_list, construction_table)
        dart_storage, posedart_dict, dart_boolean = createTranslationDarts(internal_mat, trans_mat, posedart_dict, dart_storage, distance_cutoff=distance_cutoff)

        if dart_boolean == False:
            dart_storage, posedart_dict, dart_boolean = createRotationDarts(internal_mat, rot_mat, posedart_dict, dart_storage)

        #check translation
        pass
    for key in ['rotation', 'translation']:
        if len(dart_storage[key]) > 0:
            dart_storage[key][0] = dart_storage[key][0] - dart_storage[key][0] / 10.0
    return dart_storage

    #get rotation/translation diff matrix
    #start with dihedral, loop over diffs and check overlap

def makeDartDict(internal_mat, pos_list, construction_table, dihedral_cutoff=0.5, distance_cutoff=5.5, rotation_cutoff=29.0, dart_buffer=0.9, order=['translation', 'dihedral',  'rotation']):
    """
    Makes the dictionary of darting regions used as the basis for darting,
    attempting to make a set of dihedral darts that separate the given poses.
    If all the poses are not separated at this point, translational darts are made,
    and if the remaining poses are not separated, then rotational darts are created.


    Parameters
    ----------
    internal_mat: list of Chemcoord.Zmatrixs
        list of zmats that are to be separated using this function
    pos_list: list
        list of ligand positions
    construction_table: Chemcoord construction_table object
        The construction table used to make the internal_zmat.
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
    #make diff dict
    def createDarts(function_type, internal_mat, dihedral_df, trans_mat, rot_mat, distance_cutoff, posedart_dict, dart_storage):
        if function_type == 'translation':
            return createTranslationDarts(internal_mat, trans_mat, posedart_dict, dart_storage, distance_cutoff=distance_cutoff)
        elif function_type == 'rotation':
            return createRotationDarts(internal_mat, rot_mat, posedart_dict, dart_storage, rotation_cutoff=rotation_cutoff)
        elif function_type == 'dihedral':
            return createDihedralDarts(internal_mat, dihedral_df, posedart_dict, dart_storage)
    def checkOverlap(posedart_dict):
        for posekey, posevalue in iteritems(posedart_dict):
            set_overlap_output = {}
            set_overlap = []
            set_overlap.append
            for key, value in iteritems(posevalue):
                if key == 'dihedral' and value:
                    print('value', value)
                    for key1, value1 in iteritems(value):
                        if value1:
                            set_overlap.append(value1)
                elif isinstance(value, list):
                    set_overlap.append(value)
            set_overlap = [set(i) for i in set_overlap]
            print('set_overlap', set.intersection(*set_overlap))
            set_overlap_output[posekey] = set_overlap
            for pose in set_overlap_output.keys():
                print('pose', pose, posedart_dict[pose])
        print('set_overlap', set_overlap)


    #dihedral_df = makeDihedralDifferenceDf(internal_mat, dihedral_cutoff=dihedral_cutoff)
    dihedral_df = None

    posedart_dict = {'pose_'+str(posnum):{}  for posnum, value in enumerate(internal_mat)}

    for key, value in iteritems(posedart_dict):
        value['bond'] = {}
        value['angle'] = {}
        value['dihedral'] = {}
        value['rotation'] = None
        value['translation'] = None
        try:
            for atomnum in dihedral_df['atomnum']:
                value['dihedral'][atomnum] = []
        except TypeError:
            pass


    dart_storage = {'bond':{}, 'angle':{}, 'dihedral':{}, 'translation':[], 'rotation':[]}
    dart_boolean = False
    rot_mat, trans_mat = getRotTransMatrices(internal_mat, pos_list, construction_table)

    for darttype in order:
        if not dart_boolean:
            if darttype == 'dihedral':
                dihedral_df = makeDihedralDifferenceDf(internal_mat, dihedral_cutoff=dihedral_cutoff)
                print(dihedral_df)
            dart_storage, posedart_dict, dart_boolean = createDarts(darttype, internal_mat, dihedral_df, trans_mat, rot_mat, distance_cutoff, posedart_dict, dart_storage)
            print('dart_storage after ', darttype)
            print(dart_storage)
    if not dart_boolean:
        checkOverlap(posedart_dict)
        raise ValueError('Current settings do not separate out all poses. Current separation', posedart_dict, dart_storage)
    #dart_storage, posedart_dict, dart_boolean = createDihedralDarts(internal_mat, dihedral_df, posedart_dict, dart_storage)
    for key in ['rotation', 'translation']:
        if len(dart_storage[key]) > 0:
            #dart_storage[key][0] = dart_storage[key][0] - dart_storage[key][0] / 10.0
            dart_storage[key][0] = dart_storage[key][0] * dart_buffer

    return dart_storage



def checkDart(internal_mat, current_pos, current_zmat, pos_list, construction_table, dart_storage):
    """Checks whether a given position/zmatrix (given by current_pos and current_zmat)
    fall within the dart regions defined by dart_storage.
    If any do, then this returns the labels corresponding to those darts.

    Parameters
    ----------
    internal_mat: list of Chemcoord.Zmatrixs
        list of zmats that are to be separated using this function
    current_pos: np.array
        The current position (in cartesian space) of the ligand.
    current_zmat: Chemcord.Zmatrix
        The current Zmatrix representation of the ligand.
    pos_list: list
        list of ligand positions
    construction_table: Chemcoord construction_table object
        The construction table used to make the internal_zmat.
    dart_storage: dict
        Dict containing the darts associated with `rotation`, `translation` and `dihedral`
        keys that refer to the size of the given dart, if not empty.
    Returns:
    set_output: list
        List of overlapping dart regions. Returns an empty list of there's no overlap.

    """
    def createTranslationDarts(internal_mat, trans_mat, dart_storage):
        num_poses = np.shape(trans_mat)[0]
        trans_list = [trans_mat[0,j] for j in range(1, num_poses)]
        #this removes distances less than 1.0 from being used in finding a dart
        #change if really small translational darts are desired
        #without this then dart sizes of 0 can be accepted, which don't make sense
        if len(dart_storage['translation']) > 0:
            trans_cutoff = dart_storage['translation'][0]
            trans_list = [j for j,i in enumerate(trans_list) if i < trans_cutoff]
            return trans_list
        else:
            return None

    def compareRotation(rot_mat, internal_mat, dart_storage):
        if len(dart_storage['rotation']) > 0:
            rot_cutoff = dart_storage['rotation'][0]
            num_poses = np.shape(rot_mat)[0]
            rot_list = [rot_mat[0,j] for j in range(1, num_poses)]
            rot_list = [j for j,i in enumerate(rot_list) if i < rot_cutoff]
            print('rot_cutoff', rot_cutoff, rot_list)
            return rot_list
        else:
            return None

    def compareDihedral(current_internal, internal_mat, dart_storage, inRadians=True):
        #reset counts for dictionary
        #iterate over zmat and get original value of pose

        dihedral_output = {}
        dihedral_atoms = list(dart_storage['dihedral'].keys())

        if len(dihedral_atoms) > 0:
            for atom_index in dihedral_atoms:
                dihedral_output[atom_index] = []
                current_dihedral = current_internal['dihedral'].loc[atom_index]

                for posenum, zmat in enumerate(internal_mat):
                    comparison = zmat['dihedral'].loc[atom_index]
                    dihedral_diff = abs(current_dihedral - comparison)


                    if dihedral_diff <= np.rad2deg(dart_storage['dihedral'][atom_index]):
                        dihedral_output[atom_index].append(posenum)



            dihedral_list = [set(i) for i in dihedral_output.values()]
            dihedral_list = list(set.intersection(*dihedral_list))
            if len(dihedral_list) > 0:
                return dihedral_list
            else:
                return None

    combo_list = [current_pos] + pos_list
    combo_zmat = [current_zmat] + internal_mat

    rot_mat, trans_mat = getRotTransMatrices(combo_zmat, combo_list, construction_table)
    trans_list = createTranslationDarts(combo_zmat, trans_mat, dart_storage)

    rot_list = compareRotation(rot_mat, combo_zmat, dart_storage)
    dihedral_output = compareDihedral(current_zmat, internal_mat, dart_storage)

    combined_comparison = [set(i) for i in [dihedral_output, rot_list, trans_list] if i is not None]
    try:
        set_output = list(set.intersection(*combined_comparison))
    except TypeError:
        set_output = []
    return set_output





