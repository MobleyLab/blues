import numpy as np
from future.utils import iteritems
from itertools import combinations
import pandas as pd
from blues.icdart.rottransedit import dartRotTrans
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
    a_di, b_di = a['dihedral'], b['dihedral']
    a_dir, b_dir = np.deg2rad(a_di), np.deg2rad(b_di)
    b_cos, b_sin = np.cos(b_dir), np.sin(b_dir)
    a_cos, a_sin = np.cos(a_dir), np.sin(a_dir)
    cos_diff = np.square(b_cos - a_cos)
    sin_diff = np.square(b_sin - a_sin)
    dist = np.sqrt(cos_diff + sin_diff)
    return dist

def makeStorageFrame(dataframe, num_poses):
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
    if dihedral_cutoff < 0:
        raise ValueError('Negative dihedral_cutoff distance does not make sense. Please specify a positive cutoff.')
    diff_dict = {}
    output = makeStorageFrame(internal_mat[0], len(internal_mat))
    for index, zmat in enumerate(internal_mat):
        diff_dict[index] = copy.deepcopy(output)
    for zmat in combinations(internal_mat, 2):
        dist_series = dihedralDifference(zmat[0], zmat[1])
        first_index = internal_mat.index(zmat[0])
        second_index = internal_mat.index(zmat[1])
        sort_index = diff_dict[first_index].index
        #joining is strange, to make it work properly sort by ascending index first
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
                if df['atom'].iat[i] != 'H':
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
            #temp_frame = pd.DataFrame(data=[key, di], columns=['atomnum', 'diff'])
#            temp_frame = pd.DataFrame(data={'atomnum':key, 'diff':di}, index=[entry_counter])
            temp_frame = pd.DataFrame(data={'atomnum':key, 'diff':di}, index=[entry_counter])
            entry_counter = entry_counter +1
            if ndf is None:
                ndf = temp_frame.copy()
            else:
                ndf = ndf.append(temp_frame)
            #temp_frame = pd.DataFrame(data=[key, di], columns=['atomnum', 'diff'])

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
        comparison = zmat['dihedral'].iat[atom_index]

        for other_posenum, other_zmat in enumerate(internal_mat):
            if posenum != other_posenum:
                other_comparison = other_zmat['dihedral'].iat[atom_index]
                result = compare_dihedral_edit(comparison, other_comparison, cutoff=diff_spread)
                if result == 1:
                    posedart_copy['pose_'+str(posenum)]['dihedral'][atom_index].append(other_posenum)
            else:
                pass
    return posedart_copy

def fcompare_dihedral(a, b, atomnum, cutoff=80.0, construction_table=None):
    a_di, b_di = a['dihedral'].iat[atomnum], b['dihedral'].iat[atomnum]

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
        #returns 0 if within radius
        return 0


def createDihedralDarts(internal_mat, dihedral_df, posedart_dict, dart_storage):
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
        posedart_copy = compareDihedral(internal_mat, atom_index=i, diff_spread=dihedral_df['diff'].iat[idx]/2.0, posedart_dict=posedart_dict)
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
                print('selected internal coordinates separate all poses')
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
            dart_storage['dihedral'][i] = dihedral_df['diff'].iat[idx]/2.0
            for key, value in iteritems(unison_dict):
                last_repeat[key] = unison_dict[key]


        #counts all poses and sees if no overlap (-=0)is true for all
        #If so, stop loop
        if dart_check == 0:
            return dart_storage, posedart_dict, True
    return dart_storage, posedart_dict, False

def compareTranslation(trans_mat, trans_spread, posedart_dict, dart_type='translation'):
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
    matches a pose from the original poses"""
    dart_list = [0 for p in internal_mat]
    total_counter = sum(len(v) for v in dart_storage.itervalues())
    for index, pose in enumerate(internal_mat):
        dart_counter = 0
        for selection in ['bond', 'angle', 'dihedral' ]:
            if selection == 'dihedral':
                    for atomnum, di in iteritems(dart_storage['dihedral']):
                        inorout = fcompare_dihedral(sim_mat, pose, atomnum, cutoff=di, )
                        dart_counter =+ inorout

        dart_list[index] = dart_counter
    for item in dart_list:
        if item == total_counter:
            print('pose found')

def addSet(set_list):
    add_return = None
    for i in set_list:
        if add_return is None:
            add_return = i
        elif i != None:
            add_return = add_return + i
    return add_return

def createTranslationDarts(internal_mat, trans_mat, posedart_dict, dart_storage):
    #need to know how many regions are separated to see if adding translational darts improve things
    dihedral_present = True
    #rotation_present = True
    last_repeat = {}
    unison_dict = {}
    for key, pose in iteritems(posedart_dict):
        #trans_overlap_list = [set(pose['translation'])]
        #rot_overlap_list = [set(pose['rotation'])]
        #overlap_list = di_overlap_list+trans_overlap_list+rot_overlap_list
        try:
            di_overlap_list = [set(oi) for oi in list(pose['dihedral'].values()) if len(oi) > 0 ]
            #if there's no overlaps then this will fail
            if len(di_overlap_list) > 0:
                unison = set.intersection(*di_overlap_list)
            else:
                last_repeat[key] = 0
        except AttributeError:
            unison = set([])
            last_repeat[key] = None
            dihedral_present = False

    trans_indices = np.triu_indices(len(internal_mat))
    trans_list = sorted([trans_mat[i,j] for i,j in zip(trans_indices[0], trans_indices[1])], reverse=True)

    #this removes distances less than 1.0 from being used in finding a dart
    #change if really small translational darts are desired
    #without this then dart sizes of 0 can be accepted, which don't make sense
    trans_list = [i for i in trans_list if i > 1.0]
    if len(trans_list) > 0:
        for trans_diff in trans_list:
            #updates posedart_dict with overlaps of poses for each dart
            posedart_copy = compareTranslation(trans_mat=trans_mat, trans_spread=trans_diff, posedart_dict=posedart_dict)
            dart_check = 0

            #check for duplicates
            for key, pose in iteritems(posedart_copy):
                if dihedral_present == True:
                    di_overlap_list = [set(oi) for oi in list(pose['dihedral'].values()) if len(oi) > 0 ]
                else:
                    di_overlap_list = None
                trans_overlap_list = [set(pose['translation'])]
                overlap_list = addSet([di_overlap_list, trans_overlap_list])

                try:
                    #if there's no overlaps then this will fail
                    unison = set.intersection(*overlap_list)
                except TypeError:
                    unison = set([])

                unison_dict[key] = len(unison)
                if len(unison) > 0 and last_repeat[key] is not None:
                    dart_check += 1
                    last_repeat[key] = len(unison)
                elif last_repeat[key] is None and len(unison) < len(internal_mat)-1 and len(unison) > 0:
                    dart_check += 1
                    posedart_dict = copy.deepcopy(posedart_copy)
                    #TODO decide if it should be in angles or radians
                    dart_storage['translation'] = [trans_diff]
                    last_repeat[key] = len(unison)

                elif len(unison) == 0:
                    print('selected internal coordinates separate all poses')
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

def createRotationDarts(internal_mat, rot_mat, posedart_dict, dart_storage):
    dihedral_present = True
    translation_present = True
    last_repeat = {}
    unison_dict = {}
    for key, pose in iteritems(posedart_dict):
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
    rot_list = [i for i in rot_list if i > 0.1]

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
                #TODO decide if it should be in angles or radians
                dart_storage['rotation'] = [rot_diff]
                last_repeat[key] = len(unison)

            elif len(unison) == 0:
                print('selected internal coordinates separate all poses')
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



def makeDartDict(internal_mat, pos_list, construction_table, dihedral_cutoff=0.5):
    """
    internal_mat: list of zmats
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
    #if dart_boolean is false, we need to continue looking thru rot/trans for better separation
    if dart_boolean == False:
        rot_mat, trans_mat = getRotTransMatrices(internal_mat, pos_list, construction_table)
        dart_storage, posedart_dict, dart_boolean = createRotationDarts(internal_mat, rot_mat, posedart_dict, dart_storage)
        if dart_boolean == False:
            dart_storage, posedart_dict, dart_boolean = createTranslationDarts(internal_mat, trans_mat, posedart_dict, dart_storage)
        #check translation
        pass
    for key in ['rotation', 'translation']:
        if len(dart_storage[key]) > 0:
            dart_storage[key][0] = dart_storage[key][0] - dart_storage[key][0] / 10.0
    return dart_storage

    #get rotation/translation diff matrix
    #start with dihedral, loop over diffs and check overlap

def getNumDarts(dart_storage):
    dart_counter = 0
    for value in dart_storage.values():
        if isinstance(value, list):
            if len(value) > 0:
                    dart_counter += 1
        if isinstance(value, dict):
            if len(value.keys()) > 0:
                for valuevalue in value.values():
                    dart_counter += 1
    return dart_counter


def checkDart(internal_mat, current_pos, current_zmat, pos_list, construction_table, dart_storage):

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
            print('rot_mat', rot_mat)
            print('rot_list', rot_list)

            return rot_list
        else:
            return None

    def compareDihedral(current_internal, internal_mat, dart_storage, inRadians=True):
        #reset counts for dictionary
        #iterate over zmat and get original value of pose

        dihedral_output = {}
        #num_poses = len(internal_mat)
        dihedral_atoms = list(dart_storage['dihedral'].keys())
        if len(dihedral_atoms) > 0:
            for atom_index in dihedral_atoms:
                dihedral_output[atom_index] = []
                current_dihedral = current_internal['dihedral'].iat[atom_index]

                for posenum, zmat in enumerate(internal_mat):
                    comparison = zmat['dihedral'].iat[atom_index]
                    dihedral_diff = abs(current_dihedral - comparison)


                    if dihedral_diff <= np.rad2deg(dart_storage['dihedral'][atom_index]):
                        dihedral_output[atom_index].append(posenum)


            dihedral_list = []
            for entry in dihedral_output.values():
                dihedral_list.append(*entry)
            return dihedral_list
        else:
            return None
    combo_list = [current_pos] + pos_list
    combo_zmat = [current_zmat] + internal_mat

    rot_mat, trans_mat = getRotTransMatrices(combo_zmat, combo_list, construction_table)
    trans_list = createTranslationDarts(combo_zmat, trans_mat, dart_storage)

    rot_list = compareRotation(rot_mat, combo_zmat, dart_storage)
    dihedral_output = compareDihedral(current_zmat, internal_mat, dart_storage)

    set_output = (addSet([dihedral_output, rot_list, trans_list]))
    #if set_output is not None and len(set_output) ==
    return set_output





