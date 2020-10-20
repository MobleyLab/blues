import numpy as np

def rigid_transform_3D(A, B, center):
    """
    Finds the optimal rotation matrix to overlay
    A onto B, assuming that the centroid is
    located at the position of the `center` index.

    Parameters
    ----------
    A: 3x3 np.array
        simulation positions
    B: 3x3 np.array
        comparison positions
    center: int
        Index of the center atom (either 1,2 or 3)
    Returns
    R: 3x3 np.array
        Rotation matrix to overlay A onto B
    t: float
        The distance between the centroid of A and B
    centroid_A: 1x3 np.array
        The position of centroid_A
    centroid_B: 1x3np.array
        The position of centroid_B
    centroid_difference: 1x3 np.array
        Vector corresponding to the difference
        between the centroid of A and B

    """
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    centroid_A = A[center]
    centroid_B = B[center]
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    centroid_difference = centroid_B - centroid_A
    t = -np.dot(R, centroid_A.T) + centroid_B.T

    return R, t, centroid_A, centroid_B, centroid_difference


def getRotTrans(apos, bpos, center):
    """
    Get rotation and translation of rigid pose for a three body
    system.

    Parameters
    ---------
    apos: 3x3 np.array
        simulation positions
    bpos: 3x3 np.array
        comparison positions
    center:
        The index of the center atom (either 1, 2 or 3)
    Returns
    -------
    rot: 3x3 np.array
        Rotation matrix for overlapping a onto b
    centroid_difference, float
        Difference between the centroids.
    """
    rot, trans, centa, centb, centroid_difference = rigid_transform_3D(apos, bpos, center)
    return rot, centroid_difference


def adjust_angle(a,b, radians, maintain_magnitude=True):
    """
    Adjusts angle a so that the angle between a and b
    matches the specified radians. The change in a
    occurs in the plane of a and b.

    Parameters
    ---------
    a, 1x3 np.array:
        The vector to be adjusted to change the angle
    b, 1x3 np.array:
        The vector to act as a base and creates the
        angle with a.
    radians, float:
        Radian you want to adjust the angle to.
    maintain_magnitude, boolean:
        If True adjusts the returned vector keeps
        the same magnitude as a. If false returns
        a normalized vector
    Returns
    -------
    c, 1x3 np.array:
        Vector adjusted so that the angle made
        from c and b is the value specifed by radians.
    """
    #use the projection of a onto b to get
    #parallel and perpendicular components of a in terms of b
    para = b*(a.dot(b)/b.dot(b))
    perp = a - para
    mag_para = np.linalg.norm(para)
    mag_perp = np.linalg.norm(perp)
    #determine how much to scale the perpendicular
    #component of a to adjust the angle
    scale = np.tan(float(radians))*mag_para/mag_perp
    c = para - perp*scale
    #if True, adjusts vector c to maintain same
    #total magnitude as the original (a).
    if maintain_magnitude == True:
        mag_a = np.linalg.norm(a)
        c = c / np.linalg.norm(c) * mag_a
    else:
        c = c / np.linalg.norm(c)
    return c


def dartRotTrans(binding_mode_pos, internal_zmat, binding_mode_index, comparison_index, construction_table, bond_compare=True, rigid_move=False):
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

    #choose a random binding pose
    #change symmetric atoms
    #TODO: change to take in xyz format already
    #get matching binding mode pose and get rotation/translation to that pose
    #TODO decide on making a copy or always point to same object


    #find translation differences in positions of first two atoms to reference structure
    #find the appropriate rotation to transform the structure back
    #repeat for second bond
    def findCentralAngle(buildlist):
        connection_list = []
        index_list = [0,1,2]
        for i in buildlist.index.values[:3]:
            connection_list.append(buildlist['b'][i])
        #count the number of bonds to the first buildatom
        counts = connection_list.count(construction_table.index.values[0])
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




    vector_list = findCentralAngle(construction_table)
    #find translation differences in positions of first two atoms to reference structure
    #find the appropriate rotation to transform the structure back
    #repeat for second bond
    #get first 3 new moldart positions, apply same series of rotation/translations
    build_indices = construction_table.index.values[:3]
    sim_three = binding_mode_pos[binding_mode_index][build_indices]
    ref_three = binding_mode_pos[comparison_index][build_indices]
    dart_three = binding_mode_pos[comparison_index][build_indices]
    #dart_ref = binding_mode_pos[comparison_index][build_indices]

    change_three = np.copy(sim_three)
    vec1_sim = sim_three[vector_list[0][0]] - sim_three[vector_list[0][1]]
    vec2_sim = sim_three[vector_list[1][0]] - sim_three[vector_list[1][1]]

    #calculate rotation from ref pos to sim pos

    #change angle of one vector
    ###edits

    ###
    ref_angle = internal_zmat[comparison_index]._frame['angle'][construction_table.index.values[2]]

    ad_vec = adjust_angle(vec1_sim, vec2_sim, np.radians(ref_angle), maintain_magnitude=False)
    ad_vec = ad_vec / np.linalg.norm(ad_vec) * internal_zmat[binding_mode_index]._frame['bond'][construction_table.index.values[2]]/10.
    #apply changed vector to center coordinate to get new position of first particle
    rot_mat, centroid = getRotTrans(change_three, ref_three, center=vector_list[0][1])

    nvec2_sim = vec2_sim / np.linalg.norm(vec2_sim) * internal_zmat[binding_mode_index]._frame['bond'][construction_table.index.values[2]]/10.
    rot_angle = np.rad2deg(np.arccos(( (np.trace(rot_mat)-1) )/2.0 ))
    change_three[vector_list[0][0]] = sim_three[vector_list[0][1]] + ad_vec
    change_three[vector_list[1][0]] = sim_three[vector_list[0][1]] + nvec2_sim

    centroid_orig = dart_three[vector_list[0][1]]

    dart_three = (dart_three -  np.tile(centroid_orig, (3,1))).dot(rot_mat) + np.tile(centroid_orig, (3,1)) - np.tile(centroid, (3,1))

    rot_angle = np.rad2deg(np.arccos(( (np.trace(rot_mat)-1) )/2.0 ) )
    trans_dist = np.linalg.norm(centroid)
    return rot_angle, trans_dist
