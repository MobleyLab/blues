import numpy as np

def adjust_angle(a,b, radians, maintain_magnitude=True):
    '''
    Adjusts angle a so that the angle between a and b
    matches the specified radians. The change in a
    occurs in the plane of a and b.

    Arguments
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
    '''
    #use the projection of a onto b to get
    #parallel and perpendicular components of a in terms of b
    para = b*(a.dot(b)/b.dot(b))
    perp = a - para
    mag_para = np.linalg.norm(para)
    mag_perp = np.linalg.norm(perp)
    #determine how much to scale the perpendicular
    #component of a to adjust the angle
    scale = np.tan(float(radians))*mag_para/mag_perp
    c = para + perp*scale
    #if True, adjusts vector c to maintain same
    #total magnitude as the original (a).
    if maintain_magnitude == True:
        mag_a = np.linalg.norm(a)
        c = c / np.linalg.norm(c) * mag_a
    return c

def calc_rotation_matrix(vec_ref, vec_target):
    '''calculate the rotation matrix that will rotate vec_ref to vec_target
        Note: will fail if vectors are in exact opposite directions
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
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                    ])
    I = np.identity(3)
    #actually calculate rotation matrix
    R = I + vx + vx.dot(vx)*(1/(1+vec_cos))
    return R

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
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

def getRotTrans(ares, bres):
    '''
    Get rotation and translation of rigid pose

    Arguments
    ---------
    apos: nx3 np.array
        simulation positions
    bpos: nx3 np.array
        comparison positions
    residueList
    '''
    rot, trans, centa, centb, centroid_difference = rigid_transform_3D(ares, bres)
    return rot, centroid_difference

def old_getRotTrans(apos, bpos, residueList=None):
    '''
    Get rotation and translation of rigid pose

    Arguments
    ---------
    apos: nx3 np.array
        simulation positions
    bpos: nx3 np.array
        comparison positions
    residueList
    '''
    if type(residueList) == type(None):
        residueList = self.residueList
    a_new = apos[:]
    a_res = np.zeros((len(residueList),3))
    b_res = np.zeros((len(residueList),3))
    for index, i in enumerate(residueList):
        a_res[index] = apos[i]
        b_res[index] = bpos[i]
    rot, trans, centa, centb, centroid_difference = rigid_transform_3D(a_res, b_res)
    return rot, centroid_difference
