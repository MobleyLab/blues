import numpy as np
from math import sqrt
from simtk import unit

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    print('centroid_A', centroid_A)
    print('centroid_B', centroid_B)


    # centre the points
    print('debug tile', np.tile(centroid_A, (N, 1)))
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
#    H = transpose(AA) * BB
    H = np.dot(np.transpose(AA), BB)
    print('H', H)


    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       #print("Reflection detected")
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)
    centroid_difference = centroid_B - centroid_A
    t = -np.dot(R, centroid_A.T) + centroid_B.T

    #print('t', t)
    #print('centroid_difference', centroid_difference)

    return R, t, centroid_A, centroid_B, centroid_difference

def getRotTrans(apos, bpos, residueList=None):
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
    #rot, trans, centa, centb, tedit = rigid_transform_3D(apos, bpos)
    a_new = apos[:]
    a_res = np.zeros((3,len(residueList)))
    b_res = np.zeros((3,len(residueList)))
    for index, i in enumerate(residueList):
        print('ares', a_res)
        print('apos', apos)

        print('bres', b_res)
        print('bpos', bpos)
        a_res[index] = apos[i]
        b_res[index] = bpos[i]
    rot, trans, centa, centb, centroid_difference = rigid_transform_3D(a_res, b_res)
    print('ares', a_res)
    print('TODO')
    print('rot', rot)
    print('centroid_difference', centroid_difference)
    return rot, centroid_difference

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
    #rot, trans, centa, centb, tedit = rigid_transform_3D(apos, bpos)
    a_new = apos[:]
    num_res = len(residueList)
    a_res = np.zeros((3, num_res))
    b_res = np.zeros((3, num_res))
    for index, i in enumerate(residueList):
        print('ares', a_res)
        print('apos', apos)
        a_res[index] = apos[i]
        b_res[index] = bpos[i]
    holder_rot, trans, centa, centb, holder_centroid_difference = rigid_transform_3D(a_res, b_res)
    b_removed_centroid = b_res - (np.tile(centb, (num_res, 1)))
    b_new = (np.tile(centroid_difference, (num_res, 1))).T + (np.tile(centb, (num_res, 1))).T
#    b_new = np.dot(rot, b_removed_centroid.T) + (np.tile(centroid_difference, (num_res, 1))).T + (np.tile(centb, (num_res, 1))).T
    print('b_res', b_res)
    print('b_removed_centroid', b_removed_centroid)
    print('b_new', b_new)
    print('rot', rot)
#    b_new = np.dot(rot, a_removed_centroid.T) + (np.tile(centroid_difference, (num_res, 1))).T + (np.tile(centa, (num_res, 1))).T
    b_new = b_new * unit.nanometer
    for index, i in enumerate(residueList):
        #index is index, i is residueList index
        print('a_new', a_new)
        print('b_new', b_new)
        a_new[residueList[index]] = b_new[index]
    return a_new

# Test with random data
if __name__== "__main__":
  # Random rotation and translation
  R = np.array(np.random.rand(3,3))
  t = np.array(np.random.rand(3,1))

  # make R a proper rotation matrix, force orthonormal
  U, S, Vt = np.linalg.svd(R)
  R = np.dot(U,Vt)

  # remove reflection
  if np.linalg.det(R) < 0:
     Vt[2,:] *= -1
     R = np.dot(U,Vt)

  # number of points
  n = 10

  A = np.mat(np.random.rand(n,3));
  B = np.dot(R,A.T) + np.tile(t, (1, n))
  B = B.T;

  # recover the transformation
  ret_R, ret_t, centa, centb, tedit = rigid_transform_3D(A, B)

  A2 = np.dot(ret_R,A.T) + np.tile(ret_t, (1, n))
  A2 = A2.T

  # Find the error
  err = A2 - B

  err = np.multiply(err, err)
  err = np.sum(err)
  rmse = sqrt(err/n);

  print("Points A")
  print(A)
  print("")

  print("Points B")
  print(B)
  print("")

  print("Rotation")
  print(R)
  print("")

  print("Translation")
  print(t)
  print("")

  print("RMSE:", rmse)
  print("If RMSE is near zero, the function is correct!")

