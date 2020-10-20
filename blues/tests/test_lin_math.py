from blues.moldart.lin_math import rigid_transform_3D, getRotTrans, adjust_angle
import numpy as np
import pytest

import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def test_rigid_transform_3D():
    tri1 = np.array([[0,0,0],
                    [1,0,0],
                    [0,1,0]])
    tri2 = np.array([[1,1,1],
                    [2,1,1],
                    [1,2,1]])

    tri_return = rigid_transform_3D(tri1, tri2, 0)
    #rotation matrix should be the identity
    assert np.array_equal(tri_return[0], np.identity(3))
    #the chosen centroids should be the first entry of each array
    assert np.array_equal(tri1[0], tri_return[2])
    assert np.array_equal(tri2[0], tri_return[3])
    rot_mat, trans = getRotTrans(tri1, tri2, 0)
    assert np.array_equal(rot_mat, tri_return[0])
    assert np.array_equal(trans, tri_return[-1])


def test_adjust_angle():
    vec1 = np.array([0,0,1])
    vec2 = np.array([0,1,np.sqrt(2)/2.])
    angle = np.rad2deg(angle_between(vec1, vec2))
    print('angle', angle)
    #radian_adjust = np.deg2rad(45)
    radian_adjust = np.deg2rad(25)
    #check that maintain_magnitude works
    output1 = adjust_angle(vec2,vec1, radian_adjust, maintain_magnitude=False)
    print('new angle', np.linalg.norm(output1))

    assert np.isclose(np.linalg.norm(output1), 1.0)
    output2 = adjust_angle(vec2,vec1, radian_adjust, maintain_magnitude=True)
    assert np.isclose(np.linalg.norm(output2), np.linalg.norm(vec2))
    assert np.isclose(angle_between(output1, vec1), np.deg2rad(25))

if __name__ == "__main__":
    test_adjust_angle()