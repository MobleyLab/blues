import numpy as np
from math import sqrt
a = np.array([[1,2,3],[4,5,6.5],[7,8,9]])
b = np.array([[7,3,6]])
#a = np.array([[1,3],[2,1]])
#b = np.array([7,2])
print(a)
print(a.T)
print(b.T)

ainv = np.linalg.inv(a.T)
print(ainv)
print('new', np.dot(ainv,b.T))
def changeBasis(a, b):
    '''
    a is 3x3 np.array defining vectors that create a basis
    b is 1x3 np.array define position of particle to be transformed into
        new coordinate system
    '''
    ainv = np.linalg.inv(a.T)
    print('ainv', ainv)
    print('b.T', b.T)
    changed_coord = np.dot(ainv,b.T)
    return changed_coord
    
def normalize(a):
    magnitude = sqrt(np.sum(a*a))
    unit_vec = a / magnitude
    return unit_vec
def localcoord(particle1, particle2, particle3):
    part1 = particle1 - particle1
    part2 = particle2 - particle1
    part3 = particle3 - particle1
    vec1 = normalize(part2)
    vec2 = normalize(part3)
    vec3 = np.cross(vec1,vec2)
    print('vec3', vec3, normalize(vec3))
    print('vec1', vec1, 'vec2', vec2, 'vec3', vec3)
    return vec1, vec2, vec3
def findNewCoord(particle1, particle2, particle3, center):
    vec1, vec2, vec3 = localcoord(particle1, particle2, particle3)
    basis_set = np.zeros((3,3))
    basis_set[0] = vec1
    basis_set[1] = vec2
    basis_set[2] = vec3
    print('basis_set', basis_set)
    recenter = center - particle1
    new_coord = changeBasis(basis_set, recenter)
    print('new_coord', new_coord)
#    old_coord = changeBasis(np.linalg.inv(basis_set), new_coord)
#    print('reversed', old_coord)
#    print('vec1_n', vec1_n)
#    print(np.dot(vec1,vec3))
#    print(np.dot(vec2,vec3))                                                    
#print(a[0])
findNewCoord(a[0], a[1], a[2], [1,5,3])
