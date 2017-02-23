import mdtraj as md
from rot_mat import rigid_transform_3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a = md.load('posA.pdb')
b = md.load('posB.pdb')
apos = a.xyz[0][2634:2641]
bpos = b.xyz[0][2634:2641]
apos = a.xyz[0][2634:2649]
bpos = b.xyz[0][2634:2649]

if 0:
    apos = np.mat(a.xyz[0][2634:2641])
    bpos = np.mat(b.xyz[0][2634:2641])
#apos is the simulations positions
#bpos is the comparison positions
def getRotTrans(apos, bpos, residueList=None):
    '''
    Get rotation and translation of rigid pose

    Arguments
    ---------
    apos: nx3 np.array
        simulation positions
    bpos: nx3 np.array
        comparison positoins
    residueList
    '''
    if type(residueList) = type(None):
        residueList = self.residueList
    #rot, trans, centa, centb, tedit = rigid_transform_3D(apos, bpos)
    a_new = apos[:]
    a_res = np.zeros((3,len(residueList)))
    b_res = np.zeros((3,len(residueList)))
    for index, i in enumerate(residueList):
        a_res[index] = apos[i]
        b_res[index] = bpos[i]
    rot, trans, centa, centb, centroid_difference = rigid_transform_3D(a_res, b_res)
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
    residueList
    '''
    if type(residueList) = type(None):
        residueList = self.residueList
    #rot, trans, centa, centb, tedit = rigid_transform_3D(apos, bpos)
    a_new = apos[:]
    num_res = len(residueList)
    a_res = np.zeros((3,len(residueList)))
    b_res = np.zeros((3,len(residueList)))
    for index, i in enumerate(residueList):
        a_res[index] = apos[i]
        b_res[index] = bpos[i]
    holder_rot, trans, centa, centb, holder_centroid_difference = rigid_transform_3D(a_res, b_res)
    a_removed_centroid = a_res - (np.tile(centa, (num_res, 1)))
    b_new = np.dot(rot, a_removed_centroid.T) + (np.tile(centroid_difference, (num_res, 1))).T + (np.tile(centa, (num_res, 1))).T
    for index, i in enumerate(residueList):
        #index is index, i is residueList index
        a_new[residueList] = b_new[i]
    return a_new










num_res = len(range(2634,2649))
#remove centroid
removed_centroid = apos - (np.tile(centa, (num_res, 1)))
new = np.dot(rot, removed_centroid.T) + (np.tile(tedit, (num_res, 1))).T + (np.tile(centa, (num_res, 1))).T
a2 = np.dot(rot, apos.T)
a2 = a2.T
a2 = np.asarray(a2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
apos = np.asarray(apos)
bpos = np.asarray(bpos)

ax.scatter(apos[:,0], apos[:,1], apos[:,2], color='red')
ax.scatter(bpos[:,0], bpos[:,1], bpos[:,2], color='blue')
centa = np.array(centa)
ax.scatter(centa[0], centa[1], centa[2], color='black')
ax.scatter(centb[0], centb[1], centb[2], color='black')
ax.scatter(new[0], new[1], new[2], color='orange')

#use error as a check for correctness
err = new.T - bpos
err = np.multiply(err, err)
err = np.sum(err)
rmse = np.sqrt(err/num_res)
print('rmse', rmse)




plt.show()


