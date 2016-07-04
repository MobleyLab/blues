import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import pyemma
import pyemma.coordinates as coor


main_file = 'youtput_debug1.h5'
traj_list = [main_file]
#feat = coor.featurizer(topfile)

top_file = main_file
feat = coor.featurizer(top_file)
prot_index = np.array(feat.select("(resid >= 102) and (resid <= 104) and (name == CA)"))
#added_residues = np.array(feat.select("(resid >= 132) and (resid <= 146) and (name == CA)"))
lig_coord = np.array(feat.select("(resname == 'LIG') and (mass >= 2)"))
prot_lig = np.concatenate((prot_index, lig_coord), axis=0)

#print 'prot_lig', prot_lig
#feat.add_selection(lig_heavy)
#feat.add_backbone_torsions(selstr="(resid >= 105) and (resid <= 115)")
#feat.add_backbone_torsions(selstr="(resid >= 132) and (resid <= 146)")
#feat.add_angles(np.array([[1635, 1639, 2640]]), cossin=True)
#feat.add_angles(np.array([[1635, 1639, 2640]]), cossin=False)

#feat.add_dihedrals(np.array([[1638, 2634, 2636, 2640]]), cossin=False)
feat.add_distances(prot_lig)
print feat.dimension()


inp = coor.source(traj_list, feat)

pca_obj = coor.pca(inp, dim=2)

Y = pca_obj.get_output()
print Y
print Y[0][:,0]

















traj = md.load('youtput_debug1.h5')
#angle_index = np.array([[1622, 2634, 2635]])
if 0:
#    angle_index = np.array([[1622, 2634, 2635]])
    angle_index = np.array([[1635, 1639, 2640]]) #makes symmetric groups

    angle_output = md.compute_angles(traj, angle_index)
if 0:
#    angle_index = np.array([[1622, 1638, 2634, 2635]])
#    angle_index = np.array([[1638, 1622, 2634, 2635]])
    angle_index = np.array([[1638, 1622, 2634, 2640]])  #separates into two groups
###    angle_index = np.array([[1638, 2634, 2636, 2640]])
###    angle_index = np.array([[1638, 2634, 2636, 2638]])
#    angle_index = np.array([[1622, 2635, 2636, 2640]])
#    angle_index = np.array([[1638, 1622, 2633, 2640]])  #separates into two groups (maybe four)
#    angle_index = np.array([[1638, 1622, 2634, 2640]])  #separates into two groups
#    angle_index = np.array([[1638, 1622, 2638, 2640]])  #separates into two groups
#    angle_index = np.array([[1638, 2634, 2638, 2640]])  #separates into two groups
#*    angle_index = np.array([[1638, 2634, 2636, 2640]])  #separates into four groups
#    angle_index = np.array([[1638, 2634, 2636, 2640]])  #separates into four groups
    angle_output = md.compute_dihedrals(traj, angle_index)


if 1: 
    angle_index = np.array([[1635, 1639, 2640]]) #makes symmetric groups
    angle_output = md.compute_angles(traj, angle_index)
    dihedral_index = np.array([[1638, 2634, 2636, 2640]])  #separates into four groups
    dihedral_output = md.compute_dihedrals(traj, dihedral_index)
    frames = traj.n_frames

    f, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(range(frames), angle_output, marker='x', s=2)
    print angle_output
    print dihedral_output
    ax2.hist2d(angle_output[:,0], dihedral_output[:,0])
    plt.show()













frames = traj.n_frames
print frames
f, (ax1, ax2) = plt.subplots(2)
ax1.scatter(range(frames), angle_output, marker='x', s=2)
ax2.hist(angle_output, bins=180)

#plt.scatter(range(frames), angle_output)
plt.show()

f, (ax1, ax2) = plt.subplots(2)
ax1.scatter(range(frames), Y[0][:,0], marker='x', s=2)
ax1.scatter(range(frames), Y[0][:,1], marker='x', s=2)
ax2.hist2d(Y[0][:,0], Y[0][:,1], bins=[75,75])

#ax2.hist(angle_output, bins=360)
plt.show()

