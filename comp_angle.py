import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
traj = md.load('youtput_debug1.h5')
#angle_index = np.array([[1622, 2634, 2635]])
if 0:
#    angle_index = np.array([[1622, 2634, 2635]])
    angle_index = np.array([[1635, 1639, 2640]]) #makes symmetric groups

    angle_output = md.compute_angles(traj, angle_index)
if 1:
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

frames = traj.n_frames
print frames
f, (ax1, ax2) = plt.subplots(2)
ax1.scatter(range(frames), angle_output, marker='x', s=2)
ax2.hist(angle_output, bins=180)

#plt.scatter(range(frames), angle_output)
plt.show()

