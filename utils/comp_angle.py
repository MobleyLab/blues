import mdtraj as md
import matplotlib
#matplotlib.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import numpy as np
#traj = md.load('example1.h5')
traj = md.load('less.h5')
#traj = traj[:6500]
#angle_index = np.array([[1622, 2634, 2635]])
if 0:
#    angle_index = np.array([[1622, 2634, 2635]])
    angle_index = np.array([[1635, 1639, 2640]]) #makes symmetric groups

    angle_output = md.compute_angles(traj, angle_index)
if 0:
#    angle_index = np.array([[1622, 1638, 2634, 2635]])
#    angle_index = np.array([[1638, 1622, 2634, 2635]])
###    angle_index = np.array([[1638, 1622, 2634, 2640]])  #separates into two groups
###    angle_index = np.array([[1638, 2634, 2636, 2640]])
###    angle_index = np.array([[1638, 2634, 2636, 2638]])
###    angle_index = np.array([[1622, 2635, 2636, 2640]])
#    angle_index = np.array([[1638, 1622, 2633, 2640]])  #separates into two groups (maybe four)
#    angle_index = np.array([[1638, 1622, 2634, 2640]])  #separates into two groups
#    angle_index = np.array([[1638, 1622, 2638, 2640]])  #separates into two groups
    angle_index = np.array([[1638, 2634, 2638, 2640]])  #separates into two groups
#*    angle_index = np.array([[1638, 2634, 2636, 2640]])  #separates into four groups
#    angle_index = np.array([[1638, 2634, 2636, 2640]])  #separates into four groups

if 1: #split into four
    angle_index = np.array([[1638, 2634, 2638, 2640]])  #separates into two groups
#    angle_index = np.array([[1638, 2634, 2637, 2640]])

    four_output = md.compute_dihedrals(traj, angle_index)
    counts_range = [[-3.14, -1.9], [-1.5,0], [0,1.25], [1.5,3.14]]


if 0: #split into two (not as nice)
    angle_index = np.array([[1638, 1622, 2633, 2640]])  #separates into two groups (maybe four)
#    angle_index = np.array([[1638, 1622, 2634, 2635]])

    angle_output = md.compute_dihedrals(traj, angle_index)

if 1: #split into two (nicer)
    angle_index = np.array([[1638, 1622, 2634, 2640]])  #separates into two groups


    two_output = md.compute_dihedrals(traj, angle_index)
    counts_range = [[-3.14,-1], [1,3.14]]



frames = traj.n_frames
print frames
#number of bins
num_bins=360
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.scatter(range(frames), four_output, marker='x', s=2)
ax1.set_xlabel('frame')
ax1.set_ylabel('radians')
ax2.hist(four_output, bins=num_bins)
ax2.set_xlabel('radians')
ax2.set_ylabel('counts')

ax3.scatter(range(frames), two_output, marker='x', s=2)
ax3.set_xlabel('frame')
ax3.set_ylabel('radians')
ax4.hist(two_output, bins=num_bins)
ax4.set_xlabel('radians')
ax4.set_ylabel('counts')

#plt.scatter(range(frames), angle_output)
hist_array = np.histogram(two_output, bins=num_bins)
if 1:
    counts_range = [[-3.14,-1], [1,3.14]]
    hist_counts = hist_array[0]
    bin_edges = hist_array[1]
    counts_list = []
#    counts_range = [[-3.14, -1.9], [-1.5,0], [0,1.25], [1.5,3.14]]
#    counts_range = [[-3.14, -1.9], [-1.5,0], [0,1.25], [1.5,3.14]]
#    counts_range = [[-3.14,-1], [1,3.14]]


    for entry in counts_range:
        first, last = np.digitize(entry, bin_edges)
#        print np.sum(hist_counts[first:last])
        counts_list.append(np.sum(hist_counts[first:last]))
    print counts_list
    for index, entry in enumerate(counts_list):
        counts_list[index] = float(counts_list[index])

    counts_list = np.array(counts_list)
    counts_list = counts_list/np.sum(counts_list)
    print counts_list
    #second time
    hist_array = np.histogram(four_output, bins=num_bins)

    counts_range = [[-3.14, -1.9], [-1.5,0], [0,1.25], [1.5,3.14]]
    hist_counts = hist_array[0]
    bin_edges = hist_array[1]
    counts_list = []

    for entry in counts_range:
        first, last = np.digitize(entry, bin_edges)
#        print np.sum(hist_counts[first:last])
        counts_list.append(np.sum(hist_counts[first:last]))
    print counts_list
    for index, entry in enumerate(counts_list):
        counts_list[index] = float(counts_list[index])

    counts_list = np.array(counts_list)
    counts_list = counts_list/np.sum(counts_list)
    print counts_list



#    print np.digitize([-3.14,-1.9], bin_edges)
#    print hist_array
plt.show()

