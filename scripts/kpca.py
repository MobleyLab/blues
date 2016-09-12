import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import pyemma
import pyemma.coordinates as coor
import pyemma.plots as mplt
import os
import fnmatch

def grep_folder(traj_list, folder, globname):
    path = folder
    traj_append = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(path)
        for f in fnmatch.filter(files, globname)]
#    print traj_append
    for entry in traj_append:
        traj_list.append(entry)
    return traj_list

def flatten_data(mapped_data, dim):
    assert type(dim) == int
    temp_mapped_data = np.copy(mapped_data)
    frame_counter = 0
    total_length = 0
    for index, entry in enumerate(temp_mapped_data):
    #index is used to to map to the traj_list index
    #entry is the individual trajectory (that you will iterate through)
    #keeping track of total length of the trajectory using the next two lines
        total_length = total_length + len(entry)
    #frame_membership is used to
    debug_mapped_data = np.zeros((total_length, dim))
    counter = 0
    #flatten mapped_data (which is the traj clusters)
    for array in np.copy(mapped_data):
#        print 'array', array
        len_traj = np.shape(array)[0]
        print len_traj
        for entry in range(len_traj):
#            print 'debug_mapped counter'
#            print debug_mapped_data[counter]
#            print 'array[entry]'
#            print array[entry]
            debug_mapped_data[counter] = array[entry]
            counter = counter + 1
    return debug_mapped_data

def find_frame(xRange, yRange, xList, yList):
    frame_list = []
    print type(xList)
    print xList
    for i in range(len((xList))):
        if xList[i] >= xRange[0] and xList[i] <= xRange[1] and yList[i] >= yRange[0] and yList[i] <= yRange[1]:
            frame_list.append(i)
    print frame_list
    return frame_list

def output_traj(orig_traj, outname, framelist):
    traj = md.load(orig_traj) 
    traj = traj[framelist]
    traj.save(outname)
   

main_file = 'less.h5'
traj_list = [main_file]
if 0: #finds all files
    directory_list = [x[0] for x in os.walk('../')]
    traj_list = []
    print 'directory_list', directory_list
    for entry in directory_list[1:]:
        grep_folder(traj_list=traj_list, folder=entry, globname='example1.h5')
        print traj_list
    print traj_list, len(traj_list)

#feat = coor.featurizer(topfile)

top_file = main_file
feat = coor.featurizer(top_file)
prot_index = np.array(feat.select("(resid == 102) and (name == CA)"))
#added_residues = np.array(feat.select("(resid >= 132) and (resid <= 146) and (name == CA)"))
lig_coord = np.array(feat.select("(resname == 'LIG') and (mass >= 2)"))
prot_lig = np.concatenate((prot_index, lig_coord), axis=0)

#print 'prot_lig', prot_lig
#feat.add_selection(lig_heavy)
#feat.add_backbone_torsions(selstr="(resid >= 105) and (resid <= 115)")
#feat.add_backbone_torsions(selstr="(resid >= 132) and (resid <= 146)")
#feat.add_angles(np.array([[1635, 1639, 2640]]), cossin=False)
#feat.add_angles(np.array([[1635, 1639, 2640]]), cossin=False)

feat.add_dihedrals(np.array([[1638, 1622, 2634, 2640]]), cossin=False)
#feat.add_dihedrals(np.array([[1638, 2634, 2638, 2640]]), cossin=False)
feat.add_distances(prot_lig)
print feat.dimension()


inp = coor.source(traj_list, feat)

#pca_obj = coor.pca(inp, dim=5)
pca_obj = coor.tica(inp, dim=5, lag=2)

Y = pca_obj.get_output()
print Y
print Y[0][:,0]
num_clusters = 4
input_data = coor.cluster_kmeans(data=Y, k=num_clusters, max_iter=1000)
mapped_data = [np.array(dtraj)[:,0] for dtraj in input_data.get_output()]
frame_dict = {}
for num in range(num_clusters):
    frame_dict[num] = []

for index, entry in enumerate(mapped_data[0]):
    frame_dict[entry].append(index)
for num in range(num_clusters):
    print len(frame_dict[num])
for num in range(num_clusters):
    output_name = 'tcluster_' + str(num) + '.dcd'
    print 'saving ', output_name
    output_traj(main_file, outname=output_name, framelist=frame_dict[num])
print mapped_data
traj = md.load(main_file)
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
    ax2.hist2d(angle_output[:,0], dihedral_output[:,0], bins=[40,40])
    plt.show()

frames = traj.n_frames
print frames
f, (ax1, ax2) = plt.subplots(2)
ax1.scatter(range(frames), angle_output, marker='x', s=2)
ax2.hist(angle_output, bins=180)

#plt.scatter(range(frames), angle_output)
plt.show()
print 'Y', Y
if 1: #flattens data
    Y = flatten_data(Y, dim=5)
    frames = len(Y)
xpca = Y[:,0]
ypca = Y[:,1]
f, (ax1, ax2) = plt.subplots(2)
ax1.scatter(range(frames), xpca, marker='x', s=2)
ax1.scatter(range(frames), ypca, marker='x', s=2)
ax2.hist2d(Y[:,0], Y[:,1], bins=[100,100])

#ax2.hist(angle_output, bins=360)
plt.show()

def get_counts(x_list, y_list, xpca, ypca, histbins=10):
    hist, xedges, yedges = np.histogram2d(xpca, ypca, bins=histbins)
    totsum = 0
    xfirst, xlast = np.digitize(x_list, xedges)  
    yfirst, ylast = np.digitize(y_list, yedges)
#    print 'hist', hist
    print 'doing loop'
#    print hist[xfirst:xlast]
    for i in hist[xfirst:xlast]:
#        print 'i', i
        for j in i[yfirst:ylast]:
            totsum += j
#            print j, totsum
    return totsum

#    counts_range = [[-3.14, -1.9], [-1.5,0], [0,1.25], [1.5,3.14]]
#    counts_range = [[-3.14, -1.9], [-1.5,0], [0,1.25], [1.5,3.14]]
#    counts_range = [[-3.14,-1], [1,3.14]]


#    for entry in counts_range:
#        first, last = np.digitize(entry, bin_edges)
#        print np.sum(hist_counts[first:last])
#        counts_list.append(np.sum(hist_counts[first:last]))

#x_lists = [[-0.23,-0.095], [0.095, 0.24]]
#y_lists = [[-0.4, -0.02], [0.05, 0.37]] 
#x_lists = [[-0.25,-0.1], [0.1,0.25]]
#y_lists = [[-0.48, 0.64]] 
#x_lists = [[-3.0, 0], [0.,3.1]]
#y_lists = [[-0.26, 0.02], [0.03, .275]]
x_lists = [[-4.3, -1.9]]
y_lists = [[-4.0, 0.0]]


#outlist = find_frame(xRange=x_lists[0], yRange=y_lists[0], xList=xpca, yList=ypca)
#print 'frames found'
#outframes = md.load('short.h5')
#out = outframes[outlist]
#out.save('output.h5')


pca_counts = []
pca_order = []
pca_probs = []
counter = 0
for i1 in x_lists:
    print i1
    for j1 in y_lists:
        print j1
        pca_counts.append(get_counts(x_list=i1, y_list=j1, xpca=xpca, ypca=ypca, histbins=50))
        print get_counts(x_list=i1, y_list=j1, xpca=xpca, ypca=ypca, histbins=50), i1, j1
        pca_order.append([i1, j1])
        if 0:
            outlist = find_frame(xRange=i1, yRange=j1, xList=xpca, yList=ypca)
            outname = 'pcluster_' + str(counter) + '.pdb'
            output_traj(orig_traj=main_file, framelist=outlist, outname=outname)
        counter = counter + 1
print pca_counts
sumtotal = np.sum(pca_counts)
for entry in pca_counts:
    pca_probs.append(float(entry)/float(sumtotal))
print pca_probs
print pca_order
#get_counts(x_list=[-0.1, 0.1], y_list=[-0.1,0.2], xpca=xpca, ypca=ypca, histbins=20)

if 0:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = Y[0][:,0]
    y = Y[0][:,1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=25)
    
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()
    
    ax.bar3d(xpos, ypos, hist, dx, dy, dy, color='b', zsort='average')
    
    plt.show()

