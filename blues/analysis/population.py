import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pyemma.coordinates as coor
from pyemma import plots
from blues.analysis import tools
from copy import deepcopy

class BindingModePopulation(object):
    """
    BindingModePopulation provides functionality to calculate the occupancy
    of each defined binding mode, determined from FindBindingModes().
    """

    def __init__(self, molid, n_clusters, time_per_frame=0.0352,
                frames_per_iter=1, jsonfile=None):
        """
        Initialize the BindingModePopulation class by loading the *single*
        MD trajectory.

        Parameters:
        -----------
        molid : str, specifying the name of the molecule
        n_clusters : int, the optimimal number of clusters to use according to
                     the silhouette scores.
        time_per_frame: float, time in nanoseconds per trajectory frame.
        frames_per_iter : int, number of frames per BLUES iteration. If analyzing MD simulation,
            using default setting of 1.
        jsonfile : str, specifying the path for the associated json file that stores
                   when a BLUES move is accepted.
        """
        self.molid = molid
        self.n_clusters = n_clusters

        self.time_per_frame = time_per_frame
        self.frames_per_iter = frames_per_iter

        if jsonfile:
            self.jsonfile = jsonfile
            self.acc_data = tools.parse_blues_json(self.jsonfile, self.molid)
        else:
            self.acc_data = None

        self.population = {}
        self.labeled_frames = {}
        self.bm_change_frames = []

    def calcPopulation(self, traj_cluster_labels):
        """
        *Primary function to call.*

        Calculates the population of each defined binding mode and tries to
        detect changes in ligand binding mode using the assigned trajectory data.
        This function will discard counting frames before the first detected change
        in binding mode and will not count frames if there are multiple trajectories
        that are stuck in the same ligand binding mode.

        Parameters:
        -----------
        traj_cluster_labels : list of arrays, cluster assignment for each
                              trajectory frame where every array is for each trajectory

        Returns:
        --------
        traj_populations : numpy array, number of trajectory frames assigned to each cluster
                           across the entire trajectory pool.
        """

        bm_change_frames = [self._find_bm_change(labels,range(len(labels)),self.frames_per_iter) for labels in traj_cluster_labels]
        frame_cluster_map = [self._map_frames_to_cluster(self.n_clusters, labels) for labels in traj_cluster_labels]

        frame_cluster_map_copy = deepcopy(frame_cluster_map)

        for bm, fr_map, in zip(bm_change_frames, frame_cluster_map):
            if bm: fr_map = self._discard_equil_frames(bm, fr_map)

        traj_populations = self._get_populations(frame_cluster_map)
        orig_populations = self._get_populations(frame_cluster_map_copy, dedupe=False)

        self.traj_populations = traj_populations
        self.orig_populations = orig_populations
        return traj_populations

    def _get_populations(self, frame_cluster_map, dedupe=True):
        """
        Helper function. Computes the number of frames assigned to each cluster.

        Parameters:
        -----------
        frame_cluster_map : dict, { cluster_index : list of frame indices }
        dedupe : boolean, If True, this will remove counts from trajectories that are
            stuck in the same binding mode throughout the simulation to avoid
            overcounting.

        Returns:
        --------
        traj_populations : numpy array, number of trajectory frames assigned to each cluster
                           across the entire trajectory pool.
        """
        traj_populations = []
        for idx,cl_d in enumerate(frame_cluster_map):
            populations = [ len(frames) for cl_idx,frames in cl_d.items() ]
            if dedupe: #Avoid counting trajectories that are the same (stuck)
                if populations not in traj_populations:
                    traj_populations.append(populations)
                else:
                    print('Discarding population count from trajectory file', idx)
                    print('\t', populations)
            else:
                traj_populations.append(populations)
        return np.asarray(traj_populations, dtype=int)

    def _discard_equil_frames(self, bm_change_frames, frame_cluster_map):
        """
        Helper function. Discard frames before first binding mode change to avoid
        over counting the initial binding mode.

        Parameters:
        -----------
        bm_change_frames : list of frame indices where a binding mode change was detected.
        frame_cluster_map : dict, { cluster_index : list of frame indices }

        Returns:
        ---------
        frame_cluster_map : dict, { cluster_index : list of frame indices } with discarded equil frames.
        """

        #Get frames that correspond to the initial binding mode
        # Up until the first binding mode change
        equil_frames = set([x for x in range(bm_change_frames[0])])
        for cl, frames in frame_cluster_map.items():
            frame_cluster_map[cl] = list(set(frames).difference(equil_frames))
        return frame_cluster_map

    def _map_frames_to_cluster(self, n_clusters, cluster_labels):
        """
        Helper function. Construct a dictionary that maps the frame indices to the cluster index.
        Ex.  `{ cluster_index : [frames]}`

        Parameters:
        -----------
        n_clusters : int, optimimal number of clusters according to silhouette scores
        cluster_labels : list of arrays, cluster assignment for each
            trajectory frame where every array is for each trajectory

        Returns:
        ----------
        frame_cluster_map : dict, { cluster_index : list of frame indices } with discarded equil frames.
        """
        frame_cluster_map = {n:[] for n in range(n_clusters)}
        for i, label in enumerate(cluster_labels):
            frame_cluster_map[label].append(i)
        return frame_cluster_map

    def _find_bm_change(self, cluster_labels, frame_indices, frames_per_iter):
        """
        Helper function. Obtain the frame indices in which a change in binding mode is
        detected.

        Parameters:
        ----------
        cluster_labels : list of arrays, cluster assignment for each
            trajectory frame where every array is for each trajectory
        frame_indices : list, trajectory frame indices.
        frames_per_iter : int, number of frames per BLUES iteration. If analyzing MD simulation,
            using default setting of 1.

        Returns:
        --------
        bm_change_frames : list of frame indices where a binding mode change was detected.
        """
        bm_change_frames = []
        for l in frame_indices:
            try:
                #Get the cluster labels for the frames of previous iteration
                prev_cluster_labels = [cluster_labels[l-r] for r in range(frames_per_iter)]
                prev_counts = np.bincount(prev_cluster_labels)
                prev_label = np.argmax(prev_counts)

                #Get cluster labels for frames in next iteration
                next_cluster_labels = [cluster_labels[l+r] for r in range(frames_per_iter)]
                next_counts = np.bincount(next_cluster_labels)
                next_label = np.argmax(next_counts)
                #print(l, prev_cluster_labels,prev_label, next_cluster_labels, next_label)
            except:
                pass
            #Store the frame indices when there is a change in binding mode
            if prev_label != next_label:
                bm_change_frames.append(l)
                #print(l, prev_cluster_labels,prev_label, next_cluster_labels, next_label)
        return bm_change_frames

    def barplotPopulation(self, n_clusters, traj_populations, orig_populations=None,
                       title='Ligand Binding Mode Populations', outfname=None):
        """
        Generates a bar plot of the total populations from the trajectory pool.

        Parameters:
        ----------
        n_clusters : int, optimimal number of clusters according to silhouette scores
        traj_populations : numpy array, number of trajectory frames assigned to each cluster
                           across the entire trajectory pool *after discarding equil and duplicate frames*.
        orig_populations : numpy array, number of trajectory frames assigned to each cluster
                           across the entire trajectory pool *prior to discarding frames*.
        title : str, title for plots
        outfname : str, filename prefix for saving the plot of the format `<outfname>-barpop.png`
        """

        plt.figure(figsize=(16, 9), dpi=300,tight_layout=True)
        f, ax = plt.subplots(1)
        colors = tools.get_color_list(n_clusters)

        if orig_populations is not None:
            orig_percent = orig_populations.sum(axis=0)/orig_populations.sum() * 100
            ax.bar(range(n_clusters),orig_percent, align='center', color=colors, alpha=0.5);

        total_percent = traj_populations.sum(axis=0)/traj_populations.sum() * 100
        rect = ax.bar(range(n_clusters),total_percent, align='center', color=colors, alpha=1.0);

        ax.set_xlabel("Ligand Binding Mode");
        ax.set_ylabel("% Frequency");
        ax.set_xticks(range(n_clusters));
        ax.set_title(title);
        tools.autolabel(ax, rect)

        if outfname:
            plt.savefig(outfname+'-barpop.png', dpi=300, bbox_inches='tight')
            plt.close(f)
        else:
            plt.show()

    def plotTrajPopulation(self, trajfile, labels, outfname=None):
        """
        *Primary function to call.*

        This function will plot the population for a *single* given trajectory, as opposed
        to the entire data pool.

        trajfile : str, specifying the file path for the individual trajectory
        labels : list, cluster assignment for each trajectory frame
        outfname: str, specfiying the outfile filepath to save the plot
        """
        traj, jobid = self._loadTrajectory(trajfile)
        if self.acc_data :
            acc_data = self.acc_data[jobid]
        else:
            acc_data = None

        lig_atoms = traj.top.select('resname LIG and not type H')
        dist = tools.rmsdNP(traj, traj[0], idx=np.asarray(lig_atoms))
        np.savetxt('%s-%s-labels.txt'%(outfname,jobid),labels, fmt='%d')
        self._rmsd_population_plot(self.n_clusters, distances=dist,
                               cluster_labels=labels,
                               time_per_frame=self.time_per_frame,
                               frames_per_iter=self.frames_per_iter,
                               acc_data=acc_data,
                               title='BM Population %s-%s' %(self.molid, jobid),
                               outfname=outfname)

    def _loadTrajectory(self,trajfile, topfile=None):
        """
        Convenience function for loading the trajectory file.

        Parameters:
        -----------
        trajfile : str, specifying the path for a single trajectory file to analyze.
        topfile : str, specifying the path for the topology file (pdb)

        Returns:
        --------
        traj : mdtraj.Trajectory object of the BLUES simulation.
        jobid : str, specifying the jobid for the BLUES simulation.
        """
        topfile = trajfile.replace('dcd', 'pdb')
        jobid = trajfile.split('/')[-1].split('-')[1]
        traj = md.load(trajfile, top=topfile)
        return traj, jobid

    def _rmsd_population_plot(self, n_clusters, distances,
                         cluster_labels,
                         time_per_frame,
                         frames_per_iter,
                         outfname,
                         acc_data=[],
                         title="Binding Mode Population",
                         cmap='gist_rainbow',
                         ):
        """
        Plots the occupancy of each binding mode for the given trajectory.
        Plots the RMSD of the ligand, relative to the starting position, coloring each
        datapoint to it's assigned cluster and calculates the occupancy of each ligand binding mode.
        Vertival lines on the plot correspond to a change in ligand binding mode.

        Parameters:
        -----------
        n_clusters : int, optimimal number of clusters according to silhouette scores
        distances : numpy array of the RMSD relative to the starting position.
        time_per_frame: float, time in nanoseconds per trajectory frame.
        frames_per_iter : int, number of frames per BLUES iteration. If analyzing MD simulation,
            using default setting of 1.
        cluster_labels : list, containing the cluster assignments of the trajectory frames.
        title : str, title to be used on the plots
        time_conversion : float to compute the corresponding time in each frame.
        acc_data : list, containing the timepoints when a BLUES move was accepted.
        cmap : str, specifying the matplotlib colormap to use.
        """

        T = len(cluster_labels)
        time = [time_per_frame*t for t in range(T)]

        #Convert distances from nm to angstrom (distances*10.0)
        ang_dist = 10.0*distances

        plt.figure(figsize=(16, 9), dpi=300,tight_layout=True)
        f, ax = plt.subplots(2, gridspec_kw={'height_ratios':[1,3]})

        colors = tools.get_color_list(n_clusters, cmap)
        colr_list = [ colors[x] for x in cluster_labels]

        frame_cluster_map = self._map_frames_to_cluster(n_clusters, cluster_labels)
        freq = { key:len(val)/len(cluster_labels)*100 for key,val in frame_cluster_map.items() }
        rect = ax[0].bar(range(len(freq.keys())),freq.values(), align='center', color=colors)
        tools.autolabel(ax[0], rect)
        ax[0].set_title("%s N=%s" %(title, T))
        ax[0].set_xlabel("Ligand Binding Mode")
        ax[0].set_ylabel("% Frequency")
        ax[0].set_xticks([])

        #Lineplot of RMSD relative to initial frame
        #ax[1].plot(time, 10.0*distances, 'k', linewidth=0.1)

        #Colored Scatterplot points to state of minimum RMSD
        ax[1].scatter(time, ang_dist, c=colr_list,clip_on=False,s=25, edgecolors=None)

        if acc_data is not None:
            acc_time = [time_per_frame*frames_per_iter*x for x in acc_data]
            acc_frames = [frames_per_iter*x for x in acc_data]
            for idx,fr in enumerate(acc_frames):
                ax[1].scatter(acc_time[idx], ang_dist[fr], marker='+', color='k', edgecolors='k')

            #Get timepoint when accepted BLUES move results in a change in binding mode
            bm_change_frames = self._find_bm_change(cluster_labels, acc_frames, frames_per_iter)
            #Compute stats for BLUES move proposals
            frac_acc_bm_change = len(bm_change_frames)/len(acc_data) * 100
            frac_total_bm_change = len(bm_change_frames)/len(cluster_labels) *100
            print(title)
            print('\t Accepted moves with BM change: %s / %s == %.2f %%' % (len(bm_change_frames), len(acc_data), frac_acc_bm_change))
            print('\t Total BM change: %s / %s == %.2f %%'% (len(bm_change_frames), len(cluster_labels), frac_total_bm_change))
            self.frac_acc_bm_change = frac_acc_bm_change
            self.frac_total_bm_change = frac_total_bm_change
        else: #Detect binding mode changes in MD simulations
            bm_change_frames = self._find_bm_change(cluster_labels, range(T), frames_per_iter=3)
            for idx,f in enumerate(bm_change_frames):
                try:
                    next_value = int(bm_change_frames[idx+1])
                    if f+1 == next_value:
                        bm_change_frames.remove(f)
                except:
                    pass
            print('\tTotal BM changes: %s'% len(bm_change_frames))

        bm_change_time = [f*time_per_frame for f in bm_change_frames]
        for bm in bm_change_time:
            ax[1].axvline(bm, color='k', linestyle=':', linewidth=1.0)

        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("RMSD $\AA$")
        ax[1].xaxis.grid(False)
        ax[1].yaxis.grid(False, which='major')
        plt.autoscale(enable=True, axis='both', tight=True)

        if outfname:
            plt.savefig('%s-pop.png'%outfname, dpi=300)
            plt.close(f)
        else:
            plt.show()

    def animateTICA(self, n_clusters,
                    X_traj_tica_coords, Y_traj_tica_coords,
                    x_tica_coord, y_tica_coord,
                    cluster_labels, cmap='gist_rainbow',interval=200):
        """
        Generates an animated TICA scatter plot of the trajectory over time.

        Parameters:
        -----------
        n_clusters : int, number of clusters
        X_traj_tica_coords : numpy array, TICA coordinates for the entire trajectory pool to
            be plotted on the X-axis for the contour background.
        Y_traj_tica_coords : numpy array, TICA coordiantes for the entitre trajectory pool to
            be plotted on the Y-axis for the contour background.
        x_tica_coord : numpy array, TICA coordinates belonging to the *single* trajectory to
            be animated on the X-axis.
        y_tica_coord : numpy array, TICA coordinates belonging to the *single* trajectory to
            be animated on the Y-axis.
        cluster_labels : list, cluster assignment for the *single* trajectory to
            be animated on the TICA plot.
        cmap : str, specifing the colormap to be used for the animated data points
        interval : int, delay time (milliseconds) inbetween animated data points.
        """
        from matplotlib import animation, rc
        tica_coords = np.stack((x_tica_coord,y_tica_coord),axis=-1)
        xs, ys = zip(*tica_coords)
        x, y = np.array([]), np.array([])

        colors = tools.get_color_list(n_clusters, cmap)
        colr_list = [ colors[x] for x in cluster_labels]

        fig = plt.figure(tight_layout=True)
        #ax = fig.add_subplot(111)
        plots.plot_free_energy(X_traj_tica_coords,
                               Y_traj_tica_coords,
                               cmap='nipy_spectral', cbar=False)
        pathcol = plt.scatter([],[], s=50, c=colr_list,
                               marker='X', edgecolors='black')
        def init():
            pathcol.set_offsets([[],[]])
            return [pathcol]

        def update(i, pathcol, data):
            pathcol.set_offsets(data[:i])
            return [pathcol]

        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       fargs=(pathcol, tica_coords), interval=interval,
                                       frames=len(tica_coords), blit=True, repeat=True)
        return anim
