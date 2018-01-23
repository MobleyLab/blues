import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pyemma.coordinates as coor
from blues.analysis import tools

class BindingModeOccupancy(object):
    """
    BindingModeOccupancy provides functionality to calculate the occupancy
    of each defined binding mode, determined from FindBindingModes().
    """

    def __init__(self, molid, trajfile, tica_coord, time_per_frame=0.0352,
                frames_per_iter=1, jsonfile=None, sel='resname LIG and not type H'):
        """
        Initialize the BindingModeOccupancy class by loading the *single*
        MD trajectory.

        Parameters:
        -----------
        molid : str, specifying the name of the molecule
        trajfile : str, specifying the path for a single trajectory file to analyze.
        tica_coord : numpy array, tica coordinates for the associated trajectory.
        time_per_frame: float, time in nanoseconds per trajectory frame.
        sel : str, selection str for computing the RMSD.
              Default = 'resname LIG and not type H' to compute the ligand heavy atom RMSD.

        Parameters (BLUES SIMULATIONS ONLY):
        ----------------------
        frames_per_iter : int, number of frames per BLUES iteration.
        jsonfile : str, specifying the path for the associated json file that stores
                   when a BLUES move is accepted.
        acc_data : dict, `{jobid : list}`. Dictionary where the keys correspond to
                   the jobid for the BLUES simulation for this molecule. Values
                   correspond to the time a proposed BLUES move was accepted.

        """
        self.molid = molid
        self.filename = trajfile
        self.tica_coord = tica_coord
        self.time_per_frame = time_per_frame
        self.frames_per_iter = frames_per_iter

        if jsonfile:
            self.jsonfile = jsonfile
            self.traj, self.jobid = self._loadTrajectory(trajfile)
            self.time_per_iter = time_per_frame * frames_per_iter
            acc_data = tools.convertIterToTime(self.jsonfile, self.molid, self.time_per_iter)
            self.acc_time = acc_data[self.molid][self.jobid]
        else:
            self.traj, self.jobid = self._loadTrajectory(trajfile)
            self.acc_time = None

        self.lig_atoms = self.traj.top.select('resname LIG and not type H')
        self.freq = {}
        self.labeled_frames = {}
        self.bm_change_frames = []
        print('Analyzing populations in', self.filename)

    def _loadTrajectory(self,trajfile, topfile=None):
        """
        Convenience function for loading the trajectory file.

        Parameters:
        -----------
        trajfile : str, specifying the path for a single trajectory file to analyze.
        acc_data : dict, `{jobid : list}`. Dictionary where the keys correspond to
                   the jobid for the BLUES simulation for this molecule. Values
                   correspond to the time a proposed BLUES move was accepted.
        Returns:
        --------
        traj : mdtraj.Trajectory object of the BLUES simulation.
        jobid : str, specifying the jobid for the BLUES simulation.
        acc_time : list, values correspond to the time a proposed BLUES move was accepted.
        """
        topfile = trajfile.replace('dcd', 'pdb')
        jobid = trajfile.split('/')[-1].split('-')[1]
        traj = md.load(trajfile, top=topfile)
        return traj, jobid

    def calcOccupancy(self, pcca_centers, pcca_labels, outfname=None):
        """
        Assigns the trajectory frames to a given cluster by minimizing the
        euclidean distance to the centers from PCCA. Plots the RMSD of the
        ligand, relative to the starting position, coloring each datapoint to
        it's assigned cluster and calculates the occupancy of
        each ligand binding mode.

        Parameters:
        -----------
        pcca_centers : numpy array, contains the coordinates of the cluster centers
                  from the active set states obtained from PCCA.
        pcca_labels : list, cluster assignment of the centers from PCCA.
        outfname : str, specifying the file prefix for saving the plots.
                   Will append the suffix `<outfname>-<jobid>-occ.png`
        """
        n_clusters = len(np.unique(pcca_labels))
        dist = tools.rmsdNP(self.traj, self.traj[0], idx=np.asarray(self.lig_atoms))
        labels = self.assignFramesToPCCACenters(self.tica_coord, pcca_centers, pcca_labels)

        self.plotPoseSampling(n_clusters, distances=dist,
                               cluster_labels=labels,
                               time_per_frame=self.time_per_frame,
                               frames_per_iter=self.frames_per_iter,
                               acc_time=self.acc_time,
                               title='BM Occupancies %s-%s' %(self.molid, self.jobid))
        if outfname:
            plt.savefig('%s-%s-occ.png' %(outfname, self.jobid), dpi=300)
        else:
            plt.show()
        plt.clf()

    @staticmethod
    def assignFramesToPCCACenters(tica_coord, pcca_centers, pcca_labels):
        """
        Given the TICA coordinates data for a trajectory, assign data to
        the nearest clsuter centers as defined from PCCA.

        Parameters:
        -----------
        tica_coord : numpy array, tica coordinates for the associated trajectory.
        pcca_centers : numpy array, contains the coordinates of the cluster centers
                  from the active set states obtained from PCCA.
        pcca_labels : list, cluster assignment of the centers from PCCA.

        Returns:
        -------
        traj_cluster_labels : list, cluster assignment for each trajectory frame
        """
        traj_tica_coor = np.stack((tica_coord[:,0], tica_coord[:,1]),axis=-1)
        assignments = coor.assign_to_centers(traj_tica_coor, pcca_centers)[0]
        traj_cluster_labels = []
        for a in assignments:
            traj_cluster_labels.append(pcca_labels[a])
        return traj_cluster_labels

    @staticmethod
    def assignFramesFromPCCALeaders(traj,leaders,leader_labels, sel='resname LIG and not type H'):
        """
        Assigns the trajectory frames to a given cluster by minimizing the RMSD
        to the cluster leaders.

        Parameters:
        -----------
        leaders : mdtraj.Trajectory, frames containing cluster leaders *after filtering*.
        leader_labels : list, cluster index for each cluster leader frame.
        sel : str, selection str for computing the RMSD.
              Default = 'resname LIG and not type H' to compute the ligand heavy atom RMSD.

        Returns:
        --------
        cluster_labels : list, containing the cluster assignments of the trajectory frames.
        """

        cluster_labels = []
        atom_indices = traj.top.select(sel)
        for i,frame in enumerate(traj):
            rmsds = tools.rmsdNP(leaders,frame,atom_indices)
            cluster_labels.append(leader_labels[np.argmin(rmsds)])
            tools.drawProgressBar(i/(traj.n_frames-1))
        cluster_labels = np.array(cluster_labels)

        return cluster_labels

    def _find_bm_change(self, cluster_labels, frame_indices, frames_per_iter):
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
        self.bm_change_frames = bm_change_frames
        return bm_change_frames

    def _percent_occupancy(self, n_clusters, cluster_labels):
        """Compute the percent frequency of each binding mode."""
        labeled_frames = {n:[] for n in range(n_clusters)}
        for i, label in enumerate(cluster_labels):
            labeled_frames[label].append(i)
        freq = { key:len(val)/len(cluster_labels)*100 for key,val in labeled_frames.items() }
        self.labeled_frames = labeled_frames
        self.freq = freq
        return labeled_frames, freq

    def plotPoseSampling(self, n_clusters, distances,
                         cluster_labels,
                         time_per_frame,
                         frames_per_iter,
                         acc_time=[],
                         title="Binding Mode Frequency",
                         cmap='gist_rainbow'):
        """
        Plots the occupancy of each binding mode for the given trajectory.
        Plots the RMSD of the ligand, relative to the starting position, coloring each
        datapoint to it's assigned cluster and calculates the occupancy of each ligand binding mode.
        Vertival lines on the plot correspond to a change in ligand binding mode.

        Parameters:
        -----------
        distances : numpy array of the RMSD relative to the starting position.
        cluster _labels : list, containing the cluster assignments of the trajectory frames.
        title : str, title to be used on the plots
        time_conversion : float to compute the corresponding time in each frame.
        acc_time : list, containing the timepoints when a BLUES move was accepted.
        cmap : str, specifying the matplotlib colormap to use.
        """
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            font = {'weight' : 'heavy', 'size': 14}
            for rect in rects:
                height = rect.get_height()
                ax[0].text(rect.get_x() + rect.get_width()/2., 1.0,
                        '%.1f' % height,
                        ha='center', va='bottom', fontdict=font)

        T = len(cluster_labels)
        time = [time_per_frame*t for t in range(T)]

        #Convert distances from nm to angstrom (distances*10.0)
        ang_dist = 10.0*distances

        plt.figure(figsize=(16, 9), dpi=300,tight_layout=True)
        f, ax = plt.subplots(2, gridspec_kw={'height_ratios':[1,3]})

        colors = tools.get_color_list(n_clusters, cmap)
        colr_list = [ colors[x] for x in cluster_labels]

        labeled_frames, freq = self._percent_occupancy(n_clusters, cluster_labels)
        rect = ax[0].bar(range(len(freq.keys())),freq.values(), align='center', color=colors)
        autolabel(rect)
        ax[0].set_title("%s N=%s" %(title, T))
        ax[0].set_xlabel("Ligand Binding Mode")
        ax[0].set_ylabel("% Frequency")
        ax[0].set_xticks([])

        #Lineplot of RMSD relative to initial frame
        #ax[1].plot(time, 10.0*distances, 'k', linewidth=0.1)

        #Colored Scatterplot points to state of minimum RMSD
        ax[1].scatter(time, ang_dist, c=colr_list,clip_on=False,s=25, edgecolors=None)

        acc_frame_indices = []
        if acc_time is not None:
            for t in acc_time:
                try:
                    idx = np.argwhere(t == np.asarray(time))
                    acc_frame_indices.append(int(idx[0]))
                    ax[1].scatter(t, ang_dist[idx], marker='+', color='k', edgecolors='k')
                except:
                    pass

        #Get timepoint when accepted BLUES move results in a change in binding mode
        if acc_time:
            bm_change_frames = self._find_bm_change(cluster_labels, acc_frame_indices, frames_per_iter)
            #Compute stats for BLUES move proposals
            frac_acc_bm_change = len(bm_change_frames)/len(frame_indices)
            frac_total_bm_change = len(bm_change_frames)/len(cluster_labels)
            print('\t %% Accepted moves with BM change: %.3f'% frac_acc_bm_change)
            print('\t %% Total BM change: %.3f'% frac_total_bm_change)
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
