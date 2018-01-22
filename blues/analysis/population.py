import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from . import tools

class BindingModeOccupancy(object):
    """
    BindingModeOccupancy provides functionality to calculate the occupancy
    of each defined binding mode, determined from FindBindingModes().
    """

    def __init__(self, trajfile, acc_data=None, sel='resname LIG and not type H'):
        """
        Initialize the BindingModeOccupancy class by loading the *single*
        MD trajectory.

        Parameters:
        -----------
        trajfile : str, specifying the path for a single trajectory file to analyze.
        acc_data : dict, `{jobid : list}`. Dictionary where the keys correspond to
                   the jobid for the BLUES simulation for this molecule. Values
                   correspond to the time a proposed BLUES move was accepted.
        sel : str, selection str for computing the RMSD.
              Default = 'resname LIG and not type H' to compute the ligand heavy atom RMSD.
        """

        self.filename = trajfile
        self.molid = trajfile.split('/')[0]
        if acc_data:
            self.traj, self.jobid, self.acc_time = self.loadTrajectory(trajfile, acc_data)
        else:
            self.traj, self.jobid = self.loadTrajectory(trajfile)
            self.acc_time = None
        self.lig_atoms = self.traj.top.select('resname LIG and not type H')
        self.populations = {}
        self.labeled_frames = {}
        print('\t Analyzing:', self.filename)


    def loadTrajectory(self,trajfile, acc_data=None):
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

        if acc_data:
            acc_time = acc_data[jobid]
            return traj, jobid, acc_time
        else:
            return traj, jobid

    def calcOccupancy(self, leaders, leader_labels, outfname=None):
        """
        Assigns the trajectory frames to a given cluster by minimizing the RMSD to
        the representative frames from each cluster (i.e. leaders). Plots the RMSD
        of the ligand, relative to the starting position, coloring each datapoint to
        it's assigned cluster and calculates the occupancy of each ligand binding mode.

        Parameters:
        -----------
        leaders : mdtraj.Trajectory, frames containing cluster leaders *after filtering*.
        leader_labels : list, cluster index for each cluster leader frame.
        outfname : str, specifying the file prefix for saving the plots.
                   Will append the suffix `<outfname>-<jobid>-occ.png`
        """

        n_clusters = len(np.unique(leader_labels))
        dist = tools.rmsdNP(self.traj, self.traj[0], idx=np.asarray(self.lig_atoms))
        labels = self.assignFramesFromPCCALeaders(self.traj, leaders, leader_labels)
        self.populations, self.labeled_frames = self.plotPoseSampling(distances=dist, cluster_labels=labels,
                               time_conversion=0.0352,
                               acc_time=self.acc_time,
                               title='BM Occupancies %s-%s' %(self.molid, self.jobid))
        if outfname:
            plt.savefig('%s-%s-occ.png' %(outfname, self.jobid), dpi=300)
        else:
            plt.show()
        plt.clf()

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

        #print('Assigning trajectory frames by leaders')
        cluster_labels = []
        atom_indices = traj.top.select(sel)
        for i,frame in enumerate(traj):
            rmsds = tools.rmsdNP(leaders,frame,atom_indices)
            cluster_labels.append(leader_labels[np.argmin(rmsds)])
            tools.drawProgressBar(i/(traj.n_frames-1))
        cluster_labels = np.array(cluster_labels)

        return cluster_labels

    @staticmethod
    def plotPoseSampling(distances,cluster_labels,title="Binding Mode Frequency", time_conversion=0.08,
    acc_time=[], cmap='gist_rainbow'):
        """
        Plots the occupancy of each binding mode for the given trajectory.
        Plots the RMSD of the ligand, relative to the starting position, coloring each
        datapoint to it's assigned cluster and calculates the occupancy of each ligand binding mode.

        Parameters:
        -----------
        distances : numpy array of the RMSD relative to the starting position.
        cluster _labels : list, containing the cluster assignments of the trajectory frames.
        title : str, title to be used on the plots
        time_conversion : float to compute the corresponding time in each frame.
        acc_time : list, containing the timepoints when a BLUES move was accepted.
        cmap : str, specifying the matplotlib colormap to use.
        """
        n_clusters = len(np.unique(cluster_labels))
        T = len(cluster_labels)
        plt.figure(figsize=(16, 9), dpi=300,tight_layout=True)
        f, ax = plt.subplots(2, gridspec_kw={'height_ratios':[1,3]})

        cm = plt.get_cmap(cmap)
        colors = [cm(1.*i/n_clusters) for i in range(n_clusters)]

        #Compute and plot the % frequency in bar plot
        labeled_frames = {n:[] for n in range(n_clusters)}
        for i, label in enumerate(cluster_labels):
            labeled_frames[label].append(i)
        freq = { key:len(val)/T*100 for key,val in labeled_frames.items() }
        rect = ax[0].bar(range(len(freq.keys())),freq.values(), align='center', color=colors)
                   #,tick_label=freq.keys());
        ax[0].set_xlabel("Ligand Binding Mode")
        ax[0].set_ylabel("% Frequency")
        ax[0].set_xticks([])

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

        autolabel(rect)

        colr_list = []
        for x in cluster_labels:
            colr_list.append(colors[x])


        time = [time_conversion*t for t in range(T)]
        #Lineplot of RMSD relative to initial frame
        #Convert distances from nm to angstrom (distances*10.0)
        #ax[1].plot(time, 10.0*distances, 'k', linewidth=0.1)
        #Colored Scatterplot points to state of minimum RMSD
        ang_dist = 10.0*distances
        ax[1].scatter(time, ang_dist,c=colr_list,clip_on=False,s=25, edgecolors=None)
        if acc_time:
            for t in acc_time:
                try:
                    idx = np.argwhere(t == np.asarray(time))
                    ax[1].scatter(t, ang_dist[idx], marker='+', color='k', edgecolors='k')
                except:
                    pass
        #ax[1].axvline(x=it, color='k', linestyle=':', linewidth=1.0)
        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("RMSD $\AA$")
        ax[1].xaxis.grid(False)
        ax[1].yaxis.grid(False, which='major')
        plt.autoscale(enable=True, axis='both', tight=True)
        ax[0].set_title("%s N=%s" %(title, T))

        return freq, labeled_frames
