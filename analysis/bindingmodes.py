import pyemma.coordinates as coor
from pyemma import msm, plots
from collections import defaultdict
from cycler import cycler
import mdtraj as md
import numpy as np
import sys,random, itertools, glob
import matplotlib.pyplot as plt

import matplotlib.mlab as mlab
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
from matplotlib import animation, rc
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import gridspec
from scipy import stats
import json

class ConstructMSM(object):
    """
    ConstructMSM provides functionality to select the appropriate lagtime
    through plots, TICA-transforms the input feature coordinates for
    k-means clustering to discretize the trajectories, and constructs the MSM
    for the purpose of identifying the metastable binding modes.

    Example:
    >>> from analysis import bindingmodes
    >>> import pyemma.coordinates as coor
    >>> feat = coor.featurizer(topfiles[0])
    >>> lig_atoms = feat.select("resname LIG and not type H")
    >>> feat.add_selection(lig_atoms)
    >>> inp = coor.source(trajfiles, feat)
    >>> data = bindingmodes.ConstructMSM(inp)
    >>> M = data.getMSM(data.Y, dt=8, lagtime=150)
    """

    def __init__(self, inp):
        """
        Initialize the class by loading the featurized data.

        inp : pyemma.coordinates.data.feature_reader.FeatureReader
        topfile : str, path specifying the topology file path (pdb)
        """
        print('number of trajectories = ',inp.number_of_trajectories())
        print('number of dimension = ',inp.dimension())
        self.Y = inp.get_output()
        self.topfile = inp.topfile

    def _tica(self, Y, dt, lagtime):
        """Convenience function to transform the feature coordinates
        by time-lagged independent component analysis (TICA).

        Parameters:
        -----------
        Y : list, featurized coordinates from the FeatureReader
        dt : int, trajectory timestep
        lagtime : float, choosen lag time from the implied timescales

        Returns:
        -------
        tica_coordinates : numpy array of shape [n_samples, n_features_new]
        lag : int, lag time (in multiples of dt)
        """

        lag = np.int(lagtime/dt)
        tica_obj = coor.tica(Y, lag=lag, kinetic_map=True, var_cutoff=0.90, reversible=True);
        print('TICA dimension ', tica_obj.dimension())

        self.lag = lag; self.dt = dt
        self.tica_coordinates = tica_obj.get_output()

        return self.tica_coordinates, self.lag

    def _kmeans(self, tica_coordinates, k=None, max_iter=100):
        """Convenience function to discretize the data by performing
        k-means clustering.

        Parameters:
        -----------
        tica_coordinates : numpy array of shape [n_samples, n_features_new]
        k : int, maximum number of cluster centers. When `k=None`, the number of
            data points used will be k = min(sqrt(N))
        max_iter: maximum number of iterations before stopping clustering.

        Returns:
        --------
        dtrajs : list of arrays, each array contains the
                 trajectory frames assigned to the cluster centers indices.
        centers : numpy array, contains the coordinates of the cluster centers
        index_clusters : list of arrays, For each state, all trajectory and time indexes
                        where this cluster occurs. Each row consists of a pair [i,t],
                        where `i` is the index of the trajectory and `t` is the frame index.
        """
        cl = coor.cluster_kmeans(tica_coordinates, k, max_iter);


        dtrajs = cl.dtrajs
        cc_x = cl.clustercenters[:, 0]
        cc_y = cl.clustercenters[:, 1]

        # For later use we save the discrete trajectories and cluster center coordinates
        centers = np.stack((cc_x,cc_y),axis=-1)
        self.dtrajs = dtrajs
        self.centers = centers
        self.index_clusters = cl.index_clusters

        return self.dtrajs, self.centers, self.index_clusters

    @staticmethod
    def plotImpliedTimescales(inp, dt, lag_list, outfname=None):
        """
        Plots the implied timescales for the trajectories using a
        list of lag times to try.

        inp : pyemma.coordinates.data.feature_reader.FeatureReader
        dt  : int() timestep interval from trajectory frames
        lag_list : list of lag times to try
        outfname : str() specifying the output filename. None displays plot.
        """

        lag_times = []
        for lag in lag_list:
            data = coor.tica(inp, lag, kinetic_map=True, var_cutoff=0.90, reversible=True);
            lag_times.append(data.timescales[:10])
        lag_times = np.asarray(lag_times)

        plt.plot(dt*lag_list, dt*lag_times, linewidth=2)
        plt.xlabel(r"Lag time [$ps$]", fontsize=12)
        plt.ylabel(r"Timescales [$ps$]", fontsize=12)

        if outfname:
            #plt.switch_backend('agg')
            plt.suptitle("Implied timescales for %s "% (outfname.split('/')[0]),
                         fontsize=14, fontweight='bold')
            plt.savefig(outfname+'-lagtimes.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.suptitle("Implied timescales", fontsize=14, fontweight='bold')
            plt.show()

    def getMSM(self, Y, dt, lagtime):
        """
        Runs TICA on the input feature coordinates, discretize the data
        using k-means clustering and estimates the markov model from
        the discrete trajectories.

        Parameters:
        -----------
        Y : Feautrized input data from inp.get_output()
        dt : int, trajectory timestep
        lagtime : float, choosen lag time from the implied timescales

        Returns:
        --------
        M : MaximumLikelihoodMSM, pyemma estimator object containing the
            Markov State Model and estimation information.
        """
        tica_coordinates, lag = self._tica(Y, dt, lagtime)
        dtrajs, centers, index_clusters = self._kmeans(tica_coordinates)
        self.M = msm.estimate_markov_model(dtrajs, lag)

        return self.M

class FindBindingModes(object):
    """
    FindBindingModes provides functions to analyze and determine the number of
    metastable modes from the resulting MSM data out of ConstructMSM() via
    spectral clustering from PCCA.

    Example:
    >>> data = bindingmodes.ConstructMSM(inp)
    >>> data.getMSM(data.Y, dt=8, lagtime=150)
    >>> fbm = bindingmodes.FindBindingModes(data)
    """
    def __init__(self, data):
        self.data = data
        #self.M = self.data.M
        #self.centers = self.data.centers
        #self.tica_coordinates = self.data.tica_coordinates
        self.silhouette_data = {}

    def _get_colors(self, n_clusters, cmap='gist_rainbow'):
        """Returns N colors according to the provided colormap.

        Parameters:
        -----------
        n_clusters : int, the number of metastable states/clusters
        cmap : str, specifying the matplotlib colormap to use.

        Returns:
        --------
        colors : list, containing the colors for each cluster.
        """
        colormap = plt.get_cmap(cmap)
        colors = [colormap(1.*i/n_clusters) for i in range(n_clusters)]
        return colors

    def _pcca(self, n_clusters):
        """Convenience function to run PCCA++ to compute a metastable
        decomposition of MSM states.

        Parameters:
        -----------
        n_clusters : int, the number of metastable states/clusters

        Returns:
        -----------
        pcca_sets : list of arrays, the metastable sets of active set states
                    within each metastable set using PCCA++. Each element is an
                    array with the microstate indexes.
        cluster_labels : numpy array, assignment of active set states to the
                        metastable sets from PCCA++.
        centers : numpy array, contains the coordinates of the cluster centers
                  from the active set states.
        """

        pcca = self.data.M.pcca(n_clusters)
        centers = self.data.centers[self.data.M.active_set]
        pcca_sets = pcca.metastable_sets
        cluster_labels = pcca.metastable_assignment

        return pcca_sets, cluster_labels, centers

    def _load_pcca_trajectories(self, pcca_outfiles):
        """Convenience function that loads the PCCA samples into
        an mdtraj.Trajectory object

        Parameters:
        -----------
        pcca_outfiles : list, strings specifying the trajectory files containing
                        samples obtained from PCCA++.

        Returns:
        --------
        pcca_traj : dict: `{int(cluster_index) : mdtraj.Trajectory}`
                    Each key corresponds to the cluster index and the value pair
                    references the loaded mdtraj.Trajectory.
        """

        pcca_traj = {}
        for n,f in enumerate(pcca_outfiles):
            pcca_traj[n] = md.load(f,top=self.data.topfile)
        return pcca_traj

    def _get_n_clusters(self, s_score_avg, s_score_std):
        """Helper function that selects the appropriate number of
        metastable states from the silhouette scores. Opts to use any number
        of states greater than 2 if it is within error (by ttest).

        Parameters:
        -----------
        s_score_avg : list, average silhouette scores for different numbers of metastable states.
        s_score_std : list, standard devations of silhouette scores for different number of metastable states.

        Returns:
        -------
        n_clusters : int, suggested number of clusters.
        """
        max0 = np.argmax(s_score_avg)
        #If cluster number is 2,
        # check if next highest is significantly different
        if max0 == 0:
            max1 = np.partition(s_score_avg, -2)[-2]
            max1 = int(np.argwhere(s_score_avg == max1))
            t, p = stats.ttest_ind_from_stats(s_score_avg[max0],
                                            s_score_std[max0],
                                            self.silhouette_data[self.range_n_clusters[max0]]['N_samples'],
                                            s_score_avg[max1],
                                            s_score_std[max1],
                                            self.silhouette_data[self.range_n_clusters[max1]]['N_samples'])
            if p < 0.05:
                print('\tInitial suggestion n_clusters = %s' % self.range_n_clusters[max0])
                print('\tChecking if n_clusters = %s is within error.' % self.range_n_clusters[max1])
                max0 = max1

        n_clusters = self.range_n_clusters[max0]
        print('Suggested number of cluster =', n_clusters)
        return n_clusters

    @staticmethod
    def scoreSilhouette(n_clusters, centers, cluster_labels):
        """Conveience function to calculate the silhouette scores.

        Parameters:
        -----------
        n_clusters : int, number of clusters
        centers : numpy array, contains the coordinates of the cluster centers
                  from the active set states.
        cluster_labels : numpy array, assignment of active set states to the
                        metastable sets from PCCA++.


        Returns:
        ---------
        silhouette_avg : float, average of silhouette coefficients for individual clusters.
        sample_silhouette_values : numpy array, silhouette score of individual frames.
        """
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(centers, cluster_labels)

        print("\tFor n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(centers, cluster_labels)
        return silhouette_avg, sample_silhouette_values

    def _plotSilhouette(self, ax1,ax2, n_clusters, pcca_sets, cluster_labels, centers,
                      silhouette_avg, sample_silhouette_values, cmap='gist_rainbow'):
        """
        Function to plot the silhouette scores next to the clusters resulting from
        PCCA++ projected onto the TICA coordinates.

        Parameters:
        -----------
        ax1, ax2 : matplotlib.axes.Axes
                   `ax1` contains the figure for the silhoutte plots.
                   `ax2` contains the figure for the clusters from PCCA++.
        n_clusters : int, number of clusters
        pcca_sets : list of arrays, the metastable sets of active set states
                    within each metastable set using PCCA++. Each element is an
                    array with the microstate indexes.
        cluster_labels : numpy array, assignment of active set states to the
                        metastable sets from PCCA++.
        silhouette_avg : float, average of silhouette coefficients for individual clusters.
        sample_silhouette_values : numpy array, silhouette score of individual frames.
        cmap : str, specifying the matplotlib colormap to use.
        """

        # The 1st subplot is the silhouette plot
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(centers) + (n_clusters + 1) * 10])
        y_lower = 10

        colors = self._get_colors(n_clusters, cmap)
        for i, color in enumerate(colors[:n_clusters]):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Avg Silhouette Score = %0.4f" % silhouette_avg, fontweight='bold')
        ax1.set_xlabel("Silhouette Scores")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

        # 2nd Plot showing the actual clusters formed
        plots.plot_free_energy(np.vstack(self.data.tica_coordinates)[:, 0], np.vstack(self.data.tica_coordinates)[:, 1],
                                ax=ax2, cmap=cmap, cbar=False)
        for i, color in enumerate(colors[:n_clusters]):
            ax2.scatter(centers[pcca_sets[i],0], centers[pcca_sets[i],1],
                       marker='X', c=color,s=200, edgecolors='black')
        ax2.set_title("%s State PCCA" %n_clusters, fontweight='bold')
        ax2.set_xlabel("TIC1")
        ax2.set_ylabel("TIC2")
        ax2.set_yticks([])
        ax2.set_xticks([])

    def _plotNClusters(self, ax, s_score_avg, n_clusters, outfname):
        """
        Plots a barplot of the silhouette scores.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        s_score_avg : list, average silhouette scores for different numbers of metastable states.
        s_score_std : list, standard devations of silhouette scores for different number of metastable states.
        outfname : str, specify the molecule name for the barplot title.
        """

        bar = ax.bar(self.range_n_clusters,s_score_avg, align='center', yerr=np.std(s_score_avg))
        bar[n_clusters-2].set_color('r')
        ax.set_title("Silhouette analysis for PCCA clustering on %s" % outfname.split('/')[-1],
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("# clusters")
        ax.set_ylabel("<Score>")

    def getNumBindingModes(self, range_n_clusters=range(2,10), outfname=None):
        """
        Determine the likely number of binding modes by scoring the silhouettes
        of a range of different cluster numbers. Plots the silhouettes alongside
        the PCCA++ assigned clusters to a file.

        Parameters:
        ------------
        range_n_clusters : list, range of different cluster numbers to try.
        outfname : str, specifying the file prefix for saving the plots.
                   Will append the suffix `<outfname>-tica_pcca.png`

        Returns:
        --------
        n_clusters : int, the optimimal number of clusters to use according to
                     the silhouette scores.
        """
        self.range_n_clusters = range_n_clusters

        fig = plt.figure(figsize=(16, len(range_n_clusters)*5), dpi=120)
        gs = gridspec.GridSpec(len(range_n_clusters)+1, 2)

        for idx,K in enumerate(range_n_clusters):
            pcca_sets, cluster_labels, centers = self._pcca(K)
            silhouette_avg, sample_silhouette_values = self.scoreSilhouette(K, centers, cluster_labels)

            ax1 = fig.add_subplot(gs[idx+1,0])
            ax2 = fig.add_subplot(gs[idx+1,1])
            self._plotSilhouette(ax1,ax2, K,pcca_sets,cluster_labels,centers,
                                silhouette_avg, sample_silhouette_values)

            self.silhouette_data[K] = {'PCCA_sets' : pcca_sets,
                            'Labels': cluster_labels,
                            'N_samples' : len(sample_silhouette_values),
                            'Values': sample_silhouette_values,
                            'AVG' : silhouette_avg,
                            'STD' : np.std(sample_silhouette_values)
                            }

        s_score_avg = [v['AVG'] for k,v in sorted(self.silhouette_data.items())]
        s_score_std = [v['STD'] for k,v in sorted(self.silhouette_data.items())]
        n_clusters = self._get_n_clusters(s_score_avg, s_score_std)

        ax3 = fig.add_subplot(gs[0,:])
        self._plotNClusters(ax3, s_score_avg, n_clusters, outfname)

        gs.tight_layout(fig)
        plt.savefig('{}-tica_pcca.png'.format(outfname), bbox_inches='tight')
        plt.close(fig)

        return n_clusters

    def _filter_leaders(self, leaders, leader_labels, cutoff=0.3):
        """
        Remove frames containing the representative
        binding modes of each clusters (i.e `leaders`) that are similar
        to other frames belonging to a separate cluster
        by some RMSD cutoff.

        Parameters:
        -----------
        leaders : mdtraj.Trajectory, frames containing cluster leaders
        leader_labels : list, cluster labels for each frame from leaders
        cutoff : float, cutoff RMSD distance (nm) for removing similar leader frames

        Returns:
        --------
        new_leaders : mdtraj.Trajectory, frames containing cluster leaders after filtering
        new_leader_labels : list, cluster index for each cluster leader frame.
        """
        #Get pairwise similarity between leaders
        prmsd = calcPairwiseRMSD(leaders)

        new_leader_labels = []
        keep_frames = []
        for idx, row in enumerate(prmsd):
            #Get indices of frames that are similar
            similar_frames = np.where( row < cutoff)[0]

            #Get the cluster labels for similar frames
            cluster_labels = [leader_labels[i] for i in similar_frames]

            #Check if frames are similar to frames in other clusters
            filter_mask = [ele == leader_labels[idx] for ele in cluster_labels]

            #Retain frames that are similar within clusters
            if all(filter_mask):
                keep_frames.append(idx)
                new_leader_labels.append(leader_labels[idx])

        new_leaders = leaders[keep_frames]
        #print(new_leaders, new_leader_labels)
        return new_leaders, new_leader_labels

    def _generate_leaders(self, pcca_traj, n_leaders_per_cluster=5):
        """
        Generates the mdtraj.Trajectory object containing the frames of the
        representative binding modes of each clusters (i.e `leaders`).

        Parameters:
        -----------
        pcca_traj : dict: `{int(cluster_index) : mdtraj.Trajectory}`
                    Each key corresponds to the cluster index and the value pair
                    references the loaded mdtraj.Trajectory.
        n_leaders_per_cluster : int, number of frames to sub-sample from the PCCA++ distributions.

        Returns:
        --------
        leaders : mdtraj.Trajectory, contains the sub-samples from the PCCA++ distributions
        leader_labels : list, cluster index for each cluster leader frame.
        """

        #WorkAround: Can't create a new 'boxless' traj, insert 1 frame from traj[0] into leaders.
        leaders = md.Trajectory(xyz=np.empty((0, pcca_traj[0].n_atoms, 3)),
                                topology=pcca_traj[0].topology)
        leaders = pcca_traj[0][0]
        leader_labels = []
        print('Extracting %s frames from each cluster' % n_leaders_per_cluster)
        for n,mtrj in pcca_traj.items():
            leaders = leaders.join(mtrj[np.random.choice(pcca_traj[n].n_frames, n_leaders_per_cluster)])
            leader_labels.extend([n] * n_leaders_per_cluster)
        #Cut out the inserted frame dummy frame from the workaround
        leaders = leaders[1:]

        return leaders, leader_labels

    def selectLeaders(self, pcca_outfiles, n_clusters=4, n_leaders_per_cluster=5,
                    cutoff=0.3, max_iter=100, outfname=None):
        """
        Selects the frames of representative binding modes from each cluster
        assigned from PCCA++, after filtering.

        Parameters:
        -----------
        pcca_outfiles : list, strings specifying the trajectory files containing
                        samples obtained from PCCA++.
        n_clusters : int, the optimimal number of clusters to use according to
                     the silhouette scores.
        n_leaders_per_cluster : int, number of frames to sub-sample from the PCCA++ distributions.
        cutoff : float, cutoff RMSD distance (nm) for removing similar leader frames.
        max_iter : maximum number of iterations for filtering the cluster leaders.
        outfname : str, specifying the file prefix for saving the cluster leaders.
                   Will append the suffix `<outfname>-leaders.pdb`
        Returns:
        -------
        new_leaders : mdtraj.Trajectory, frames containing cluster leaders after filtering
        new_leader_labels : list, cluster index for each cluster leader frame.
        """

        pcca_traj = self. _load_pcca_trajectories(pcca_outfiles)
        leaders, leader_labels = self._generate_leaders(pcca_traj, n_leaders_per_cluster)
        new_leaders, new_leader_labels = self._filter_leaders(leaders, leader_labels, cutoff)

        for i in range(max_iter):
            n_clusters = len(np.unique(leader_labels))
            k_clusters, k_count = np.unique(new_leader_labels, return_counts=True)
            k_clusters = len(k_clusters)
            if k_clusters != n_clusters:
                print('WARNING: '
                      'Filtering resulted in {k} clusters < original {n} clusters. '
                      'Re-drawing from PCCA samples'.format(k=k_clusters, n=n_clusters))
                leaders, leader_labels = self._generate_leaders(pcca_traj, n_leaders_per_cluster)
                new_leaders, new_leader_labels = self._filter_leaders(leaders, leader_labels, cutoff)
            elif any(k_count == 1):
                indices = np.where( k_count == 1)[0]
                print('WARNING: '
                      'Clusters {indices} contain only 1 sample. '
                      'Re-drawing from PCCA samples'.format(indices=indices))
                leaders, leader_labels = self._generate_leaders(pcca_traj, n_leaders_per_cluster)
                new_leaders, new_leader_labels = self._filter_leaders(leaders, leader_labels, cutoff)
            elif i == max_iter-1:
                raise Exception('ERROR: Reached maximum number of iterations. '
                      'Clusters may be too similar. Try lowering cutoff or n_clusters.')
            else:
                leader_difference = len(leader_labels) - len(new_leader_labels)
                print('Filtering removed %s frames' % (leader_difference))
                break

        with open('%s-leaders.txt' % outfname, 'w') as lead_txt:
            lead_txt.write('%s' % new_leader_labels)
        new_leaders.save('%s-leaders.pdb' %outfname)

        return new_leaders, new_leader_labels

    def savePCCASamples(self, inp, n_clusters, outfname, n_samples=100):
        """
        Draws samples from each of the clusters according to the PCCA++ membership
        probabilities.

        Parameters:
        -----------
        inp: pyemma.coordinates.data.feature_reader.FeatureReader
        n_clusters : int, the optimimal number of clusters to use according to
                     the silhouette scores.
        outfname : str, specifying the file prefix for saving the PCCA++ samples.
                   Will append the suffix `<outfname>-pcca<cluster_index>_samples.dcd`
        n_samples : number of samples to draw from the PCCA++ distributions.

        Returns:
        --------
        pcca_outfiles : list, strings specifying the trajectory files containing
                        samples obtained from PCCA++.
        """

        self.data.M.pcca(n_clusters)
        pcca_dist = self.data.M.metastable_distributions
        pcca_samples = self.data.M.sample_by_distributions(pcca_dist, n_samples)
        outfiles = []
        for N in range(n_clusters):
            outfiles.append('%s-pcca%s_samples.dcd' % (outfname,N))
        pcca_outfiles = coor.save_trajs(inp, pcca_samples, outfiles=outfiles)
        print('Storing %s PCCA samples each to: \n\t%s' % (n_samples, '\n\t'.join(pcca_outfiles)))

        return pcca_outfiles

class BindingModeOccupancy(object):
    """
    BindingModeOccupancy provides functionality to calculate the occupancy
    of each defined binding mode, determined from FindBindingModes().
    """

    def __init__(self, trajfile, acc_data, sel='resname LIG and not type H'):
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
        self.traj, self.jobid, self.acc_time = self.loadTrajectory(trajfile, acc_data)
        self.lig_atoms = self.traj.top.select('resname LIG and not type H')
        print('\t Analyzing:', self.filename)


    def loadTrajectory(self,trajfile, acc_data):
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
        jobid = trajfile.split('-')[1]
        #Every iteration stores 5 frames
        acc_time = acc_data[jobid]
        traj = md.load(trajfile, top=topfile)
        return traj, jobid, acc_time

    def calcOccupancy(self, leaders, leader_labels, outfname):
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
        dist = rmsdNP(self.traj, self.traj[0], idx=np.asarray(self.lig_atoms))
        labels = self.assignFramesFromPCCALeaders(self.traj, leaders, leader_labels)
        self.plotPoseSampling(distances=dist, cluster_labels=labels,
                                n_clusters=n_clusters,
                               time_conversion=0.0352,
                               acc_it=self.acc_time,
                               title='BM Occupancies %s-%s' %(self.molid, self.jobid))
        plt.savefig('%s-%s-occ.png' %(outfname, self.jobid), dpi=300)

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
            rmsds = rmsdNP(leaders,frame,atom_indices)
            cluster_labels.append(leader_labels[np.argmin(rmsds)])
            drawProgressBar(i/(traj.n_frames-1))
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


def animatePoseSampling(distances,cluster_labels,title='Simulation', outfname='scat_rmsd',fps=30, interval=50, size=50, n_clusters=4, xlim=0, ylim=0, acc_it=[],cmap='gist_rainbow'):
    N = len(cluster_labels)
    fig = plt.figure(3,figsize=(8, 6), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111)
    #plt.figure(figsize=(8, 6), dpi=300,tight_layout=True)
    #f, ax = plt.subplots(111)
    time = [0.0042*t for t in range(N)]
    x1 = np.asarray(time)
    y1 = np.asarray(10.0*distances)
    x2 = time
    y2 = list(10.0*distances)

    if not xlim: xlim = np.max(x1)
    if not ylim: ylim = np.max(y1)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"RMSD $\AA$")
    ax.set_xlim([0,xlim])
    ax.set_ylim([0,ylim])
    if acc_it:
        for it in acc_it:
            ax.axvline(x=it, color='k', linestyle='--')

    NUM_COLORS = n_clusters
    cm = plt.get_cmap(cmap)
    colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    colr_list = []
    #cmap = get_cmap(n_clusters*2)
    for x in cluster_labels:
        colr_list.append(colors[x])

    plots = [ax.scatter([], [],s=size, c=colr_list),
             ax.plot([], [], 'k-', linewidth=0.25,animated=True)[0]]
    if acc_it:
        for it in acc_it:
            ax.axvline(x=it, color='k', linestyle='--', linewidth=0.5)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='NLM'), bitrate=-1)

    def init():
        for idx,plot in enumerate(plots):
            if idx == 0:
                plot.set_offsets([])

            elif idx ==1:
                plot.set_data([],[])
        return plots

    def animate(i):
        data = np.hstack((x1[:i,np.newaxis], y1[:i, np.newaxis]))
        plots[0].set_offsets(data)
        plots[1].set_data(x2[:i],y2[:i])
        fig.suptitle('\n%s Time = %0.1fns' %(title,time[i]))
        drawProgressBar(i/(N-1))
        return plots

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x1),
                                   interval=50, blit=False, repeat=False)
    anim.save(outfname+'.mp4',writer=writer)
    HTML(anim.to_html5_video())

def check_bound_ligand(traj, cutoff=0.8, query_sel='resname LIG'):
    """
    Helper function to check if the ligand is still bound throughout the trajectory.

    Parameters:
    -----------
    traj : mdtraj.Trajectory
    cutoff : cutoff distance in nanometers
    query_sel : ligand selection string (Def: 'resname LIG')

    Returns:
    --------
    bound : boolean, True = ligand remains bound or False = ligand unbinds
    """
    matches = md.compute_neighbors(traj, cutoff,
                     query_indices=traj.top.select(query_sel),
                    haystack_indices=traj.top.select('protein'))
    bound = True
    for i, m in enumerate(matches):
        if not len(m):
            bound = False
    return bound


def convertIterToTime(jsonfile, time_per_iter=0):
    """
    Helper function that reads the json file containing the metadata of
    each BLUES simulation and converts the accepted move iteration number to time.

    Paramters:
    ----------
    jsonfile : str, filepath for json file containing the BLUES simualtion metadata.
    time_per_iter : float, conversion factor for converting the iteration number to time.
                    Ex. If every iteration stores 5 frames, every frame stores 0.0352ns
                    time_per_iter = 5 * 0.0352

    Returns:
    --------
    acc_data : dict, nested dictionary of BLUES metadata `{ molid : {jobid : list }}`
    """

    acc_data = {}
    fpath = str(jsonfile)
    molid = fpath.split('/')[-1].split('-')[0]

    with open(fpath, 'r') as jsf:
        data = json.load(jsf)
        if time_per_iter:
            data = {k: sorted(list(map(lambda x: float(x)*time_per_iter, v))) for k,v in data.items() if v != []}
        else:
            data = {k: sorted(list(map(int,v))) for k,v in data.items() if v != []}
        acc_data[molid] = data
    return acc_data

def get_cmap(N):
    ''' Returns a function that maps each index in 0, 1, ...
        N-1 to a distinct RGB color.'''
    color_norm  = mcolors.Normalize(vmin=0.0, vmax=N-1.0)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap='gist_rainbow')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def drawProgressBar(percent, barLen = 20):
    """Prints the progress of a `for` loop."""
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

def rmsdNP(traj, ref, idx=list()):
    """Computes the RMSD of a given mdtraj.Trajectory object
    *without alignment* to a reference frame."""
    return np.sqrt(3*np.sum(np.square(traj.xyz[:,idx,:] - ref.xyz[:,idx,:]),
                          axis=(1, 2))/idx.shape[0])

def calcPairwiseRMSD(traj, sel='resname LIG and not type H'):
    """Computes the pairwise RMSD of the frames within a given mdtraj.Trajectory object."""
    #print("Calculating Pairwise RMSD for selection: '%s'" % selection)
    #Compute pairwise RMSD using NumPy, which does NOT center the ligand atoms
    distances = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        distances[i] = rmsdNP(traj, traj[i],traj.top.select(sel))
        #drawProgressBar(i/(traj.n_frames-1))
    #print(np.shape(distances))
    return distances
