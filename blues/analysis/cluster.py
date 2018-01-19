import pyemma.coordinates as coor
from pyemma import plots
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import gridspec
from scipy import stats
from blues.analysis import tools

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
        self.silhouette_pcca = {}

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

    def _pcca(self, n_clusters, n_samples=100):
        """Convenience function to run PCCA++ to compute a metastable
        decomposition of MSM states.

        Parameters:
        -----------
        n_clusters : int, the number of metastable states/clusters
        n_samples : number of samples to draw from the PCCA++ distributions.

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
        pcca_dist = self.data.M.metastable_distributions
        pcca_samples = self.data.M.sample_by_distributions(pcca_dist, n_samples)
        return pcca_sets, cluster_labels, centers, pcca_dist, pcca_samples

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
            pcca_traj[n] = md.load(f,top=self.data.inp.topfile)
        return pcca_traj

    def _get_n_clusters(self, silhouette_pcca):
        """Helper function that selects the appropriate number of
        metastable states from the silhouette scores. Opts to use any number
        of states greater than 2 if it is within error (by ttest).

        Parameters:
        -----------
        silhouette_pcca : dict, dictionary containing silhouette scores of PCCA assignments

        Returns:
        -------
        n_clusters : int, suggested number of clusters.
        """

        s_score_avg = [v['AVG'] for k,v in sorted(silhouette_pcca.items())]
        s_score_std = [v['STD'] for k,v in sorted(silhouette_pcca.items())]
        range_n_clusters = [k for k in sorted(silhouette_pcca.keys())]

        max0 = np.argmax(s_score_avg)
        #If cluster number is 2,
        # check if next highest is significantly different
        if max0 == 0:
            max1 = np.partition(s_score_avg, -2)[-2]
            max1 = int(np.argwhere(s_score_avg == max1)[0])
            print('\tInitial suggestion n_clusters = %s' % range_n_clusters[max0])
            print('\tChecking if n_clusters = %s is within error.' % range_n_clusters[max1])
            t, p = stats.ttest_ind_from_stats(s_score_avg[max0],
                                            s_score_std[max0],
                                            silhouette_pcca[range_n_clusters[max0]]['N_samples'],
                                            s_score_avg[max1],
                                            s_score_std[max1],
                                            silhouette_pcca[range_n_clusters[max1]]['N_samples'])
            if p < 0.05:
                max0 = max1

        n_clusters = range_n_clusters[max0]
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
                                ax=ax2, cmap='nipy_spectral', cbar=False)
        for i, color in enumerate(colors[:n_clusters]):
            ax2.scatter(centers[pcca_sets[i],0], centers[pcca_sets[i],1],
                       marker='X', c=color,s=200, edgecolors='black')
        ax2.set_title("%s State PCCA" %n_clusters, fontweight='bold')
        ax2.set_xlabel("TIC1")
        ax2.set_ylabel("TIC2")
        ax2.set_yticks([])
        ax2.set_xticks([])

    def _plotNClusters(self, ax, silhouette_pcca, n_clusters, outfname):
        """
        Plots a barplot of the silhouette scores.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
        silhouette_pcca : dict, dictionary containing silhouette scores of PCCA assignments
        n_clusters : int, optimimal number of clusters according to silhouette scores
        outfname : str, specify the molecule name for the barplot title.
        """
        s_score_avg = [v['AVG'] for k,v in sorted(silhouette_pcca.items())]
        s_score_std = [v['STD'] for k,v in sorted(silhouette_pcca.items())]
        range_n_clusters = [k for k in sorted(silhouette_pcca.keys())]

        bar = ax.bar(range_n_clusters,s_score_avg, align='center', yerr=np.std(s_score_avg))
        bar[n_clusters-2].set_color('r')
        ax.set_title("Silhouette analysis for PCCA clustering on %s" % outfname.split('/')[-1],
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("# clusters")
        ax.set_ylabel("<Score>")

    def getNumBindingModes(self, range_n_clusters=range(2,10), n_samples=100, outfname=None):
        """
        Determine the likely number of binding modes by scoring the silhouettes
        of a range of different cluster numbers. Plots the silhouettes alongside
        the PCCA++ assigned clusters to a file.

        Parameters:
        ------------
        range_n_clusters : list, range of different cluster numbers to try.
        n_samples : int, number of samples to draw from PCCA distributions.
        outfname : str, specifying the file prefix for saving the plots.
                   Will append the suffix `<outfname>-tica_pcca.png`

        Returns:
        --------
        n_clusters : int, the optimimal number of clusters to use according to
                     the silhouette scores.
        """

        fig = plt.figure(figsize=(16, len(range_n_clusters)*5), dpi=120)
        gs = gridspec.GridSpec(len(range_n_clusters)+1, 2)

        silhouette_pcca = {}
        for idx,K in enumerate(range_n_clusters):
            pcca_sets, cluster_labels, centers, pcca_dist, pcca_samples = self._pcca(K, n_samples)
            silhouette_avg, sample_silhouette_values = self.scoreSilhouette(K, centers, cluster_labels)

            ax1 = fig.add_subplot(gs[idx+1,0])
            ax2 = fig.add_subplot(gs[idx+1,1])
            self._plotSilhouette(ax1,ax2, K,pcca_sets,cluster_labels,centers,
                                silhouette_avg, sample_silhouette_values)

            silhouette_pcca[K] = {'PCCA_sets' : pcca_sets,
                            'Centers' : centers,
                            'Distribution' : pcca_dist,
                            'Samples' : pcca_samples,
                            'Labels': cluster_labels,
                            'N_samples' : len(sample_silhouette_values),
                            'Values': sample_silhouette_values,
                            'AVG' : silhouette_avg,
                            'STD' : np.std(sample_silhouette_values)
                            }

        n_clusters = self._get_n_clusters(silhouette_pcca)

        ax3 = fig.add_subplot(gs[0,:])
        self._plotNClusters(ax3, silhouette_pcca, n_clusters, outfname)

        gs.tight_layout(fig)
        plt.savefig('{}-tica_pcca.png'.format(outfname), bbox_inches='tight')
        plt.close(fig)

        self.range_n_clusters = range_n_clusters
        self.silhouette_pcca = silhouette_pcca
        return n_clusters

    def savePCCASamples(self, n_clusters, outfname,inp=None, pcca_samples=None):
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

        Returns:
        --------
        pcca_outfiles : list, strings specifying the trajectory files containing
                        samples obtained from PCCA++.
        """

        if not inp: inp = self.data.inp
        if pcca_samples is None: pcca_samples = self.silhouette_pcca[n_clusters]['Samples']

        outfiles = ['%s-pcca%s_samples.dcd' % (outfname,N) for N in range(n_clusters)]
        pcca_outfiles = coor.save_trajs(inp, pcca_samples, outfiles=outfiles)
        print('Storing %s PCCA samples each to: \n\t%s' % (len(pcca_samples), '\n\t'.join(pcca_outfiles)))

        return pcca_outfiles

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
        prmsd = tools.calcPairwiseRMSD(leaders)

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
