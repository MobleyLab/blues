import pyemma.coordinates as coor
from pyemma.msm import estimate_markov_model
import numpy as np
import matplotlib.pyplot as plt

class ConstructMSM(object):
    """
    ConstructMSM provides convenience functions for using PyEMMA.
    This class assists in selecting the appropriate lagtime
    through plots, TICA-transforms the input feature coordinates for
    k-means clustering to discretize the trajectories, and constructs the MSM
    for the purpose of identifying the metastable binding modes.

    Example:
    >>> from blues.analysis import msm
    >>> import pyemma.coordinates as coor
    >>> feat = coor.featurizer(topfiles[0])
    >>> lig_atoms = feat.select("resname LIG and not type H")
    >>> feat.add_selection(lig_atoms)
    >>> inp = coor.source(trajfiles, feat)
    >>> data = msm.ConstructMSM(inp)

    >>> dt = 8; lag_list = np.arange(1, 40,5)

    #To view the implied timescales plot using a range of lagtimes
    >>> data.plotImpliedTimescales(data.Y, dt, lag_list, outfname)

    #Use the plots to select the apprioriate lagtime and generate the MSM.
    >>> M = data.getMSM(data.Y, dt, lagtime=150)
    """

    def __init__(self, inp):
        """
        Initialize the class by loading the featurized data.

        inp : pyemma.coordinates.data.feature_reader.FeatureReader
        """
        print('number of trajectories = ',inp.number_of_trajectories())
        print('number of dimension = ',inp.dimension())
        self.inp = inp
        self.Y = inp.get_output()

    def getMSM(self, Y, dt, lagtime, k=None, fixed_seed=False):
        """
        *Primary function to call.*

        Runs TICA on the input feature coordinates, discretize the data
        using k-means clustering and estimates the markov model from
        the discrete trajectories.

        Parameters:
        -----------
        Y : Feautrized input data from inp.get_output()
        dt : int, trajectory timestep
        lagtime : float, choosen lag time from the implied timescales
        k : int, maximum number of cluster centers. When `k=None`, the number of
            data points used will be k = min(sqrt(N))
        fixed_seed : bool or (positive) integer – if set to True, the random seed gets fixed resulting in deterministic behavior;
            default is false. If an integer >= 0 is given, use this to initialize the random generator.

        Returns:
        --------
        M : MaximumLikelihoodMSM, pyemma estimator object containing the
            Markov State Model and estimation information.
        """
        tica_coordinates, lag = self._tica(Y, dt, lagtime)
        dtrajs, centers, index_clusters = self._kmeans(tica_coordinates,k,fixed_seed=fixed_seed)
        self.M = estimate_markov_model(dtrajs, lag)

        return self.M

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
        tica_obj = coor.tica(Y, lag=lag, kinetic_map=True, reversible=True);
        print('TICA dimension ', tica_obj.dimension())
        print(tica_obj.cumvar)
        self.lag = lag; self.dt = dt
        self.tica_coordinates = tica_obj.get_output()

        return self.tica_coordinates, self.lag

    def _kmeans(self, tica_coordinates, k=None, max_iter=100, fixed_seed=False):
        """Convenience function to discretize the data by performing
        k-means clustering.

        Parameters:
        -----------
        tica_coordinates : numpy array of shape [n_samples, n_features_new]
        k : int, maximum number of cluster centers. When `k=None`, the number of
            data points used will be k = min(sqrt(N))
        max_iter: maximum number of iterations before stopping clustering.
        fixed_seed : bool or (positive) integer – if set to True, the random seed gets fixed resulting in deterministic behavior;
            default is false. If an integer >= 0 is given, use this to initialize the random generator.

        Returns:
        --------
        dtrajs : list of arrays, each array contains the
                 trajectory frames assigned to the cluster centers indices.
        centers : numpy array, contains the coordinates of the cluster centers
        index_clusters : list of arrays, For each state, all trajectory and time indexes
                        where this cluster occurs. Each row consists of a pair [i,t],
                        where `i` is the index of the trajectory and `t` is the frame index.
        """
        cl = coor.cluster_kmeans(tica_coordinates, k, max_iter,fixed_seed=fixed_seed);
        dtrajs = cl.dtrajs
        self.dtrajs = dtrajs
        self.centers = cl.clustercenters
        self.index_clusters = cl.index_clusters

        return self.dtrajs, self.centers, self.index_clusters

    @staticmethod
    def plotImpliedTimescales(inp, dt, lag_list, outfname=None):
        """
        Plots the implied timescales for the trajectories using a
        list of lag times to try.

        inp : pyemma.coordinates.data.feature_reader.FeatureReader
            Featurized coordinate data.
        dt  : int(), timestep interval from trajectory frames
        lag_list : list of lag times to try
        outfname : str(), specifying the output filename. None displays plot.
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
