import unittest
import numpy as np
import mdtraj as md
import pickle
from blues.analysis import cluster
from blues import utils
import shutil, tempfile
np.random.seed(0)

class FindBindingModesTester(unittest.TestCase):
    """
    Test the FindBindingModes class.
    """
    def setUp(self):
        self.n_clusters = 2
        self.n_samples = 10
        msm_pkl = utils.get_data_filename('blues.analysis', 'tests/data/t4-tol-msm.pkl')
        with open(msm_pkl, 'rb') as f:
            self.data = pickle.load(f)
        self.fbm = cluster.FindBindingModes(self.data)

        pcca_pkl = utils.get_data_filename('blues.analysis', 'tests/data/silhouette_pcca.pkl')
        with open(pcca_pkl, 'rb') as fpcca:
            self.silhouette_pcca= pickle.load(fpcca)

        self.test_dir = tempfile.mkdtemp()

    def test_pcca(self):
        pcca_sets, cluster_labels, centers, pcca_dist, pcca_samples = self.fbm._pcca(self.n_clusters, self.n_samples)
        #Check pcca states are equal to the given number of clusters
        self.assertIsInstance(pcca_sets, list)
        self.assertEqual(len(pcca_sets), self.n_clusters)

        #Check cluster labels are equal to the given number of clusters
        self.assertIsInstance(cluster_labels, np.ndarray)
        self.assertEqual(len(np.unique(cluster_labels)), self.n_clusters)

        #Check that the resulting pcca centers is less than or equal to the
        #initial centers from k-means clustering. (PCCA may delete disconnected points)
        self.assertIsInstance(centers, np.ndarray)
        self.assertLessEqual(len(centers), len(self.data.centers))

        #Check that PCCA probabilities is present for each center and sums to 1.
        self.assertIsInstance(pcca_dist, np.ndarray)
        self.assertSequenceEqual( list([self.n_clusters, len(centers)]), list(pcca_dist.shape))
        self.assertEqual(np.int(np.sum(pcca_dist)/self.n_clusters), 1)

        #Check that each cluster contains the given number of samples.
        self.assertIsInstance(pcca_samples, np.ndarray )
        samples = []
        samples.extend([self.n_samples] *self.n_clusters)
        self.assertSequenceEqual(list(map(len,pcca_samples)), samples)

    #def test_score_silhouette(self):
    #    centers = self.silhouette_pcca[self.n_clusters]['Centers']
    #    cluster_labels = self.silhouette_pcca[self.n_clusters]['Labels']

    #    silhouette_avg, sample_silhouette_values = self.fbm.scoreSilhouette(self.n_clusters, centers, cluster_labels)

        #Check silhouette scoring is same as values in pickled data.
    #    self.assertAlmostEqual(silhouette_avg, self.silhouette_pcca[self.n_clusters]['AVG'], places=5)
    #    self.assertSequenceEqual(list(sample_silhouette_values), list(self.silhouette_pcca[self.n_clusters]['Values']))

    def test_get_n_clusters(self):
        #Check method that gets cluster number by silhouette score matches expected value.
        n_clusters = self.fbm._get_n_clusters(self.silhouette_pcca)
        self.assertEqual(n_clusters, self.n_clusters)

    #def test_save_pcca(self):
        #kwargs = { 'n_clusters' : self.n_clusters,
        #           'outfname' : '%s/t4-tol' % self.test_dir,
        #           'inp' : self.fbm.data.inp,
        #           'pcca_samples' : self.silhouette_pcca[self.n_clusters]['Samples'] }

        #Check method that saves PCCA samples to trajectory file
        #pcca_outfiles = self.fbm.savePCCASamples(**kwargs)
        #self.assertEqual(len(pcca_outfiles), self.n_clusters)

    def test_select_leaders(self):
        pcca0 = utils.get_data_filename('blues.analysis', 'tests/data/t4-tol-pcca0_samples.dcd')
        pcca1 = utils.get_data_filename('blues.analysis', 'tests/data/t4-tol-pcca1_samples.dcd')
        pdbfile = utils.get_data_filename('blues.analysis', 'tests/data/run03-centered.pdb')
        pcca_outfiles = [pcca0, pcca1]
        leader_kwargs = { 'pcca_outfiles' : pcca_outfiles,
                          'topfile' : pdbfile,
                          'n_clusters' : self.n_clusters,
                          'n_leaders_per_cluster' : 3,
                          'cutoff' : 0.3,
                          'max_iter' : 10,
                          'outfname' : '%s/t4-tol' % self.test_dir}

        #Check method that filters the representative structures from PCCA
        leaders, leader_labels = self.fbm.selectLeaders(**leader_kwargs)
        self.assertEqual(len(np.unique(leader_labels)), self.n_clusters)
        self.assertIsInstance(leaders, md.Trajectory)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

if __name__ == "__main__":
        unittest.main()
