import unittest
import pyemma.coordinates as coor
from pyemma.msm import estimate_markov_model
import numpy as np
import pickle
from blues.analysis import msm, cluster
from blues import utils
np.random.seed(0)

class ConstructMSMTester(unittest.TestCase):
    """
    Test the ConstructMSM class.
    """
    def setUp(self):

        trajfiles = utils.get_data_filename('blues.analysis', 'tests/data/run03-centered.dcd')
        pdbfile = utils.get_data_filename('blues.analysis', 'tests/data/run03-centered.pdb')
        feat = coor.featurizer(pdbfile)
        atom_index = np.array([2634, 2635, 2636, 2637, 2638, 2639, 2640, 1605, 1622, 1638, 1658, 1675, 1692, 1700, 1714, 1728, 1735, 1751, 1761, 1768, 1788])
        paired_index = feat.pairs(atom_index)
        feat.add_distances(paired_index)
        inp = coor.source(trajfiles, feat)

        self.data = msm.ConstructMSM(inp)
        self.dt = 30
        self.lagtime = 200

    def test_getMSM(self):
        tica_coordinates, lag = self.data._tica(self.data.Y, self.dt, self.lagtime)
        #Check for array of tica coordinates
        self.assertIsInstance(tica_coordinates[0][0],np.ndarray)
        #Check lagtime gets converted to units of dt
        self.assertEqual(lag, np.int(self.lagtime/self.dt))

        #Check for appropriate number of centers when k is given.
        dtrajs, centers, index_clusters = self.data._kmeans(tica_coordinates, k=50)
        self.assertEqual(len(centers), 50)
        self.assertEqual(len(index_clusters), len(centers))

        dtrajs, centers, index_clusters = self.data._kmeans(tica_coordinates)
        #Check that all trajectory frames have been discretized.
        n_frames = self.data.inp.n_frames_total()
        self.assertEqual(len(dtrajs[0]),n_frames)
        #Check for the appropriate number of centers when None is given.
        k_clusters = np.int(np.sqrt(n_frames))

        self.assertEqual(len(centers), k_clusters)
        self.assertEqual(len(index_clusters), len(centers))

        #Check MSM still contains all trajectory frames
        M = estimate_markov_model(dtrajs, lag)
        self.assertEqual(len(M.dtrajs_full[0]), n_frames)

if __name__ == "__main__":
        unittest.main()
