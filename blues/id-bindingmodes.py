from analysis import msm, utils, cluster, population
import os, glob, traceback, pickle
from argparse import ArgumentParser
import pyemma.coordinates as coor
import numpy as np

def main(fpath, molid, args):
    prefix = os.path.join(fpath,molid,molid)
    trajfiles = glob.glob(prefix+'*-centered.dcd')
    topfiles = glob.glob(prefix+'*-centered.pdb')
    jsonfile = os.path.join(prefix+'-acc_its.json')

    #Pre-process trajectory files, check for unbound ligands
    trajfiles = utils.check_bound_ligand(trajfiles,
                                        topfiles[0])

    #Select features to analyze in trajectory
    feat = coor.featurizer(topfiles[0])
    lig_atoms = feat.select("resname LIG and not type H")
    feat.add_selection(lig_atoms)
    inp = coor.source(trajfiles, feat)

    #Define some input parameters
    dt = 8 #Frames in MD taken every 8ps == 0.008 ns
    lag_list = np.arange(1, 40,5) #Give a range of lag times to try

    #Initailize object to assign trajectories to clusters
    data = msm.ConstructMSM(inp)

    #Select the apprioriate lagtime from the implied timescale plots
    #data.plotImpliedTimescales(data.Y, dt, lag_list, outfname=prefix)

    #Selecting a lagtime of 150ps, every BLUES iteration is 176ps
    lagtime = 150

    #Discretize the trajectories
    #Estimate our Markov model from the discrete trajectories
    data.getMSM(data.Y, dt, lagtime)

    #Analyze our assigned clusters by Silhouette method
    fbm = cluster.FindBindingModes(data)

    #Get the optimimal number of clusters by silhouette score
    n_clusters = fbm.getNumBindingModes(range_n_clusters=range(2,10), outfname=prefix)

    #Draw samples from the PCCA metastable distributions
    pcca_outfiles = fbm.savePCCASamples(n_clusters, outfname=prefix)
    leaders, leader_labels = fbm.selectLeaders(pcca_outfiles, n_clusters, outfname=prefix)

    #Every iteration stores 5 frames, every frame stores 0.0352ns
    time_per_iter = 5 * 0.0352
    acc_data = utils.convertIterToTime(jsonfile, molid, time_per_iter)

    for t in trajfiles:
        bmo = population.BindingModeOccupancy(t, acc_data[molid])
        bmo.calcOccupancy(leaders, leader_labels, outfname=prefix)


parser = ArgumentParser()
parser.add_argument('-f','--file_path', dest='fpath', type=str,
              help='parent directory of BLUES simluations')
parser.add_argument('-o','--output', dest='outfname', type=str, default="blues",
                  help='Filename for output DCD')
parser.add_argument('-m','--molid', dest='molid', type=str,
              help='molecule ID')
parser.add_argument('--show_progress_bars', action='store_true',default=False, dest='show_progress_bars')
args = parser.parse_args()

try:
    main(args.fpath, args.molid, args)
except Exception as e:
    print('\nERROR!!!', args.molid)
    with open('%s.err' %args.molid, 'w') as errfile:
        errfile.write('### %s \n' % args.molid)
        errmsg = traceback.format_exc()
        errfile.write(errmsg)
        print('\n'+str(errmsg))
