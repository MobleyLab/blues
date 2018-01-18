from analysis import bindingmodes
import mdtraj as md
import numpy as np
import os, glob, traceback
from optparse import OptionParser
import pyemma.coordinates as coor

def main(fpath, molid):
    prefix = os.path.join(fpath,molid,molid)
    print(prefix)
    trajfiles = glob.glob(prefix+'*-centered.dcd')
    topfiles = glob.glob(prefix+'*.pdb')

    #Pre-process trajectory files, check for unbound ligands
    #for trj in trajfiles:
    #    traj = md.load(trj, top=topfiles[0])
    #    if not bindingmodes.check_bound_ligand(traj):
    #        print('Detected unbound ligand in:', trj)
    #        with open('unbound-lig.err', 'a') as errfile:
    #            errfile.write(trj+'\n')
    #        trajfiles.remove(trj)

    #Select features to analyze in trajectory
    feat = coor.featurizer(topfiles[0])
    lig_atoms = feat.select("resname LIG and not type H")
    feat.add_selection(lig_atoms)
    inp = coor.source(trajfiles, feat)

    #Define some input parameters
    dt = 8 #Frames in MD taken every 8ps == 0.008 ns
    lag_list = np.arange(1, 40,5) #Give a range of lag times to try

    #Initailize object to assign trajectories to clusters
    data = bindingmodes.ConstructMSM(inp)

    #Select the apprioriate lagtime from the implied timescale plots
    #data.plotImpliedTimescales(data.Y, dt, lag_list, outfname=prefix)

    #Selecting a lagtime of 150ps, every BLUES iteration is 176ps
    lagtime = 150

    #Discretize the trajectories
    #Estimate our Markov model from the discrete trajectories
    data.getMSM(data.Y, dt, lagtime)

    #Analyze our assigned clusters by Silhouette method
    fbm = bindingmodes.FindBindingModes(data)
    #fbm = bindingmodes.FindBindingModes(data.M, data.centers, data.tica_coordinates)

    #Get the optimimal number of clusters by silhouette score
    n_clusters = fbm.getNumBindingModes(range_n_clusters=range(2,10), outfname=prefix)

    #Draw samples from the PCCA metastable distributions
    pcca_outfiles = fbm.savePCCASamples(inp, n_clusters, outfname=prefix)
    leaders, leader_labels = fbm.selectLeaders(pcca_outfiles, n_clusters, outfname=prefix)

    #Every iteration stores 5 frames, every frame stores 0.0352ns
    time_per_iter = 5 * 0.0352
    jsonfile = glob.glob('json/%s*.json' %molid)[0]
    acc_data = bindingmodes.convertIterToTime(jsonfile, time_per_iter)

    for t in trajfiles:
        bmo = bindingmodes.BindingModeOccupancy(t, acc_data[molid])
        bmo.calcOccupancy(leaders, leader_labels, outfname=prefix)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f','--file_path', dest='fpath', type='str',
                  help='parent directory of BLUES simluations')
    parser.add_option('-m','--molid', dest='molid', type='str',
                  help='molecule ID')
    (options, args) = parser.parse_args()

    fpath = options.fpath
    mol = options.molid

    try:
        print('\n### Analyzing', mol)
        main(fpath,mol)
    except Exception as e:
        print('\nERROR!!!', mol)
        with open('%s.err' %mol, 'w') as errfile:
            errfile.write('### %s \n' % mol)
            errmsg = traceback.format_exc()
            errfile.write(errmsg)
            print('\n'+str(errmsg))
