import sys, json
import mdtraj as md
import numpy as np
import matplotlib.mlab as mlab
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
from matplotlib import animation, rc

def centerTrajectory(traj, outfname, remove_solvent=True):
    """Take an mdtraj.Trajectory object and centers the system for visualization.
    By default, this will remove solvent molecules."""

    if remove_solvent:
        #Remove the solvent to shrink the filesize
        traj = traj.remove_solvent()
    #Recenter/impose periodicity to the system
    anchor = traj.top.guess_anchor_molecules()[0]
    imgd = traj.image_molecules(anchor_molecules=[anchor])
    #Let's align by the protein atoms in reference to the first frame
    prt_idx = imgd.top.select('protein')
    superposed = imgd.superpose(reference=imgd,frame=0,atom_indices=prt_idx)
    #Save out the aligned frames
    superposed.save_dcd(outfname+'-centered.dcd')
    superposed[0].save_pdb(outfname+'-centered.pdb')

def check_bound_ligand(trajfiles,topfile, cutoff=0.8,
                    sel='resname LIG', ref_sel='protein'):
    """
    Helper function to check if the ligand is still bound throughout the trajectories.

    Parameters:
    -----------
    trajfiles : list of strings, path for trajectory file
    topfile : str, path for topology file (pdb)
    cutoff : cutoff distance in nanometers
    query_sel : ligand selection string (Def: 'resname LIG')

    Returns:
    --------
    trajfiles : list of strings, path for trajectory file after removing bad trajectories.
    """

    print('Checking %s trajectories for unbound ligands' % len(trajfiles))
    errfile = open('unbound-lig-trajs.err', 'w')
    for idx,trj in enumerate(trajfiles):
        traj = md.load(trj, top=topfile)
        matches = md.compute_neighbors(traj, cutoff,
                         query_indices=traj.top.select(sel),
                        haystack_indices=traj.top.select(ref_sel))
        matches = list(map(len, matches))
        if 0 in matches:
            print('\t WARNING!!! Detected unbound ligand in:', trj)
            errfile.write('%s\n'%trj)
            trajfiles.remove(trj)

    errfile.close()
    return trajfiles

def convertIterToTime(jsonfile, molid, time_per_iter=0):
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
