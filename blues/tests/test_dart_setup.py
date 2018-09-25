from blues import utils
import mdtraj as md
import unittest
import parmed
import tempfile
from mdtraj.formats.xyzfile import XYZTrajectoryFile
from mdtraj.utils import in_units_of
import chemcoord as cc
import copy
import numpy as np
from blues.moldart.darts import makeDartDict, checkDart

class DartSetupTester(unittest.TestCase):
    def __init__():
        setup()
    def setup(self):
        pdb_files = [ [utils.get_data_filename('blues', 'tests/data/posA.pdb')], [utils.get_data_filename('blues', 'tests/data/posB.pdb')]]
        prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')

        self.struct = parmed.load_file(prmtop, xyz=inpcrd)
        pdb_files = [ [utils.get_data_filename('blues', 'tests/data/posA.pdb')], [utils.get_data_filename('blues', 'tests/data/posB.pdb')]]

        traj = md.load(inpcrd, top=prmtop)
        fit_atoms = traj.top.select("resid 50 to 155 and name CA")
        fit_atoms = traj.top.select("protein")
        self.atom_indices = traj.top.select("resname LIG")

        self.binding_mode_traj = []
        #positions of only the ligand atoms
        self.binding_mode_pos = []
        #fit atoms are the atom indices which should be fit to to remove rot/trans changes
        self.fit_atoms = fit_atoms
        #chemcoord cartesian xyz of the ligand atoms
        self.internal_xyz = []
        #chemcoord internal zmatrix representation of the ligand atoms
        self.internal_zmat = []
        self.buildlist = None
        #ref traj is the reference md trajectory used for superposition
        self.ref_traj = None
        #sim ref corresponds to the simulation positions
        self.sim_ref = None
        #dictionary of how to break up darting regions
        self.darts = None
        #tracks how many times moves are attempted and when ligand is within darting regions
        self.moves_attempted = 0
        self.times_within_dart = 0
        #if a subset of waters are to be frozen the number is specified here

        #find pdb inputs inputs
        if len(pdb_files) <= 1:
            raise ValueError('Should specify at least two pdbs in pdb_files for darting to be beneficial')
        self.dart_groups = list(range(len(pdb_files)))
        #chemcoords reads in xyz files only, so we need to use mdtraj
        #to get the ligand coordinates in an xyz file
        with tempfile.NamedTemporaryFile(suffix='.xyz') as t:
            fname = t.name
            traj = md.load(pdb_files[0]).atom_slice(self.atom_indices)
            xtraj = XYZTrajectoryFile(filename=fname, mode='w')
            xtraj.write(xyz=in_units_of(traj.xyz, traj._distance_unit, xtraj.distance_unit),
                        types=[i.element.symbol for i in traj.top.atoms] )
            xtraj.close()
            xyz = cc.Cartesian.read_xyz(fname)
            self.buildlist = xyz.get_construction_table()


        #use the positions from the structure to be used as a reference for
        #superposition of the rest of poses
        with tempfile.NamedTemporaryFile(suffix='.pdb') as t:
            fname = t.name
            self.struct.save(fname, overwrite=True)
            struct_traj = md.load(fname)
        ref_traj = md.load(pdb_files[0])[0]
        num_atoms = ref_traj.n_atoms
        self.ref_traj = copy.deepcopy(struct_traj)
        #add the trajectory and xyz coordinates to a list
        for j, pdb_file in enumerate(pdb_files):
            traj = copy.deepcopy(self.ref_traj)
            pdb_traj = md.load(pdb_file)[0]
            traj.xyz[0][:num_atoms] = pdb_traj.xyz[0]
            traj.superpose(reference=ref_traj, atom_indices=fit_atoms,
                ref_atom_indices=fit_atoms
                )
            self.binding_mode_traj.append(copy.deepcopy(traj))
            #get internal representation
            self.internal_xyz.append(copy.deepcopy(xyz))
            #take the xyz coordinates from the poses and update
            #the chemcoords cartesian xyz class to match
            for index, entry in enumerate(['x', 'y', 'z']):
                for i in range(len(self.atom_indices)):
                    sel_atom = self.atom_indices[i]
                    self.internal_xyz[j]._frame.at[i, entry] = self.binding_mode_traj[j].xyz[0][:,index][sel_atom]*10
            self.internal_zmat.append(self.internal_xyz[j].get_zmat(construction_table=self.buildlist))
        #set the binding_mode_pos by taking the self.atom_indices indices
        self.binding_mode_pos = [np.asarray(atraj.xyz[0])[self.atom_indices]*10.0 for atraj in self.binding_mode_traj]
        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])
        self.sim_ref = copy.deepcopy(self.binding_mode_traj[0])
        self.darts = makeDartDict(self.internal_zmat, self.binding_mode_pos, self.buildlist)
        
        #if transition_matrix is None:
        #    self.transition_matrix = np.ones((len(pdb_files), len(pdb_files)))
        #    np.fill_diagonal(self.transition_matrix, 0)
        #else:
        #    self.transition_matrix = transition_matrix
        #self.transition_matrix = self._checkTransitionMatrix(self.transition_matrix, self.dart_groups)
        #row_sums = self.transition_matrix.sum(axis=1)
        #self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
        #if np.shape(self.transition_matrix) != (len(pdb_files), len(pdb_files)):
        #    raise ValueError('Transition matrix should be an nxn matrix, where n is the length of pdb_files')
    def test_makeDartDict(self):
        self.darts = makeDartDict(self.internal_zmat, self.binding_mode_pos, self.buildlist)
        print(self.darts)
    def test_stuff(self):
        self.setup()
        print(self.buildlist)
if __name__ == "__main__":
        unittest.main()

