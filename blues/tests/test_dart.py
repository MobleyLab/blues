from __future__ import print_function
from blues.moves import MolDartMove, MoveEngine
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from blues.settings import Settings
import parmed
import mdtraj as md
from simtk import unit
import simtk.openmm as mm
import numpy as np
import unittest
from simtk.openmm import app
import tempfile
from mdtraj.formats.xyzfile import XYZTrajectoryFile
from mdtraj.utils import in_units_of
import chemcoord as cc
import copy
from blues.moldart.darts import makeDartDict, checkDart



#from simtk.openmm.app import OBC2
#@unittest.skip(reason="no way of currently testing this")
class MolEdit(MolDartMove):
    def __init__(self, *args, **kwargs):
        super(MolEdit, self).__init__(*args, **kwargs)
    def _initializeSystem(self, system, integrator):
        new_sys, new_int = super(MolEdit, self).initializeSystem(system, integrator)
        force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", 5.0*unit.kilocalories_per_mole/unit.angstroms**2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")
        aatoms = []
        for i, atom_crd in enumerate(self.structure.positions):
            if self.structure.atoms[i].name in ('CA', 'C', 'N'):
                force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
                aatoms.append(i)
        new_sys.addForce(force)
        return new_sys, new_int

#@unittest.skip(reason="no way of currently testing this")
class DartTester(unittest.TestCase):
    """
    Tests that the ic dart move is reversible

    """
    #@unittest.skip(reason="no way of currently testing this")
    def setUp(self):
        yaml_cfg = """
            output_dir: .
            outfname: ala-dipep-vac
            Logger:
              level: info
              stream: True

            system:
              nonbonded: NoCutoff
              constraints: HBonds

            simulation:
              dt: 0.002 * picoseconds
              friction: 1 * 1/picoseconds
              temperature: 300 * kelvin
              nIter: 1
              nstepsMD: 10
              nstepsNC: 10

            md_reporters:
              stream:
                title: md
                reportInterval: 1
                totalSteps: 10 # nIter * nstepsMD
                step: True
                speed: True
                progress: True
                remainingTime: True
                currentIter : True
            ncmc_reporters:
              stream:
                title: ncmc
                reportInterval: 1
                totalSteps: 10 # Use nstepsNC
                step: True
                speed: True
                progress: True
                remainingTime: True
                protocolWork : True
                alchemicalLambda : True
                currentIter : True
        """
        yaml_cfg = Settings(yaml_cfg)
        cfg = yaml_cfg.asDict()

        #Define some options
        opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.0005,
                'nIter' : 4, 'nstepsNC' : 10, 'nstepsMD' : 10,
                'nonbondedMethod' : 'CutoffNonPeriodic', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
                'trajectory_interval' : 1000, 'reporter_interval' : 1000,
                'platform' : 'CPU',
                'outfname' : 't4-tol',
                'nprop':1,
                'prop_lambda':0.10,
                'implicitSolvent': None,
                'verbose' : False,
                }
        self.opt = opt
        prmtop = utils.get_data_filename('blues', 'tests/data/vacVA.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/VAn68.pdb')

        self.struct = parmed.load_file(prmtop, xyz=inpcrd)
        pdb_files = [ [utils.get_data_filename('blues', 'tests/data/VA68.pdb')], [utils.get_data_filename('blues', 'tests/data/VAn68.pdb')]]

        fit_atoms = [0, 4, 16, 18, 20, 26]
        system_cfg = { 'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}
        self.systems = SystemFactory(self.struct, list(range(29)), system_cfg)

        self.ligand = MolDartMove(structure=self.struct, resname=['ALA','VAL'],
                                          pdb_files=pdb_files,
                                          fit_atoms=fit_atoms,
                                          restraints=False,
                                          restrained_receptor_atoms=[622, 2592, 2425],
                                          rigid_move=False,
                                          rigid_darts='rigid_darts'
                                          )

        # Initialize object that proposes moves.
        self.engine = MoveEngine(self.ligand)
        # Generate the MD, NCMC, ALCHEMICAL Simulation objects
        simulations = SimulationFactory(self.systems, self.engine, cfg['simulation'],
                                    cfg['md_reporters'], cfg['ncmc_reporters'])
        self.blues = BLUESSimulation(simulations)


    #@unittest.skip(reason="no way of currently testing this")
    def test_dartreverse(self):

        #get context and set positions to end, see if get same positions as beginning
        begin_traj = md.load(utils.get_data_filename('blues', 'tests/data/dart_start.pdb' ))
        end_traj = md.load(utils.get_data_filename('blues', 'tests/data/dart_end.pdb'))

        end_pos = end_traj.openmm_positions(0)

        self.blues._md_sim.context.setPositions(end_pos)
        begin_compare = self.ligand.move(self.blues._md_sim.context).getState(getPositions=True).getPositions(asNumpy=True)
        #check that the reverse of the move gives the same positions
        assert np.allclose(begin_compare._value, begin_traj.openmm_positions(0)._value, rtol=1e-4, atol=1e-4)


    def test_checkTransitionMatrix(self):
        """Check that the transition matrix accounts for the change in the acceptance ratio correctly"""
        dart_group =[1,1,1]
        #should fail because not reversible
        matrix_1 = np.array([[1,0,1],
                            [1,0.5,1],
                            [1,1,1]])
        #should fail because negative value
        matrix_2 = np.array([[1,-1,1],
                            [1,0.5,1],
                            [1,1,1]])
        #should pass
        matrix_3 = np.array([[1,1,1],
                            [1,0.5,1],
                            [1,1,1]])



        self.assertRaises(ValueError, self.ligand._checkTransitionMatrix, matrix_1, dart_group)
        self.assertRaises(ValueError, self.ligand._checkTransitionMatrix, matrix_2, dart_group)
        self.ligand._checkTransitionMatrix(matrix_3, dart_group)

    #@unittest.skip(reason="no way of currently testing this")
    def test_transition_matrix(self):
        self.ligand.acceptance_ratio=1
        self.ligand.transition_matrix = np.array([[0, 1],[0.1,0.9]])
        begin_traj = md.load(utils.get_data_filename('blues', 'tests/data/VA68.pdb' ))
        self.blues._md_sim.context.setPositions(begin_traj.openmm_positions(0))
        self.ligand.move(self.blues._md_sim.context).getState(getPositions=True).getPositions(asNumpy=True)

        assert self.ligand.acceptance_ratio == 0.1

#@unittest.skip(reason="no way of currently testing this")
class BoreschRestraintTester(unittest.TestCase):
    """
    Tests that the ic dart move is reversible

    """
    def setUp(self):
        #Define some options
        yaml_cfg = """
            output_dir: .
            outfname: ala-dipep-vac
            Logger:
              level: info
              stream: True

            system:
              nonbonded: NoCutoff
              constraints: HBonds

            simulation:
              dt: 0.002 * picoseconds
              friction: 1 * 1/picoseconds
              temperature: 300 * kelvin
              nIter: 1
              nstepsMD: 10
              nstepsNC: 10

            md_reporters:
              stream:
                title: md
                reportInterval: 1
                totalSteps: 10 # nIter * nstepsMD
                step: True
                speed: True
                progress: True
                remainingTime: True
                currentIter : True
            ncmc_reporters:
              stream:
                title: ncmc
                reportInterval: 1
                totalSteps: 10 # Use nstepsNC
                step: True
                speed: True
                progress: True
                remainingTime: True
                protocolWork : True
                alchemicalLambda : True
                currentIter : True
        """
        yaml_cfg = Settings(yaml_cfg)
        cfg = yaml_cfg.asDict()
        prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')

        self.struct = parmed.load_file(prmtop, xyz=inpcrd)
        pdb_files = [ [utils.get_data_filename('blues', 'tests/data/posA.pdb')], [utils.get_data_filename('blues', 'tests/data/posB.pdb')]]

        traj = md.load(inpcrd, top=prmtop)
        fit_atoms = traj.top.select("resid 50 to 155 and name CA")
        fit_atoms = traj.top.select("protein")

        self.ligand = MolEdit(structure=self.struct, resname='LIG',
                                          pdb_files=pdb_files,
                                          fit_atoms=fit_atoms,
                                          restraints='boresch',
                                          #restraints=None,

                                          restrained_receptor_atoms=[1605, 1735, 1837],
                                          )
        system_cfg = { 'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}

        self.systems = SystemFactory(self.struct, self.ligand.atom_indices, system_cfg)

        # Initialize object that proposes moves.
        self.ligand_mover = MoveEngine(self.ligand)
        self.engine = MoveEngine(self.ligand)
        # Generate the MD, NCMC, ALCHEMICAL Simulation objects
        simulations = SimulationFactory(self.systems, self.engine, cfg['simulation'],
                                    cfg['md_reporters'], cfg['ncmc_reporters'])
        self.blues = BLUESSimulation(simulations)



    def test_restraints(self):
        forces = self.blues._ncmc_sim.system.getForces()
        n_boresch = np.sum([1 if isinstance(i, mm.CustomCompoundBondForce) else 0 for i in forces])

        assert n_boresch == 2

class DartSetupTester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DartSetupTester, self).__init__(*args, **kwargs)

    def setUp(self):
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

    def test_makeDartDict(self):
        """Tests if the dart made are expected"""
        self.darts = makeDartDict(self.internal_zmat, self.binding_mode_pos, self.buildlist)
        print('darts', self.darts)
        print('rotation', self.darts['rotation'])
        assert np.isclose(64.41, self.darts['rotation'][0])


if __name__ == "__main__":
        unittest.main()
