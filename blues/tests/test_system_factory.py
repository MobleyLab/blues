import unittest, parmed, yaml
from blues import utils
from blues.simulation import SystemFactory
from simtk import openmm, unit
from simtk.openmm import app

class SystemFactoryTester(unittest.TestCase):
    """
    Test the SystemFactory class.
    """
    def setUp(self):
        # Load the waterbox with toluene into a structure.
        self.prmtop = utils.get_data_filename('blues', 'tests/data/TOL-parm.prmtop')
        self.inpcrd = utils.get_data_filename('blues', 'tests/data/TOL-parm.inpcrd')
        self.structure = parmed.load_file(self.prmtop, xyz=self.inpcrd)
        self.atom_indices = utils.atomIndexfromTop('LIG', self.structure.topology)

        self.system_cfg = { 'nonbondedMethod' : app.PME,
                       'nonbondedCutoff' : 8.0*unit.angstroms,
                       'constraints'  : app.HBonds }
        self.systems = SystemFactory(self.structure, self.atom_indices, self.system_cfg)


    def test_generate_systems(self):
        # Create the OpenMM system
        print('Creating OpenMM System')
        md_system = SystemFactory.generateSystem(self.structure, **self.system_cfg)

        # Check that we get an openmm.System
        self.assertIsInstance(md_system, openmm.System)
        # Check atoms in system is same in input parmed.Structure
        self.assertEqual(md_system.getNumParticles(), len(self.structure.atoms))

        # Create the OpenMM system
        print('Creating OpenMM Alchemical System')
        alch_system = SystemFactory.generateAlchSystem(md_system, self.atom_indices)

        # Check that we get an openmm.System
        self.assertIsInstance(alch_system, openmm.System)
        # Check atoms in system is same in input parmed.Structure
        self.assertEqual(alch_system.getNumParticles(), len(self.structure.atoms))

    def test_atom_selections(self):
        atom_indices = self.systems._amber_selection_to_atom_indices_(self.structure, ':LIG')

        print('Testing AMBER selection parser')
        self.assertIsInstance(atom_indices, list)
        self.assertEqual(len(atom_indices), len(self.atom_indices))

        print('Testing atoms from AMBER selection with parmed.Structure')
        atom_list = SystemFactory._print_atomlist_from_atom_indices_(self.structure, atom_indices)
        atom_selection = [ self.structure.atoms[i] for i in atom_indices]
        self.assertEqual(atom_selection, atom_list)

    def test_restrain_postions(self):
        print('Testing positional restraints')
        no_restr = self.systems.md.getForces()

        md_system_restr = SystemFactory.restrain_positions(self.structure, self.systems.md, ':LIG')
        restr = md_system_restr.getForces()

        #Check that forces have been added to the system.
        self.assertNotEqual(len(restr), len(no_restr))
        #Check that it has added the CustomExternalForce
        self.assertIsInstance(restr[-1], openmm.CustomExternalForce)

    def test_freeze_atoms(self):
        print('Testing freeze_atoms')
        masses = [self.systems.md.getParticleMass(i) for i in self.atom_indices]
        frzn_lig = SystemFactory.freeze_atoms(self.structure, self.systems.md, ':LIG')
        massless = [frzn_lig.getParticleMass(i) for i in self.atom_indices]
        #Check that masses have been zeroed
        self.assertNotEqual(massless,masses)

    def test_freeze_radius(self):
        print('Testing freeze_radius')
        freeze_cfg = { 'freeze_center' : ':LIG',
                       'freeze_solvent' : ':WAT,Cl-',
                       'freeze_distance' : 5.0 * unit.angstroms}
        frzn_sys = SystemFactory.freeze_radius(self.structure, self.systems.md,
                                              **freeze_cfg)

        selection = "({freeze_center}<:{freeze_distance._value})&!({freeze_solvent})".format(**freeze_cfg)
        site_idx = SystemFactory._amber_selection_to_atom_indices_(self.structure, selection)
        #Invert that selection to freeze everything but the binding site.
        freeze_idx = set(range(self.systems.md.getNumParticles())) - set(site_idx)

        #Check that the ligand has NOT been frozen
        lig_masses = [frzn_sys.getParticleMass(i) for i in self.atom_indices]
        self.assertNotEqual(lig_masses, 0)
        #Check that the binding site has NOT been frozen
        masses = [frzn_sys.getParticleMass(i) for i in site_idx]
        self.assertNotEqual(masses,0)
        #Check that the selection has been frozen
        massless = set([frzn_sys.getParticleMass(i)._value for i in freeze_idx])
        self.assertEqual(list(massless)[0],0)


if __name__ == '__main__':
        unittest.main()
