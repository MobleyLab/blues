import unittest, parmed
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from simtk.openmm import app
from blues.moves import SideChainMove
from blues.moves import MoveEngine
from openmmtools import testsystems
import simtk.unit as unit
import numpy as np
from unittest import skipUnless

try:
    import openeye.oechem as oechem
    if not oechem.OEChemIsLicensed():
        raise ImportError("Need License for OEChem! SideChainMove class will be unavailable.")
    try:
        import oeommtools.utils as oeommtools
    except ImportError:
        raise ImportError('Could not import oeommtools. SideChainMove class will be unavailable.')
    has_openeye = True
except ImportError:
    has_openeye = False
    print('Could not import openeye-toolkits. SideChainMove class will be unavailable.')


@skipUnless(has_openeye, 'Cannot test SideChainMove without openeye-toolkits and oeommtools.')
class SideChainTesterDunbrak(unittest.TestCase):
    """
    Test the SmartDartMove.move() function.
    """

    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues', 'tests/data/vacDivaline.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/vacDivaline.inpcrd')
        self.struct = parmed.load_file(prmtop, xyz=inpcrd)

        self.sidechain = SideChainMove(self.struct, [1], bias_range=15.0)
        self.engine = MoveEngine(self.sidechain)
        self.engine.selectMove()

        self.system_cfg = {'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}
        self.systems = SystemFactory(self.struct, self.sidechain.atom_indices, self.system_cfg)

        self.cfg = {
            'dt': 0.002 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 300 * unit.kelvin,
            'nIter': 1,
            'nstepsMD': 1,
            'nstepsNC': 4,
            'alchemical_functions': {
                'lambda_sterics':
                'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics':
                'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            }
        }

        self.simulations = SimulationFactory(self.systems, self.engine, self.cfg)

    def test_getRotBondAtoms(self):
        vals = [v for v in self.sidechain.rot_atoms[1]['chis'][1]['atms2mv']]
        assert len(vals) == 11
        #Ensure it selects 1 rotatable bond in Valine
        assert len(self.sidechain.rot_bonds) == 1

    def test_sidechain_move(self):
        #check to make sure the dart stays within the specified ranges during the simulation
        atom_indices = [v for v in self.sidechain.rot_atoms[1]['chis'][1]['atms2mv']]

        for i in range(20):
            before_context = self.simulations.ncmc.context
            dobefore = self.sidechain.beforeMove(before_context)
            before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
                asNumpy=True)[atom_indices, :]
            self.simulations.ncmc.context = self.engine.runEngine(self.simulations.ncmc.context)
            after_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
                asNumpy=True)[atom_indices, :]

            #Check that our system has performed the move correctly
            # Integrator must step for context to update positions
            # Remove the first two atoms in check as these are the anchor atoms and are not rotated.
            pos_compare = np.not_equal(before_move, after_move)[2:, :].all()
            positions = self.simulations.ncmc.context.getState(getPositions=True).getPositions(asNumpy=True)
            bond_check = False
            bonds_in_bins = []
            #check the current angle after the move and make sure it's within range of the dihedral bins specified
            for residx in self.sidechain.rot_atoms.keys():
                resname = self.sidechain.rot_atoms[residx]['res_name']
                for chi in self.sidechain.rot_atoms[residx]['chis']:
                    dihed_atoms = [self.sidechain.rot_atoms[residx]['chis'][chi]['dihed_atms']]
                    curr_angle = self.sidechain.getDihedral(positions,dihed_atoms)
                    bin_ct=0
                    for bin in self.sidechain.rot_atoms[residx]['chis'][chi]['bin_pref']:
                        bin_ct+=1
                        if self.sidechain.is_in_bin(curr_angle,bin):

                            bin_idx = bin_ct-1
                            bonds_in_bins.append([(self.sidechain.rot_atoms[residx]['chis'][chi]),curr_angle[0][0],bin_idx])
                            bond_check = True
                            break

            assert bond_check == True

            assert pos_compare

class SideChainTesterRandom(unittest.TestCase):
    """
    Test the SmartDartMove.move() function.
    """

    def setUp(self):
        # Obtain topologies/positions
        prmtop = utils.get_data_filename('blues', 'tests/data/vacDivaline.prmtop')
        inpcrd = utils.get_data_filename('blues', 'tests/data/vacDivaline.inpcrd')
        self.struct = parmed.load_file(prmtop, xyz=inpcrd)

        self.sidechain = SideChainMove(self.struct, [1])
        self.engine = MoveEngine(self.sidechain)
        self.engine.selectMove()

        self.system_cfg = {'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}
        self.systems = SystemFactory(self.struct, self.sidechain.atom_indices, self.system_cfg)

        self.cfg = {
            'dt': 0.002 * unit.picoseconds,
            'friction': 1 * 1 / unit.picoseconds,
            'temperature': 300 * unit.kelvin,
            'nIter': 1,
            'nstepsMD': 1,
            'nstepsNC': 4,
            'alchemical_functions': {
                'lambda_sterics':
                'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                'lambda_electrostatics':
                'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            }
        }

        self.simulations = SimulationFactory(self.systems, self.engine, self.cfg)

    def test_getRotBondAtoms(self):
        vals = [v for v in self.sidechain.rot_atoms[1]['chis'][1]['atms2mv']]
        assert len(vals) == 11
        #Ensure it selects 1 rotatable bond in Valine
        assert len(self.sidechain.rot_bonds) == 1

    def test_sidechain_move(self):
        atom_indices = [v for v in self.sidechain.rot_atoms[1]['chis'][1]['atms2mv']]
        before_context = self.simulations.ncmc.context
        dobefore = self.sidechain.beforeMove(before_context)
        before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[atom_indices, :]
        self.simulations.ncmc.context = self.engine.runEngine(self.simulations.ncmc.context)
        after_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(
            asNumpy=True)[atom_indices, :]

        #Check that our system has run dynamics
        # Integrator must step for context to update positions
        # Remove the first two atoms in check as these are the anchor atoms and are not rotated.
        pos_compare = np.not_equal(before_move, after_move)[2:, :].all()
        assert pos_compare

if __name__ == "__main__":
    unittest.main()
