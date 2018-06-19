import unittest, parmed
from blues import utils
from blues.simulation import SystemFactory, SimulationFactory, BLUESSimulation
from simtk.openmm import app
from blues.moves import SideChainMove
from blues.engine import MoveEngine
from openmmtools import testsystems
import simtk.unit as unit
import numpy as np
from unittest import skipIf
from mdtraj.utils.delay_import import import_

try:
    oechem = utils.import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    HAVE_OE = True
except Exception as e:
    HAVE_OE = False
    openeye_exception_message = str(e)


@skipIf(not HAVE_OE, "Cannot test openeye module without OpenEye tools.\n" + openeye_exception_message)
class SideChainTester(unittest.TestCase):
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

        self.system_cfg = { 'nonbondedMethod': app.NoCutoff, 'constraints': app.HBonds}
        self.systems = SystemFactory(self.struct, self.sidechain.atom_indices, self.system_cfg)

        self.cfg = { 'dt' : 0.002 * unit.picoseconds,
                'friction' : 1 * 1/unit.picoseconds,
                'temperature' : 300 * unit.kelvin,
                'nIter': 1,
                'nstepsMD': 1,
                'nstepsNC': 4,
                'alchemical_functions' : {
                    'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                    'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
            }

        self.simulations = SimulationFactory(self.systems, self.engine, self.cfg)

    def test_getRotBondAtoms(self):
        vals = [v for v in self.sidechain.rot_atoms[1].values()][0]
        self.assertEqual(len(vals), 11)
        #Ensure it selects 1 rotatable bond in Valine
        self.assertEqual(len(self.sidechain.rot_bonds), 1)

    def test_sidechain_move(self):
        atom_indices = [v for v in self.sidechain.rot_atoms[1].values()][0]
        before_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(asNumpy=True)[atom_indices,:]
        self.simulations.ncmc.context = self.engine.runEngine(self.simulations.ncmc.context)
        after_move = self.simulations.ncmc.context.getState(getPositions=True).getPositions(asNumpy=True)[atom_indices,:]

        #Check that our system has run dynamics
        # Integrator must step for context to update positions
        # Remove the first two atoms in check as these are the anchor atoms and are not rotated.
        pos_compare = np.not_equal(before_move, after_move)[2:,:].all()
        self.assertTrue(pos_compare)

if __name__ == "__main__":
        unittest.main()
