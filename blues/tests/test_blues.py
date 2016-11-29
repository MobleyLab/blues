import blues
from simtk import unit
from openmmtools import testsystems


# Super basic test for blues

def test_basic_ncmc():
    """Test package loads properly and NCMC sets up."""
    test_system = testsystems.AlanineDipeptideVacuum()
    test_temp = 300*unit.kelvins
    test_res_list=[1]
    ncmc_test = blues.ncmc.SimNCMC(temperature=test_temp, residueList=test_res_list)
    # Need to add a test here that something is true about the output -- this just checks that it runs, but it would be good to check that the output is valid somehow or something. Pytest likes to use assert, i.e.
    # assert ncmc_test == 2

def test_basic_integrator():
    """Test that integrator loads properly."""
    test_system = testsystems.AlanineDipeptideVacuum()
    test_temp = 300*unit.kelvins
    test_res_list=[1]
    integrator_test = blues.ncmc_switching.NCMCVVAlchemicalIntegrator(temperature=test_temp, system=test_system.system, functions=[])
    # Should add something checking the output...

def test_basic_smartdart():
    """Test that smart darting loads properly."""
    test_system = testsystems.AlanineDipeptideVacuum()
    test_temp = 300*unit.kelvins
    test_res_list=[1]
    smartdart_test = blues.smartdart.SmartDarting(temperature=test_temp, residueList=test_res_list)

