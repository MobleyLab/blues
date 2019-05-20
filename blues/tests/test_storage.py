import pytest
import parmed
import fnmatch
import numpy
import logging
from openmmtools.cache import ContextCache
from openmmtools.states import ThermodynamicState
from blues.systemfactories import *
from simtk.openmm import app
from simtk import unit
from blues.storage import *


def test_add_logging_level():
    print("Testing adding logger level")
    addLoggingLevel('TRACE', logging.DEBUG - 5)
    assert True == hasattr(logging, 'TRACE')


def test_init_logger():
    print('Testing logger initialization')
    logger = logging.getLogger(__name__)
    level = logger.getEffectiveLevel()
    logger = init_logger(logger, level=logging.INFO, stream=True)
    new_level = logger.getEffectiveLevel()
    assert level != new_level


def test_netcdf4storage():
    nc_storage = NetCDF4Storage('test.nc')
    assert True == hasattr(nc_storage, 'describeNextReport')
    assert True == hasattr(nc_storage, 'report')


def test_statedatastorage():
    state_storage = BLUESStateDataStorage('test.log')
    assert True == hasattr(state_storage, 'describeNextReport')
    assert True == hasattr(state_storage, 'report')
