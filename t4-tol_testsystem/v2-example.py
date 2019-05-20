import argparse
from simtk import unit, openmm
from openmmtools import cache, alchemy
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from openmmtools import storage

from blues import utils
import parmed
import logging
import os, sys, copy
import numpy as np
from blues import utils
from blues.ncmc import *
from blues.storage import *
from blues.systemfactories import *

finfo = np.finfo(np.float32)
rtol = finfo.precision
logger = logging.getLogger(__name__)
logger.setLevel(logi)
logging.getLogger("parmed").setLevel(logging.ERROR)
logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='Restart file name')
parser.add_argument('-j', '--jobname', type=str, help="store jobname")
parser.add_argument('-n', '--nIter', default=1, type=int, help="number of Iterations")
parser.add_argument('-s', '--nsteps', default=100, type=int, help="number of steps")
parser.add_argument('-r', '--reportInterval', default=100, type=int, help="reportInterval")
args = parser.parse_args()

# Define parameters
outfname = args.jobname
temperature = 300 * unit.kelvin
collision_rate = 1 / unit.picoseconds
timestep = 4.0 * unit.femtoseconds
n_steps = args.nsteps
reportInterval = args.reportInterval
nIter = args.nIter

context_cache = cache.ContextCache()
prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')  #TOL-parm
inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
tol = parmed.load_file(prmtop, xyz=inpcrd)
tol.system = tol.createSystem(nonbondedMethod=openmm.app.PME,
                              nonbondedCutoff=10 * unit.angstrom,
                              constraints=openmm.app.HBonds,
                              hydrogenMass=3.024 * unit.dalton,
                              rigidWater=True,
                              removeCMMotion=True,
                              flexibleConstraints=True,
                              splitDihedrals=False)

# Create our State objects
sampler_state = SamplerState(positions=tol.positions)
thermodynamic_state = ThermodynamicState(system=tol.system, temperature=temperature)

# Create our AlchemicalState
alchemical_atoms = utils.atomIndexfromTop('LIG', tol.topology)
toluene_alchemical_system = generateAlchSystem(tol.system, alchemical_atoms)
alchemical_state = alchemy.AlchemicalState.from_system(toluene_alchemical_system)
alch_thermodynamic_state = ThermodynamicState(system=toluene_alchemical_system, temperature=temperature)
alch_thermodynamic_state = CompoundThermodynamicState(alch_thermodynamic_state, composable_states=[alchemical_state])
alch_thermodynamic_state.topology = tol.topology

context, integrator = context_cache.get_context(thermodynamic_state)
utils.print_host_info(context)

nc_reporter = NetCDF4Storage(outfname + '_MD.nc', reportInterval)
logger = init_logger(logger, level=logging.INFO, outfname=outfname, stream=True)
state_reporter = BLUESStateDataStorage(logger,
                                       reportInterval,
                                       title='md',
                                       step=True,
                                       speed=True,
                                       progress=True,
                                       totalSteps=int(n_steps * nIter))

state_reporter1 = BLUESStateDataStorage(logger,
                                        reportInterval,
                                        title='ncmc',
                                        step=True,
                                        speed=True,
                                        progress=True,
                                        totalSteps=int(n_steps * nIter))

# Iniitialize our Move set
rot_move = RandomLigandRotationMove(timestep,
                                    n_steps,
                                    atom_subset=alchemical_atoms,
                                    context_cache=context_cache,
                                    reporters=[state_reporter1])
langevin_move = ReportLangevinDynamicsMove(timestep,
                                           collision_rate,
                                           n_steps,
                                           reassign_velocities=True,
                                           context_cache=context_cache,
                                           reporters=[nc_reporter, state_reporter])
sampler = BLUESSampler(atom_subset=alchemical_atoms,
                       thermodynamic_state=thermodynamic_state,
                       sampler_state=sampler_state,
                       ncmc_move=rot_move,
                       dynamics_move=langevin_move,
                       topology=tol.topology)
sampler.run(nIter)
