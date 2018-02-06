"""
example.py: Provides an example script to run BLUES and
benchmark the run on a given platform

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley

* Benchmarking related code adapted from:
https://github.com/pandegroup/openmm/blob/master/examples/benchmark.py
(Authors: Peter Eastman)
"""

from __future__ import print_function
from blues.mold import MolDart
from blues.moves import RandomLigandRotationMove
from blues.engine import MoveEngine
from blues import utils
from blues.simulation import Simulation, SimulationFactory
import parmed
from simtk import openmm
from optparse import OptionParser
import mdtraj as md
from simtk import unit

def runNCMC(platform_name):
    #Define some options
    opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
            'nIter' : 1, 'nstepsNC' : 2000, 'nstepsMD' : 500,
            'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
            'trajectory_interval' : 1000, 'reporter_interval' : 1000,
            'platform' : platform_name,
            #'verbose' : True,
            'outfname' : 't4-tol',
            'nprop':10,
            'freeze_distance' : 10.0,
            #'write_ncmc' : 1,
            #'ncmc_traj': True
            }

    #Generate the ParmEd Structure
    prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
    inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
    struct = parmed.load_file(prmtop, xyz=inpcrd)
    struct = parmed.load_file(prmtop, xyz='posA.pdb')

    #Define the 'model' object we are perturbing here.
    # Calculate particle masses of object to be moved
    traj = md.load(inpcrd, top=prmtop)
    fit_atoms = traj.top.select("resid 50 to 155 and name CA")
    fit_atoms = traj.top.select("protein")
    ligand = MolDart(structure=struct, resname='LIG',
                                      pdb_files=['posB.pdb', 'posA.pdb'],
                                      #pdb_files=['posA.pdb', 'posB.pdb'],
                                      fit_atoms=fit_atoms)
#    ligand = RandomLigandRotationMove(struct, 'LIG')
#    ligand.calculateProperties()

    # Initialize object that proposes moves.
    ligand_mover = MoveEngine(ligand)

    # Generate the MD, NCMC, ALCHEMICAL Simulation objects
    simulations = SimulationFactory(struct, ligand_mover, **opt)
    simulations.createSimulationSet()

    blues = Simulation(simulations, ligand_mover, **opt)
    traj_reporter = openmm.app.DCDReporter('traj'+'-nc{}.dcd'.format(1000), opt['trajectory_interval'])
    simulations.md.reporters.append(traj_reporter)
    blues.run(opt['nIter'])

parser = OptionParser()
parser.add_option('-f', '--force', action='store_true', default=False,
                  help='run BLUES example without GPU platform')
(options, args) = parser.parse_args()

platformNames = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]

if 'OpenCL' in platformNames:
    runNCMC('OpenCL')
elif 'CUDA' in platformNames:
    runNCMC('CUDA')
else:
    if options.force:
        runNCMC('CPU')
    else:
        print('WARNING: Could not find a valid CUDA/OpenCL platform. BLUES is not recommended on CPUs.')
        print("To run on CPU: 'python blues/example.py -f'")
