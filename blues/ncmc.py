"""
ncmc.py: Provides the core simulation setup class objects SimulationFactory)
required to run the BLUES engine

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function
from openmmtools import alchemy
import numpy as np

from simtk import unit, openmm
from simtk.openmm import app

from blues import utils
from blues.ncmc_switching import NCMCVVAlchemicalIntegrator

import sys, parmed, math, copy
import numpy as np
import mdtraj
from mdtraj.reporters import HDF5Reporter



class SimulationFactory(object):
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run.
    Ex.
        from blues.ncmc import SimulationFactory
        sims = SimulationFactory(structure, model, **opt)
        sims.createSimulationSet()
    """
    def __init__(self, structure, model, **opt):
        """Requires a parmed.Structure of the entire system and the ncmc.Model
        object being perturbed.

        Options is expected to be a dict of values. Ex:
        nIter=5, nstepsNC=50, nstepsMD=10000,
        temperature=300, friction=1, dt=0.002,
        nonbondedMethod='PME', nonbondedCutoff=10, constraints='HBonds',
        trajectory_interval=1000, reporter_interval=1000, platform=None,
        verbose=False"""

        #Structure of entire system
        self.structure = structure
        #Atom indicies for model
        self.atom_indices = model.atom_indices

        self.system = None
        self.alch_system = None
        self.md = None
        self.alch  = None
        self.nc  = None

        self.opt = opt
    def generateAlchSystem(self, system, atom_indices):
        """Returns the OpenMM System for alchemical perturbations.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list
            Atom indicies of the model.
        """
        import logging
        logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)
        factory = alchemy.AbsoluteAlchemicalFactory()
        alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices)
        alch_system = factory.create_alchemical_system(system, alch_region)
        return alch_system

    def generateSystem(self, structure, nonbondedMethod='PME', nonbondedCutoff=10,
                       constraints='HBonds', **opt):
        """Returns the OpenMM System for the reference system.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        opt : optional parameters (i.e. cutoffs/constraints)
        """
        system = structure.createSystem(nonbondedMethod=eval("app.%s" % nonbondedMethod),
                            nonbondedCutoff=nonbondedCutoff*unit.angstroms,
                            constraints=eval("app.%s" % constraints) )
        return system

    def generateSimFromStruct(self, structure, system, nIter, nstepsNC, nstepsMD,
                             temperature=300, dt=0.002, friction=1,
                             reporter_interval=1000,
                             ncmc=False, platform=None,
                             verbose=False, printfile=sys.stdout,  **opt):
        """Used to generate the OpenMM Simulation objects given a ParmEd Structure.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system :
        opt : optional parameters (i.e. cutoffs/constraints)
        atom_indices : list
            Atom indicies of the model.
        """
        if ncmc:
            #During NCMC simulation, lambda parameters are controlled by function dict below
            # Keys correspond to parameter type (i.e 'lambda_sterics', 'lambda_electrostatics')
            # 'lambda' = step/totalsteps where step corresponds to current NCMC step,
            functions = { 'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                          'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }

            integrator = NCMCVVAlchemicalIntegrator(temperature*unit.kelvin,
                                                    system,
                                                    functions,
                                                    nsteps=nstepsNC,
                                                    direction='insert',
                                                    timestep=0.001*unit.picoseconds,
                                                    steps_per_propagation=1)
        else:
            integrator = openmm.LangevinIntegrator(temperature*unit.kelvin,
                                                   friction/unit.picosecond,
                                                   dt*unit.picoseconds)
        #TODO SIMPLIFY TO 1 LINE.
        #Specifying platform properties here used for local development.
        if platform is None:
            #Use the fastest available platform
            simulation = app.Simulation(structure.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName(platform)
            #prop = dict(DeviceIndex='2') # For local testing with multi-GPU Mac.
            simulation = app.Simulation(structure.topology, system, integrator, platform) #, prop)

        if verbose:
            # OpenMM platform information
            mmver = openmm.version.version
            mmplat = simulation.context.getPlatform()
            print('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()), file=printfile)

            # Host information
            # ._asdict() is incompatible with py2.7
            #from platform import uname
            #for k,v in uname()._asdict().items():
            #    print(k, ':', v, file=printfile)

            # Platform properties
            for prop in mmplat.getPropertyNames():
                val = mmplat.getPropertyValue(simulation.context, prop)
                print(prop, ':', val, file=printfile)

        # Set initial positions/velocities
        # Will get overwritten from saved State.
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)

        #TODO MOVE SIMULATION REPORTERS TO OWN FUNCTION.
        simulation.reporters.append(app.StateDataReporter(sys.stdout, separator="\t",
                                    reportInterval=reporter_interval,
                                    step=True, totalSteps=nIter*nstepsMD,
                                    time=True, speed=True, progress=True,
                                    elapsedTime=True, remainingTime=True))

        return simulation

    def createSimulationSet(self):
        """Function used to generate the 3 OpenMM Simulation objects."""
        self.system = self.generateSystem(self.structure, **self.opt)
        self.alch_system = self.generateAlchSystem(self.system, self.atom_indices)

        self.md = self.generateSimFromStruct(self.structure, self.system, **self.opt)
        self.alch = self.generateSimFromStruct(self.structure, self.system,  **self.opt)
        self.nc = self.generateSimFromStruct(self.structure, self.alch_system,
                                            ncmc=True, **self.opt)
