from simtk import unit, openmm
from simtk.openmm import app
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState

import blues.utils as utils
import blues.ncmc as ncmc
import blues.ncmc_switching as ncmc_switching
from blues.smartdart import SmartDarting

import sys
import numpy as np
import mdtraj
from mdtraj.reporters import HDF5Reporter
from datetime import datetime
from optparse import OptionParser

class SimulationFactory(object):
    def __init__(self, structure, atom_indices):
        self.structure = structure
        self.atom_indices = atom_indices
        self.system = None
        self.alch_system = None
        self.md = None
        self.alch  = None
        self.nc  = None
            #Defines ncmc move eqns for lambda peturbation of sterics/electrostatics
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }

    def generateAlchSystem(self, system, atom_indices):
        # Generate Alchemical System
        factory = AbsoluteAlchemicalFactory(system, atom_indices,
                                            annihilate_sterics=True,
                                            annihilate_electrostatics=True)
        alch_system = factory.createPerturbedSystem()
        self.alch_system = alch_system
        return self.alch_system

    def generateSystem(self, structure, **opt):
        system = structure.createSystem(nonbondedMethod=eval("app.%s" % opt['nonbondedMethod']),
                            nonbondedCutoff=opt['nonbondedCutoff']*unit.angstroms,
                            constraints=eval("app.%s" % opt['constraints']),
                            flexibleConstraints=False)
        self.system = system
        return self.system

    def generateSimFromStruct(self, structure, system, ncmc=False, printfile=sys.stdout, **opt):
        if ncmc:
            integrator = ncmc_switching.NCMCVVAlchemicalIntegrator(opt['temperature']*unit.kelvin,
                                                       system,
                                                       self.functions,
                                                       nsteps=opt['nstepsNC'],
                                                       direction='insert',
                                                       timestep=0.001*unit.picoseconds,
                                                       steps_per_propagation=1)
        else:
            integrator = openmm.LangevinIntegrator(opt['temperature']*unit.kelvin,
                                                   opt['friction']/unit.picosecond,
                                                   opt['dt']*unit.picoseconds)

        if opt['platform'] is None:
            #Use the fastest available platform
            simulation = app.Simulation(structure.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName(opt['platform'])
            prop = dict(DeviceIndex='2') # For local testing with multi-GPU Mac.
            simulation = app.Simulation(structure.topology, system, integrator, platform, prop)

        # OpenMM platform information
        mmver = openmm.version.version
        mmplat = simulation.context.getPlatform()
        print('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()), file=printfile)

        if opt['verbose']:
            # Host information
            from platform import uname
            for k,v in uname()._asdict().items():
                print(k, ':', v, file=printfile)

            # Platform properties
            for prop in mmplat.getPropertyNames():
                val = mmplat.getPropertyValue(simulation.context, prop)
                print(prop, ':', val, file=printfile)

        # Set initial positions/velocities
        # Will get overwritten from saved State.
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(opt['temperature']*unit.kelvin)
        simulation.reporters.append(app.StateDataReporter(sys.stdout, separator="\t",
                                    reportInterval=opt['reporter_interval'],
                                    step=True, totalSteps=opt['nIter']*opt['nstepsMD'],
                                    time=True, speed=True, progress=True,
                                    elapsedTime=True, remainingTime=True))

        return simulation

    def createSimulationSet(self, opt):
        system = self.generateSystem(self.structure, **opt)
        alch_system = self.generateAlchSystem(self.system, self.atom_indices)

        sim_dict = {}
        self.md = self.generateSimFromStruct(self.structure, system, **opt)
        self.alch = self.generateSimFromStruct(self.structure, system,  **opt)
        self.nc = self.generateSimFromStruct(self.structure, alch_system, ncmc=True,  **opt)
