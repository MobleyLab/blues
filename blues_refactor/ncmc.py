"""
ncmc.py: Provides the Simulation class for running the NCMC simulation.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley

version: 0.0.2 (WIP-Refactor)
"""

from __future__ import print_function
import sys
from simtk.openmm.app import *
from simtk.openmm import *
from blues.ncmc_switching import *
import simtk.unit as unit
import mdtraj
import math
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np
from openmmtools import testsystems

import blues_refactor.utils as utils
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
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run.
    Ex.
        from blues.ncmc import SimulationFactory
        sims = SimulationFactory(structure, atom_indices, **opt)
        sims.createSimulationSet()
    """
    def __init__(self, structure=None, atom_indices=[], **opt):
        self.structure = structure
        self.atom_indices = atom_indices
        self.system = None
        self.alch_system = None
        self.md = None
        self.alch  = None
        self.nc  = None

        self.opt = opt
        #Defines ncmc move eqns for lambda peturbation of sterics/electrostatics
        self.functions = { 'lambda_sterics' : 'step(0.199999-lambda) + step(lambda-0.2)*step(0.8-lambda)*abs(lambda-0.5)*1/0.3 + step(lambda-0.800001)',
                           'lambda_electrostatics' : 'step(0.2-lambda)- 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }

    @staticmethod
    def generateAlchSystem(system, atom_indices):
        """Returns the OpenMM System for alchemical perturbations.

        Parameters
        ----------
        context : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list
            Atom indicies of the model.
        """
        factory = AbsoluteAlchemicalFactory(system, atom_indices,
                                            annihilate_sterics=True,
                                            annihilate_electrostatics=True)
        alch_system = factory.createPerturbedSystem()

        return alch_system

    @staticmethod
    def generateSystem(structure, nonbondedMethod='PME', nonbondedCutoff=10,
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
                            constraints=eval("app.%s" % constraints),
                            flexibleConstraints=False)
        #self.system = system
        return system

    @staticmethod
    def generateSimFromStruct(structure, system, functions, ncmc=False, printfile=sys.stdout, **opt):
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
            integrator = ncmc_switching.NCMCVVAlchemicalIntegrator(opt['temperature']*unit.kelvin,
                                                       system,
                                                       functions,
                                                       nsteps=opt['nstepsNC'],
                                                       direction='insert',
                                                       timestep=0.001*unit.picoseconds,
                                                       steps_per_propagation=1)
        else:
            integrator = openmm.LangevinIntegrator(opt['temperature']*unit.kelvin,
                                                   opt['friction']/unit.picosecond,
                                                   opt['dt']*unit.picoseconds)
        ###TODO SIMPLIFY TO 1 LINE.
        #Specifying platform properties here used for local development.
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

        ###TODO MOVE SIMULATION REPORTERS TO OWN FUNCTION.
        simulation.reporters.append(app.StateDataReporter(sys.stdout, separator="\t",
                                    reportInterval=opt['reporter_interval'],
                                    step=True, totalSteps=opt['nIter']*opt['nstepsMD'],
                                    time=True, speed=True, progress=True,
                                    elapsedTime=True, remainingTime=True))

        return simulation

    def createSimulationSet(self):
        """Function used to generate the 3 OpenMM Simulation objects."""
        self.system = self.generateSystem(self.structure, **self.opt)
        self.alch_system = self.generateAlchSystem(self.system, self.atom_indices)

        self.md = self.generateSimFromStruct(self.structure, system, **self.opt)
        self.alch = self.generateSimFromStruct(self.structure, system,  **self.opt)
        self.nc = self.generateSimFromStruct(self.structure, alch_system,
                                            self.functions, ncmc=True,  **self.opt)

class ModelProperties(object):
    """ModelProperties provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.

    Current methods calculate the object's atomic masses and center of masss.
    Calculating the object's center of mass will get the positions and total mass.
    Ex.
        import blues.ncmc as ncmc
        model = ncmc.ModelProperties(nc_sim, atom_indices)
        model.calculateCOM()
        print(model.center_of_mass)
        print(model.totalmass, model.masses, model.positions)
    """

    def __init__(self, nc_sim, atom_indices):
        """Initialize the model.

        Parameters
        ----------
        nc_sim : openmm.app.simulation.Simulation
            The OpenMM Simulation object corresponding to the NCMC simulation.
        atom_indices : list
            Atom indicies of the model.
        """
        self.nc_sim = nc_sim
        self.atom_indices = atom_indices

        self.totalmass = 0
        self.masses = []
        self.positions = None
        self.center_of_mass = None

    def getMasses(self, context, atom_indices):
        """Returns a list of masses of the atoms in the model.

        Parameters
        ----------
        context : openmm.openmm.Context
            The OpenMM Context corresponding to the NCMC simulation.
        atom_indices : list
            Atom indicies of the model.
        """
        masses = unit.Quantity(np.zeros([len(atom_indices),1],np.float32), unit.dalton)
        system = context.getSystem()
        for ele, idx in enumerate(atom_indices):
            masses[ele] = system.getParticleMass(idx)
        self.totalmass = masses.sum()
        self.masses = masses
        return self.masses

    def getTotalMass(self, masses):
        """Returns total mass of model.

        Parameters
        ----------
        masses : list
            List of atom masses of model
        """
        self.totalmass = self.masses.sum()
        return self.totalmass

    def getPositions(self, context, atom_indices):
        """Returns a numpy.array of atom positions of the model given the
        simulation Context.

        Parameters
        ----------
        context : openmm.openmm.Context
            The OpenMM Context corresponding to the NCMC simulation.
        atom_indices : list
            Atom indicies of the model.
        """
        state = context.getState(getPositions=True)
        coordinates = state.getPositions(asNumpy=True) / unit.nanometers
        positions = unit.Quantity( np.zeros([len(atom_indices),3],np.float32), unit.nanometers)
        for ele, idx in enumerate(atom_indices):
            positions[ele,:] = unit.Quantity(coordinates[idx], unit.nanometers)
        self.positions = positions
        return self.positions

    def calculateCOM(self):
        """Calculates the center of mass of the model."""
        context = self.nc_sim.context
        atom_indices = self.atom_indices
        #Update masses for current context
        masses = self.getMasses(context, atom_indices)
        totalmass = self.getTotalMass(masses)
        positions = self.getPositions(context, atom_indices)
        center_of_mass =  (masses / totalmass * positions).sum(0)
        self.center_of_mass = center_of_mass

class MoveProposal(object):
    """MoveProposal provides perturbation functions for the model during the NCMC
    simulation. Current supported methods: random rotation.

    Ex.
        from blues.ncmc import MoveProposal
        mover = MoveProposal(nc_sim, model, 'random_rotation', nstepsNC)

        #Get the dictionary of proposed moves
        mover.nc_move
    """
    def __init__(self, nc_sim, model, method, nstepsNC):
        """Initialize the MovePropsal object that contains functions to perturb
        the model in the NCMC simulation.

        Parameters
        ----------
        nc_sim : openmm.simulation
            The OpenMM Simulation object corresponding to the NCMC simulation.
        model : blues.ncmc.ModelProperties object
            The object to be perturbed in the NCMC simulation.
        method : str
            A string of the perturbation function name (i.e 'random_rotation')
        nstepsNC : int
            An integer value for the number of NCMC steps performed.
        """
        supported_methods = ['random_rotation']
        if method not in supported_methods:
            raise Exception("Method %s not implemented" % method)
        else:
            self.nc_sim = nc_sim
            self.model = model
            self.nc_move = { 'method' : None , 'step' : 0}
            self.setMove(method, nstepsNC)

    def random_rotation(self):
        """Function that performs a random rotation about the center of mass of
        the model.
        """
        atom_indices = self.model.atom_indices
        model_pos = self.model.positions
        com = self.model.com
        reduced_pos = model_pos - com

        # Store initial positions of entire system
        initial_positions = self.nc_sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        positions = copy.deepcopy(initial_positions)

        # Define random rotational move on the ligand
        rand_quat = mdtraj.utils.uniform_quaternion()
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)

        #multiply lig coordinates by rot matrix and add back COM translation from origin
        rot_move =  np.dot(reduced_pos, rand_rotation_matrix) * unit.nanometers + com

        # Update ligand positions in nc_sim
        for index, atomidx in enumerate(atom_indices):
            positions[atomidx] = rot_move[index]
        self.nc_sim.context.setPositions(positions)
        return self.nc_sim

    def setMove(self, method, step):
        """Returns the dictionary that defines the perturbation methods to be
        performed on the model object and the step number to perform it at."""
        self.nc_move['method']  = getattr(MoveProposal, method)
        self.nc_move['step'] = int(step) / 2 - 1
        return self.nc_move

class Simulation(object):
    """Simulation class provides the functions that perform the BLUES run.

    Ex.
        import blues.ncmc
        blues = ncmc.Simulation(sims, model, mover, **opt)
        blues.run()

    """
    def __init__(self, sims, model, mover, **opt):
        """Initialize the BLUES Simulation object.

        Parameters
        ----------
        sims : blues.ncmc.SimulationFactory object
            SimulationFactory Object which carries the 3 required
            OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.
        model : blues.ncmc.ModelProperties object
            ModelProperties object that represents the model to be perturbed
            in the NCMC simulation.
        mover : blues.ncmc.MoveProposal object
            MoveProposal object which contains the dict of moves performed
            in the NCMC simulation.
        """
        self.md_sim = sims.md
        self.alch_sim = sims.alch
        self.nc_context = sims.nc.context
        self.nc_integrator = sims.nc.context._integrator
        self.model = model
        self.nc_move = mover.nc_move

        self.accept = 0
        self.reject = 0
        self.accept_ratio = 0

        self.nIter = int(opt['nIter'])
        self.nstepsNC = int(opt['nstepsNC'])
        self.nstepsMD = int(opt['nstepsMD'])

        self.current_iter = 0
        self.current_stepNC = 0
        self.current_stepMD = 0

        self.current_state = { 'md'   : { 'state0' : {}, 'state1' : {} },
                               'nc'   : { 'state0' : {}, 'state1' : {} },
                               'alch' : { 'state0' : {}, 'state1' : {} }
                            }

        self.temperature = opt['temperature']
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * opt['temperature']
        beta = 1.0 / kT
        self.beta = beta

        self.work_keys = ['total_work', 'lambda', 'shadow_work',
                          'protocol_work', 'Eold', 'Enew','Epert']

        self.state_keys = { 'getPositions' : True,
                       'getVelocities' : True,
                       'getForces' : False,
                       'getEnergy' : True,
                       'getParameters': True,
                       'enforcePeriodicBox' : True}

    def setSimState(self, simkey, stateidx, stateinfo):
        """Stores the dict of Positions, Velocities, Potential/Kinetic energies
        of the state before and after a NCMC step or iteration.

        Parameters
        ----------
        simkey : str (key: 'md', 'nc', 'alch')
            Key corresponding to the simulation.
        stateidx : int (key: 'state0' or 'state1')
            Key corresponding to the state information being stored.
        stateinfo : dict
            Dictionary containing the State information.
        """
        self.current_state[simkey][stateidx] = stateinfo

    def setStateConditions(self):
        """Stores the dict of current state of the MD and NCMC simulations.
        Dict contains the Positions, Velocities, Potential/Kinetic Energies
        of the current state.
        Sets the NCMC simulation Positions/Velocities to
        the current state of the MD simulation.
        """
        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        nc_state0 = self.getStateInfo(self.nc_context, self.state_keys)
        self.nc_context.setPositions(md_state0['positions'])
        self.nc_context.setVelocities(md_state0['velocities'])
        self.setSimState('md', 'state0', md_state0)
        self.setSimState('nc', 'state0', nc_state0)

    def getStateInfo(self, context, parameters):
        """Function that gets the State information from the given context and
        list of parameters to queuey it with.
        Returns a dict of the data from the State.

        Parameters
        ----------
        context : openmm.Context
            Context of the OpenMM Simulation to query.
        parameters : list
            Default: [ positions, velocities, potential_energy, kinetic_energy ]
            A list that defines what information to get from the context State.
        """
        stateinfo = {}
        state  = context.getState(**parameters)
        stateinfo['iter'] = int(self.current_iter)
        stateinfo['positions'] =  state.getPositions(asNumpy=True)
        stateinfo['velocities'] = state.getVelocities(asNumpy=True)
        stateinfo['potential_energy'] = state.getPotentialEnergy()
        stateinfo['kinetic_energy'] = state.getKineticEnergy()
        return stateinfo

    def getWorkInfo(self, nc_integrator, parameters):
        """Function that obtains the work and energies from the NCMC integrator.

        Returns a dict of the specified parameters.

        Parameters
        ----------
        nc_integrator : openmm.Context.Integrator
            The integrator from the NCMC Context
        parameters : list
            list containing strings of the values to get from the integrator.
            Default : ['total_work', 'lambda', 'shadow_work',
                       'protocol_work', 'Eold', 'Enew','Epert']
        """
        workinfo = {}
        for param in parameters:
            workinfo['param'] = nc_integrator.getGlobalVariableByName(param)
        return workinfo

    def chooseMove(self):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.current_state['md']['state0']
        nc_state0 = self.current_state['nc']['state0']
        nc_state1 = self.current_state['nc']['state1']

        log_ncmc = self.nc_integrator.getLogAcceptanceProbability(self.nc_context)
        randnum =  math.log(np.random.random())

        ### Compute Alchemical Correction Term
        if not np.isnan(log_ncmc):
            self.alch_sim.context.setPositions(nc_state1['positions'])
            alch_state1 = self.getStateInfo(self.alch_sim.context, self.state_keys)
            self.setSimState('alch', 'state1', alch_state1)

            n1_PE = alch_state1['potential_energy'] - nc_state1['potential_energy']
            n_PE = md_state0['potential_energy'] - nc_state0['potential_energy']
            correction_factor = (-1.0/self.nc_integrator.kT)*( n1_PE - n_PE )
            #print('correction_factor', correction_factor)
            log_ncmc = log_ncmc + correction_factor

        if log_ncmc > randnum:
            self.accept += 1
            print('NCMC MOVE ACCEPTED: log_ncmc {} > randnum {}'.format(log_ncmc, randnum) )
            #print('accCounter', float(self.accept)/float(stepsdone+1), self.accept)
            self.md_sim.context.setPositions(nc_state1['positions'])
        else:
            self.reject += 1
            print('NCMC MOVE REJECTED: {} < {}'.format(log_ncmc, randnum) )
            #print('ncmc PE', newinfo['potential_energy'], 'old PE', md_PE0)
            self.nc_context.setPositions(md_state0['positions'])

        self.nc_integrator.reset()
        self.md_sim.context.setVelocitiesToTemperature(self.temperature)

    def simulateNCMC(self):
        """Function that performs the NCMC simulation."""
        for nc_step in range(self.nstepsNC):
            try:
                self.current_stepNC = int(nc_step)
                # Calculate Work/Energies Before Step
                work_initial = self.getWorkInfo(self.nc_integrator, self.work_keys)
                # Attempt NCMC Move
                if nc_step == self.nc_move['step']:
                    print('[Iter {}] Performing NCMC {} move'.format(
                    self.current_iter, self.nc_move['method'].__name__))
                    self.nc_move['method']
                # Do 1 NCMC step
                self.nc_integrator.step(1)
                # Calculate Work/Energies After Step.
                work_final = self.getWorkInfo(self.nc_integrator, self.work_keys)
            except Exception as e:
                print(e)
                break

        nc_state1 = self.getStateInfo(self.nc_context, self.state_keys)
        self.setSimState('nc', 'state1', nc_state1)

    def simulateMD(self):
        """Function that performs the MD simulation."""
        md_state0 = self.current_state['md']['state0']
        try:
            self.md_sim.step(self.nstepsMD)
            self.current_stepMD = self.md_sim.currentStep
        except Exception as e:
            print(e)
            stateinfo = self.getStateInfo(self.md_sim.context, self.state_keys)
            last_x, last_y = np.shape(md_state0['positions'])
            reshape = (np.reshape(md_state0['positions'], (1, last_x, last_y))).value_in_unit(unit.nanometers)
            print('potential energy before NCMC', md_state0['potential_energy'])
            print('kinetic energy before NCMC', md_state0['kinetic_energy'])

            last_top = mdtraj.Topology.from_openmm(self.md_sim.topology)
            broken_frame = mdtraj.Trajectory(xyz=reshape, topology=last_top)
            broken_frame.save_pdb('MD-blues_fail-iter{}_md{}.pdb'.format(self.current_iter, self.current_stepMD))
            exit()

        md_state1 = self.getStateInfo(self.md_sim.context, self.state_keys)
        self.setSimState('md', 'state1', md_state1)
        # Set NC poistions to last positions from MD
        self.nc_context.setPositions(md_state1['positions'])
        self.nc_context.setVelocities(md_state1['velocities'])

    def run(self):
        """Function that runs the BLUES engine that iterates of the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state.
        """
        #set inital conditions
        self.setStateConditions()
        for n in range(self.nIter):
            self.current_iter = int(n)
            self.setStateConditions()
            self.simulateNCMC()
            self.chooseMove()
            self.simulateMD()

        # END OF NITER
        self.accept_ratio = self.accept/float(self.nIter)
        print('Acceptance Ratio', self.accept_ratio)
        print('numsteps ', self.nstepsNC)
