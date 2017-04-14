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
import blues.ncmc_switching as ncmc_switching
from blues.smartdart import SmartDarting

import sys, parmed
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
    def __init__(self, structure, atom_indices=[], **opt):
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

    def generateAlchSystem(self, system, atom_indices):
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
                            constraints=eval("app.%s" % constraints))
        return system

    def generateSimFromStruct(self, structure, system, ncmc=False, platform=None,
                              verbose=False, printfile=sys.stdout, **opt):
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
                                                       self.functions,
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
        if platform is None:
            #Use the fastest available platform
            simulation = app.Simulation(structure.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName(platform)
            prop = dict(DeviceIndex='2') # For local testing with multi-GPU Mac.
            simulation = app.Simulation(structure.topology, system, integrator, platform, prop)

        if verbose:
            # OpenMM platform information
            mmver = openmm.version.version
            mmplat = simulation.context.getPlatform()
            print('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()), file=printfile)

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

        self.md = self.generateSimFromStruct(self.structure, self.system, **self.opt)
        self.alch = self.generateSimFromStruct(self.structure, self.system,  **self.opt)
        self.nc = self.generateSimFromStruct(self.structure, self.alch_system,
                                            ncmc=True, **self.opt)

class ModelProperties(object):
    """ModelProperties provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.

    Current methods calculate the object's atomic masses and center of masss.
    Calculating the object's center of mass will get the positions and total mass.
    Ex.
        from blues.ncmc import ModelProperties
        model = ModelProperties(structure, 'LIG')
        model.calculateProperties()

    Attributes
    ----------
    model.resname : string specifying the residue name of the ligand
    model.atom_indices : list of atom indicies of the ligand.
    model.structure : parmed.Structure of the ligand selected by resname.
    model.masses : list of particle masses of the ligand with units.
    model.totalmass : integer of the total mass of the ligand.
    model.center_of_mass : np.array of calculated center of mass of the ligand
    """

    def __init__(self, structure, resname='LIG'):
        """Initialize the model.

        Parameters
        ----------
        resname : str
            String specifying the resiue name of the ligand.
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        """
        self.resname = resname
        self.atom_indices = self.getAtomIndices(self.resname, structure)
        self.structure = structure[self.atom_indices]

        self.totalmass = 0
        self.masses = []
        self.center_of_mass = None
        self.positions = self.structure.positions

    def getAtomIndices(self, resname, structure):
        """
        Get atom indices of a ligand from ParmEd Structure.
        Arguments
        ---------
        resname : str
            String specifying the resiue name of the ligand.
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        Returns
        -------
        atom_indices : list of ints
            list of atoms in the coordinate file matching lig_resname
        """
        atom_indices = []
        topology = structure.topology
        for atom in topology.atoms():
            if str(resname) in atom.residue.name:
                atom_indices.append(atom.index)
        return atom_indices

    def getMasses(self, structure):
        """Returns a list of masses of the atoms in the model.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        """
        masses = unit.Quantity(np.zeros([len(structure.atoms),1],np.float32), unit.dalton)
        for idx in range(len(structure.atoms)):
            masses[idx] = structure.atoms[idx].mass * unit.dalton
        return masses

    def getTotalMass(self, masses):
        """Returns total mass of model.

        Parameters
        ----------
        masses : numpy.array
            np.array of particle masses
        """
        totalmass = masses.sum()
        return totalmass

    def getCenterOfMass(self, structure, masses):
        """Returns the calculate center of mass of the ligand as a np.array

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        masses : numpy.array
            np.array of particle masses
        """
        dim = structure.positions.unit
        coordinates = np.asarray(structure.positions._value, np.float32)
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * dim
        return center_of_mass

    def calculateProperties(self):
        """Function to quickly calculate available properties."""
        self.masses = self.getMasses(self.structure)
        self.totalmass = self.getTotalMass(self.masses)
        self.center_of_mass = self.getCenterOfMass(self.structure, self.masses)


class MoveProposal(object):
    """MoveProposal provides perturbation functions for the model during the NCMC
    simulation. Current supported methods: random rotation.

    Ex.
        from blues.ncmc import MoveProposal
        mover = MoveProposal(nc_sim, model, 'random_rotation', nstepsNC)

        #Get the dictionary of proposed moves
        mover.nc_move
    """
    def __init__(self, model, method, nstepsNC):
        """Initialize the MovePropsal object that contains functions to perturb
        the model in the NCMC simulation.

        Parameters
        ----------
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
            self.nc_move = { 'method' : None , 'step' : 0}
        self.setMove(method, nstepsNC)

    @staticmethod
    def random_rotation(model, nc_context):
        """Function that performs a random rotation about the center of mass of
        the model.
        """
        atom_indices = model.atom_indices
        model_pos = model.positions
        com = model.center_of_mass
        reduced_pos = model_pos - com
        #nc_sim = self.nc_sim
        # Store initial positions of entire system
        initial_positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
        positions = copy.deepcopy(initial_positions)

        # Define random rotational move on the ligand
        rand_quat = mdtraj.utils.uniform_quaternion()
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)

        #multiply lig coordinates by rot matrix and add back COM translation from origin
        rot_move =  np.dot(reduced_pos, rand_rotation_matrix) * unit.nanometers + com

        # Update ligand positions in nc_sim
        for index, atomidx in enumerate(atom_indices):
            positions[atomidx] = rot_move[index]
        nc_context.setPositions(positions)
        return nc_context

    def setMove(self, method, step=None):
        """Returns the dictionary that defines the perturbation methods to be
        performed on the model object and the step number to perform it at."""
        nc_move = {}
        nc_move['method']  = getattr(MoveProposal, method)
        nc_move['step'] = int(step) / 2 - 1
        self.nc_move = nc_move
        return nc_move

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
                if int(self.nc_move['step']) == nc_step:
                    print('[Iter {}] Performing NCMC {} move'.format(
                    self.current_iter, self.nc_move['method'].__name__))

                    #Do move
                    self.nc_context = self.nc_move['method'](model=self.model, nc_context=self.nc_context)

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
