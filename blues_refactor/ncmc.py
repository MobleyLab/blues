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

class MovePropsal(object):

    def __init__(self, nc_sim, model, method, nstepsNC):
        supported_methods = ['random_rotation']
        if method not in supported_methods:
            raise Exception("Method %s not implemented" % method)
        else:
            self.nc_sim = nc_sim
            self.model = model
            self.nc_move = { 'method' : None , 'step' : 0}
            self.setMove(method, nstepsNC)

    def random_rotation(self):
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
        self.nc_move['method']  = getattr(MovePropsal, method)
        self.nc_move['step'] = int(step) / 2 - 1
        return self.nc_move

class Simulation(object):
    def __init__(self, sims, model, nc_move, **opt):
        self.md_sim = sims.md
        self.alch_sim = sims.alch
        self.nc_context = sims.nc.context
        self.nc_integrator = sims.nc.context._integrator
        self.nc_move = nc_move
        self.model = model
        self.accept = 0
        self.reject = 0
        self.accept_ratio = 0

        self.nIter = int(opt['nIter'])
        self.nstepsNC = int(opt['nstepsNC'])
        self.nstepsMD = int(opt['nstepsMD'])

        self.current_stepNC = 0
        self.current_iter = 0
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
        self.current_state[simkey][stateidx] = stateinfo

    def getStateInfo(self, context, parameters):
        stateinfo = {}
        state  = context.getState(**parameters)
        stateinfo['iter'] = int(self.current_iter)
        stateinfo['positions'] =  state.getPositions(asNumpy=True)
        stateinfo['velocities'] = state.getVelocities(asNumpy=True)
        stateinfo['potential_energy'] = state.getPotentialEnergy()
        stateinfo['kinetic_energy'] = state.getKineticEnergy()
        return stateinfo

    def getWorkInfo(self, nc_integrator, parameters):
        workinfo = {}
        for param in parameters:
            workinfo['param'] = nc_integrator.getGlobalVariableByName(param)
        return workinfo

    def chooseMove(self):
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
            self.md_sim.context.setVelocities(nc_state1['velocities'])
        else:
            self.reject += 1
            print('NCMC MOVE REJECTED: {} < {}'.format(log_ncmc, randnum) )
            #print('ncmc PE', newinfo['potential_energy'], 'old PE', md_PE0)
            self.nc_context.setPositions(md_state0['positions'])
            self.nc_context.setVelocities(-md_state0['velocities'])

        self.nc_integrator.reset()

    def simulateNCMC(self):
        for nc_step in range(self.nstepsNC):
            try:
                self.current_stepNC = int(nc_step)
                # Calculate Work/Energies Before Step
                work_initial = self.getWorkInfo(self.nc_integrator, self.work_keys)
                # Attempt NCMC Move
                if nc_step == self.nc_move['step']:
                    print('[Iter {}] Performing NCMC move: {} at NC step {}'.format(
                    self.current_iter, self.nc_move['method'].__name__, self.nc_move['step'] ) )
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
        md_state0 = self.current_state['md']['state0']
        try:
            self.md_sim.step(self.nstepsMD)
            self.current_stepMD = self.md_sim.currentStep
        except Exception as e:
            raise(e)
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

    def setStateConditions(self):
        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        nc_state0 = self.getStateInfo(self.nc_context, self.state_keys)
        self.nc_context.setPositions(md_state0['positions'])
        self.nc_context.setVelocities(md_state0['velocities'])
        self.setSimState('md', 'state0', md_state0)
        self.setSimState('nc', 'state0', nc_state0)

    def run(self):
        #set inital conditions
        self.setStateConditions()
        for n in range(self.nIter):
            self.current_iter = int(n)
            self.setStateConditions
            self.simulateNCMC()
            self.chooseMove()
            self.simulateMD()

        # END OF NITER
        self.accept_ratio = self.accept/float(self.nIter)
        print('Acceptance Ratio', self.accept_ratio)
        print('numsteps ', self.nstepsNC)
