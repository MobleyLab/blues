"""
simulation.py: Provides the Simulation class object that runs the BLUES engine

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""
import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
#from simtk.unit import *
import parmed, math
import mdtraj
import sys, time
from datetime import datetime
from openmmtools import alchemy
from blues.integrators import AlchemicalExternalLangevinIntegrator
import logging

def init_logger(outfname='blues'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s-%(asctime)s %(message)s',  "%H:%M:%S")
    # Write to File
    fh = logging.FileHandler(outfname+'.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream to terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

class SimulationFactory(object):
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run.
    Ex.
        from blues.ncmc import SimulationFactory
        sims = SimulationFactory(structure, move_engine, **opt)
        sims.createSimulationSet()
    #TODO: add functionality for handling multiple alchemical regions
    """
    def __init__(self, structure, move_engine, **opt):
        """Requires a parmed.Structure of the entire system and the ncmc.Model
        object being perturbed.

        Options is expected to be a dict of values. Ex:
        nIter=5, nstepsNC=50, nstepsMD=10000,
        temperature=300, friction=1, dt=0.002,
        nonbondedMethod='PME', nonbondedCutoff=10, constraints='HBonds',
        trajectory_interval=1000, reporter_interval=1000, platform=None,
        verbose=False"""
        if 'Logger' in opt:
            self.log = opt['Logger']
        else:
            self.log = init_logger(opt['outfname'])

        #Structure of entire system
        self.structure = structure
        #Atom indicies from move_engine
        #TODO: change atom_indices selection for multiple regions
        self.atom_indices = move_engine.moves[0].atom_indices
        self.move_engine = move_engine
        self.system = None
        self.alch_system = None
        self.md = None
        self.alch  = None
        self.nc  = None

        self.opt = opt
        for k,v in opt.items():
            self.log.info('Options: {} = {}'.format(k,v))

    def _zero_allother_masses(self, system, indexlist):
        num_atoms = system.getNumParticles()
        for index in range(num_atoms):
            if index in indexlist:
                pass
            else:
                system.setParticleMass(index, 0*unit.daltons)
        return system

    def generateAlchSystem(self, system, atom_indices,
                            freeze_distance=0, freeze_center='LIG', freeze_solvent='HOH,NA,CL',
                            **opt):
        """Returns the OpenMM System for alchemical perturbations.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list
            Atom indicies of the move.
        freeze_center : str
            AmberMask selection for the center in which to select atoms for zeroing their masses. Default: LIG
        freeze_distance : float
            Distance (angstroms) to select atoms for retaining their masses. Atoms outside the set distance will have their masses set to 0.0. Default: 5.0
        freeze_solvent : str
            AmberMask selection in which to select solvent atoms for zeroing their masses. Default: HOH,NA,CL
        """
        logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)
        factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True)
        alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices)
        #alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices, annihilate_electrostatics=True, annihilate_sterics=True)
        alch_system = factory.create_alchemical_system(system, alch_region)


        if freeze_distance:
            #Atom selection for zeroing protein atom masses
            mask = parmed.amber.AmberMask(self.structure,"(:%s<:%f)&!(:%s)" % (freeze_center,freeze_distance,freeze_solvent))
            site_idx = [i for i in mask.Selected()]
            self.log.info('Zeroing mass of %s atoms %.1f Angstroms from LIG' % (len(site_idx), freeze_distance))
            alch_system = self._zero_allother_masses(alch_system, site_idx)
        else:
            pass

        return alch_system

    def generateSystem(self, structure, nonbondedMethod='PME', nonbondedCutoff=10,
                       constraints='HBonds', hydrogenMass=None, **opt):
        """Returns the OpenMM System for the reference system.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        opt : optional parameters (i.e. cutoffs/constraints)
        """
        if hydrogenMass:
            self.log.info('HMR Settings: \n\ttimestep: {} \n\tconstraints: {} \n\tHydrogenMass: {}*unit.dalton'.format(opt['dt'], constraints, hydrogenMass))
            hydrogenMass = hydrogenMass*unit.dalton
        else:
            hydrogenMass = None

        system = structure.createSystem(nonbondedMethod=eval("app.%s" % nonbondedMethod),
                            nonbondedCutoff=nonbondedCutoff*unit.angstroms,
                            constraints=eval("app.%s" % constraints),
                            hydrogenMass=hydrogenMass)

        return system

    def generateSimFromStruct(self, structure, move_engine, system, nIter, nstepsNC, nstepsMD,
                             temperature=300, dt=0.002, friction=1,
                             reporter_interval=1000, nprop=1, prop_lambda=0.3,
                             ncmc=False, platform=None, verbose=True,
                             **opt):
        """Used to generate the OpenMM Simulation objects given a ParmEd Structure.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system :
        opt : optional parameters (i.e. cutoffs/constraints)
        atom_indices : list
            Atom indicies of the move.
        prop_lambda : float (Default = 0.3)
            Defines the region in which to add extra propagation
            steps during the NCMC simulation from the midpoint 0.5.
            i.e. A value of 0.3 will add extra steps from lambda 0.2 to 0.8.
        nprop : int (Default: 1)
            Controls the number of propagation steps to add in the lambda
            region defined by `prop_lambda`
        """
        if ncmc:
            #During NCMC simulation, lambda parameters are controlled by function dict below
            # Keys correspond to parameter type (i.e 'lambda_sterics', 'lambda_electrostatics')
            # 'lambda' = step/totalsteps where step corresponds to current NCMC step,
            functions = { 'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                          'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }
            integrator = AlchemicalExternalLangevinIntegrator(alchemical_functions=functions,
                                   splitting= "H V R O R V H",
                                   temperature=temperature*unit.kelvin,
                                   nsteps_neq=nstepsNC,
                                   timestep=dt*unit.picoseconds,
                                   nprop=nprop,
                                   prop_lambda=prop_lambda
                                   )

            for move in move_engine.moves:
                system, integrator = move.initializeSystem(system, integrator)

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
            simulation = app.Simulation(structure.topology, system, integrator, platform)

        if verbose:
            # OpenMM platform information
            mmver = openmm.version.version
            mmplat = simulation.context.getPlatform()
            self.log.debug('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()))

            # Platform properties
            for prop in mmplat.getPropertyNames():
                val = mmplat.getPropertyValue(simulation.context, prop)
                self.log.debug('PlatformProperties: {} - {}'.format(prop,val))

        # Set initial positions/velocities
        # Will get overwritten from saved State.
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)

        return simulation

    def createSimulationSet(self):
        """Function used to generate the 3 OpenMM Simulation objects."""
        self.system = self.generateSystem(self.structure, **self.opt)
        self.alch_system = self.generateAlchSystem(self.system, self.atom_indices, **self.opt)
        self.md = self.generateSimFromStruct(self.structure, self.move_engine, self.system, **self.opt)
        self.alch = self.generateSimFromStruct(self.structure, self.move_engine, self.system,  **self.opt)
        self.nc = self.generateSimFromStruct(self.structure, self.move_engine, self.alch_system,
                                            ncmc=True, **self.opt)


class Simulation(object):
    """Simulation class provides the functions that perform the BLUES run.

    Ex.
        import blues.ncmc
        blues = ncmc.Simulation(sims, move_engine, **opt)
        blues.run()

    """
    def __init__(self, simulations, move_engine, **opt):
        """Initialize the BLUES Simulation object.

        Parameters
        ----------
        sims : blues.ncmc.SimulationFactory object
            SimulationFactory Object which carries the 3 required
            OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.
        move_engine : blues.ncmc.MoveEngine object
            MoveProposal object which contains the dict of moves performed
            in the NCMC simulation.
        """
        if 'Logger' in opt:
            self.log = opt['Logger']
        else:
            self.log = logging.getLogger(__name__)

        self.opt = opt
        self.md_sim = simulations.md
        self.alch_sim = simulations.alch
        self.nc_context = simulations.nc.context
        self.nc_sim = simulations.nc
        self.nc_integrator = simulations.nc.context._integrator
        self.move_engine = move_engine

        self.accept = 0
        self.reject = 0
        self.accept_ratio = 0
        self.move_ct = 0

        #if nstepsNC not specified, set it to 0
        #will be caught if NCMC simulation is run
        if (self.opt['nstepsNC'] % 2) != 0:
            raise ValueError('nstepsNC needs to be even to ensure the protocol is symmetric (currently %i)' % (nstepsNC))
        else:
            self.movestep = int(self.opt['nstepsNC']) / 2

        self.current_iter = 0
        self.current_state = { 'md'   : { 'state0' : {}, 'state1' : {} },
                               'nc'   : { 'state0' : {}, 'state1' : {} },
                               'alch' : { 'state0' : {}, 'state1' : {} }
                            }
        #attach ncmc reporter if specified
        if 'ncmc_traj' in self.opt:
            # Add reporter to NCMC Simulation useful for debugging:
            self.ncmc_reporter = app.dcdreporter.DCDReporter('{ncmc_traj}.dcd'.format(**self.opt), 1)
            self.nc_sim.reporters.append(self.ncmc_reporter)
        else:
            pass


        #controls how many mc moves are performed during each iteration
        if 'mc_per_iter' in opt:
            self.mc_per_iter = opt['mc_per_iter']
        else:
            self.mc_per_iter = 1

        #specify nc integrator variables to report in verbose output
        self.work_keys = [ 'lambda', 'shadow_work',
                          'protocol_work', 'Eold', 'Enew']

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

    def _getSimulationInfo(self):
        """Prints out simulation timing and related information."""
        self.log.info('Iterations = %s' % self.opt['nIter'])
        self.log.info('Timestep = %s ps' % self.opt['dt'])
        self.log.info('NCMC Steps = %s' % self.opt['nstepsNC'])

        prop_lambda = self.nc_integrator._prop_lambda
        prop_range = round(prop_lambda[1] - prop_lambda[0],4)
        if prop_range >= 0.0:
            self.log.info('\tAdding {} extra propgation steps in lambda [{}, {}]'.format(self.opt['nprop'], prop_lambda[0],prop_lambda[1]))
            #Get number of NCMC steps before extra propagation
            normal_ncmc_steps = round(prop_lambda[0] * self.opt['nstepsNC'],4)

            #Get number of NCMC steps for extra propagation
            extra_ncmc_steps = (prop_range * self.opt['nstepsNC']) * self.opt['nprop']

            self.log.info('\tLambda: 0.0 -> %s = %s NCMC Steps' % (prop_lambda[0],normal_ncmc_steps))
            self.log.info('\tLambda: %s -> %s = %s NCMC Steps' % (prop_lambda[0],prop_lambda[1],extra_ncmc_steps))
            self.log.info('\tLambda: %s -> 1.0 = %s NCMC Steps' % (prop_lambda[1],normal_ncmc_steps))

            #Get total number of NCMC steps including extra propagation
            total_ncmc_steps = (normal_ncmc_steps * 2.0) + extra_ncmc_steps
            self.log.info('\t%s NCMC Steps/iter' % total_ncmc_steps)

        else:
            total_ncmc_steps = self.opt['nstepsNC']

        #Total NCMC simulation time
        time_ncmc_steps = total_ncmc_steps * self.opt['dt']
        self.log.info('\t%s NCMC ps/iter' % time_ncmc_steps)

        #Total MD simulation time
        time_md_steps = self.opt['nstepsMD'] * self.opt['dt']
        self.log.info('MD Steps = %s' % self.opt['nstepsMD'])
        self.log.info('\t%s MD ps/iter' % time_md_steps)

        #Total BLUES simulation time
        totaltime = (time_ncmc_steps + time_md_steps) * self.opt['nIter']
        self.log.info('Total Simulation Time = %s ps' % totaltime)
        self.log.info('\tTotal NCMC time = %s ps' % (int(time_ncmc_steps) * int(self.opt['nIter'])))
        self.log.info('\tTotal MD time = %s ps' % (int(time_md_steps) * int(self.opt['nIter'])))

        #Get trajectory frame interval timing for BLUES simulation
        frame_iter = self.opt['nstepsMD'] / self.opt['trajectory_interval']
        timetraj_frame = (time_ncmc_steps + time_md_steps) / frame_iter
        self.log.info('\tTrajectory Interval = %s ps' % timetraj_frame)
        self.log.info('\t\t%s frames/iter' % frame_iter )

    def getStateInfo(self, context, parameters):
        """Function that gets the State information from the given context and
        list of parameters to query it with.
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
            workinfo[param] = nc_integrator.getGlobalVariableByName(param)
        return workinfo

    def writeFrame(self, simulation, outfname):
        """Extracts a ParmEd structure and writes the frame given
        an OpenMM Simulation object"""
        topology = simulation.topology
        system = simulation.context.getSystem()
        state = simulation.context.getState(getPositions=True,
                                            getVelocities=True,
                                            getParameters=True,
                                            getForces=True,
                                            getParameterDerivatives=True,
                                            getEnergy=True,
                                            enforcePeriodicBox=True)


        # Generate the ParmEd Structure
        structure = parmed.openmm.load_topology(topology, system,
                                   xyz=state.getPositions())

        structure.save(outfname,overwrite=True)
        self.log.info('\tSaving Frame to: %s' % outfname)

    def acceptRejectNCMC(self, temperature=300, write_move=False, **opt):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.current_state['md']['state0']
        nc_state0 = self.current_state['nc']['state0']
        nc_state1 = self.current_state['nc']['state1']

        log_ncmc = self.nc_integrator.getLogAcceptanceProbability(self.nc_context)
        randnum =  math.log(np.random.random())

        # Compute Alchemical Correction Term
        if np.isnan(log_ncmc) == False:
            self.alch_sim.context.setPositions(nc_state1['positions'])
            alch_state1 = self.getStateInfo(self.alch_sim.context, self.state_keys)
            self.setSimState('alch', 'state1', alch_state1)
            correction_factor = (nc_state0['potential_energy'] - md_state0['potential_energy'] + alch_state1['potential_energy'] - nc_state1['potential_energy']) * (-1.0/self.nc_integrator.kT)
            log_ncmc = log_ncmc + correction_factor

        if self.move_engine.moves[self.move_engine.selected_move].bin_boolean == False:
            print("Bin Boolean false")

        if log_ncmc > randnum and self.move_engine.moves[self.move_engine.selected_move].bin_boolean:
            self.accept += 1
            self.log.info('NCMC MOVE ACCEPTED: log_ncmc {} > randnum {}'.format(log_ncmc, randnum) )
            self.md_sim.context.setPositions(nc_state1['positions'])
            if write_move:
            	self.writeFrame(self.md_sim, '{}acc-it{}.pdb'.format(self.opt['outfname'],self.current_iter))

        else:
            self.reject += 1
            self.log.info('NCMC MOVE REJECTED: log_ncmc {} < {}'.format(log_ncmc, randnum) )
            self.nc_context.setPositions(md_state0['positions'])

        self.nc_integrator.reset()
        self.md_sim.context.setVelocitiesToTemperature(temperature)

    def simulateNCMC(self, nstepsNC=5000, ncmc_traj=None,
                    reporter_interval=1000, verbose=False, **opt):
        """Function that performs the NCMC simulation."""
        self.log.info('[Iter %i] Advancing %i NCMC steps...' % (self.current_iter, nstepsNC))
        #choose a move to be performed according to move probabilities
        #TODO: will have to change to work with multiple alch region
        self.move_engine.selectMove()
        move_idx = self.move_engine.selected_move
        move_name = self.move_engine.moves[move_idx].__class__.__name__
        for nc_step in range(int(nstepsNC)):
            start = time.time()
            self._initialSimulationTime = self.nc_context.getState().getTime()
            try:
                #Attempt anything related to the move before protocol is performed
                if nc_step == 0:
                    self.nc_context = self.move_engine.moves[self.move_engine.selected_move].beforeMove(self.nc_context)
                # Attempt selected MoveEngine Move at the halfway point
                #to ensure protocol is symmetric
                if self.movestep == nc_step:
                    #Do move
                    self.log.info('Performing %s...' % move_name)
                    self.nc_context = self.move_engine.runEngine(self.nc_context)

                # Do 1 NCMC step with the integrator
                self.nc_integrator.step(1)

                ###DEBUG options at every NCMC step
                if verbose:
                    # Print energies at every step
                    work = self.getWorkInfo(self.nc_integrator, self.work_keys)
                    self.log.debug('%s' % work)
                if ncmc_traj:
                    self.ncmc_reporter.report(self.nc_sim, self.nc_context.getState(getPositions=True, getVelocities=True))

                #Attempt anything related to the move after protocol is performed
                if nc_step == nstepsNC-1:
                    self.nc_context = self.move_engine.moves[self.move_engine.selected_move].afterMove(self.nc_context)

            except Exception as e:
                self.log.error(e)
                self.move_engine.moves[self.move_engine.selected_move]._error(self.nc_context)
                break

            self._report(start, nc_step)

        nc_state1 = self.getStateInfo(self.nc_context, self.state_keys)
        self.setSimState('nc', 'state1', nc_state1)

    def simulateMD(self, nstepsMD=5000, **opt):
        """Function that performs the MD simulation."""

        self.log.info('[Iter %i] Advancing %i MD steps...' % (self.current_iter, nstepsMD))

        md_state0 = self.current_state['md']['state0']
        try:
            self.md_sim.step(nstepsMD)
        except Exception as e:
            self.log.error(e)
            self.log.error('potential energy before NCMC: %s' % md_state0['potential_energy'])
            self.log.error('kinetic energy before NCMC: %s' % md_state0['kinetic_energy'])
            #Write out broken frame
            self.writeFrame(self.md_sim, 'MD-fail-it%s-md%i.pdb' %(self.current_iter, self.md_sim.currentStep))
            exit()

        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        self.setSimState('md', 'state0', md_state0)
        # Set NC poistions to last positions from MD
        self.nc_context.setPositions(md_state0['positions'])
        self.nc_context.setVelocities(md_state0['velocities'])

    def _report(self,start,nc_step):
        end = time.time()
        headers = ['Step', 'Speed (ns/day)', 'Acc. Moves', 'Iter']
        if self.current_iter == 0 and nc_step == 0:
            self.log.info('[NCMC] "%s"' % ('"'+'\t'+'"').join(headers))
        if nc_step % self.opt['reporter_interval'] == 0 or nc_step+1 == self.opt['nstepsNC']:
            elapsed = end-start
            elapsedDays = (elapsed/86400.0)
            elapsedNs = (self.nc_context.getState().getTime()-self._initialSimulationTime).value_in_unit(unit.nanosecond)
            speed = (elapsedNs/elapsedDays)
            speed = "%.3g" % speed
            values = [nc_step, speed, self.accept, self.current_iter]
            self.log.info('\t\t'.join(str(v) for v in values))

    def evalDihedral(self, positions):
        topology = mdtraj.Topology.from_openmm(self.md_sim.topology)
        traj = mdtraj.Trajectory(np.asarray(positions),topology)
        #traj.xyz = np.asarray(positions)
        indices = np.array([[1735,1737,1739,1741]])
        dihedralangle = mdtraj.compute_dihedrals(traj, indices)
        if -1.3 <= dihedralangle <= -0.9:
            eval = True
        elif -3.14159 <= dihedralangle <= -2.94159:
            eval = True
        elif 0.9 <= dihedralangle <= 1.3:
            eval = True
        elif 2.94159 <= dihedralangle <= 3.14159:
            eval = True
        else:
            eval = False
        if eval == False:
            print("no ncmc --> dihedral not ok")
        if eval == True:
            print("Dihedral ok --> NCMC proceed")
            print("In the simulation.py script, this is the dihedral angle %f" %(dihedralangle))
        return(eval)

    def run(self, nIter=100):
        """Function that runs the BLUES engine to iterate over the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state.
        """
        self.log.info('Running %i BLUES iterations...' % (nIter))
        self._getSimulationInfo()
        #set inital conditions
        self.setStateConditions()

        #
        while self.move_ct <= nIter:
            self.current_iter = int(self.move_ct)
            positions = self.nc_context.getState(getPositions=True).getPositions(asNumpy=True)
            if self.evalDihedral(positions):
            #for n in range(int(nIter)):
                #self.current_iter = int(n)
                self.setStateConditions()
                self.simulateNCMC(**self.opt)
                self.acceptRejectNCMC(**self.opt)
                self.move_ct += 1
            self.simulateMD(**self.opt)

        # END OF NITER
        self.accept_ratio = self.accept/float(self.move_ct)
        self.log.info('Acceptance Ratio: %s' % self.accept_ratio)
        self.log.info('nIter: %s ' % nIter)

    def simulateMC(self):
        """Function that performs the MC simulation."""

        #choose a move to be performed according to move probabilities
        self.move_engine.selectMove()
        #change coordinates according to Moves in MoveEngine
        new_context = self.move_engine.runEngine(self.md_sim.context)
        md_state1 = self.getStateInfo(new_context, self.state_keys)
        self.setSimState('md', 'state1', md_state1)

    def acceptRejectMC(self, temperature=300, **opt):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.current_state['md']['state0']
        md_state1 = self.current_state['md']['state1']
        log_mc = (md_state1['potential_energy'] - md_state0['potential_energy']) * (-1.0/self.nc_integrator.kT)
        randnum =  math.log(np.random.random())

        if log_mc > randnum:
            self.accept += 1
            self.log.info('MC MOVE ACCEPTED: log_mc {} > randnum {}'.format(log_mc, randnum) )
            self.md_sim.context.setPositions(md_state1['positions'])
        else:
            self.reject += 1
            self.log.info('MC MOVE REJECTED: log_mc {} < {}'.format(log_mc, randnum) )
            self.md_sim.context.setPositions(md_state0['positions'])
        self.log_mc = log_mc
        self.md_sim.context.setVelocitiesToTemperature(temperature)



    def runMC(self, nIter=100):
        """Function that runs the BLUES engine to iterate over the actions:
        perform proposed move, accepts/rejects move,
        then performs the MD simulation from the accepted or rejected state.
        """

        #set inital conditions
        self.setStateConditions()
        for n in range(nIter):
            self.current_iter = int(n)
            for i in range(self.mc_per_iter):
                self.setStateConditions()
                self.simulateMC()
                self.acceptRejectMC(**self.opt)
            self.simulateMD(**self.opt)
