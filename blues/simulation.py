"""
simulation.py: Provides the Simulation class object that runs the BLUES engine

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""
import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import parmed, math
import mdtraj
import sys
from openmmtools import alchemy
from blues.integrators import AlchemicalExternalLangevinIntegrator
import logging

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
        self.logger = logging.getLogger(__name__)
        #Structure of entire system
        self.structure = structure
        #Atom indicies from move_engine
        #TODO: change atom_indices selection for multiple regions
        self.atom_indices = move_engine.moves[0].atom_indices

        self.system = None
        self.alch_system = None
        self.md = None
        self.alch  = None
        self.nc  = None

        self.opt = opt

    def _zero_allother_masses(self, system, indexlist):
        num_atoms = system.getNumParticles()
        for index in range(num_atoms):
            if index in indexlist:
                pass
            else:
                system.setParticleMass(index, 0*unit.daltons)
        return system

    def generateAlchSystem(self, system, atom_indices, freeze_distance=0, **opt):
        """Returns the OpenMM System for alchemical perturbations.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list
            Atom indicies of the move.
        """
        logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)
        factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True)
        alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices)
        alch_system = factory.create_alchemical_system(system, alch_region)


        if freeze_distance:
            #Atom selection for zeroing protein atom masses
            mask = parmed.amber.AmberMask(self.structure,"(:LIG<:%f)&!(:HOH,NA,CL)" % freeze_distance )
            site_idx = [i for i in mask.Selected()]
            self.logger.info('Zeroing mass of %s protein atoms %.1f Angstroms from LIG' % (len(site_idx), freeze_distance))
            alch_system = self._zero_allother_masses(alch_system, site_idx)
        else:
            pass

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
                                   )

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
            simulation = app.Simulation(structure.topology, system, integrator, platform)#, prop)

        if self.logger.isEnabledFor(logging.DEBUG):
            # OpenMM platform information
            mmver = openmm.version.version
            mmplat = simulation.context.getPlatform()
            self.logger.debug('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()))

            # Platform properties
            for prop in mmplat.getPropertyNames():
                val = mmplat.getPropertyValue(simulation.context, prop)
                self.logger.debug('PlatformProperties: {} - {}'.format(prop,val))

        # Set initial positions/velocities
        # Will get overwritten from saved State.
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)

        return simulation

    def createSimulationSet(self):
        """Function used to generate the 3 OpenMM Simulation objects."""
        self.system = self.generateSystem(self.structure, **self.opt)
        self.alch_system = self.generateAlchSystem(self.system, self.atom_indices, **self.opt)
        self.md = self.generateSimFromStruct(self.structure, self.system, **self.opt)
        self.alch = self.generateSimFromStruct(self.structure, self.system,  **self.opt)
        self.nc = self.generateSimFromStruct(self.structure, self.alch_system,
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
        self.logger = logging.getLogger(__name__)
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

        #if nstepsNC not specified, set it to 0
        #will be caught if NCMC simulation is run
        if (self.opt['nstepsNC'] % 2) != 0:
            raise ValueError('nstepsNC needs to be even to ensure the protocol is symmetric (currently %i)' % (nstepsNC))
        else:
            self.movestep = int(self.opt['nstepsNC']) / 2

        #Get Lambda step parameters for extra propagation
        self._getAlchStepParameters()

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

    def _getAlchStepParameters(self):
        initial_lambda = self.nc_integrator.getGlobalVariableByName('lambda')
        initial_step = self.nc_integrator.getGlobalVariableByName('step')
        initial_lambda_step = self.nc_integrator.getGlobalVariableByName('lambda_step')
        self.nc_integrator.step(1)
        final_lambda = self.nc_integrator.getGlobalVariableByName('lambda')
        final_step = self.nc_integrator.getGlobalVariableByName('step')
        final_lambda_step = self.nc_integrator.getGlobalVariableByName('lambda_step')
        self.nc_integrator.reset()

        self.d_lambda = final_lambda - initial_lambda
        self.d_step = final_step - initial_step
        self.d_lambda_step = final_lambda_step - initial_lambda_step

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
        self.logger.info('\tSaving Frame to: %s' % outfname)

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

        if log_ncmc > randnum:
            self.accept += 1
            self.logger.info('NCMC MOVE ACCEPTED: log_ncmc {} > randnum {}'.format(log_ncmc, randnum) )
            self.md_sim.context.setPositions(nc_state1['positions'])
            if write_move:
                self.writeFrame(self.md_sim, 'acc-it%s.pdb' %(self.current_iter))
        else:
            self.reject += 1
            self.logger.info('NCMC MOVE REJECTED: log_ncmc {} < {}'.format(log_ncmc, randnum) )
            self.nc_context.setPositions(md_state0['positions'])

        self.nc_integrator.reset()
        self.md_sim.context.setVelocitiesToTemperature(temperature)

    def simulateNCMC(self, nstepsNC=5000, nprop=5, ncmc_traj=None,
                    reporter_interval=1000, **opt):
        """Function that performs the NCMC simulation."""

        self.logger.info('[Iter %i] Starting NCMC Simulation with %i NCMC steps...' % (self.current_iter, nstepsNC))

        #choose a move to be performed according to move probabilities
        #TODO: will have to change to work with multiple alch regions
        self.move_engine.selectMove()
        for nc_step in range(int(nstepsNC)):
            try:
                # Attempt selected MoveEngine Move at the halfway point
                #to ensure protocol is symmetric
                if self.movestep == nc_step:
                    #Do move
                    self.logger.info('Step = %s Performing NCMC move...' % nc_step)
                    self.nc_context = self.move_engine.runEngine(self.nc_context)

                # Add extra propagation steps
                if nprop > 1:
                    current_lambda = self.nc_integrator.getGlobalVariableByName('lambda')
                    if current_lambda >= 0.2 and current_lambda <= 0.8:
                        for n in range(int(nprop)):
                            self.nc_integrator.setGlobalVariableByName('lambda', self.nc_integrator.getGlobalVariableByName('lambda') - self.d_lambda)
                            self.nc_integrator.setGlobalVariableByName('step', self.nc_integrator.getGlobalVariableByName('step') - self.d_step)
                            self.nc_integrator.setGlobalVariableByName('lambda_step', self.nc_integrator.getGlobalVariableByName('lambda_step') - self.d_lambda_step)
                            self.nc_integrator.step(1)

                # Do 1 NCMC step with the integrator
                self.nc_integrator.step(1)

                # Print out NCMC info to show progress.
                if nc_step % reporter_interval == 0:
                    workinfo = self.getWorkInfo(self.nc_integrator, ['step','lambda','lambda_step', 'protocol_work'])
                    self.logger.info('Step = {step}  Lambda = {lambda}  Work = {protocol_work} Lambda_step = {lambda_step}'.format(**workinfo))


                ###DEBUG options at every NCMC step
                if self.logger.isEnabledFor(logging.DEBUG):
                    # Print energies at every step
                    work = self.getWorkInfo(self.nc_integrator, self.work_keys)
                    self.logger.debug('%s' % work)
                if ncmc_traj:
                    self.ncmc_reporter.report(self.nc_sim, self.nc_sim.context.getState(getPositions=True, getVelocities=True))

            except Exception as e:
                self.logger.error(e)
                break

        nc_state1 = self.getStateInfo(self.nc_context, self.state_keys)
        self.setSimState('nc', 'state1', nc_state1)

    def simulateMD(self, nstepsMD=5000, **opt):
        """Function that performs the MD simulation."""

        self.logger.info('[Iter %i] Starting MD Simulation with %i MD steps...' % (self.current_iter, nstepsMD))

        md_state0 = self.current_state['md']['state0']
        try:
            self.md_sim.step(nstepsMD)
        except Exception as e:
            self.logger.error(e)
            self.logger.error('potential energy before NCMC: %s' % md_state0['potential_energy'])
            self.logger.error('kinetic energy before NCMC: %s' % md_state0['kinetic_energy'])
            #Write out broken frame
            self.writeFrame(self.md_sim, 'MD-fail-it%s-md%i.pdb' %(self.current_iter, self.md_sim.currentStep))
            exit()

        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        self.setSimState('md', 'state0', md_state0)
        # Set NC poistions to last positions from MD
        self.nc_context.setPositions(md_state0['positions'])
        self.nc_context.setVelocities(md_state0['velocities'])

    def run(self, nIter=100):
        """Function that runs the BLUES engine to iterate over the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state.
        """
        self.logger.info('Running %i BLUES iterations...' % (nIter))
        #set inital conditions
        self.setStateConditions()
        for n in range(int(nIter)):
            self.current_iter = int(n)
            self.setStateConditions()
            self.simulateNCMC(**self.opt)
            self.acceptRejectNCMC(**self.opt)
            self.simulateMD(**self.opt)

        # END OF NITER
        self.accept_ratio = self.accept/float(nIter)
        self.logger.info('Acceptance Ratio: %s' % self.accept_ratio)
        self.logger.info('nIter: %s ' % nIter)

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
            self.logger.info('MC MOVE ACCEPTED: log_mc {} > randnum {}'.format(log_mc, randnum) )
            self.md_sim.context.setPositions(md_state1['positions'])
        else:
            self.reject += 1
            self.logger.info('MC MOVE REJECTED: log_mc {} < {}'.format(log_mc, randnum) )
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
