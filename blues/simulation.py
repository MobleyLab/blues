"""
simulation.py: Provides the Simulation class object that runs the BLUES engine

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""
from __future__ import print_function
import numpy as np
from simtk import unit
from simtk.openmm import app
import parmed, math
import mdtraj


class Simulation(object):
    """Simulation class provides the functions that perform the BLUES run.

    Ex.
        import blues.ncmc
        blues = ncmc.Simulation(sims, model, mover, **opt)
        blues.run()

    """
    def __init__(self, simulations, mover, **opt):
        """Initialize the BLUES Simulation object.

        Parameters
        ----------
        sims : blues.ncmc.SimulationFactory object
            SimulationFactory Object which carries the 3 required
            OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.
        mover : blues.ncmc.MoveEngine object
            MoveProposal object which contains the dict of moves performed
            in the NCMC simulation.
        """
        self.md_sim = simulations.md
        self.alch_sim = simulations.alch
        self.nc_context = simulations.nc.context
        self.nc_sim = simulations.nc
        self.nc_integrator = simulations.nc.context._integrator
        self.move_engine = mover

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

        if 'verbose' in opt:
            self.verbose = opt['verbose']
        else:
            self.verbose = False

        #attach ncmc reporter if specified
        if 'write_ncmc' in opt:
            self.write_ncmc = opt['write_ncmc']
            if 'ncmc_outfile' in opt:
                self.ncmc_outfile = opt['ncmc_outfile']
            else:
                self.ncmc_outfile = 'ncmc_output.dcd'
        else:
            self.write_ncmc = None

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
        print('Saving Frame to', outfname)

    def chooseMove(self):
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
            print('NCMC MOVE ACCEPTED: log_ncmc {} > randnum {}'.format(log_ncmc, randnum) )
            self.md_sim.context.setPositions(nc_state1['positions'])
        else:
            self.reject += 1
            print('NCMC MOVE REJECTED: {} < {}'.format(log_ncmc, randnum) )
            self.nc_context.setPositions(md_state0['positions'])

        self.nc_integrator.reset()
        self.md_sim.context.setVelocitiesToTemperature(self.temperature)

    def simulateNCMC(self, verbose=False, write_ncmc=False):
        """Function that performs the NCMC simulation."""
        #append nc reporter at the first step
        if (self.current_iter == 0) and (write_ncmc):
            self.ncmc_reporter = app.dcdreporter.DCDReporter(self.ncmc_outfile, 1)
            self.nc_sim.reporters.append(self.ncmc_reporter)
        for nc_step in range(self.nstepsNC):
            try:
                self.current_stepNC = int(nc_step)
                # Calculate Work/Energies Before Step
                if verbose:
                    work_initial = self.getWorkInfo(self.nc_integrator, self.work_keys)

                # Attempt NCMC Move
                if self.nstepsNC / 2 == nc_step:

                    #Do move
                    print('[Iter {}] Performing NCMC move'.format(self.current_iter))
                    try:
                        self.nc_context = self.move_engine.runEngine(self.nc_context)
                    except Exception as e:
                        #error handling in the case where trying the move raises
                        #an exception
                        #TODO figure out how to do this without using SystemExit
                        print(e)
                        raise SystemExit

                    if write_ncmc and (nc_step+1) % write_ncmc == 0:
                        self.ncmc_reporter.report(self.nc_sim, self.nc_sim.context.getState(getPositions=True, getVelocities=True))

                # Do 1 NCMC step
                self.nc_integrator.step(1)
                if write_ncmc and (nc_step+1) % write_ncmc == 0:
                    self.ncmc_reporter.report(self.nc_sim, self.nc_sim.context.getState(getPositions=True, getVelocities=True))

                if verbose:
                    # Calculate Work/Energies After Step.
                    work_final = self.getWorkInfo(self.nc_integrator, self.work_keys)
                    print('Initial work:', work_initial)
                    print('Final work:', work_final)
                    #TODO write out frame regardless if accepted/REJECTED
                    # Embed in move function or here???
                    #self.writeFrame(self.md_sim, 'MD-iter{}.pdb'.format(self.current_iter))

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
            last_x, last_y = np.shape(md_state0['positions'])
            reshape = (np.reshape(md_state0['positions'], (1, last_x, last_y))).value_in_unit(unit.nanometers)
            print('potential energy before NCMC', md_state0['potential_energy'])
            print('kinetic energy before NCMC', md_state0['kinetic_energy'])

            last_top = mdtraj.Topology.from_openmm(self.md_sim.topology)
            broken_frame = mdtraj.Trajectory(xyz=reshape, topology=last_top)
            broken_frame.save_pdb('MD-blues_fail-iter{}_md{}.pdb'.format(self.current_iter, self.current_stepMD))
            exit()

        md_state0 = self.getStateInfo(self.md_sim.context, self.state_keys)
        self.setSimState('md', 'state0', md_state0)
        # Set NC poistions to last positions from MD
        self.nc_context.setPositions(md_state0['positions'])
        self.nc_context.setVelocities(md_state0['velocities'])

    def run(self):
        """Function that runs the BLUES engine to iterate over the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state.
        """
        #set inital conditions
        self.setStateConditions()
        for n in range(self.nIter):
            self.current_iter = int(n)
            self.setStateConditions()
            self.simulateNCMC(verbose=self.verbose, write_ncmc=self.write_ncmc)
            self.chooseMove()
            self.simulateMD()

        # END OF NITER
        self.accept_ratio = self.accept/float(self.nIter)
        print('Acceptance Ratio', self.accept_ratio)
        print('nIter ', self.nIter)
