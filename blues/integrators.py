from openmmtools.integrators import LangevinIntegrator
import numpy
import logging
import re

import simtk.unit

import simtk.unit as units
import simtk.openmm as mm

from openmmtools.constants import kB

#TODO use Nonequilibrium baseclass directly
class AlchemicalLangevinSplittingIntegrator(LangevinIntegrator):
    """Allows nonequilibrium switching based on force parameters specified in alchemical_functions.
    Propagator is based on Langevin splitting, as described below.
    One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt
        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass
        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal
    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive but slow-fluctuating forces. Since forces are only
    evaluated in the V step, we represent this by including in our "alphabet" V0, V1, ...)
    When the system contains holonomic constraints, these steps are confined to the constraint
    manifold.
    Examples
    --------
        - VVVR
            splitting="O V R V O"
        - BAOAB:
            splitting="V R O R V"
        - g-BAOAB, with K_r=3:
            splitting="V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            splitting="V0 V1 R R O R R V1 R R O R R V1 V0"
    Attributes
    ----------
    _kinetic_energy : str
        This is 0.5*m*v*v by default, and is the expression used for the kinetic energy
    References
    ----------
    [Nilmeier, et al. 2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7
    """

    def __init__(self,
                 alchemical_functions,
                 splitting="V R O R V",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True,
                 direction="forward",
                 steps_per_propagation=1,
                 nsteps_neq=100):
        """
        Parameters
        ----------
        alchemical_functions : dict of strings
            key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
            string that depends on the variable "lambda"
        splitting : string, default: "V R O R V"
            Sequence of R, V, O (and optionally V{i}), and { }substeps to be executed each timestep.
            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            ( will cause metropolization, and must be followed later by a ).
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Fictitious "bath" temperature
        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep
        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver
        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`
        measure_heat : boolean, default: True
            Accumulate the heat exchanged with the bath in each step, in the global `heat`
        direction : str, default: "forward"
            Whether to move the global lambda parameter from 0 to 1 (forward) or 1 to 0 (reverse).
        nsteps_neq : int, default: 100
            Number of steps in nonequilibrium protocol. Default 100
        """

        self._alchemical_functions = alchemical_functions
        self._direction = direction
        self._n_steps_neq = nsteps_neq

        # collect the system parameters.
        self._system_parameters = {system_parameter for system_parameter in alchemical_functions.keys()}

        # call the base class constructor
        super(AlchemicalLangevinSplittingIntegrator, self).__init__(splitting=splitting, temperature=temperature,
                                                                    collision_rate=collision_rate, timestep=timestep,
                                                                    constraint_tolerance=constraint_tolerance,
                                                                    measure_shadow_work=measure_shadow_work,
                                                                    measure_heat=measure_heat,
                                                                    )

        # add some global variables relevant to the integrator
        self.add_global_variables(nsteps=nsteps_neq)

    def update_alchemical_parameters_step(self):
        """
        Update Context parameters according to provided functions.
        """
        for context_parameter in self._alchemical_functions:
            if context_parameter in self._system_parameters:
                self.addComputeGlobal(context_parameter, self._alchemical_functions[context_parameter])

    def alchemical_perturbation_step(self):
        """
        Add alchemical perturbation step, accumulating protocol work.
        """
        # Store initial potential energy
        self.addComputeGlobal("Eold", "Epert")

        # Use fractional state
        if self._direction == 'forward':
            self.addComputeGlobal('lambda', '(step+1)/nsteps')
        elif self._direction == 'reverse':
            self.addComputeGlobal('lambda', '(nsteps - step - 1)/nsteps')

        # Update all slaved alchemical parameters
        self.update_alchemical_parameters_step()

        # Accumulate protocol work
        self.addComputeGlobal("Enew", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)/kT")

    def sanity_check(self, splitting, allowed_characters="H{}RVO0123456789"):
        super(AlchemicalLangevinSplittingIntegrator, self).sanity_check(splitting, allowed_characters=allowed_characters)

    def substep_function(self, step_string, measure_shadow_work, measure_heat, n_R, force_group_nV, mts):
        """Take step string, and add the appropriate R, V, O, M, or H step with appropriate parameters.
        The step string input here is a single character (or character + number, for MTS)
        Parameters
        ----------
        step_string : str
            R, O, V, or Vn (where n is a nonnegative integer specifying force group)
        measure_shadow_work : bool
            Whether the steps should measure shadow work
        measure_heat : bool
            Whether the O step should measure heat
        n_R : int
            The number of R steps per integrator step
        force_group_nV : dict
            The number of V steps per integrator step per force group. {0: nV} if not mts
        mts : bool
            Whether the integrator is a multiple timestep integrator
        """

        if step_string == "O":
            self.O_step(measure_heat)
        elif step_string == "R":
            self.R_step(measure_shadow_work, n_R)
        elif step_string == "{":
            self.begin_metropolize()
        elif step_string == "}":
            self.metropolize()
        elif step_string[0] == "V":
            # get the force group for this update--it's the number after the V
            force_group = step_string[1:]
            self.V_step(force_group, measure_shadow_work, force_group_nV, mts)
        elif step_string == "H":
            self.alchemical_perturbation_step()

    def add_integrator_steps(self, splitting, measure_shadow_work, measure_heat, ORV_counts, force_group_nV, mts):
        """
        Override the base class to insert reset steps around the integrator.
        """
        #if the step is zero,
        self.beginIfBlock('step = 0')
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.reset_work_step()
        self.alchemical_reset_step()
        self.addComputeGlobal("Epert", "energy")
        self.endBlock()

        #call the superclass function to insert the appropriate steps, provided the step number is less than n_steps
        self.beginIfBlock("step < nsteps")
        super(AlchemicalLangevinSplittingIntegrator, self).add_integrator_steps(splitting, measure_shadow_work,
                                                                                measure_heat, ORV_counts,
                                                                                force_group_nV, mts)

        #increment the step number
        self.addComputeGlobal("step", "step + 1")

        self.endBlock()


    def add_global_variables(self, nsteps):
        """Add the appropriate global parameters to the CustomIntegrator. nsteps refers to the number of
        total steps in the protocol.
        Parameters
        ----------
        nsteps : int, greater than 0
            The number of steps in the switching protocol.
        """
        self.addGlobalVariable('Eold', 0) #old energy value before perturbation
        self.addGlobalVariable('Enew', 0) #new energy value after perturbation
        self.addGlobalVariable('Epert', 0) #holder energy value after integrator step to keep track of non-alchemical work
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addGlobalVariable('kinetic', 0.0) # kinetic energy
        self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
        self.addGlobalVariable('step', 0) # current NCMC step number
        self.addGlobalVariable('protocol_work', 0) # work performed by NCMC protocol
        self.addGlobalVariable('pstep')

    def alchemical_reset_step(self):
        """
        Reset the alchemical lambda to its starting value
        This is 1 for reverse and 0 for forward
        """
        if self._direction == "forward":
            self.addComputeGlobal("lambda", "0")
        if self._direction == "reverse":
            self.addComputeGlobal("lambda", "1")

        self.addComputeGlobal("protocol_work", "0.0")
        if self._measure_shadow_work:
            self.addComputeGlobal("shadow_work", "0.0")
        self.addComputeGlobal("step", "0.0")
        #add all dependent parameters
        self.update_alchemical_parameters_step()

    def reset_work_step(self):
        """
        This step resets work statistics that have been accumulated.
        """
        self.addComputeGlobal("protocol_work", "0.0")
        self.addComputeGlobal("shadow_work", "0.0")

    def reset_integrator(self):
        """
        Manually reset the work statistics and step
        """
        self.setGlobalVariableByName("step", 0)

