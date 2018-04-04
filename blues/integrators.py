from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator
import simtk

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = simtk.unit.kilojoules_per_mole

class AlchemicalExternalLangevinIntegrator(AlchemicalNonequilibriumLangevinIntegrator):
    """
    Allows nonequilibrium switching based on force parameters specified in alchemical_functions.
    A variable named lambda is switched from 0 to 1 linearly throughout the nsteps of the protocol.
    The functions can use this to create more complex protocols for other global parameters.

    As opposed to `openmmtools.integrators.AlchemicalNonequilibriumLangevinIntegrator`,
    which this inherits from, the AlchemicalExternalLangevinIntegrator integrator also takes
    into account work done outside the nonequilibrium switching portion(between integration steps).
    For example if a molecule is rotated between integration steps, this integrator would
    correctly account for the work caused by that rotation.

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
        - g-BAOAB:
            splitting="R V O H O V R"
        - VVVR
            splitting="O V R H R V O"
        - VV
            splitting="V R H R V"
        - An NCMC algorithm with Metropolized integrator:
            splitting="O { V R H R V } O"
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
                 splitting="R V O H O V R",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True,
                 nsteps_neq=100,
                 nprop=1,
                 prop_lambda=0.3,
                 *args, **kwargs):
        """
        Parameters
        ----------
        alchemical_functions : dict of strings
            key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
            string that depends on the variable "lambda"
        splitting : string, default: "H V R O V R H"
            Sequence of R, V, O (and optionally V{i}), and { }substeps to be executed each timestep. There is also an H option,
            which increments the global parameter `lambda` by 1/nsteps_neq for each step.
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
        nsteps_neq : int, default: 100
            Number of steps in nonequilibrium protocol. Default 100
        prop_lambda : float (Default = 0.3)
            Defines the region in which to add extra propagation
            steps during the NCMC simulation from the midpoint 0.5.
            i.e. A value of 0.3 will add extra steps from lambda 0.2 to 0.8.
        nprop : int (Default: 1)
            Controls the number of propagation steps to add in the lambda
            region defined by `prop_lambda`.
        """
        #store parameters into a dictionary so they can easily be accessed
        #in the case that a new integrator created by the Moves class
        #needs to access them
        self.args = args
        self.kwargs = kwargs
        self.kwargs['alchemical_functions'] = alchemical_functions
        self.kwargs['splitting'] = splitting
        self.kwargs['temperature'] = temperature
        self.kwargs['collision_rate'] = collision_rate
        self.kwargs['timestep'] = timestep
        self.kwargs['constraint_tolerance'] = constraint_tolerance
        self.kwargs['measure_shadow_work'] = measure_shadow_work
        self.kwargs['measure_heat'] = measure_heat
        self.kwargs['nsteps_neq'] = nsteps_neq
        self.kwargs['nprop'] = nprop
        self.kwargs['prop_lambda'] = prop_lambda

        # call the base class constructor
        super(AlchemicalExternalLangevinIntegrator, self).__init__(alchemical_functions=alchemical_functions,
                                                               splitting=splitting, temperature=temperature,
                                                               collision_rate=collision_rate, timestep=timestep,
                                                               constraint_tolerance=constraint_tolerance,
                                                               measure_shadow_work=measure_shadow_work,
                                                               measure_heat=measure_heat,
                                                               nsteps_neq=nsteps_neq
                                                               )

        self._prop_lambda = self._get_prop_lambda(prop_lambda)

        # add some global variables relevant to the integrator
        kB = simtk.unit.BOLTZMANN_CONSTANT_kB * simtk.unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        self.addGlobalVariable("perturbed_pe", 0)
        self.addGlobalVariable("unperturbed_pe", 0)
        self.addGlobalVariable("first_step", 0)
        self.addGlobalVariable("nprop", nprop)
        self.addGlobalVariable("prop", 1)
        self.addGlobalVariable("prop_lambda_min", self._prop_lambda[0])
        self.addGlobalVariable("prop_lambda_max", self._prop_lambda[1])
        self._registered_step_types['H'] = (self._add_alchemical_perturbation_step, False)
        self.addGlobalVariable("debug", 0)
#        try:
#            self.getGlobalVariableByName("lambda_restraints")
#        except:
#            self.addGlobalVariable("lambda_restraints", 0)


        try:
            self.getGlobalVariableByName("shadow_work")
        except:
            self.addGlobalVariable('shadow_work', 0)

    def _get_prop_lambda(self, prop_lambda):
        prop_lambda_max = round(prop_lambda + 0.5,4)
        prop_lambda_min = round(0.5 - prop_lambda,4)
        prop_range = prop_lambda_max - prop_lambda_min

        #Set values to outside [0, 1.0] to skip IfBlock
        if prop_range <= 0.0:
            prop_lambda_min = 2.0
            prop_lambda_max = -1.0

        return prop_lambda_min, prop_lambda_max

    def updateRestraints(self):
        #self.addComputeGlobal("Eold", "energy")
        #self.addComputeGlobal('lambda_restraints', 'max(0, 1-(1/0.5)*abs(lambda-0.5))')
        self.addComputeGlobal('lambda_restraints', 'max(0, 1-(1/0.30)*abs(lambda-0.5))')
        #self.addComputeGlobal('lambda_restraints', '0')
        #self.addComputeGlobal('lambda_restraints', 'max(0, 1-(1/0.15)*abs(lambda-0.5))')
        #self.addComputeGlobal('lambda_restraints', 'max(0, 1-(1/0.10)*abs(lambda-0.5))')
        #self.addComputeGlobal("Enew", "energy")
        #self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)")

    def _add_integrator_steps(self):
        """
        Override the base class to insert reset steps around the integrator.
        """

        # First step: Constrain positions and velocities and reset work accumulators and alchemical integrators
        self.beginIfBlock('step = 0')
        self.addComputeGlobal("perturbed_pe", "energy")
        self.addComputeGlobal("unperturbed_pe", "energy")
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self._add_reset_protocol_work_step()
        self._add_alchemical_reset_step()
        self.endBlock()

        # Main body
        #try:
        #    self.getGlobalVariableByName("lambda_restraints")
        #except:
        #    self.addGlobalVariable("lambda_restraints", 0)

        if self._n_steps_neq == 0:
            # If nsteps = 0, we need to force execution on the first step only.
            self.beginIfBlock('step = 0')
            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            self.addComputeGlobal("step", "step + 1")
            self.endBlock()
        else:
            #call the superclass function to insert the appropriate steps, provided the step number is less than n_steps
            self.beginIfBlock("step < nsteps")#
            self.addComputeGlobal("perturbed_pe", "energy")
            self.beginIfBlock("first_step < 1")##
            #TODO write better test that checks that the initial work isn't gigantic
            self.addComputeGlobal("first_step", "1")
            self.addComputeGlobal("unperturbed_pe", "energy")
            self.endBlock()##
            #initial iteration
            self.addComputeGlobal("protocol_work", "protocol_work + (perturbed_pe - unperturbed_pe)")
            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            #if more propogation steps are requested
            self.beginIfBlock("lambda > prop_lambda_min")###
            self.beginIfBlock("lambda <= prop_lambda_max")####

            self.beginWhileBlock("prop < nprop")#####
            self.addComputeGlobal("prop", "prop + 1")

            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            self.endBlock()#####
            self.endBlock()####
            self.endBlock()###
            #ending variables to reset
            self.addComputeGlobal("unperturbed_pe", "energy")
            self.addComputeGlobal("step", "step + 1")
            self.addComputeGlobal("prop", "1")

            self.endBlock()#

    def _add_alchemical_perturbation_step(self):
        """
        Add alchemical perturbation step, accumulating protocol work.
        TODO: Extend this to be able to handle force groups?
        """
        # Store initial potential energy
        self.beginIfBlock("prop = 1")
        self.addComputeGlobal("debug", "debug + 1")
        self.addComputeGlobal("Eold", "energy")

        # Update lambda and increment that tracks updates.
        self.addComputeGlobal('lambda', '(lambda_step+1)/n_lambda_steps')
        self.addComputeGlobal('lambda_step', 'lambda_step + 1')

        # Update all slaved alchemical parameters
        self._add_update_alchemical_parameters_step()

        # Accumulate protocol work
        self.addComputeGlobal("Enew", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)")
        self.endBlock()

    def getLogAcceptanceProbability(self, context):
        #TODO remove context from arguments if/once ncmc_switching is changed
        protocol = self.getGlobalVariableByName("protocol_work")
        shadow = self.getGlobalVariableByName("shadow_work")
        logp_accept = -1.0*(protocol + shadow)*_OPENMM_ENERGY_UNIT / self.kT
        return logp_accept

    def reset(self):
        self.setGlobalVariableByName("step", 0)
        self.setGlobalVariableByName("lambda", 0.0)
        self.setGlobalVariableByName("protocol_work", 0.0)
        self.setGlobalVariableByName("shadow_work", 0.0)
        self.setGlobalVariableByName("first_step", 0)
        self.setGlobalVariableByName("perturbed_pe", 0.0)
        self.setGlobalVariableByName("unperturbed_pe", 0.0)
        self.setGlobalVariableByName("prop", 1)
        super(AlchemicalExternalLangevinIntegrator, self).reset()
