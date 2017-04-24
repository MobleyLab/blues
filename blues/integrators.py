from openmmtools.integrators import NonequilibriumLangevinIntegrator
import simtk

class NonequilibriumExternalLangevinIntegrator(NonequilibriumLangevinIntegrator):
    """Allows nonequilibrium switching based on force parameters specified in alchemical_functions.
    A variable named lambda is switched from 0 to 1 linearly throughout the nsteps of the protocol.
    The functions can use this to create more complex protocols for other global parameters.
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
                 steps_per_propagation=1):
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
        """

#        self._alchemical_functions = alchemical_functions
#        self._n_steps_neq = nsteps_neq

        # collect the system parameters.
#        self._system_parameters = {system_parameter for system_parameter in alchemical_functions.keys()}

        # call the base class constructor
        super(EditIntegrator, self).__init__(alchemical_functions=alchemical_functions,
                                                               splitting=splitting, temperature=temperature,
                                                               collision_rate=collision_rate, timestep=timestep,
                                                               constraint_tolerance=constraint_tolerance,
                                                               measure_shadow_work=measure_shadow_work,
                                                               measure_heat=measure_heat,
                                                               nsteps_neq=nsteps_neq
                                                               )

        # add some global variables relevant to the integrator
        ##self.add_global_variables(nsteps=nsteps_neq)
        kB = simtk.unit.BOLTZMANN_CONSTANT_kB * simtk.unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        self.kT = kT
        self.addGlobalVariable("perturbed_pe", 0)
        self.addGlobalVariable("unperturbed_pe", 0)
        self.addGlobalVariable("first_step", 0)
        self.addGlobalVariable("psteps", steps_per_propagation)
        self.addGlobalVariable("pstep", 0)

    def add_integrator_steps(self, splitting, measure_shadow_work, measure_heat, ORV_counts, force_group_nV, mts):
        self.addComputeGlobal("perturbed_pe", "energy")
        # Assumes no perturbation is done before doing the initial MD step.
        self.beginIfBlock("first_step < 1")
        self.addComputeGlobal("first_step", "1")
        self.addComputeGlobal("unperturbed_pe", "energy")
        self.endBlock()
        self.addComputeGlobal("perturbed_pe", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (perturbed_pe - unperturbed_pe)")
        #repeat BAOAB integration psteps number of times per lambda
        self.addComputeGlobal("pstep", "0")
        self.beginWhileBlock('pstep < psteps')
        super(EditIntegrator, self).add_integrator_steps(splitting, measure_shadow_work, measure_heat, ORV_counts, force_group_nV, mts)
        self.beginIfBlock("pstep < psteps - 1")
        self.addComputeGlobal("step", "step - 1")
        self.endBlock()
        self.addComputeGlobal("pstep", "pstep + 1")
        self.endBlock()
        #update unperturbed_pe
        self.addComputeGlobal("unperturbed_pe", "energy")

    def getLogAcceptanceProbability(self, context):
        protocol = self.getGlobalVariableByName("protocol_work")
        shadow = self.getGlobalVariableByName("shadow_work")
        logp_accept = -1.0*(protocol + shadow)
        return logp_accept

    def reset(self):
        self.setGlobalVariableByName("step", 0)
        self.setGlobalVariableByName("lambda", 0.0)
#        self.setGlobalVariableByName("total_work", 0.0)
        self.setGlobalVariableByName("protocol_work", 0.0)
        self.setGlobalVariableByName("shadow_work", 0.0)
        self.setGlobalVariableByName("first_step", 0)
