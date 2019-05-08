"""
Provides classes for setting up and running the BLUES simulation.

- `SystemFactory` : setup and modifying the OpenMM System prior to the simulation.
- `SimulationFactory` : generates the OpenMM Simulations from the System.
- `BLUESSimulation` : runs the NCMC+MD hybrid simulation.
- `MonteCarloSimulation` : runs a pure Monte Carlo simulation.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, Meghan Osato, David L. Mobley
"""

import logging
import math
import sys
import copy

import numpy as np
import parmed
from openmmtools import alchemy
from simtk import openmm, unit
from simtk.openmm import app
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from openmmtools import cache

from blues import utils
from blues.integrators import AlchemicalExternalLangevinIntegrator

finfo = np.finfo(np.float32)
rtol = finfo.precision
logger = logging.getLogger(__name__)


class SystemFactory(object):
    """
    SystemFactory contains methods to generate/modify the OpenMM System object
    required for generating the openmm.Simulation using a given
    parmed.Structure()

    Examples
    --------
    Load Parmed Structure, select move type, initialize `MoveEngine`, and
    generate the openmm.Systems

    >>> structure = parmed.load_file('eqToluene.prmtop', xyz='eqToluene.inpcrd')
    >>> ligand = RandomLigandRotationMove(structure, 'LIG')
    >>> ligand_mover = MoveEngine(ligand)
    >>> systems = SystemFactory(structure, ligand.atom_indices, config['system'])

    The MD and alchemical Systems are generated and stored as an attribute

    >>> systems.md
    >>> systems.alch

    Freeze atoms in the alchemical system

    >>> systems.alch = SystemFactory.freeze_atoms(systems.alch,
                                            freeze_distance=5.0,
                                            freeze_center='LIG'
                                            freeze_solvent='HOH,NA,CL')

    Parameters
    ----------
    structure : parmed.Structure
        A chemical structure composed of atoms, bonds, angles, torsions, and
        other topological features.
    atom_indices : list of int
        Atom indicies of the move or designated for which the nonbonded forces
        (both sterics and electrostatics components) have to be alchemically
        modified.
    config : dict, parameters for generating the `openmm.System` for the MD
        and NCMC simulation. For complete parameters, see docs for `generateSystem`
        and `generateAlchSystem`
    """

    def __init__(self, structure, atom_indices, config=None):
        self.structure = structure
        self.atom_indices = atom_indices
        self._config = config

        #If parameters for generating the openmm.System is given, make them.
        if self._config:
            if 'alchemical' in self._config.keys():
                self.alch_config = self._config.pop('alchemical')
            else:
                #Use function defaults if none is provided
                self.alch_config = {}
            self.md = SystemFactory.generateSystem(self.structure, **self._config)
            self.alch = SystemFactory.generateAlchSystem(self.md, self.atom_indices, **self.alch_config)

    @staticmethod
    def amber_selection_to_atomidx(structure, selection):
        """
        Converts AmberMask selection [amber-syntax]_ to list of atom indices.

        Parameters
        ----------
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        selection : str
            AmberMask selection that gets converted to a list of atom indices.

        Returns
        -------
        mask_idx : list of int
            List of atom indices.

        References
        ----------
        .. [amber-syntax] J. Swails, ParmEd Documentation (2015). http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax

        """
        mask = parmed.amber.AmberMask(structure, str(selection))
        mask_idx = [i for i in mask.Selected()]
        return mask_idx

    @staticmethod
    def atomidx_to_atomlist(structure, mask_idx):
        """
        Goes through the structure and matches the previously selected atom
        indices to the atom type.

        Parameters
        ----------
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        mask_idx : list of int
            List of atom indices.

        Returns
        -------
        atom_list : list of atoms
            The atoms that were previously selected in mask_idx.
        """
        atom_list = []
        for i, at in enumerate(structure.atoms):
            if i in mask_idx:
                atom_list.append(structure.atoms[i])
        logger.debug('\nFreezing {}'.format(atom_list))
        return atom_list

    @classmethod
    def generateSystem(cls, structure, **kwargs):
        """
        Construct an OpenMM System representing the topology described by the
        prmtop file. This function is just a wrapper for parmed Structure.createSystem().

        Parameters
        ----------
        structure : parmed.Structure()
            The parmed.Structure of the molecular system to be simulated
        nonbondedMethod : cutoff method
            This is the cutoff method. It can be either the NoCutoff,
            CutoffNonPeriodic, CutoffPeriodic, PME, or Ewald objects from the
            simtk.openmm.app namespace
        nonbondedCutoff : float or distance Quantity
            The nonbonded cutoff must be either a floating point number
            (interpreted as nanometers) or a Quantity with attached units. This
            is ignored if nonbondedMethod is NoCutoff.
        switchDistance : float or distance Quantity
            The distance at which the switching function is turned on for van
            der Waals interactions. This is ignored when no cutoff is used, and
            no switch is used if switchDistance is 0, negative, or greater than
            the cutoff
        constraints : None, app.HBonds, app.HAngles, or app.AllBonds
            Which type of constraints to add to the system (e.g., SHAKE). None
            means no bonds are constrained. HBonds means bonds with hydrogen are
            constrained
        rigidWater : bool=True
            If True, water is kept rigid regardless of the value of constraints.
            A value of False is ignored if constraints is not None.
        implicitSolvent : None, app.HCT, app.OBC1, app.OBC2, app.GBn, app.GBn2
            The Generalized Born implicit solvent model to use.
        implicitSolventKappa : float or 1/distance Quantity = None
            This is the Debye kappa property related to modeling saltwater
            conditions in GB. It should have units of 1/distance (1/nanometers
            is assumed if no units present). A value of None means that kappa
            will be calculated from implicitSolventSaltConc (below)
        implicitSolventSaltConc : float or amount/volume Quantity=0 moles/liter
            If implicitSolventKappa is None, the kappa will be computed from the
            salt concentration. It should have units compatible with mol/L
        temperature : float or temperature Quantity = 298.15 kelvin
            This is only used to compute kappa from implicitSolventSaltConc
        soluteDielectric : float=1.0
            The dielectric constant of the protein interior used in GB
        solventDielectric : float=78.5
            The dielectric constant of the water used in GB
        useSASA : bool=False
            If True, use the ACE non-polar solvation model. Otherwise, use no
            SASA-based nonpolar solvation model.
        removeCMMotion : bool=True
            If True, the center-of-mass motion will be removed periodically
            during the simulation. If False, it will not.
        hydrogenMass : float or mass quantity = None
            If not None, hydrogen masses will be changed to this mass and the
            difference subtracted from the attached heavy atom (hydrogen mass
            repartitioning)
        ewaldErrorTolerance : float=0.0005
            When using PME or Ewald, the Ewald parameters will be calculated
            from this value
        flexibleConstraints : bool=True
            If False, the energies and forces from the constrained degrees of
            freedom will NOT be computed. If True, they will (but those degrees
            of freedom will *still* be constrained).
        verbose : bool=False
            If True, the progress of this subroutine will be printed to stdout
        splitDihedrals : bool=False
            If True, the dihedrals will be split into two forces -- proper and
            impropers. This is primarily useful for debugging torsion parameter
            assignments.

        Returns
        -------
        openmm.System
            System formatted according to the prmtop file.

        Notes
        -----
        This function calls prune_empty_terms if any Topology lists have
        changed.
        """
        return structure.createSystem(**kwargs)

    @classmethod
    def generateAlchSystem(cls,
                           system,
                           atom_indices,
                           softcore_alpha=0.5,
                           softcore_a=1,
                           softcore_b=1,
                           softcore_c=6,
                           softcore_beta=0.0,
                           softcore_d=1,
                           softcore_e=1,
                           softcore_f=2,
                           annihilate_electrostatics=True,
                           annihilate_sterics=False,
                           disable_alchemical_dispersion_correction=True,
                           alchemical_pme_treatment='direct-space',
                           suppress_warnings=True,
                           **kwargs):
        """Returns the OpenMM System for alchemical perturbations.
        This function calls `openmmtools.alchemy.AbsoluteAlchemicalFactory` and
        `openmmtools.alchemy.AlchemicalRegion` to generate the System for the
        NCMC simulation.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list of int
            Atom indicies of the move or designated for which the nonbonded forces
            (both sterics and electrostatics components) have to be alchemically
            modified.
        annihilate_electrostatics : bool, optional
            If True, electrostatics should be annihilated, rather than decoupled
            (default is True).
        annihilate_sterics : bool, optional
            If True, sterics (Lennard-Jones or Halgren potential) will be annihilated,
            rather than decoupled (default is False).
        softcore_alpha : float, optional
            Alchemical softcore parameter for Lennard-Jones (default is 0.5).
        softcore_a, softcore_b, softcore_c : float, optional
            Parameters modifying softcore Lennard-Jones form. Introduced in
            Eq. 13 of Ref. [TTPham-JChemPhys135-2011]_ (default is 1).
        softcore_beta : float, optional
            Alchemical softcore parameter for electrostatics. Set this to zero
            to recover standard electrostatic scaling (default is 0.0).
        softcore_d, softcore_e, softcore_f : float, optional
            Parameters modifying softcore electrostatics form (default is 1).
        disable_alchemical_dispersion_correction : bool, optional, default=True
            If True, the long-range dispersion correction will not be included for the alchemical
            region to avoid the need to recompute the correction (a CPU operation that takes ~ 0.5 s)
            every time 'lambda_sterics' is changed. If using nonequilibrium protocols, it is recommended
            that this be set to True since this can lead to enormous (100x) slowdowns if the correction
            must be recomputed every time step.
        alchemical_pme_treatment : str, optional, default = 'direct-space'
            Controls how alchemical region electrostatics are treated when PME is used.
            Options are 'direct-space', 'coulomb', 'exact'.
            - 'direct-space' only models the direct space contribution
            - 'coulomb' includes switched Coulomb interaction
            - 'exact' includes also the reciprocal space contribution, but it's
            only possible to annihilate the charges and the softcore parameters
            controlling the electrostatics are deactivated. Also, with this
            method, modifying the global variable `lambda_electrostatics` is
            not sufficient to control the charges. The recommended way to change
            them is through the `AlchemicalState` class.

        Returns
        -------
        alch_system : alchemical_system
            System to be used for the NCMC simulation.

        References
        ----------
        .. [TTPham-JChemPhys135-2011] T. T. Pham and M. R. Shirts, J. Chem. Phys 135, 034114 (2011). http://dx.doi.org/10.1063/1.3607597
        """
        if suppress_warnings:
            #Lower logger level to suppress excess warnings
            logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)

        #Disabled correction term due to increased computational cost
        factory = alchemy.AbsoluteAlchemicalFactory(
            disable_alchemical_dispersion_correction=disable_alchemical_dispersion_correction,
            alchemical_pme_treatment=alchemical_pme_treatment)
        alch_region = alchemy.AlchemicalRegion(
            alchemical_atoms=atom_indices,
            softcore_alpha=softcore_alpha,
            softcore_a=softcore_a,
            softcore_b=softcore_b,
            softcore_c=softcore_c,
            softcore_beta=softcore_beta,
            softcore_d=softcore_d,
            softcore_e=softcore_e,
            softcore_f=softcore_f,
            annihilate_electrostatics=annihilate_electrostatics,
            annihilate_sterics=annihilate_sterics)

        alch_system = factory.create_alchemical_system(system, alch_region)
        return alch_system

    @classmethod
    def restrain_positions(cls, structure, system, selection="(@CA,C,N)", weight=5.0, **kwargs):
        """
        Applies positional restraints to atoms in the openmm.System
        by the given parmed selection [amber-syntax]_.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        selection : str, Default = "(@CA,C,N)"
            AmberMask selection to apply positional restraints to
        weight : float, Default = 5.0
            Restraint weight for xyz atom restraints in kcal/(mol A^2)

        Returns
        -------
        system : openmm.System
            Modified with positional restraints applied.

        """
        mask_idx = cls.amber_selection_to_atomidx(structure, selection)

        logger.info("{} positional restraints applied to selection: '{}' ({} atoms) on {}".format(
            weight, selection, len(mask_idx), system))
        # define the custom force to restrain atoms to their starting positions
        force = openmm.CustomExternalForce('k_restr*periodicdistance(x, y, z, x0, y0, z0)^2')
        # Add the restraint weight as a global parameter in kcal/mol/A^2
        force.addGlobalParameter("k_restr", weight)
        #force.addGlobalParameter("k_restr", weight*unit.kilocalories_per_mole/unit.angstroms**2)
        # Define the target xyz coords for the restraint as per-atom (per-particle) parameters
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        for i, atom_crd in enumerate(structure.positions):
            if i in mask_idx:
                logger.debug(i, structure.atoms[i])
                force.addParticle(i, atom_crd.value_in_unit(unit.nanometers))
        system.addForce(force)

        return system

    @classmethod
    def freeze_atoms(cls, structure, system, freeze_selection=":LIG", **kwargs):
        """
        Zeroes the masses of atoms from the given parmed selection [amber-syntax]_.
        Massless atoms will be ignored by the integrator and will not change
        positions.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        freeze_selection : str, Default = ":LIG"
            AmberMask selection for the center in which to select atoms for
            zeroing their masses.
            Defaults to freezing protein backbone atoms.

        Returns
        -------
        system : openmm.System
            The modified system with the selected atoms
        """
        mask_idx = cls.amber_selection_to_atomidx(structure, freeze_selection)
        logger.info("Freezing selection '{}' ({} atoms) on {}".format(freeze_selection, len(mask_idx), system))

        cls.atomidx_to_atomlist(structure, mask_idx)
        system = utils.zero_masses(system, mask_idx)
        return system

    @classmethod
    def freeze_radius(cls,
                      structure,
                      system,
                      freeze_distance=5.0 * unit.angstrom,
                      freeze_center=':LIG',
                      freeze_solvent=':HOH,NA,CL',
                      **kwargs):
        """
        Zero the masses of atoms outside the given raidus of
        the `freeze_center` parmed selection [amber-syntax]_. Massless atoms will be ignored by the
        integrator and will not change positions.This is intended to freeze
        the solvent and protein atoms around the ligand binding site.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        freeze_distance : float, Default = 5.0
            Distance (angstroms) to select atoms for retaining their masses.
            Atoms outside the set distance will have their masses set to 0.0.
        freeze_center : str, Default = ":LIG"
            AmberMask selection for the center in which to select atoms for
            zeroing their masses. Default: LIG
        freeze_solvent : str, Default = ":HOH,NA,CL"
            AmberMask selection in which to select solvent atoms for zeroing
            their masses.

        Returns
        -------
        system : openmm.System
            Modified system with masses outside the `freeze center` zeroed.

        """
        N_atoms = system.getNumParticles()
        #Select the LIG and atoms within 5 angstroms, except for WAT or IONS (i.e. selects the binding site)
        if hasattr(freeze_distance, '_value'): freeze_distance = freeze_distance._value
        selection = "(%s<:%f)&!(%s)" % (freeze_center, freeze_distance, freeze_solvent)
        logger.info('Inverting parmed selection for freezing: %s' % selection)
        site_idx = cls.amber_selection_to_atomidx(structure, selection)
        #Invert that selection to freeze everything but the binding site.
        freeze_idx = set(range(N_atoms)) - set(site_idx)

        #Check if freeze selection has selected all atoms
        if len(freeze_idx) == N_atoms:
            err = 'All %i atoms appear to be selected for freezing. Check your atom selection.' % len(freeze_idx)
            logger.error(err)
            sys.exit(1)

        freeze_threshold = 0.98
        if len(freeze_idx) / N_atoms == freeze_threshold:
            err = '%.0f%% of your system appears to be selected for freezing. Check your atom selection' % (
                100 * freeze_threshold)
            logger.error(err)
            sys.exit(1)

        #Ensure that the freeze selection is larger than the center selection of atoms
        center_idx = cls.amber_selection_to_atomidx(structure, freeze_center)
        if len(site_idx) <= len(center_idx):
            err = "%i unfrozen atoms is less than (or equal to) the number of atoms used as the selection center '%s' (%i atoms). Check your atom selection." % (
                len(site_idx), freeze_center, len(center_idx))
            logger.error(err)
            sys.exit(1)

        freeze_warning = 0.80
        if len(freeze_idx) / N_atoms == freeze_warning:
            warn = '%.0f%% of your system appears to be selected for freezing. This may cause unexpected behaviors.' % (
                100 * freeze_warning)
            logger.warm(warn)
            sys.exit(1)

        #Ensure that the freeze selection is larger than the center selection point
        center_idx = cls.amber_selection_to_atomidx(structure, freeze_center)
        if len(site_idx) <= len(center_idx):
            err = "%i unfrozen atoms is less than (or equal to) the number of atoms from the selection center '%s' (%i atoms). Check your atom selection." % (
                len(site_idx), freeze_center, len(center_idx))
            logger.error(err)
            sys.exit(1)

        logger.info("Freezing {} atoms {} Angstroms from '{}' on {}".format(
            len(freeze_idx), freeze_distance, freeze_center, system))

        cls.atomidx_to_atomlist(structure, freeze_idx)
        system = utils.zero_masses(system, freeze_idx)
        return system

    @classmethod
    def addBarostat(cls, system, temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres, frequency=25, **kwargs):
        """
        Adds a MonteCarloBarostat to the MD system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        pressure : int, configional, default=None
            Pressure (atm) for Barostat for NPT simulations.
        frequency : int, default=25
            Frequency at which Monte Carlo pressure changes should be attempted (in time steps)

        Returns
        -------
        system : openmm.System
            The OpenMM System with the MonteCarloBarostat attached.
        """
        logger.info('Adding MonteCarloBarostat with {}. MD simulation will be {} NPT.'.format(pressure, temperature))
        # Add Force Barostat to the system
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, frequency))
        return system

class BLUESSampler(object):
    def __init__(self,
                 atom_subset=None,
                 thermodynamic_state=None,
                 sampler_state=None,
                 dynamics_move=None,
                 ncmc_move=None,
                 platform=None,
                 topology=None,
                 verbose=False):
        """
        Create an MCMC sampler.
        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to simulate
        sampler_state : SamplerState
            The initial sampler state to simulate from.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, this platform will be used
        ncfile : netCDF4.Dataset, optional, default=None
            NetCDF storage file.
        """

        if thermodynamic_state is None:
            raise Exception("'thermodynamic_state' must be specified")
        if sampler_state is None:
            raise Exception("'sampler_state' must be specified")

        self.atom_subset = atom_subset
        # Make a deep copy of the state so that initial state is unchanged.
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        self.alch_thermodynamic_state = self._get_alchemical_state(thermodynamic_state)
        self.sampler_state = sampler_state

        self.ncmc_move = ncmc_move
        self.dynamics_move = dynamics_move

        #NML: Attach topology to thermodynamic_states
        self.thermodynamic_state.topology = topology
        self.alch_thermodynamic_state.topology = topology

        # Initialize
        self.accept = False
        self.iteration = 0
        self.n_accepted = 0

        self.verbose = verbose
        self.platform = platform

    def _get_alchemical_state(self, thermodynamic_state):
        alch_system = SystemFactory.generateAlchSystem(thermodynamic_state.get_system(), self.atom_subset)
        alch_state = alchemy.AlchemicalState.from_system(alch_system)
        alch_thermodynamic_state = ThermodynamicState(alch_system, thermodynamic_state.temperature)
        alch_thermodynamic_state = CompoundThermodynamicState(alch_thermodynamic_state,
                                                     composable_states=[alch_state])

        return alch_thermodynamic_state


    def _acceptRejectMove(self):
        # Create MD context with the final positions from NCMC simulation
        integrator = self.dynamics_move._get_integrator(self.thermodynamic_state)
        context, integrator = self.dynamics_move.context_cache.get_context(self.thermodynamic_state, integrator)
        self.sampler_state.apply_to_context(context, ignore_velocities=True)
        alch_energy = self.thermodynamic_state.reduced_potential(context)

        correction_factor = (self.ncmc_move.initial_energy - self.dynamics_move.final_energy + alch_energy - self.ncmc_move.final_energy)
        logp_accept = self.ncmc_move.logp_accept
        randnum = math.log(np.random.random())
        #print("logP {} + corr {}".format(logp_accept, correction_factor))
        logp_accept = logp_accept + correction_factor
        if (not np.isnan(logp_accept) and logp_accept > randnum):
            logger.debug('NCMC MOVE ACCEPTED: logP {}'.format(logp_accept))
            self.accept = True
            self.n_accepted += 1
        else:
            logger.debug('NCMC MOVE REJECTED: logP {}'.format(logp_accept))
            self.accept = False

            # Restore original positions.
            self.sampler_state.positions = self.ncmc_move.initial_positions

    def equil(self, n_iterations=1):
        self.dynamics_move.totalSteps = int(self.dynamics_move.n_steps*n_iterations)
        # Set initial conditions by running 1 iteration of MD first
        for iteration in range(n_iterations):
            self.dynamics_move.apply(self.thermodynamic_state, self.sampler_state)
        self.dynamics_move.currentStep = 0
        self.iteration +=1

    def run(self, n_iterations=1):
        """
        Run the sampler for the specified number of iterations
        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        self.ncmc_move.totalSteps = int(self.ncmc_move.n_steps*n_iterations)
        self.dynamics_move.totalSteps = int(self.dynamics_move.n_steps*n_iterations)

        if self.iteration == 0:
            # Set initial conditions by running 1 iteration of MD first
            self.equil(1)

        self.iteration = 0
        for iteration in range(n_iterations):
            if self.verbose:
                print("." * 80)
                print("BLUES Sampler iteration %d" % self.iteration)

            #print('NCMC Simulation')
            self.ncmc_move.apply(self.alch_thermodynamic_state, self.sampler_state)

            self._acceptRejectMove()

            #print('MD Simulation')
            self.dynamics_move.apply(self.thermodynamic_state, self.sampler_state)

            # Increment iteration count
            self.iteration += 1

            if self.verbose:
                print("." * 80)

        #print('n_accepted', self.n_accepted)
        #print('iteration', self.iteration)

# class BLUESSimulation(object):
#     """BLUESSimulation class provides methods to execute the NCMC+MD
#     simulation.
#
#     Parameters
#     ----------
#     simulations : blues.simulation.SimulationFactory object
#         SimulationFactory Object which carries the 3 required
#         OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.
#     config : dict
#         Dictionary of parameters for configuring the OpenMM Simulations
#         If None, will search for configuration parameters on the `simulations`
#         object.
#
#     Examples
#     --------
#     Create our SimulationFactory object and run `BLUESSimulation`
#
#     >>> sim_cfg = { 'platform': 'OpenCL',
#                     'properties' : { 'OpenCLPrecision': 'single',
#                                      'OpenCLDeviceIndex' : 2},
#                     'nprop' : 1,
#                     'propLambda' : 0.3,
#                     'dt' : 0.001 * unit.picoseconds,
#                     'friction' : 1 * 1/unit.picoseconds,
#                     'temperature' : 100 * unit.kelvin,
#                     'nIter': 1,
#                     'nstepsMD': 10,
#                     'nstepsNC': 10,}
#     >>> simulations = SimulationFactory(systems, ligand_mover, sim_cfg)
#     >>> blues = BLUESSimulation(simulations)
#     >>> blues.run()
#
#     """
#
#     def __init__(self, simulations, config=None):
#         self._move_engine = simulations._move_engine
#         self._md_sim = simulations.md
#         self._alch_sim = simulations.alch
#         self._ncmc_sim = simulations.ncmc
#
#         # Check if configuration has been specified in `SimulationFactory` object
#         if not config:
#             if hasattr(simulations, 'config'):
#                 self._config = simulations.config
#         else:
#             #Otherwise take specified config
#             self._config = config
#         if self._config:
#             self._printSimulationTiming()
#
#         self.accept = 0
#         self.reject = 0
#         self.acceptRatio = 0
#         self.currentIter = 0
#
#         #Dict to keep track of each simulation state before/after each iteration
#         self.stateTable = {'md': {'state0': {}, 'state1': {}}, 'ncmc': {'state0': {}, 'state1': {}}}
#
#         #specify nc integrator variables to report in verbose output
#         self._integrator_keys_ = ['lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew']
#
#         self._state_keys = {
#             'getPositions': True,
#             'getVelocities': True,
#             'getForces': False,
#             'getEnergy': True,
#             'getParameters': True,
#             'enforcePeriodicBox': True
#         }
#
#     @classmethod
#     def getStateFromContext(cls, context, state_keys):
#         """Gets the State information from the given context and
#         list of state_keys to query it with.
#
#         Returns the state data as a dict.
#
#         Parameters
#         ----------
#         context : openmm.Context
#             Context of the OpenMM Simulation to query.
#         state_keys : list
#             Default: [ positions, velocities, potential_energy, kinetic_energy ]
#             A list that defines what information to get from the context State.
#
#         Returns
#         -------
#         stateinfo : dict
#             Current positions, velocities, energies and box vectors of the context.
#         """
#
#         stateinfo = {}
#         state = context.getState(**state_keys)
#         stateinfo['positions'] = state.getPositions(asnp=True)
#         stateinfo['velocities'] = state.getVelocities(asnp=True)
#         stateinfo['potential_energy'] = state.getPotentialEnergy()
#         stateinfo['kinetic_energy'] = state.getKineticEnergy()
#         stateinfo['box_vectors'] = state.getPeriodicBoxVectors()
#         return stateinfo
#
#     @classmethod
#     def getIntegratorInfo(cls,
#                           ncmc_integrator,
#                           integrator_keys=['lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew']):
#         """Returns a dict of alchemical/ncmc-swtiching data from querying the the NCMC
#         integrator.
#
#         Parameters
#         ----------
#         ncmc_integrator : openmm.Context.Integrator
#             The integrator from the NCMC Context
#         integrator_keys : list
#             list containing strings of the values to get from the integrator.
#             Default = ['lambda', 'shadow_work', 'protocol_work', 'Eold', 'Enew','Epert']
#
#         Returns
#         -------
#         integrator_info : dict
#             Work values and energies from the NCMC integrator.
#         """
#         integrator_info = {}
#         for key in integrator_keys:
#             integrator_info[key] = ncmc_integrator.getGlobalVariableByName(key)
#         return integrator_info
#
#     @classmethod
#     def setContextFromState(cls, context, state, box=True, positions=True, velocities=True):
#         """Update a given Context from the given State.
#
#         Parameters
#         ----------
#         context : openmm.Context
#             The Context to be updated from the given State.
#         state : openmm.State
#             The current state (box_vectors, positions, velocities) of the
#             Simulation to update the given context.
#
#         Returns
#         -------
#         context : openmm.Context
#             The updated Context whose box_vectors, positions, and velocities
#             have been updated.
#         """
#         # Replace ncmc data from the md context
#         if box:
#             context.setPeriodicBoxVectors(*state['box_vectors'])
#         if positions:
#             context.setPositions(state['positions'])
#         if velocities:
#             context.setVelocities(state['velocities'])
#         return context
#
#     def _printSimulationTiming(self):
#         """Prints the simulation timing and related information."""
#
#         dt = self._config['dt'].value_in_unit(unit.picoseconds)
#         nIter = self._config['nIter']
#         nprop = self._config['nprop']
#         propLambda = self._config['propLambda']
#         propSteps = self._config['propSteps']
#         nstepsNC = self._config['nstepsNC']
#         nstepsMD = self._config['nstepsMD']
#
#         force_eval = nIter * (propSteps + nstepsMD)
#         time_ncmc_iter = propSteps * dt
#         time_ncmc_total = time_ncmc_iter * nIter
#         time_md_iter = nstepsMD * dt
#         time_md_total = time_md_iter * nIter
#         time_iter = time_ncmc_iter + time_md_iter
#         time_total = time_iter * nIter
#
#         msg = 'Total BLUES Simulation Time = %s ps (%s ps/Iter)\n' % (time_total, time_iter)
#         msg += 'Total Force Evaluations = %s \n' % force_eval
#         msg += 'Total NCMC time = %s ps (%s ps/iter)\n' % (time_ncmc_total, time_ncmc_iter)
#
#         # Calculate number of lambda steps inside/outside region with extra propgation steps
#         steps_in_prop = int(nprop * (2 * math.floor(propLambda * nstepsNC)))
#         steps_out_prop = int((2 * math.ceil((0.5 - propLambda) * nstepsNC)))
#
#         prop_lambda_window = self._ncmc_sim.context._integrator._prop_lambda
#         # prop_range = round(prop_lambda_window[1] - prop_lambda_window[0], 4)
#         if propSteps != nstepsNC:
#             msg += '\t%s lambda switching steps within %s total propagation steps.\n' % (nstepsNC, propSteps)
#             msg += '\tExtra propgation steps between lambda [%s, %s]\n' % (prop_lambda_window[0],
#                                                                            prop_lambda_window[1])
#             msg += '\tLambda: 0.0 -> %s = %s propagation steps\n' % (prop_lambda_window[0], int(steps_out_prop / 2))
#             msg += '\tLambda: %s -> %s = %s propagation steps\n' % (prop_lambda_window[0], prop_lambda_window[1],
#                                                                     steps_in_prop)
#             msg += '\tLambda: %s -> 1.0 = %s propagation steps\n' % (prop_lambda_window[1], int(steps_out_prop / 2))
#
#         msg += 'Total MD time = %s ps (%s ps/iter)\n' % (time_md_total, time_md_iter)
#
#         #Get trajectory frame interval timing for BLUES simulation
#         if 'md_trajectory_interval' in self._config.keys():
#             frame_iter = nstepsMD / self._config['md_trajectory_interval']
#             timetraj_frame = (time_ncmc_iter + time_md_iter) / frame_iter
#             msg += 'Trajectory Interval = %s ps/frame (%s frames/iter)' % (timetraj_frame, frame_iter)
#
#         logger.info(msg)
#
#     def _setStateTable(self, simkey, stateidx, stateinfo):
#         """Updates `stateTable` (dict) containing:  Positions, Velocities, Potential/Kinetic energies
#         of the state before and after a NCMC step or iteration.
#
#         Parameters
#         ----------
#         simkey : str (key: 'md', 'ncmc', 'alch')
#             Key corresponding to the simulation.
#         stateidx : str (key: 'state0' or 'state1')
#             Key corresponding to the state information being stored.
#         stateinfo : dict
#             Dictionary containing the State information.
#         """
#         self.stateTable[simkey][stateidx] = stateinfo
#
#     def _syncStatesMDtoNCMC(self):
#         """Retrieves data on the current State of the MD context to
#         replace the box vectors, positions, and velocties in the NCMC context.
#         """
#         # Retrieve MD state from previous iteration
#         md_state0 = self.getStateFromContext(self._md_sim.context, self._state_keys)
#         self._setStateTable('md', 'state0', md_state0)
#
#         # Sync MD state to the NCMC context
#         self._ncmc_sim.context = self.setContextFromState(self._ncmc_sim.context, md_state0)
#
#     def _stepNCMC(self, nstepsNC, moveStep, move_engine=None):
#         """Advance the NCMC simulation.
#
#         Parameters
#         ----------
#         nstepsNC : int
#             The number of NCMC switching steps to advance by.
#         moveStep : int
#             The step number to perform the chosen move, which should be half
#             the number of nstepsNC.
#         move_engine : blues.moves.MoveEngine
#             The object that executes the chosen move.
#
#         """
#
#         logger.info('Advancing %i NCMC switching steps...' % (nstepsNC))
#         # Retrieve NCMC state before proposed move
#         ncmc_state0 = self.getStateFromContext(self._ncmc_sim.context, self._state_keys)
#         self._setStateTable('ncmc', 'state0', ncmc_state0)
#
#         #choose a move to be performed according to move probabilities
#         #TODO: will have to change to work with multiple alch region
#         if not move_engine: move_engine = self._move_engine
#         self._ncmc_sim.currentIter = self.currentIter
#         move_engine.selectMove()
#
#         lastStep = nstepsNC - 1
#         for step in range(int(nstepsNC)):
#             try:
#                 #Attempt anything related to the move before protocol is performed
#                 if not step:
#                     self._ncmc_sim.context = move_engine.selected_move.beforeMove(self._ncmc_sim.context)
#
#                 # Attempt selected MoveEngine Move at the halfway point
#                 #to ensure protocol is symmetric
#                 if step == moveStep:
#                     if hasattr(logger, 'report'):
#                         logger.info = logger.report
#                     #Do move
#                     logger.info('Performing %s...' % move_engine.move_name)
#                     self._ncmc_sim.context = move_engine.runEngine(self._ncmc_sim.context)
#
#                 # Do 1 NCMC step with the integrator
#                 self._ncmc_sim.step(1)
#
#                 #Attempt anything related to the move after protocol is performed
#                 if step == lastStep:
#                     self._ncmc_sim.context = move_engine.selected_move.afterMove(self._ncmc_sim.context)
#
#             except Exception as e:
#                 logger.error(e)
#                 move_engine.selected_move._error(self._ncmc_sim.context)
#                 break
#
#         # ncmc_state1 stores the state AFTER a proposed move.
#         ncmc_state1 = self.getStateFromContext(self._ncmc_sim.context, self._state_keys)
#         self._setStateTable('ncmc', 'state1', ncmc_state1)
#
#     def _computeAlchemicalCorrection(self):
#         """Computes the alchemical correction term from switching between the NCMC
#         and MD potentials."""
#         # Retrieve the MD/NCMC state before the proposed move.
#         md_state0_PE = self.stateTable['md']['state0']['potential_energy']
#         ncmc_state0_PE = self.stateTable['ncmc']['state0']['potential_energy']
#
#         # Retreive the NCMC state after the proposed move.
#         ncmc_state1 = self.stateTable['ncmc']['state1']
#         ncmc_state1_PE = ncmc_state1['potential_energy']
#
#         # Set the box_vectors and positions in the alchemical simulation to after the proposed move.
#         self._alch_sim.context = self.setContextFromState(self._alch_sim.context, ncmc_state1, velocities=False)
#
#         # Retrieve potential_energy for alch correction
#         alch_PE = self._alch_sim.context.getState(getEnergy=True).getPotentialEnergy()
#         correction_factor = (ncmc_state0_PE - md_state0_PE + alch_PE - ncmc_state1_PE) * (
#             -1.0 / self._ncmc_sim.context._integrator.kT)
#
#         return correction_factor
#
#     def _acceptRejectMove(self, write_move=False):
#         """Choose to accept or reject the proposed move based
#         on the acceptance criterion.
#
#         Parameters
#         ----------
#         write_move : bool, default=False
#             If True, writes the proposed NCMC move to a PDB file.
#         """
#         work_ncmc = self._ncmc_sim.context._integrator.getLogAcceptanceProbability(self._ncmc_sim.context)
#         randnum = math.log(np.random.random())
#
#         # Compute correction if work_ncmc is not NaN
#         if not np.isnan(work_ncmc):
#             correction_factor = self._computeAlchemicalCorrection()
#             logger.debug(
#                 'NCMCLogAcceptanceProbability = %.6f + Alchemical Correction = %.6f' % (work_ncmc, correction_factor))
#             work_ncmc = work_ncmc + correction_factor
#
#         if work_ncmc > randnum:
#             self.accept += 1
#             logger.info('NCMC MOVE ACCEPTED: work_ncmc {} > randnum {}'.format(work_ncmc, randnum))
#
#             # If accept move, sync NCMC state to MD context
#             ncmc_state1 = self.stateTable['ncmc']['state1']
#             self._md_sim.context = self.setContextFromState(self._md_sim.context, ncmc_state1, velocities=False)
#
#             if write_move:
#                 utils.saveSimulationFrame(self._md_sim, '{}acc-it{}.pdb'.format(self._config['outfname'],
#                                                                                 self.currentIter))
#
#         else:
#             self.reject += 1
#             logger.info('NCMC MOVE REJECTED: work_ncmc {} < {}'.format(work_ncmc, randnum))
#
#             # If reject move, do nothing,
#             # NCMC simulation be updated from MD Simulation next iteration.
#
#             # Potential energy should be from last MD step in the previous iteration
#             md_state0 = self.stateTable['md']['state0']
#             md_PE = self._md_sim.context.getState(getEnergy=True).getPotentialEnergy()
#             if not math.isclose(md_state0['potential_energy']._value, md_PE._value, rel_tol=float('1e-%s' % rtol)):
#                 logger.error(
#                     'Last MD potential energy %s != Current MD potential energy %s. Potential energy should match the prior state.'
#                     % (md_state0['potential_energy'], md_PE))
#                 sys.exit(1)
#
#     def _resetSimulations(self, temperature=None):
#         """At the end of each iteration:
#
#         1. Reset the step number in the NCMC context/integrator
#         2. Set the velocities to random values chosen from a Boltzmann distribution at a given `temperature`.
#
#         Parameters
#         ----------
#         temperature : float
#             The target temperature for the simulation.
#
#         """
#         if not temperature:
#             temperature = self._md_sim.context._integrator.getTemperature()
#
#         self._ncmc_sim.currentStep = 0
#         self._ncmc_sim.context._integrator.reset()
#
#         #Reinitialize velocities, preserving detailed balance?
#         self._md_sim.context.setVelocitiesToTemperature(temperature)
#
#     def _stepMD(self, nstepsMD):
#         """Advance the MD simulation.
#
#         Parameters
#         ----------
#         nstepsMD : int
#             The number of steps to advance the MD simulation.
#         """
#         logger.info('Advancing %i MD steps...' % (nstepsMD))
#         self._md_sim.currentIter = self.currentIter
#         # Retrieve MD state before proposed move
#         # Helps determine if previous iteration placed ligand poorly
#         md_state0 = self.stateTable['md']['state0']
#
#         for md_step in range(int(nstepsMD)):
#             try:
#                 self._md_sim.step(1)
#             except Exception as e:
#                 logger.error(e, exc_info=True)
#                 logger.error('potential energy before NCMC: %s' % md_state0['potential_energy'])
#                 logger.error('kinetic energy before NCMC: %s' % md_state0['kinetic_energy'])
#                 #Write out broken frame
#                 utils.saveSimulationFrame(self._md_sim,
#                                           'MD-fail-it%s-md%i.pdb' % (self.currentIter, self._md_sim.currentStep))
#                 sys.exit(1)
#
#     def run(self, nIter=0, nstepsNC=0, moveStep=0, nstepsMD=0, temperature=300, write_move=False, **config):
#         """Executes the BLUES engine to iterate over the actions:
#         Perform NCMC simulation, perform proposed move, accepts/rejects move,
#         then performs the MD simulation from the NCMC state, niter number of times.
#         **Note:** If the parameters are not given explicitly, will look for the parameters
#         in the provided configuration on the `SimulationFactory` object.
#
#         Parameters
#         ----------
#         nIter : int, default = None
#             Number of iterations of NCMC+MD to perform.
#         nstepsNC : int
#             The number of NCMC switching steps to advance by.
#         moveStep : int
#             The step number to perform the chosen move, which should be half
#             the number of nstepsNC.
#         nstepsMD : int
#             The number of steps to advance the MD simulation.
#         temperature : float
#             The target temperature for the simulation.
#         write_move : bool, default=False
#             If True, writes the proposed NCMC move to a PDB file.
#
#         """
#         if not nIter: nIter = self._config['nIter']
#         if not nstepsNC: nstepsNC = self._config['nstepsNC']
#         if not nstepsMD: nstepsMD = self._config['nstepsMD']
#         if not moveStep: moveStep = self._config['moveStep']
#
#         logger.info('Running %i BLUES iterations...' % (nIter))
#         for N in range(int(nIter)):
#             self.currentIter = N
#             logger.info('BLUES Iteration: %s' % N)
#             self._syncStatesMDtoNCMC()
#             self._stepNCMC(nstepsNC, moveStep)
#             self._acceptRejectMove(write_move)
#             self._resetSimulations(temperature)
#             self._stepMD(nstepsMD)
#
#         # END OF NITER
#         self.acceptRatio = self.accept / float(nIter)
#         logger.info('Acceptance Ratio: %s' % self.acceptRatio)
#         logger.info('nIter: %s ' % nIter)
#
#
# class MonteCarloSimulation(BLUESSimulation):
#     """Simulation class provides the functions that perform the MonteCarlo run.
#
#     Parameters
#     ----------
#         simulations : SimulationFactory
#             Contains 3 required OpenMM Simulationobjects
#         config : dict, default = None
#             Dict with configuration info.
#     """
#
#     def __init__(self, simulations, config=None):
#         super(MonteCarloSimulation, self).__init__(simulations, config)
#
#     def _stepMC_(self):
#         """Function that performs the MC simulation.
#         """
#
#         #choose a move to be performed according to move probabilities
#         self._move_engine.selectMove()
#         #change coordinates according to Moves in MoveEngine
#         new_context = self._move_engine.runEngine(self._md_sim.context)
#         md_state1 = self.getStateFromContext(new_context, self._state_keys)
#         self._setStateTable('md', 'state1', md_state1)
#
#     def _acceptRejectMove(self, temperature=None):
#         """Function that chooses to accept or reject the proposed move.
#         """
#         md_state0 = self.stateTable['md']['state0']
#         md_state1 = self.stateTable['md']['state1']
#         work_mc = (md_state1['potential_energy'] - md_state0['potential_energy']) * (
#             -1.0 / self._ncmc_sim.context._integrator.kT)
#         randnum = math.log(np.random.random())
#
#         if work_mc > randnum:
#             self.accept += 1
#             logger.info('MC MOVE ACCEPTED: work_mc {} > randnum {}'.format(work_mc, randnum))
#             self._md_sim.context.setPositions(md_state1['positions'])
#         else:
#             self.reject += 1
#             logger.info('MC MOVE REJECTED: work_mc {} < {}'.format(work_mc, randnum))
#             self._md_sim.context.setPositions(md_state0['positions'])
#         self._md_sim.context.setVelocitiesToTemperature(temperature)
#
#     def run(self, nIter=0, mc_per_iter=0, nstepsMD=0, temperature=300, write_move=False):
#         """Function that runs the BLUES engine to iterate over the actions:
#         perform proposed move, accepts/rejects move,
#         then performs the MD simulation from the accepted or rejected state.
#
#         Parameters
#         ----------
#         nIter : None or int, optional default = None
#             The number of iterations to perform. If None, then
#             uses the nIter specified in the opt dictionary when
#             the Simulation class was created.
#         mc_per_iter : int, default = 1
#             Number of Monte Carlo iterations.
#         nstepsMD : int, default = None
#             Number of steps the MD simulation will advance
#         write_move : bool, default = False
#             Writes the move if True
#         """
#         if not nIter: nIter = self._config['nIter']
#         if not nstepsMD: nstepsMD = self._config['nstepsMD']
#         #controls how many mc moves are performed during each iteration
#         if not mc_per_iter: mc_per_iter = self._config['mc_per_iter']
#
#         self._syncStatesMDtoNCMC()
#         for N in range(nIter):
#             self.currentIter = N
#             logger.info('MonteCarlo Iteration: %s' % N)
#             for i in range(mc_per_iter):
#                 self._syncStatesMDtoNCMC()
#                 self._stepMC_()
#                 self._acceptRejectMove(temperature)
#             self._stepMD(nstepsMD)
