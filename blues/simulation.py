"""
simulation.py: Provides the Simulation class object that runs the BLUES engine

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""
import numpy as np
from simtk import unit, openmm
from simtk.openmm import app
import parmed, math
from openmmtools import alchemy
from blues.integrators import AlchemicalExternalLangevinIntegrator
from blues import utils
import os, copy, yaml, logging, sys, itertools
import mdtraj
from blues import utils
from blues import reporters
from simtk.openmm import app
logger = logging.getLogger(__name__)

class SystemFactory(object):
    """
    SystemFactory contains methods to generate/modify the OpenMM System object required for
    generating the openmm.Simulation using a given parmed.Structure()

    Usage Example
    -------------
    #Load Parmed Structure
    structure = parmed.load_file('eqToluene.prmtop', xyz='eqToluene.inpcrd')

    #Select move type
    ligand = RandomLigandRotationMove(structure, 'LIG')
    #Iniitialize object that selects movestep
    ligand_mover = MoveEngine(ligand)

    #Generate the openmm.Systems
    systems = SystemFactory(structure, ligand.atom_indices, config['system'])

    #The MD and alchemical Systems are generated and stored as an attribute
    systems.md
    systems.alch

    #Freeze atoms in the alchemical system
    systems.alch = SystemFactory.freeze_atoms(systems.alch,
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

    @classmethod
    def generateSystem(cls, structure, **kwargs):
        """
        Construct an OpenMM System representing the topology described by the
        prmtop file. This function is just a wrapper for parmed Structure.createSystem().

        Parameters
        ----------
        structure : parmed.Structure()
            The parmed.Structure of the molecular system to be simulated

        Kwargs
        -------
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

        Notes
        -----
        This function calls prune_empty_terms if any Topology lists have changed
        """
        return structure.createSystem(**kwargs)

    @classmethod
    def generateAlchSystem(cls, system, atom_indices,
                            softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6,
                            softcore_beta=0.0, softcore_d=1, softcore_e=1, softcore_f=2,
                            annihilate_electrostatics=True, annihilate_sterics=False,
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

        Kwargs
        ------
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
            Eq. 13 of Ref. [1] (default is 1).
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
            Options are ['direct-space', 'coulomb', 'exact'].
            - 'direct-space' only models the direct space contribution
            - 'coulomb' includes switched Coulomb interaction
            - 'exact' includes also the reciprocal space contribution, but it's
                only possible to annihilate the charges and the softcore parameters
                controlling the electrostatics are deactivated. Also, with this
                method, modifying the global variable `lambda_electrostatics` is
                not sufficient to control the charges. The recommended way to change
                them is through the `AlchemicalState` class.
        References
        ----------
        [1] Pham TT and Shirts MR. Identifying low variance pathways for free
        energy calculations of molecular transformations in solution phase.
        JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597
        """
        if suppress_warnings:
            #Lower logger level to suppress excess warnings
            logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)

        #Disabled correction term due to increased computational cost
        factory = alchemy.AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=disable_alchemical_dispersion_correction,
                                                    alchemical_pme_treatment=alchemical_pme_treatment)
        alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices,
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

    @staticmethod
    def _amber_selection_to_atom_indices_(structure, selection):
        """
        Converts AmberMask selection to list of atom indices.
        Parameters
        ----------
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        selection : str
            AmberMask selection that gets converted to a list of atom indices.
        """
        mask = parmed.amber.AmberMask(structure, str(selection))
        mask_idx = [i for i in mask.Selected()]
        return mask_idx

    @staticmethod
    def _print_atomlist_from_atom_indices_(structure, mask_idx):
        """
        Goes through the structure and matches the previously selected atom
        indices to the atom type.
        Parameters
        ----------
        structure : parmed.Structure()
            Structure of the system, used for atom selection.
        mask_idx : list of int
            List of atom indices.
        """
        atom_list = []
        for i, at in enumerate(structure.atoms):
            if i in mask_idx:
                atom_list.append(structure.atoms[i])
        logger.debug('\nFreezing {}'.format(atom_list))
        return atom_list

    @classmethod
    def restrain_positions(cls, structure, system, selection="(@CA,C,N)", weight=5.0, **kwargs):
        """
        Applies positional restraints to the given openmm.System.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.

        Kwargs
        -------
        selection : str, Default = "(@CA,C,N)"
            AmberMask selection to apply positional restraints to
        weight : float, Default = 5.0
            Restraint weight for xyz atom restraints in kcal/(mol A^2)

        References
        -----
        Amber mask syntax: http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax
        """
        mask_idx = cls._amber_selection_to_atom_indices_(structure, selection)

        logger.info("{} positional restraints applied to selection: '{}' ({} atoms) on {}".format(weight, selection, len(mask_idx), system))
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
        Function that will zero the masses of atoms from the given selection.
        Massless atoms will be ignored by the integrator and will not change positions.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.

        Kwargs
        -------
        freeze_selection : str, Default = ":LIG"
            AmberMask selection for the center in which to select atoms for zeroing their masses.
            Defaults to freezing protein backbone atoms.

        References
        -----
        Amber mask syntax: http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax
        """
        mask_idx = cls._amber_selection_to_atom_indices_(structure, freeze_selection)
        logger.info("Freezing selection '{}' ({} atoms) on {}".format(freeze_selection, len(mask_idx), system))

        cls._print_atomlist_from_atom_indices_(structure, mask_idx)
        system = utils.zero_masses(system, mask_idx)
        return system

    @classmethod
    def freeze_radius(cls, structure, system, freeze_distance=5.0*unit.angstrom,
                    freeze_center=':LIG', freeze_solvent=':HOH,NA,CL', **kwargs):
        """
        Function that will zero the masses of atoms outside the given raidus of
        the `freeze_center` selection. Massless atoms will be ignored by the
        integrator and will not change positions.This is intended to freeze
        the solvent and protein atoms around the ligand binding site.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object to be modified.
        structure : parmed.Structure()
            Structure of the system, used for atom selection.

        Kwargs
        -------
        freeze_center : str, Default = ":LIG"
            AmberMask selection for the center in which to select atoms for zeroing their masses. Default: LIG
        freeze_distance : float, Default = 5.0
            Distance (angstroms) to select atoms for retaining their masses.
            Atoms outside the set distance will have their masses set to 0.0.
        freeze_solvent : str, Default = ":HOH,NA,CL"
            AmberMask selection in which to select solvent atoms for zeroing their masses.

        References
        -----
        Amber mask syntax: http://parmed.github.io/ParmEd/html/amber.html#amber-mask-syntax
        """
        #Select the LIG and atoms within 5 angstroms, except for WAT or IONS (i.e. selects the binding site)
        if hasattr(freeze_distance, '_value'): freeze_distance = freeze_distance._value
        selection = "(%s<:%f)&!(%s)" % (freeze_center,freeze_distance,freeze_solvent)
        site_idx = cls._amber_selection_to_atom_indices_(structure, selection)
        #Invert that selection to freeze everything but the binding site.
        freeze_idx = set(range(system.getNumParticles())) - set(site_idx)

        logger.info("Freezing {} atoms {} Angstroms from '{}' on {}".format(len(freeze_idx), freeze_distance, freeze_center, system))

        cls._print_atomlist_from_atom_indices_(structure, freeze_idx)
        system = utils.zero_masses(system, freeze_idx)
        return system

class SimulationFactory(object):
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run. This class can take a
    list of reporters for the MD or NCMC simulation in the arguments
    `md_reporters` or `ncmc_reporters`.

    Usage Example
    -------------
    #Load Parmed Structure
    structure = parmed.load_file('eqToluene.prmtop', xyz='eqToluene.inpcrd')

    #Select move type
    ligand = RandomLigandRotationMove(structure, 'LIG')
    #Iniitialize object that selects movestep
    ligand_mover = MoveEngine(ligand)

    #Generate the openmm.Systems
    systems = SystemFactory(structure, ligand.atom_indices, config['system'])

    #Generate the OpenMM Simulations
    #Explicit dict of simulation configuration parameters
    sim_cfg = { 'platform': 'OpenCL',
                'properties' : { 'OpenCLPrecision': 'single',
                                  'OpenCLDeviceIndex' : 2},
                'nprop' : 1,
                'propLambda' : 0.3,
                'dt' : 0.001 * unit.picoseconds,
                'friction' : 1 * 1/unit.picoseconds,
                'temperature' : 100 * unit.kelvin,
                'nIter': 1,
                'nstepsMD': 10,
                'nstepsNC': 10,}
    simulations = SimulationFactory(systems, ligand_mover, sim_cfg])

    #Access the MD/NCMC simulation objects separately with `simulations.md` or
    `simulations.ncmc`

    # If a configuration is provided at on initialization, it will call
    # `generateSimulationSet()` for convenience. Otherwise, the class can be
    # instantiated like a normal python class:

    simulations = SimulationFactory(systems, ligand_mover)
    hasattr(simulations, 'md')
    hasattr(simulations, 'ncmc')
    >>> False
    >>> False

    simulations.generateSimulationSet(sim_cfg)
    hasattr(simulations, 'md')
    hasattr(simulations, 'ncmc')
    >>> False
    >>> False

    # After generating the Simulations, attach your own reporters by providing
    # the reporters in a list. Be sure to attach to either the MD or NCMC simulation.

    from simtk.openmm.app import StateDataReporter
    reporters = [ StateDataReporter('test.log', 5) ]
    simulations.md = simulations.attachReporters( simulations.md, reporters)

    Parameters
    ----------
    systems : blues.simulation.SystemFactory object
        The object containing the MD and alchemical openmm.Systems
    move_engine : blues.engine.MoveEngine object
        MoveProposal object which contains the dict of moves performed
        in the NCMC simulation.
    config : dict of parameters for the simulation
        #TODO: SET DEFAULTS OR MAKE THESE REQUIRED
        nIter, nstepsNC, nstepsMD, nprop, propLambda, temperature, dt, propSteps, write_move
    md_reporters : (optional) list of Reporter objects for the MD openmm.Simulation
    ncmc_reporters : (optional) list of Reporter objects for the NCMC openmm.Simulation
    """
    def __init__(self, systems, move_engine, config=None, md_reporters=None, ncmc_reporters=None):
        #Hide these properties since they exist on the SystemsFactory object
        self._structure = systems.structure
        self._system = systems.md
        self._alch_system = systems.alch
        #Atom indicies from move_engine
        #TODO: change atom_indices selection for multiple regions
        self._atom_indices = move_engine.moves[0].atom_indices
        self._move_engine = move_engine
        self.config = config

        #If parameters for generating the openmm.Simulation are given, make them.
        if self.config:
            try:
                self.generateSimulationSet()
            except Exception as e:
                logger.exception(e)
                raise e

        if md_reporters:
            self._md_reporters = md_reporters
            self.md = SimulationFactory.attachReporters(self.md, self._md_reporters)
        if ncmc_reporters:
            self._ncmc_reporters = ncmc_reporters
            self.ncmc = SimulationFactory.attachReporters(self.ncmc, self._ncmc_reporters)

    @classmethod
    def addBarostat(cls, system, temperature=300*unit.kelvin, pressure=1*unit.atmospheres, frequency=25, **kwargs):
        """
        Adds a MonteCarloBarostat to the MD system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.

        Kwargs
        ------
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        pressure : int, configional, default=None
            Pressure (atm) for Barostat for NPT simulations.
        frequency : int, default=25
            Frequency at which Monte Carlo pressure changes should be attempted (in time steps)
        """
        logger.info('Adding MonteCarloBarostat with {}. MD simulation will be {} NPT.'.format(pressure, temperature))
        # Add Force Barostat to the system
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature, frequency))
        return system

    @classmethod
    def generateIntegrator(cls, temperature=300*unit.kelvin, dt=0.002*unit.picoseconds, friction=1, **kwargs):
        """
        Generates a LangevinIntegrator for the Simulations.

        Kwargs
        ----------
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        friction: float, default=1
            friction coefficient which couples to the heat bath, measured in 1/ps
        dt: int, configional, default=0.002
            The timestep of the integrator to use (in ps).
        """
        integrator = openmm.LangevinIntegrator(temperature, friction, dt)
        return integrator

    @classmethod
    def generateNCMCIntegrator(cls, nstepsNC=0,
                               alchemical_functions={'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                               'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'},
                               splitting="H V R O R V H",
                               temperature=300*unit.kelvin,
                               dt=0.002*unit.picoseconds,
                               nprop=1,
                               propLambda=0.3, **kwargs):
        """
        Generates the AlchemicalExternalLangevinIntegrator using openmmtools.

        Parameters
        -----------
        nstepsNC : int, optional, default=1000
            The number of NCMC relaxation steps to use.

        Kwargs
        ------
        alchemical_functions : dict of strings,
            key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
            string that depends on the variable "lambda"
            Default = {'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                      'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda)
                                                  + 1/0.2*(lambda-0.8)*step(lambda-0.8)'}
        splitting : string, default: "H V R O R V H"
            Sequence of R, V, O (and optionally V{i}), and { }substeps to be executed each timestep. There is also an H option,
            which increments the global parameter `lambda` by 1/nsteps_neq for each step.
            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            ( will cause metropolization, and must be followed later by a ).
        temperature : float, default=300
            temperature (Kelvin) to be simulated at.
        dt: int, optional, default=0.002
            The timestep of the integrator to use (in ps).
        nprop : int (Default: 1)
            Controls the number of propagation steps to add in the lambda
            region defined by `propLambda`
        propLambda: float, optional, default=0.3
            The range which additional propogation steps are added,
            defined by [0.5-propLambda, 0.5+propLambda].
        """
        #During NCMC simulation, lambda parameters are controlled by function dict below
        # Keys correspond to parameter type (i.e 'lambda_sterics', 'lambda_electrostatics')
        # 'lambda' = step/totalsteps where step corresponds to current NCMC step,
        ncmc_integrator = AlchemicalExternalLangevinIntegrator(
                                alchemical_functions=alchemical_functions,
                               splitting=splitting,
                               temperature=temperature,
                               nsteps_neq=nstepsNC,
                               timestep=dt,
                               nprop=nprop,
                               prop_lambda=propLambda)
        return ncmc_integrator

    @classmethod
    def generateSimFromStruct(cls, structure, system, integrator, platform=None, properties={}, **kwargs):
        """Used to generate the OpenMM Simulation objects from a given parmed.Structure()

        Parameters
        ----------
        structure : parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        integrator : openmm.Integrator
            The OpenMM Integrator object for the simulation.
        platform : str, default = None
            Valid choices: 'Auto', 'OpenCL', 'CUDA'
            If None is specified, the fastest available platform will be used.
        """
        #Specifying platform properties here used for local development.
        if platform is None:
            #Use the fastest available platform
            simulation = app.Simulation(structure.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName(platform)
            #Make sure key/values are strings
            properties = { str(k) : str(v) for k,v in properties.items()}
            simulation = app.Simulation(structure.topology, system, integrator, platform, properties)

        # Set initial positions/velocities
        if structure.box_vectors:
            simulation.context.setPeriodicBoxVectors(*structure.box_vectors)
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(integrator.getTemperature())

        return simulation

    @staticmethod
    def attachReporters(simulation, reporter_list):
        """Attach the list of reporters to the Simulation object"""
        for rep in reporter_list:
            simulation.reporters.append(rep)
        return simulation

    def generateSimulationSet(self, config=None):
        """Function used to generate the 3 OpenMM Simulation objects."""
        if not config: config = self.config

        #Construct MD Integrator and Simulation
        self.integrator = self.generateIntegrator(**config)

        #Check for pressure parameter to set simulation to NPT
        if 'pressure' in config.keys():
            self._system = self.addBarostat(self._system, **config)
            logger.warning('NCMC simulation will NOT have pressure control. NCMC will use pressure from last MD state.')
        else:
            logger.info('MD simulation will be {} NVT.'.format(config['temperature']))
        self.md = self.generateSimFromStruct(self._structure, self._system, self.integrator, **config)

        #Alchemical Simulation is used for computing correction term from MD simulation.
        alch_integrator = self.generateIntegrator(**config)
        self.alch = self.generateSimFromStruct(self._structure, self._system, alch_integrator, **config)

        #If the moveStep hasn't been calculated, recheck the NCMC steps.
        if 'moveStep' not in config.keys():
            logger.warning('Did not find `moveStep` in configuration. Checking NCMC paramters')
            ncmc_parameters = utils.calculateNCMCSteps(**config)
            for k,v in ncmc_parameters.items():
                config[k]  = v
            self.config = config

        #Construct NCMC Integrator and Simulation
        self.ncmc_integrator = self.generateNCMCIntegrator(**config)

        #Initialize the Move Engine with the Alchemical System and NCMC Integrator
        for move in self._move_engine.moves:
            self._alch_system, self.ncmc_integrator = move.initializeSystem(self._alch_system, self.ncmc_integrator)
        self.ncmc = self.generateSimFromStruct(self._structure, self._alch_system, self.ncmc_integrator, **config)
        utils.print_host_info(self.ncmc)

class BLUESSimulation(object):
    """Simulation class provides the functions that perform the BLUES run.
    """
    def __init__(self, simulations, config=None):
        """Initialize the BLUES Simulation object.

        Parameters
        ----------
        simulations : blues.simulation.SimulationFactory object
            SimulationFactory Object which carries the 3 required
            OpenMM Simulation objects (MD, NCMC, ALCH) required to run BLUES.

        """
        self._move_engine = simulations._move_engine
        self._md_sim = simulations.md
        self._alch_sim = simulations.alch
        self._ncmc_sim = simulations.ncmc
        self.temperature = simulations.md.integrator.getTemperature()

        # Check if configuration has been specified in `SimulationFactory` object
        if not config:
            if hasattr(simulations, 'config'):
                self._config = simulations.config
        else:
            #Otherwise take specified config
            self._config = config
        if self._config:
            self._print_simulation_timing_()

        self.accept = 0
        self.reject = 0
        self.acceptRatio = 0
        self.currentIter = 0

        #Dict to keep track of each simulation state before/after each iteration
        self.stateTable = { 'md'   : { 'state0' : {}, 'state1' : {} },
                            'ncmc' : { 'state0' : {}, 'state1' : {} }
                          }

        #specify nc integrator variables to report in verbose output
        self._integrator_keys_= [ 'lambda', 'shadow_work',
                          'protocol_work', 'Eold', 'Enew']

        self._state_keys_ = { 'getPositions' : True,
                       'getVelocities' : True,
                       'getForces' : False,
                       'getEnergy' : True,
                       'getParameters': True,
                       'enforcePeriodicBox' : True}

    @classmethod
    def getStateFromContext(cls, context, state_keys):
        """Function that gets the State information from the given context and
        list of state_keys to query it with.

        Returns the state data as a dict.

        Parameters
        ----------
        context : openmm.Context
            Context of the OpenMM Simulation to query.
        state_keys : list
            Default: [ positions, velocities, potential_energy, kinetic_energy ]
            A list that defines what information to get from the context State.
        """
        stateinfo = {}
        state  = context.getState(**state_keys)
        stateinfo['positions'] =  state.getPositions(asNumpy=True)
        stateinfo['velocities'] = state.getVelocities(asNumpy=True)
        stateinfo['potential_energy'] = state.getPotentialEnergy()
        stateinfo['kinetic_energy'] = state.getKineticEnergy()
        stateinfo['box_vectors'] = state.getPeriodicBoxVectors()
        return stateinfo

    @classmethod
    def getIntegratorInfo(cls, ncmc_integrator,
                    integrator_keys=[ 'lambda', 'shadow_work',
                                      'protocol_work', 'Eold', 'Enew']):
        """Returns a dict of alchemical/ncmc-swtiching data from querying the the NCMC
        integrator.

        Parameters
        ----------
        ncmc_integrator : openmm.Context.Integrator
            The integrator from the NCMC Context
        integrator_keys : list
            list containing strings of the values to get from the integrator.
            Default : ['total_work', 'lambda', 'shadow_work',
                       'protocol_work', 'Eold', 'Enew','Epert']
        """
        integrator_info = {}
        for key in integrator_keys:
            integrator_info[key] = ncmc_integrator.getGlobalVariableByName(key)
        return integrator_info

    @classmethod
    def setContextFromState(cls, context, state):
        # Replace ncmc data from the md context
        context.setPeriodicBoxVectors(*state['box_vectors'])
        context.setPositions(state['positions'])
        context.setVelocities(state['velocities'])
        return context

    def _print_simulation_timing_(self):
        """Prints the simulation timing and related information."""
        dt = self._config['dt'].value_in_unit(unit.picoseconds)
        nIter = self._config['nIter']
        nprop  = self._config['nprop']
        propLambda = self._config['propLambda']
        propSteps = self._config['propSteps']
        nstepsNC = self._config['nstepsNC']
        nstepsMD = self._config['nstepsMD']

        force_eval = nIter * (propSteps + nstepsMD)
        time_ncmc_iter =  propSteps * dt
        time_ncmc_total = time_ncmc_iter * nIter
        time_md_iter = nstepsMD * dt
        time_md_total = time_md_iter * nIter
        time_iter = time_ncmc_iter + time_md_iter
        time_total = time_iter * nIter

        msg =  'Total BLUES Simulation Time = %s ps (%s ps/Iter)\n' % (time_total, time_iter)
        msg += 'Total Force Evaluations = %s \n' % force_eval
        msg += 'Total NCMC time = %s ps (%s ps/iter)\n' % (time_ncmc_total, time_ncmc_iter)

        # Calculate number of lambda steps inside/outside region with extra propgation steps
        steps_in_prop = int(  nprop * (2 * math.floor( propLambda * nstepsNC ) )  )
        steps_out_prop = int( (2 * math.ceil( (0.5 - propLambda) * nstepsNC )  )  )

        prop_lambda_window = self._ncmc_sim.context._integrator._prop_lambda
        prop_range = round(prop_lambda_window[1] - prop_lambda_window[0],4)
        if propSteps != nstepsNC:
                msg += '\t%s lambda switching steps within %s total propagation steps.\n' % (nstepsNC, propSteps)
                msg += '\tExtra propgation steps between lambda [%s, %s]\n' % (prop_lambda_window[0], prop_lambda_window[1])
                msg += '\tLambda: 0.0 -> %s = %s propagation steps\n' % (prop_lambda_window[0], int(steps_out_prop/2))
                msg += '\tLambda: %s -> %s = %s propagation steps\n' % (prop_lambda_window[0], prop_lambda_window[1], steps_in_prop)
                msg += '\tLambda: %s -> 1.0 = %s propagation steps\n' % (prop_lambda_window[1], int(steps_out_prop/2))

        msg += 'Total MD time = %s ps (%s ps/iter)\n' % (time_md_total, time_md_iter)

        #Get trajectory frame interval timing for BLUES simulation
        if 'md_trajectory_interval' in self._config.keys():
            frame_iter = nstepsMD / self._config['md_trajectory_interval']
            timetraj_frame = (time_ncmc_iter + time_md_iter) / frame_iter
            msg += 'Trajectory Interval = %s ps/frame (%s frames/iter)' % (timetraj_frame, frame_iter)

        logger.info(msg)

    def _set_stateTable_(self, simkey, stateidx, stateinfo):
        """Updates `stateTable` (dict) containing:  Positions, Velocities, Potential/Kinetic energies
        of the state before and after a NCMC step or iteration.

        Parameters
        ----------
        simkey : str (key: 'md', 'ncmc', 'alch')
            Key corresponding to the simulation.
        stateidx : str (key: 'state0' or 'state1')
            Key corresponding to the state information being stored.
        stateinfo : dict
            Dictionary containing the State information.
        """
        self.stateTable[simkey][stateidx] = stateinfo

    def _sync_states_md_to_ncmc_(self):
        """Retrieves data on the current State of the MD context to
        replace the box vectors, positions, and velocties in the NCMC context.

        """
        # Retrieve the state data from the MD/NCMC contexts before proposed move
        md_state0 = self.getStateFromContext(self._md_sim.context, self._state_keys_)
        self._set_stateTable_('md', 'state0', md_state0)


        ncmc_state0 = self.getStateFromContext(self._ncmc_sim.context, self._state_keys_)
        self._set_stateTable_('ncmc', 'state0', ncmc_state0)

        # Replace ncmc context data from the md context
        self._ncmc_sim.context = self.setContextFromState(self._ncmc_sim.context, md_state0)

    def _stepNCMC_(self, nstepsNC, moveStep, move_engine=None):
        """Function that advances the NCMC simulation."""

        logger.info('Advancing %i NCMC switching steps...' % (nstepsNC))
        #choose a move to be performed according to move probabilities
        #TODO: will have to change to work with multiple alch region
        if not move_engine: move_engine = self._move_engine
        self._ncmc_sim.currentIter = self.currentIter
        move_engine.selectMove()
        lastStep = nstepsNC-1
        for step in range(int(nstepsNC)):
            try:
                #Attempt anything related to the move before protocol is performed
                if not step:
                    self._ncmc_sim.context = move_engine.selected_move.beforeMove(self._ncmc_sim.context)

                # Attempt selected MoveEngine Move at the halfway point
                #to ensure protocol is symmetric
                if step == moveStep:
                    if hasattr(logger, 'report'):
                        logger.info = logger.report
                    #Do move
                    logger.info('Performing %s...' % move_engine.move_name)
                    self._ncmc_sim.context = move_engine.runEngine(self._ncmc_sim.context)

                # Do 1 NCMC step with the integrator
                self._ncmc_sim.step(1)

                #DEBUG options at every NCMC step
                logger.debug('%s' % self.getIntegratorInfo(self._ncmc_sim.context._integrator, self._integrator_keys_))

                #Attempt anything related to the move after protocol is performed
                if step == lastStep:
                    self._ncmc_sim.context = move_engine.selected_move.afterMove(self._ncmc_sim.context)

            except Exception as e:
                logger.error(e)
                move_engine.selected_move._error(self._ncmc_sim.context)
                break

        # ncmc_state1 stores the state AFTER a proposed move.
        ncmc_state1 = self.getStateFromContext(self._ncmc_sim.context, self._state_keys_)
        self._set_stateTable_('ncmc', 'state1', ncmc_state1)

    def _compute_alchemical_correction_(self):
        # Retrieve the MD/NCMC state before the proposed move.
        md_state0_PE = self.stateTable['md']['state0']['potential_energy']
        ncmc_state0_PE = self.stateTable['ncmc']['state0']['potential_energy']

        # Retreive the NCMC state after the proposed move.
        ncmc_state1 = self.stateTable['ncmc']['state1']

        # Set the box_vectors and positions in the alchemical simulation to after the proposed move.
        self._alch_sim.context = self.setContextFromState(self._alch_sim.context, ncmc_state1)

        # Retrieve potential_energy for alch correction
        alch_PE = self._alch_sim.context.getState(getEnergy=True).getPotentialEnergy()

        correction_factor = (ncmc_state0_PE - md_state0_PE + alch_PE - ncmc_state1['potential_energy']) * (-1.0/self._ncmc_sim.context._integrator.kT)
        logger.debug('Alchemical Correction = %.6f' % correction_factor)

        return correction_factor

    def _accept_reject_move_(self, write_move=False):
        """Function that chooses to accept or reject the proposed move based
        on the acceptance criterion.
        """
        work_ncmc = self._ncmc_sim.context._integrator.getLogAcceptanceProbability(self._ncmc_sim.context)
        randnum =  math.log(np.random.random())

        # Compute correction if work_ncmc is not NaN
        if not np.isnan(work_ncmc):
            correction_factor = self._compute_alchemical_correction_()
            work_ncmc = work_ncmc + correction_factor

        if work_ncmc > randnum:
            self.accept += 1
            logger.info('NCMC MOVE ACCEPTED: work_ncmc {} > randnum {}'.format(work_ncmc, randnum) )

            # If accept move, sync MD context from NCMC after move.
            ncmc_state1 = self.stateTable['ncmc']['state1']
            self._md_sim.context = self.setContextFromState(self._md_sim.context, ncmc_state1)

            if write_move:
            	utils.saveSimulationFrame(self._md_sim, '{}acc-it{}.pdb'.format(self._config['outfname'], self.currentIter))

        else:
            self.reject += 1
            logger.info('NCMC MOVE REJECTED: work_ncmc {} < {}'.format(work_ncmc, randnum) )

            #If reject move, reset positions in ncmc context to before move
            md_state0 = self.stateTable['md']['state0']
            self._ncmc_sim.context = self.setContextFromState(self._ncmc_sim.context, md_state0)

    def _stepMD_(self, nstepsMD):
        """Function that advances the MD simulation."""
        logger.info('Advancing %i MD steps...' % (nstepsMD))
        self._md_sim.currentIter = self.currentIter
        #Retrieve MD state before proposed move
        # Helps determine if previous iteration placed ligand poorly
        md_state0 = self.stateTable['md']['state0']

        for md_step in range(int(nstepsMD)):
            try:
                self._md_sim.step(1)
            except Exception as e:
                logger.error(e, exc_info=True)
                logger.error('potential energy before NCMC: %s' % md_state0['potential_energy'])
                logger.error('kinetic energy before NCMC: %s' % md_state0['kinetic_energy'])
                #Write out broken frame
                utils.saveSimulationFrame(self._md_sim, 'MD-fail-it%s-md%i.pdb' %(self.currentIter, self._md_sim.currentStep))
                sys.exit(1)

        #If MD finishes okay, update stateTable
        md_state0 = self.getStateFromContext(self._md_sim.context, self._state_keys_)
        self._set_stateTable_('md', 'state0', md_state0)

        # Set NCMD poistions to last state from MD
        self._ncmc_sim.context = self.setContextFromState(self._ncmc_sim.context, md_state0)
        self._set_stateTable_('ncmc', 'state0', md_state0)

    def _reset_simulations_(self, temperature=None):
        """At the end of each iteration:
           1) Reset the step number in the NCMC context/integrator
           2) Set the velocities to random values chosen from a
              Boltzmann distribution at a given `temperature`.
        """
        if not temperature:
            temperature = self._md_sim.context._integrator.getTemperature()

        self._ncmc_sim.currentStep = 0
        self._ncmc_sim.context._integrator.reset()

        #Reinitialize velocities, preserving detailed balance?
        self._md_sim.context.setVelocitiesToTemperature(temperature)

    def run(self, nIter=None, nstepsNC=None, moveStep=None, nstepsMD=None, write_move=False, **config):
        """Function that runs the BLUES engine to iterate over the actions:
        Perform NCMC simulation, perform proposed move, accepts/rejects move,
        then performs the MD simulation from the NCMC state niter number of times.

        Parameters
        ----------
        nIter: int
            Number of iterations of NCMC+MD to perform.

        """
        if not nIter: nIter = self._config['nIter']
        if not nstepsNC: nstepsNC = self._config['nstepsNC']
        if not nstepsMD: nstepsMD = self._config['nstepsMD']
        if not moveStep: moveStep = self._config['moveStep']

        logger.info('Running %i BLUES iterations...' % (nIter))
        #set inital conditions
        self._sync_states_md_to_ncmc_()
        for N in range(int(nIter)):
            self.currentIter = N
            logger.info('BLUES Iteration: %s' % N)
            self._sync_states_md_to_ncmc_()
            self._stepNCMC_(nstepsNC, moveStep)
            self._accept_reject_move_(write_move)
            self._stepMD_(nstepsMD)
            self._reset_simulations_()

        # END OF NITER
        self.acceptRatio = self.accept/float(nIter)
        logger.info('Acceptance Ratio: %s' % self.acceptRatio)
        logger.info('nIter: %s ' % nIter)

class MonteCarloSimulation(BLUESSimulation):
    def __init__(self, simulations, config=None):
        super(MonteCarloSimulation, self).__init__(simulations, config)

    def _stepMC_(self):
        """Function that performs the MC simulation."""

        #choose a move to be performed according to move probabilities
        self._move_engine.selectMove()
        #change coordinates according to Moves in MoveEngine
        new_context = self._move_engine.runEngine(self._md_sim.context)
        md_state1 = self.getStateFromContext(new_context, self._state_keys_)
        self._set_stateTable_('md', 'state1', md_state1)

    def _accept_reject_move_(self):
        """Function that chooses to accept or reject the proposed move.
        """
        md_state0 = self.stateTable['md']['state0']
        md_state1 = self.stateTable['md']['state1']
        work_mc = (md_state1['potential_energy'] - md_state0['potential_energy']) * (-1.0/self._ncmc_sim.context._integrator.kT)
        randnum =  math.log(np.random.random())

        if log_mc > randnum:
            self.accept += 1
            logger.info('MC MOVE ACCEPTED: work_mc {} > randnum {}'.format(work_mc, randnum) )
            self._md_sim.context.setPositions(md_state1['positions'])
        else:
            self.reject += 1
            logger.info('MC MOVE REJECTED: work_mc {} < {}'.format(work_mc, randnum) )
            self._md_sim.context.setPositions(md_state0['positions'])
        self._md_sim.context.setVelocitiesToTemperature(temperature)

    def run(self, nIter, mc_per_iter=1, nstepsMD=None, write_move=False):
        """Function that runs the BLUES engine to iterate over the actions:
        perform proposed move, accepts/rejects move,
        then performs the MD simulation from the accepted or rejected state.

        Parameters
        ----------
        nIter: None or int, optional, default=None
            The number of iterations to perform. If None, then
            uses the nIter specified in the opt dictionary when
            the Simulation class was created.
        """
        if not nIter: nIter = self._config['nIter']
        if not nstepsMD: nstepsMD = self._config['nstepsMD']
        #controls how many mc moves are performed during each iteration
        if not mc_per_iter: mc_per_iter = self._config['mc_per_iter']

        self._sync_states_md_to_ncmc_()
        for N in range(nIter):
            self.currentIter = N
            logger.info('MonteCarlo Iteration: %s' % N)
            for i in range(mc_per_iter):
                self._sync_states_md_to_ncmc_()
                self._stepMC_()
                self._accept_reject_move_(write_move)
            self._stepMD_(nstepsMD)
