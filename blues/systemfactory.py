"""SystemFactory contains methods to generate/modify the OpenMM System object."""

import sys
import logging
from simtk import unit, openmm
from openmmtools import alchemy
from blues import utils
logger = logging.getLogger(__name__)


def generateAlchSystem(system,
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
    """Return the OpenMM System for alchemical perturbations.

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
    .. [TTPham-JChemPhys135-2011] T. T. Pham and M. R. Shirts,
    J. Chem. Phys 135, 034114 (2011). http://dx.doi.org/10.1063/1.3607597
    """
    if suppress_warnings:
        # Lower logger level to suppress excess warnings
        logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)

    # Disabled correction term due to increased computational cost
    factory = alchemy.AbsoluteAlchemicalFactory(
        disable_alchemical_dispersion_correction=disable_alchemical_dispersion_correction,
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

def zero_masses(system, atomList=None):
    """
    Zeroes the masses of specified atoms to constrain certain degrees of freedom.

    Arguments
    ---------
    system : openmm.System
        system to zero masses
    atomList : list of ints
        atom indicies to zero masses

    Returns
    -------
    system : openmm.System
        The modified system with massless atoms.

    """
    for index in (atomList):
        system.setParticleMass(index, 0 * unit.daltons)
    return system

def restrain_positions(structure, system, selection="(@CA,C,N)", weight=5.0, **kwargs):
    """Apply positional restraints to atoms in the openmm.System by the given parmed selection [amber-syntax]_.

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
    mask_idx = utils.amber_selection_to_atomidx(structure, selection)

    logger.info("{} positional restraints applied to selection: '{}' ({} atoms) on {}".format(
        weight, selection, len(mask_idx), system))
    # define the custom force to restrain atoms to their starting positions
    force = openmm.CustomExternalForce('k_restr*periodicdistance(x, y, z, x0, y0, z0)^2')
    # Add the restraint weight as a global parameter in kcal/mol/A^2
    force.addGlobalParameter("k_restr", weight)
    # force.addGlobalParameter("k_restr", weight*unit.kilocalories_per_mole/unit.angstroms**2)
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


def freeze_atoms(structure, system, freeze_selection=":LIG", **kwargs):
    """Zero the masses of atoms from the given parmed selection [amber-syntax]_.

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
    mask_idx = utils.amber_selection_to_atomidx(structure, freeze_selection)
    logger.info("Freezing selection '{}' ({} atoms) on {}".format(freeze_selection, len(mask_idx), system))

    utils.atomidx_to_atomlist(structure, mask_idx)
    system = zero_masses(system, mask_idx)
    return system


def freeze_radius(
                  structure,
                  system,
                  freeze_distance=5.0 * unit.angstrom,
                  freeze_center=':LIG',
                  freeze_solvent=':HOH,NA,CL',
                  **kwargs):
    """Zero the masses of atoms outside the given raidus of the `freeze_center` parmed selection [amber-syntax]_.

    Massless atoms will be ignored by the integrator and will not change
    positions. This is intended to freeze the solvent and protein atoms around
    the ligand binding site.

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
    # Select the LIG and atoms within 5 angstroms, except for WAT or IONS (i.e. selects the binding site)
    if hasattr(freeze_distance, '_value'):
        freeze_distance = freeze_distance._value
    selection = "(%s<:%f)&!(%s)" % (freeze_center, freeze_distance, freeze_solvent)
    logger.info('Inverting parmed selection for freezing: %s' % selection)
    site_idx = utils.amber_selection_to_atomidx(structure, selection)
    # Invert that selection to freeze everything but the binding site.
    freeze_idx = set(range(N_atoms)) - set(site_idx)
    center_idx = utils.amber_selection_to_atomidx(structure, freeze_center)

    freeze_threshold = 0.90
    freeze_warning = 0.75
    freeze_ratio = len(freeze_idx) / N_atoms

    # Ensure that the freeze selection is larger than the center selection of atoms
    if len(site_idx) == len(center_idx):
        err = "%i unfrozen atoms is equal to the number of atoms used as the selection center '%s' (%i atoms). Check your atom selection." % (len(site_idx), freeze_center, len(center_idx))
        logger.error(err)

    # Check if freeze selection has selected all atoms
    elif len(freeze_idx) == N_atoms:
        err = 'All %i atoms appear to be selected for freezing. Check your atom selection.' % len(freeze_idx)
        logger.error(err)

    elif freeze_ratio >= freeze_threshold:
        err = '%.0f%% of your system appears to be selected for freezing. Check your atom selection' % (
            100 * freeze_threshold)
        logger.error(err)

    elif freeze_warning <= freeze_ratio <= freeze_threshold:
        warn = '%.0f%% of your system appears to be selected for freezing. This may cause unexpected behaviors.' % (
            100 * freeze_ratio)
        logger.warning(warn)


    logger.info("Freezing {} atoms {} Angstroms from '{}' on {}".format(len(freeze_idx), freeze_distance,
                                                                        freeze_center, system))

    utils.atomidx_to_atomlist(structure, freeze_idx)
    system = zero_masses(system, freeze_idx)
    return system


def addBarostat(system, temperature=300 * unit.kelvin, pressure=1 * unit.atmospheres, frequency=25, **kwargs):
    """Add a MonteCarloBarostat to the MD system.

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
