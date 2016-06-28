import simtk.openmm as openmm
import simtk.unit as unit
def computeAlchemicalCorrection( unmodified_system, alchemical_system, initial_positions, final_positions, direction='insert', temperature=300.0 * unit.kelvin):
    """
    Compute log probability for correction from transforming real system to/from alchemical system.
    If direction is `insert`, the contribution at `final_positions` is computed as (real - alchemical).
    If direction is `delete`, the contribution at `initial_positions` is computed as (alchemical - real).
    Parameters
    ----------
    unmodified_system : simtk.unit.System
        Real fully-interacting system.
    alchemical_system : simtk.unit.System
        Alchemically modified system in fully-interacting form.
    initial_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
        The initial positions before NCMC switching.
    final_positions : simtk.unit.Quantity of dimensions [nparticles,3] with units compatible with angstroms
        The final positions after NCMC switching.
    direction : str, optional, default='insert'
        Direction of topology proposal to use for identifying alchemical atoms (allowed values: ['insert', 'delete'])
    Returns
    -------
    logP_alchemical_correction : float
        The log acceptance probability of the switch
    """

    if direction not in ['insert', 'delete']:
        raise Exception("'direction' must be one of ['insert', 'delete']; was '%s' instead" % direction)

    def computePotentialEnergy(system, positions):
        """
        Compute potential energy of the specified system object at the specified positions.
        Constraints are applied before the energy is computed.
        Parameters
        ----------
        system : simtk.openmm.System
            The System object for which the potential energy is to be computed.
        positions : simtk.unit.Quantity with dimension [natoms, 3] with units of distance.
            Positions of the atoms for which energy is to be computed.
        Returns
        -------
        potential : simtk.unit.Quantity with units of energy
            The computed potential energy
        """
        # Create dummy integrator.
        integrator = openmm.VerletIntegrator(0.002*unit.picoseconds)
        # Set the constraint tolerance if specified.
        # Create a context on the specified platform.
        context = openmm.Context(alchemical_system, integrator)
        context.setPositions(positions)
        context.applyConstraints(integrator.getConstraintTolerance())
        # Compute potential energy.
        potential = context.getState(getEnergy=True).getPotentialEnergy()
        # Clean up context and integrator.
        del context, integrator
        # Return potential energy.
        return potential

    # Compute correction from transforming real system to/from alchemical system
    if direction == 'delete':
        alchemical_potential_correction = computePotentialEnergy(alchemical_system, initial_positions) - computePotentialEnergy(unmodified_system, initial_positions)
    elif direction == 'insert':
        alchemical_potential_correction = computePotentialEnergy(unmodified_system, final_positions) - computePotentialEnergy(alchemical_system, final_positions)
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        beta = 1.0 / kT
    logP_alchemical_correction = -beta * alchemical_potential_correction

    return logP_alchemical_correction

