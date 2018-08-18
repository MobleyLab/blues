from openmmtools import testsystems
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
from simtk.openmm.app import Simulation
import simtk
from blues.integrators import AlchemicalExternalLangevinIntegrator


def test_nonequilibrium_external_integrator():
    """Moves the first atom during the third step of the integration
    and checks to see that the AlchemicalExternalLangevinIntegrator
    finds the correct energy change.
    """
    testsystem = testsystems.AlanineDipeptideVacuum()
    functions = {'lambda_sterics': '1', 'lambda_electrostatics': '1'}
    factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
    alanine_vacuum = testsystem.system
    alchemical_region = AlchemicalRegion(
        alchemical_atoms=range(22),
        annihilate_electrostatics=True,
        annihilate_sterics=True)
    alanine_alchemical_system = factory.create_alchemical_system(
        reference_system=alanine_vacuum, alchemical_regions=alchemical_region)
    nsteps_neq = 6
    integrator = AlchemicalExternalLangevinIntegrator(
        alchemical_functions=functions,
        timestep=0.05 * simtk.unit.femtoseconds,
        nsteps_neq=nsteps_neq,
        measure_shadow_work=False,
        steps_per_propagation=2)

    simulation = Simulation(testsystem.topology, alanine_alchemical_system,
                            integrator)
    simulation.context.setPositions(testsystem.positions)
    for i in range(nsteps_neq):
        simulation.step(1)
        protocol_work = simulation.integrator.getGlobalVariableByName(
            "protocol_work")
        if i == 3:
            #perform the displacement of an atom
            state = simulation.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True)
            pos[0, 1] = pos[0, 1] + 0.5 * simtk.unit.nanometers
            simulation.context.setPositions(pos)
    protocol_work = simulation.integrator.getLogAcceptanceProbability(
        simulation.context)
    print(protocol_work)
    #find the work done for that displacement
    #protocol work is around 221.0 (in kT units), so acceptance is around -221.0
    assert -220.0 > protocol_work
    assert -223.0 < protocol_work


if __name__ == '__main__':
    test_nonequilibrium_external_integrator()
