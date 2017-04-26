from openmmtools import testsystems, alchemy
from openmmtools.alchemy import AlchemicalFactory, AlchemicalRegion
from simtk.openmm.app import Simulation
import simtk
from blues.integrators import NonequilibriumExternalLangevinIntegrator
def test_nonequilibrium_external_integrator():
    testsystem = testsystems.AlanineDipeptideVacuum()
    functions = { 'lambda_sterics' : '1', 'lambda_electrostatics' : '1'}
    factory = AlchemicalFactory(consistent_exceptions=False)
    alanine_vacuum = testsystem.system
    alchemical_region = AlchemicalRegion(alchemical_atoms=range(22),
                        annihilate_electrostatics=True,  annihilate_sterics=True)
    alanine_alchemical_system = factory.create_alchemical_system(reference_system=alanine_vacuum,
                                                                 alchemical_regions=alchemical_region)
    nsteps_neq = 6
    integrator = NonequilibriumExternalLangevinIntegrator(alchemical_functions=functions,
                            timestep=0.05*simtk.unit.femtoseconds,
                            nsteps_neq=nsteps_neq,
                            measure_shadow_work=False,
                            steps_per_propagation=2)

    simulation = Simulation(testsystem.topology, alanine_alchemical_system, integrator)
    simulation.context.setPositions(testsystem.positions)
    for i in range(nsteps_neq):
        simulation.step(1)
        protocol_work = simulation.integrator.getGlobalVariableByName("protocol_work")
        if i == 3:
            state = simulation.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True)
            pos[0,1] = pos[0,1] + 0.5*simtk.unit.nanometers
            simulation.context.setPositions(pos)
    print(protocol_work)
    #protocol work is around 221.0
    assert 220.0 < protocol_work
    assert 223.0 > protocol_work

if __name__ == '__main__':
    test_nonequilibrium_external_integrator()
