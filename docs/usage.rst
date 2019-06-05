Usage
=====

This package takes advantage of non-equilibrium candidate Monte Carlo moves (NCMC) to help sample between different ligand binding modes using the OpenMM simulation package. One goal for this package is to allow for easy additions of other moves of interest, which will be covered below.

The integrator of `BLUES` contains the framework necessary for NCMC. Specifically, the integrator class calculates the work done during a NCMC move. It also controls the lambda scaling of parameters. The integrator that BLUES uses inherits from `openmmtools.integrators.AlchemicalNonequilibriumLangevinIntegrator` to keep track of the work done outside integration steps, allowing Monte Carlo (MC) moves to be incorporated together with the NCMC thermodynamic perturbation protocol. Currently, the `openmmtools.alchemy` package is used to generate the lambda parameters for the ligand, allowing alchemical modification of the sterics and electrostatics of the system.

The `BLUESSampler` class in `ncmc.py` serves as a wrapper for running NCMC+MD simulations. To run the hybrid simulation, the `BLUESSampler` class requires defining two moves for running the (1) MD simulation and (2) the NCMC protcol. These moves are defined in the `ncmc.py` module. A simple example is provided below.

Example
-------
Using the BLUES framework requires the use of a **ThermodynamicState** and **SamplerState** from `openmmtools` which we import from `openmmtools.states`:

.. code-block:: python

   from openmmtools.states import ThermodynamicState, SamplerState
   from openmmtools.testsystems import TolueneVacuum
   from blues.ncmc import *
   from simtk import unit


Create the states for a toluene molecule in vacuum.

.. code-block:: python

   tol = TolueneVacuum()
   thermodynamic_state = ThermodynamicState(tol.system, temperature=300*unit.kelvin)
   sampler_state = SamplerState(positions=tol.positions)


Define our langevin dynamics move for the MD simulation portion and then our NCMC move which performs a random rotation. Here, we use a customized LangevinDynamicsMove which allows us to store information from the MD simulation portion.

.. code-block:: python

   dynamics_move = ReportLangevinDynamicsMove(n_steps=10)
   ncmc_move = RandomLigandRotationMove(n_steps=10, atom_subset=list(range(15)))


Provide the `BLUESSampler` class with an `openmm.Topology` and these objects to run the NCMC+MD simulation.

.. code-block:: python

   sampler = BLUESSampler(thermodynamic_state=thermodynamic_state,
                         sampler_state=sampler_state,
                         dynamics_move=dynamics_move,
                         ncmc_move=ncmc_move,
                         topology=tol.topology)
   sampler.run(n_iterations=1)



.. role:: raw-html(raw)
   :format: html

Let us know if you have any problems or suggestions through our :raw-html:`<a class="github-button" href="https://github.com/mobleylab/blues/issues" data-icon="octicon-issue-opened" data-size="large" data-show-count="false" aria-label="Issue mobleylab/blues on GitHub">Issue</a><script async defer src="https://buttons.github.io/buttons.js"></script>` tracker.
