# Diagram of classes in BLUES
![Class diagram](class-diagram.png)

## Legend
- ![#97D017](https://placehold.it/15/97D077/000000?text=+) `User Input`
- ![#B3B3B3](https://placehold.it/15/B3B3B3/000000?text=+) `Core BLUES Objects`
    - Class attributes are represented with : `+ <name>`
    - Class functions are represented with :  **+ function()**
- ![#6C8EBF](https://placehold.it/15/6C8EBF/000000?text=+) `Simulation() functions`

### ![#97D017](https://placehold.it/15/97D077/000000?text=+) `User Input` and ![#B3B3B3](https://placehold.it/15/B3B3B3/000000?text=+) `Core BLUES Objects`
Before running the BLUES simulation, the user is expected to have a forcefield parameterized `parmed.Structure` of the solvated protein:ligand system.
For example, in the `blues/example.py` script we have:

```python    
#Generate the ParmEd Structure
prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')
inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
struct = parmed.load_file(prmtop, xyz=inpcrd)
```

3 other inputs are required to generate the 3 core BLUES objects (described in more detail below):
- `Model()`
- `MoveProposal()`
- `SimulationFactory()`

In the `Model()` class, the user specifies the residue name of the _model_ object that we will be rotating in the BLUES simulation.
In the example, we are rotating the toluene ligand in T4 lysozyme which has the residue name `'LIG'`.

```python
#Define the 'model' object we are perturbing here.
ligand = ncmc.Model(struct, 'LIG')
ligand.calculateProperties()
```

In the `MoveProposal()`, the user chooses the _move_ type and the step number to perform the move.
To select the move type, the user must provide a string of the function name corresponding to the move.
In the example, we use the string `random_rotation` to perform a rotation about the ligand's center of mass on the step `opt['nstepsNC']` set from the options dictionary.

```python
# Initialize object that proposes moves.
ligand_mover = ncmc.MoveProposal(ligand, 'random_rotation', opt['nstepsNC'])
```

Now that we have selected the ligand, defined the NCMC move, and created their corresponding Python objects, we generate the OpenMM Simulations in the `SimulationFactory()`.
This class takes in 3 inputs:
 1. `parmed.Structure` of the solvated protein:ligand system.
 2. `ncmc.Model` to obtain the atom indices to generate the alchemical system.
 3. `opt` is a dictionary of various simulation parameters (see below).

Snippet from the example script below:
```python
opt = { 'temperature' : 300.0, 'friction' : 1, 'dt' : 0.002,
        'nIter' : 10, 'nstepsNC' : 10, 'nstepsMD' : 5000,
        'nonbondedMethod' : 'PME', 'nonbondedCutoff': 10, 'constraints': 'HBonds',
        'trajectory_interval' : 1000, 'reporter_interval' : 1000,
        'platform' : 'OpenCL',
        'verbose' : True }
# Generate the MD, NCMC, ALCHEMICAL Simulation objects
simulations = ncmc.SimulationFactory(struct, ligand, **opt)
simulations.createSimulationSet()
```

### ![#6C8EBF](https://placehold.it/15/6C8EBF/000000?text=+) `Simulation() functions`
In order to run the BLUES simulation, we provide the `Simulation()` class the 3 core objects described above.
From the example:

```python    
blues = Simulation(simulations, ligand, ligand_mover, **opt)
blues.run()
```

In each NCMC iteration, there are 4 general stages:
 1. `setStateConditions()`: store the current state of our Simulations to a dict.
 2. `simulateNCMC()` : Alchemically scale our ligand interactions and perform the rotational move.
 3. `chooseMove()` : Accept/reject move based on Metropolis criterion.
 4. `simulateMD()` : Allow the system to advance in time.

Described below are the simulation stages described in more detail:
#### `simulateNCMC`: Performing the rotational move.
At this stage we operate on all 3 of the BLUES core objects.
First, we advance the NCMC simulation by referencing `simulations.nc` generated from the `SimulationFactory()` class.
As we take steps in the NCMC simulation, the ligand interactions are alchemically scaled off/on.

From `ligand_mover`, we reference the dict of moves in generated from the `MoveProposal()` class.
When the NCMC simulation is on the specified step to perform the move, we call the perturbation method from the dict using the key: `method`.
In our example, we reference the `MoveProposal.random_rotation()` function.
By default, we perform the rotation half-way through the number of NCMC steps.

When we make the call to the `MoveProposal.random_rotation()` method, we provide it the `ligand` generated from the `Model()` class and the OpenMM Context of the NCMC simulation.
In the `random_rotation()` call, we update the `ligand.positions` from the current Simulation state to obtain the `ligand.center_of_mass` and perform the random rotation about the ligand's center of mass.
In the diagram, these attributes are marked with a `*` to denote that these are dynamic variables that **need** to be updated with each step.

#### `chooseMove`: Metropolis-Hastings Acceptance or Rejection.
After performing the rotational move in the NCMC simulation, we accept or reject the move in accordance to the Metropolis criterion.
Then, we obtain the `LogAcceptanceProbability` from the NCMC integrator and add in the alchemical correction factor.

The alchemical correction factor is obtained by setting the positions `simulations.alch` (from the `SimulationFactory()` class) to that of the current state of the NCMC simulation.
Then, we compute the difference in potential energy from the previous state.

If the corrected log acceptance probability is greater than a randomly generated number we accept the move, otherwise the move is rejected.
On move acceptance, we set the positions of `simulations.md` to the current (i.e rotated) positions.
On rejection, we reset the positions of `simulations.nc` back to their positions before the NCMC simulation.
In either case, after the positions have been set, we randomly set the velocities of `simulations.md` by the temperature.

#### `simulateMD` : Relaxing the system.
After the move has been accepted or rejected, we simply allow the system to relax by advancing the MD simulation, referenced from `simulations.md`.
After the MD simulation has completed the specified number of steps, we set the positions and velocities of `simulations.nc` to that of the final state of the MD simulation.

In regards to velocities, it may be important to note:
- *Before the MD simulation*, velocities are randomly initialized by the selected temperature.
- *After the MD simulation*, the NCMC simulation uses velocities from the final state of the MD simulation.

# Implementing other non-equilibrium moves
To implement a new move into the BLUES framework, additions need to be made to 2 classes:
`MoveProposal()` and `Model()`.

Under the `MoveProposal()` class, define a new function that will perform your new move type.
Some general guidelines to follow when implementing a new function:
1. Inputs should be the OpenMM Context corresponding to the NCMC simulation and the `model` object generated from the `Model()` class.  
2. Be sure to update the relevant attributes to the `model` object (i.e `model.positions`)

Under the `Model()` class, add any related functions required for your move.
These functions should strictly be methods that store/calculate some properties of the ligand being perturbed.
In our example, we need the ligand's center of mass, which we obtain from the class's function `getCenterOfMass()`


For example, the implementation of random_rotation is shown below:

```python
def random_rotation(model, nc_context):
    initial_positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
    positions = copy.deepcopy(initial_positions)

    model.positions = positions[model.atom_indices]
    model.center_of_mass = model.getCenterOfMass(model.positions, model.masses)
    reduced_pos = model.positions - model.center_of_mass

    # Define random rotational move on the ligand
    rand_quat = mdtraj.utils.uniform_quaternion()
    rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
    #multiply lig coordinates by rot matrix and add back COM translation from origin
    rot_move =  np.dot(reduced_pos, rand_rotation_matrix) * positions.unit + model.center_of_mass

    # Update ligand positions in nc_sim
    for index, atomidx in enumerate(model.atom_indices):
        positions[atomidx] = rot_move[index]
    nc_context.setPositions(positions)
    positions = nc_context.getState(getPositions=True).getPositions(asNumpy=True)
    model.positions = positions[model.atom_indices]

    return model, nc_context
```

In short, `MoveProposal()` will operate on the NCMC simulation's Context by actually performing the move, using the current context to update the ligands positions. `Model()` is intended to provide functionality for storing attributes required to do the move itself.
