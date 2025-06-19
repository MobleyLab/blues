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
- `Move()`
- `MoveEngine()`
- `SimulationFactory()`

The `Move()` class arguments can vary dramatically between subclasses, but the inputs generally allow the selection of particular atoms to be moved during the NCMC simluation. One method that all `Move()` classes have in common is `move()`, which takes in a context and changes the positions of the specified atoms, in a way that depends on the particular move.
In the example, we are rotating the toluene ligand in T4 lysozyme using the `RandomLigandRotationMove` which specifes the ligand atoms to be moved by specifying the resname:`'LIG'`.


```python
from blues.move import RandomLigandRotationMove
#Define the 'Move' object we are perturbing here.
ligand = RandomLigandRotationMove(struct, 'LIG')
ligand.calculateProperties()
```

`MoveEngine()`, defines what types of moves will be performed during the NCMC protocol with what probability.
`MoveEngine` takes in either a `Move` or list of `Move` objects. If a list is passed in, then one of those `Move.move()` methods will be selected during an NCMC iteration depending on a probability defined by the `probabilities` argument. If no probability is specified, each move is given equal weight.
In the example we have just a single move so we just pass that into the `MoveEngine` directly.

```python
# Initialize object that proposes moves.
from blues.moves import MoveEngine
ligand_mover = MoveEngine(ligand)
```

Now that we have selected the ligand, defined the NCMC move, and created their corresponding Python objects, we generate the OpenMM Simulations in the `SimulationFactory()`.
This class takes in 3 inputs:
 1. `parmed.Structure` of the solvated protein:ligand system.
 2. `MoveEngine` to obtain the atom indices to generate the alchemical system.
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
 3. `acceptReject()` : Accept/reject move based on Metropolis criterion.
 4. `simulateMD()` : Allow the system to advance in time.

Described below are the simulation stages described in more detail:
#### `simulateNCMC`: Performing the rotational move.
At this stage we operate on all 3 of the BLUES core objects.
First, we advance the NCMC simulation by referencing `simulations.nc` generated from the `SimulationFactory()` class.
As we take steps in the NCMC simulation, the ligand interactions are alchemically scaled off/on.

Before we take any steps, a `Move` object is randomly chosen from `ligand_mover.moves` to determine which atoms need to be alchemically treated.
We perform the rotation (and moves in general) half-way through the number of NCMC steps to ensure the protocol is symmetric, to help maintain detailed balance.

In the diagram,  attributes are marked with a `*` to denote that these are dynamic variables that **need** to be updated with each step.

#### `acceptReject`: Metropolis-Hastings Acceptance or Rejection.
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


### Implementing Custom Moves
Users can implement their own MC moves into NCMC by inheriting from an appropriate `blues.moves.Move` class and constructing a custom `move()` method that only takes in an Openmm context object as a parameter. The `move()` method will then access the positions of that context, change those positions, then update the positions of that context. For example if you would like to add a move that randomly translates a set of coordinates the code would look similar to this pseudocode:

```python
from blues.moves import Move
class TranslationMove(Move):
    def __init__(self, atom_indices):
        self.atom_indices = atom_indices
    def move(context):
        positions = context.context.getState(getPositions=True).getPositions(asNumpy=True)
        #get positions from context
        #use some function that translates atom_indices
        newPositions = RandomTranslation(positions[self.atom_indices])
        context.setPositions(newPositions)
        return context
```

###Combining Moves together
If you're interested in combining moves together sequentially–say you'd like to perform a rotation and translation move together–instead of coding up a new `Move` class that performs that, you can instead leverage the functionality of existing `Move`s using the `CombinationMove` class. `CombinationMove` takes in a list of instantiated `Move` objects. The `CombinationMove`'s `move()` method perfroms the moves in either listed or reverse order. Replicating a rotation and translation move on t, then, can effectively be done by passing in an instantiated TranslationMove (from the pseudocode example above) and RandomLigandRotation.
One important non-obvious thing to note about the CombinationMove class is that to ensure detailed balance is maintained, moves are done half the time in listed order and half the time in the reverse order.
```
