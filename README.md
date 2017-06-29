# `BLUES`: Binding modes of Ligands Using Enhanced Sampling
<img align="right" src="./images/blues.png" width="300">

This package takes advantage of non-equilibrium candidate Monte Carlo moves (NCMC) to help sample between different ligand binding modes.

Latest release: [![DOI](https://zenodo.org/badge/62096511.svg)](https://zenodo.org/badge/latestdoi/62096511)


## Manifest

* `utils/` - some helper scripts for various things (not directly associated with BLUES)
* `blues/` - simple toolkit illustrating the use of RJMCMC to sample over SMARTS-specified atom types; also contains forcefield.py for handling SMIRFF forcefield format.
* `run_scripts/` - example scripts to run blues
* `systems/` - some example systems to run blues on.

## Prerequisites
BLUES compatible with MacOSX/Linux with Python 2.7/3.5
Install [miniconda](http://conda.pydata.org/miniconda.html) according to your systems

## Installation
Recommended: Install from conda
```bash
conda install -c mobleylab blues
```

Install from source
```bash
git clone git@github.com:MobleyLab/blues.git
python setup.py install
```

## Documentation

### BLUES using NCMC

This package takes advantage of non-equilibrium candidate Monte Carlo moves (NCMC) to help sample between different ligand binding modes using the OpenMM simulation package.  One goal for this package is to allow for easy additions of other moves of interest, which will be covered below.

### Example Use
An example of how to set up a simulation sampling the binding modes of toluene bound to T4 lysozyme using NCMC and a rotational move can be found in `blues/example.py`

### Actually using BLUES
The heart of the package is found in `blues/ncmc_switching.py`. This holds the framework for NCMC. Particularly, it contains the integrator class that calculates the work done during the NCMC move. It also controls the lambda scaling of parameters. Currently the openmmtools.alchemy package is used to generate the lambda parameters for the ligand, allowing alchemical modification of the sterics and electrostatics of the system.
The `Simulation` class in `blues/simulation.py` serves as a wrapper for running NCMC simulations.


###### For a detailed explanation of the BLUES framework or implementing new moves, check out the [README](devdocs/README.md) in devdocs.
To see an example of how to use BLUES check blues/example.py for an example script



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

### Combining Moves together
If you're interested in combining moves together sequentially–say you'd like to perform a rotation and translation move together–instead of coding up a new `Move` class that performs that, you can instead leverage the functionality of existing `Move`s using the `CombinationMove` class. `CombinationMove` takes in a list of instantiated `Move` objects. The `CombinationMove`'s `move()` method perfroms the moves in either listed or reverse order. Replicating a rotation and translation move on t, then, can effectively be done by passing in an instantiated TranslationMove (from the pseudocode example above) and RandomLigandRotation.
One important non-obvious thing to note about the CombinationMove class is that to ensure detailed balance is maintained, moves are done half the time in listed order and half the time in the reverse order.
```

## Versions:
- Version 0.0.1: Basic BLUES functionality/package
- [Version 0.0.2](http://dx.doi.org/10.5281/zenodo.438714): Maintenance release fixing a critical bug and improving organization as a package.
- [Version 0.0.3](http://doi.org/10.5281/zenodo.569065): Refactored BLUES functionality and design.

## Acknowledgements
We would like to thank Patrick Grinaway and John Chodera for their basic code framework for NCMC in OpenMM (see https://github.com/choderalab/perses/tree/master/perses/annihilation), and John Chodera and Christopher Bayly for their helpful discussions.
