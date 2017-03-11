# `BLUES`: Binding modes of Ligands Using Enhanced Sampling
<img align="right" src="./images/blues.png">

This package takes advantage of non-candidate equilibrium monte carlo moves (NCMC) to help sample between different ligand binding modes.

This also provides a prototype and validation of the SMIRFF SMIRKS-based force field format, along with classes to parameterize OpenMM systems given [SMIRFF `.ffxml` format files](https://github.com/open-forcefield-group/smarty/blob/master/The-SMIRFF-force-field-format.md) as provided here.

## Manifest

* `utils/` - some helper scripts for various things (not directly associated with BLUES)
* `blues/` - simple toolkit illustrating the use of RJMCMC to sample over SMARTS-specified atom types; also contains forcefield.py for handling SMIRFF forcefield format.
* `run_scripts/` - example scripts to run blues
* `systems/` - some example systems to run blues on.

## Prerequisites

Install [miniconda](http://conda.pydata.org/miniconda.html) first. On `osx` with `bash`, just run:
```
bash Miniconda3-latest-MacOSX-x86_64.sh
```
from the command-line after downloading.

Install other conda dependencies, after adding the omnia channel:
```
conda config add --channels omnia
conda install --yes omnia mdtraj
conda install --yes omnia openmmtools
conda install --yes omnia alchemy
```


## Installation

Install `blues` from the `blues/` directory with:
```bash
pip install .
```

## Documentation

## BLUES using NCMC

This package takes advantage of non-candidate equilibrium monte carlo moves (NCMC) to help sample between different ligand binding modes using the OpenMM simulation package. Currently the innate functionality is found in `blues/ncmc.py`, which utilizes a lambda coupling to alter the sterics and electrostatics of the ligand over the course of the ncmc move. One goal for this package is to allow for easy additions of other moves of interest, which will be covered below.

## Actually using BLUES
The heart of the package is found in `blues/ncmc_switching.py`. This holds the framework for NCMC. Particularly, it contains the integrator class that calculates the work done during the NCMC move. It also controls the lambda scaling of parameters. Currently the alchemy package is used to generate the lambda parameters for the ligand, which can potentially modify 5 parameters–sterics, electrostatics, bonds, angles, and torsions.
The class SimNCMC in `blues/ncmc.py` serves as a wrapper for running ncmc simulations. SimNCMC.runSim() actually runs the simulation.

## Example Use
An example of how to set up a simulation sampling the binding modes of toluene bound to T4 lysozyme using NCMC and a rotational move can be found in `examples/example.py`

## Implementing other non-equilibrium moves
An optional argument for SimNCMC.runSim() , movekey allows users to implement their own moves into the simulation workflow. To add your own moves make a subclass of SimNCMC. The custom method can use self.nc_context or self.nc_integrator to access the NCMC context or NCMC integrator in the runSim() method. Then, insert the method into the movekey. As an example, say you implemented a smart darting move in a method called smartdart(). Then for the keymove argument use [[smartdart, [range]]], where range is a list of integers that specify when you want to apply the move. In general the keymove argument takes a list of lists. The first item in the list references the function, while the second references a list which specifies what steps you want to apply the function.
##Acknowledgements
We would like to thank Patrick Grinaway and John Chodera for their basic code framework for NCMC in OpenMM (see https://github.com/choderalab/perses/tree/master/perses/annihilation), and John Chodera and Christopher Bayly for their helpful discussions.
