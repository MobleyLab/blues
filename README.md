# `BLUES`: Binding modes of Ligands Using Enhanced Sampling
<img align="right" src="./images/blues.png" width="300">

This package takes advantage of non-equilibrium candidate Monte Carlo moves (NCMC) to help sample between different ligand binding modes.


Latest release:
[![Build Status](https://travis-ci.org/MobleyLab/blues.svg?branch=master)](https://travis-ci.org/MobleyLab/blues)
[![Documentation Status](https://readthedocs.org/projects/mobleylab-blues/badge/?version=master](https://mobleylab-blues.readthedocs.io/en/master/?badge=master)
[![codecov](https://codecov.io/gh/MobleyLab/blues/branch/master/graph/badge.svg)](https://codecov.io/gh/MobleyLab/blues)
[![Anaconda-Server Badge](https://anaconda.org/mobleylab/blues/badges/version.svg)](https://anaconda.org/mobleylab/blues)
 [![DOI](https://zenodo.org/badge/62096511.svg)](https://zenodo.org/badge/latestdoi/62096511)


## Citations
#### Publications
- [Gill, S; Lim, N. M.; Grinaway, P.; Rustenburg, A. S.; Fass, J.; Ross, G.; Chodera, J. D.; Mobley, D. L. “Binding Modes of Ligands Using Enhanced Sampling (BLUES): Rapid Decorrelation of Ligand Binding Modes Using Nonequilibrium Candidate Monte Carlo”](https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b11820) - Journal of Physical Chemistry B. February 27, 2018
- [Burley, K. H., Gill, S. C., Lim, N. M., & Mobley, D. L. "Enhancing Sidechain Rotamer Sampling Using Non-Equilibrium Candidate Monte Carlo"](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b01018) - Journal of Chemical Theory and Computation. January 24, 2019

#### Preprints
- [BLUES v1](https://chemrxiv.org/articles/Binding_Modes_of_Ligands_Using_Enhanced_Sampling_BLUES_Rapid_Decorrelation_of_Ligand_Binding_Modes_Using_Nonequilibrium_Candidate_Monte_Carlo/5406907) - ChemRxiv September 19, 2017
- [BLUES v2](https://doi.org/10.26434/chemrxiv.5406907.v2) - ChemRxiv September 25, 2017

## Manifest
* `blues/` -  Source code and example scripts for BLUES toolkit
* `devtools/` - Developer tools and documentation for conda, travis, and issuing a release
* `docs/` - Documentation
* `images/` - Images/logo for repository

## Prerequisites
BLUES is compatible with MacOSX/Linux with Python>=3.6 (blues<=1.1 still works with Python 2.7)
Install [miniconda](http://conda.pydata.org/miniconda.html) according to your system.

## Requirements
Starting from v1.2, you will need the OpenEye toolkits and related tools:
```bash
conda install -c openeye/label/Orion -c omnia oeommtools packmol

# Requires OpenEye License
conda install -c openeye openeye-toolkits
```

## Installation
[ReadTheDocs: Installation](https://mobleylab-blues.readthedocs.io/en/latest/installation.html)

Recommended: Install releases from conda
```bash
conda install -c mobleylab blues
```

Development builds: contains latest commits/PRs not yet issued in a point release
```bash
conda install -c mobleylab/label/dev blues
```

Install from source (NOT RECOMMENDED)
```bash
# Clone the BLUES repository
git clone git@github.com:MobleyLab/blues.git

# Install some dependencies
conda install -c omnia -c conda-forge openmmtools openmm pymbar numpy cython

# Install BLUES package from the top directory
pip install -e .

# To validate your BLUES installation run the tests.
pip instal -e .[tests]
pytest -v -s
```

## Documentation
For documentation on the BLUES modules see [ReadTheDocs: Modules](https://mobleylab-blues.readthedocs.io/en/latest/module_doc.html)

### Usage
This package takes advantage of non-equilibrium candidate Monte Carlo moves (NCMC) to help sample between different ligand binding modes using the OpenMM simulation package. One goal for this package is to allow for easy additions of other moves of interest, which will be covered below.

The integrator of `BLUES` contains the framework necessary for NCMC. Specifically, the integrator class calculates the work done during a NCMC move. It also controls the lambda scaling of parameters. The integrator that BLUES uses inherits from `openmmtools.integrators.AlchemicalNonequilibriumLangevinIntegrator` to keep track of the work done outside integration steps, allowing Monte Carlo (MC) moves to be incorporated together with the NCMC thermodynamic perturbation protocol. Currently, the `openmmtools.alchemy` package is used to generate the lambda parameters for the ligand, allowing alchemical modification of the sterics and electrostatics of the system.

The `BLUESSampler` class in `ncmc.py` serves as a wrapper for running NCMC+MD simulations. To run the hybrid simulation, the `BLUESSampler` class requires defining two moves for running the (1) MD simulation and (2) the NCMC protcol. These moves are defined in the `ncmc.py` module. A simple example is provided below.

#### Example
Using the BLUES framework requires the use of a **ThermodynamicState** and **SamplerState** from `openmmtools` which we import from `openmmtools.states`:

```python
from openmmtools.states import ThermodynamicState, SamplerState
from openmmtools.testsystems import TolueneVacuum
from blues.ncmc import *
from simtk import unit
```

Create the states for a toluene molecule in vacuum.
```python
tol = TolueneVacuum()
thermodynamic_state = ThermodynamicState(tol.system, temperature=300*unit.kelvin)
sampler_state = SamplerState(positions=tol.positions)
```

Define our langevin dynamics move for the MD simulation portion and then our NCMC move which performs a random rotation. Here, we use a customized LangevinDynamicsMove which allows us to store information from the MD simulation portion.

```python
dynamics_move = ReportLangevinDynamicsMove(n_steps=10)
ncmc_move = RandomLigandRotationMove(n_steps=10, atom_subset=list(range(15)))
```

Provide the `BLUESSampler` class with an `openmm.Topology` and these objects to run the NCMC+MD simulation.
```python
sampler = BLUESSampler(thermodynamic_state=thermodynamic_state,
                      sampler_state=sampler_state,
                      dynamics_move=dynamics_move,
                      ncmc_move=ncmc_move,
                      topology=tol.topology)
sampler.run(n_iterations=1)
```

### Implementing custom NCMC moves
Users can implement their own MC moves into NCMC by inheriting from an appropriate `blues.ncmc.NCMCMove` class and overriding the `_propose_positions()` method that only takes in and returns a positions array.
Updating the positions in the context is handled by the `BLUESSampler` class.

With blues>=0.2.5, the API has been redesigned to allow compatibility with moves implemented in [`openmmtools.mcmc`](https://openmmtools.readthedocs.io/en/0.18.1/mcmc.html#mcmc-move-types). Users can take MCMC moves and turn them into NCMC moves without having to write new code. Simply, override the `_get_integrator()` method to use the `blues.integrator.AlchemicalExternalLangevinIntegrator` provided in this module. For example:

```python
from blues.ncmc import NCMCMove
from openmmtools.mcmc import MCDisplacementMove
class NCMCDisplacementMove(MCDisplacementMove, NCMCMove):
    def _get_integrator(self, thermodynamic_state):
        return NCMCMove._get_integrator(self,thermodynamic_state)
```

## Versions:
- Version 0.0.1: Basic BLUES functionality/package
- [Version 0.0.2](http://dx.doi.org/10.5281/zenodo.438714): Maintenance release fixing a critical bug and improving organization as a package.
- [Version 0.0.3](http://dx.doi.org/10.5281/zenodo.569065): Refactored BLUES functionality and design.
- [Version 0.0.4](http://dx.doi.org/10.5281/zenodo.569074): Minor bug fixes plus a functionality problem on some GPU configs.
- [Version 0.1.0](http://dx.doi.org/10.5281/zenodo.837900): Refactored move proposals, added Monte Carlo functionality, Smart Darting moves, and changed alchemical integrator.
- [Version 0.1.1](https://doi.org/10.5281/zenodo.1028925): Features to boost move acceptance such as freezing atoms in the NCMC simulation and adding extra propagation steps in the alchemical integrator.
- [Version 0.1.2](https://doi.org/10.5281/zenodo.1040364): Incorporation of SideChainMove functionality (Contributor: Kalistyn Burley)
- [Version 0.1.3](https://doi.org/10.5281/zenodo.1048250): Improvements to simulation logging functionality and parameters for extra propagation.
- [Version 0.2.0](https://doi.org/10.5281/zenodo.1284568): YAML support, API changes, custom reporters.
- [Version 0.2.1](https://doi.org/10.5281/zenodo.1288925): Bug fix in alchemical correction term
- [Version 0.2.2](https://doi.org/10.5281/zenodo.1324415): Bug fixes for OpenEye tests and restarting from the YAML; enhancements to the Logger and package installation.
- [Version 0.2.3](https://zenodo.org/badge/latestdoi/62096511): Improvements to Travis CI, fix in velocity synicng, and add tests for checking freezing selection.
- [Version 0.2.4](): Addition of a simple test that can run on CPU.
- [Version 0.2.5](): API redesign for compatibility with `openmmtools`

## Acknowledgements
We would like to thank Patrick Grinaway and John Chodera for their basic code framework for NCMC in OpenMM (see https://github.com/choderalab/perses/tree/master/perses/annihilation), and John Chodera and Christopher Bayly for their helpful discussions.
