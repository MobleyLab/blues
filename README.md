[![Build Status](https://travis-ci.org/open-forcefield-group/smarty.svg?branch=master)](https://travis-ci.org/open-forcefield-group/smarty)

# `BLUES`: Binding modes of Ligands Using Enhanced Sampling

This package takes advantage of non-candidate equilibrium monte carlo moves (NCMC) to help sample between different ligand binding modes.

This also provides a prototype and validation of the SMIRFF SMIRKS-based force field format, along with classes to parameterize OpenMM systems given [SMIRFF `.ffxml` format files](https://github.com/open-forcefield-group/smarty/blob/master/The-SMIRFF-force-field-format.md) as provided here.

## Manifest

* `utils/` - some helper scripts for various things (not directly associated with BLUES)
* `blues/` - simple toolkit illustrating the use of RJMCMC to sample over SMARTS-specified atom types; also contains forcefield.py for handling SMIRFF forcefield format.
* `run_scripts/` - example scripts to run blues
* `systems/` - some example systems to run blues on.

## Prerequisites

Install [miniconda](http://conda.pydata.org/miniconda.html) first. On `osx` with `bash`, this is:
```

Install other conda dependencies:
```
conda install --yes omnia mdtraj
conda install --yes omnia openmmtools
conda install --yes omnia alchemy
```

NOTE: We'll add a better way to install these dependencies via `conda` soon.

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
The following is an example of how to set up a simulation sampling the binding modes of toluene bound to T4 lysozyme using NCMC and a rotational move.
```
from simtk.openmm.app import *
from simtk.openmm import *
import simtk import unit
from ncmc_switching import *
import mdtraj as md
from openmmtools import testsystems
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np
from mdtraj.reporters import HDF5Reporter
from smartdart import SmartDarting
from ncmc import *

##load systems and make alchemical systems
coord_file = 'eqToluene.inpcrd'
top_file =   'eqToluene.prmtop'
prmtop = openmm.app.AmberPrmtopFile(top_file)
inpcrd = openmm.app.AmberInpcrdFile(coord_file)
temp_system = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*unit.nanometer, constraints=openmm.app.HBonds)
testsystem = testsystems.TestSystem
testsystem.system = temp_system 
testsystem.topology = prmtop.topology
testsystem.positions = inpcrd.
#helper function to get list of ligand atoms
lig_atoms = get_lig_residues(lig_resname='LIG', coord_file=coord_file, top_file=top_file)
#create alchemical system using alchemy functions
factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=ligand_atoms, annihilate_sterics=True, annihilate_electrostatics=True)
alchemical_system = factory.createPerturbedSystem()
##set up OpenMM simulations
temperature = 300.0*unit.kelvin
#functions describes how lambda scales with nstepsNC
functions = { 'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))', 'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)' }
nc_integrator = NCMCVVAlchemicalIntegrator(temperature, alchemical_system, functions, nsteps=nstepsNC, direction='insert', timestep=0.001*unit.picoseconds)
md_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
md_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=md_integrator)
#dummy_simulation is used to perform alchemical corrections and serves as a reporter for ncmc moves
dummy_integrator = openmm.openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
dummy_simulation = openmm.app.simulation.Simulation(topology=testsystem.topology, system=testsystem.system, integrator=dummy_integrator)
md_simulation.context.setPositions(testsystem.positions)
md_simulation.context.setVelocitiesToTemperature(temperature)
#add reporters
md_simulation.reporters.append(openmm.app.dcdreporter.DCDReporter('traj.dcd', 1000))
md_simulation.reporters.append(HDF5Reporter('traj.h5', 1000))
practice_run = SimNCMC(temperature=temperature, residueList=lig_atoms)
#set nc attributes
numIter = 100
nstepsNC = 100
nstepsMD = 1000

practice_run.runSim(md_simulation, nc_context, nc_integrator, dummy_simulation, movekey=ncmove, niter=numIter, nstepsNC=nstepsNC, nstepsMD=nstepsMD)



```




## Implementing other non-equilibrium moves
An optional argument for SimNCMC.runSim() , movekey allows users to implement their own moves into the simulation workflow. To add your own moves make a subclass of SimNCMC. The custom method can use self.nc_context or self.nc_integrator to access the NCMC context or NCMC integrator in the runSim() method. Then, insert the method into the movekey. As an example, say you implemented a smart darting move in a method called smartdart(). Then for the keymove argument use [[smartdart, [range]]], where range is a list of integers that specify when you want to apply the move. In general the keymove argument takes a list of lists. The first item in the list references the function, while the second references a list which specifies what steps you want to apply the function.

Check out the example in `examples/parm@frosst/`:

Atom types are specified by SMARTS matches with corresponding parameter names.

First, we start with a number of initial "base types" which are essentially indestructible (often generic) atom types, specified in `atomtypes/basetypes.smarts`:
```
% atom types
[#1]    hydrogen
[#6]    carbon
[#7]    nitrogen
[#8]    oxygen
[#9]    fluorine
[#15]   phosphorous
[#16]   sulfur
[#17]   chlorine
[#35]   bromine
[#53]   iodine
```
Note that lines beginning with `%` are comment lines.

We also specify a number of starting types, "initial types" which can be the same or different from the base types. These follow the same format, and `atomtypes/basetypes.smarts` can be reused unless alternate behavior is desired (such as starting from more sophisticated initial types).

We have two sampler options for SMARTY which differ in how focused the sampling is. The original sampler samples over all elements/patterns at once, whereas the elemental sampler focuses on sampling only one specific element. The principle of sampling is the same; the only change is in which elements we sample over. To sample only over a single element, such as oxygen, for example, we use the elemental sampler to focus on that element.


### Original sampler

Command line example: `smarty --samplertype original --basetypes=examples/AlkEtOH/atomtypes/basetypes.smarts --initialtypes=examples/AlkEtOH/atomtypes/basetypes.smarts --decorators=examples/AlkEtOH/atomtypes/new-decorators.smarts  --molecules=examples/AlkEtOH/molecules/test_filt1_tripos.mol2 --reference=examples/AlkEtOH/molecules/test_filt1_ff.mol2 --iterations 100 --temperature=0`

The original sampler is the default option. Here, smarty samples SMARTS patterns covering all elements contained in the set.

Atom type creation moves has two options, one is using simple decorators (`--decoratorbehavior=simple-decorators`) and the other is combinatorial decorators (default).

 The first option (simple-decorators) attempt to split off a new atom type from a parent atom type by combining (via an "and" operator, `&`) the parent atom type with a "decorator".
The decorators are listed in `AlkEtOH/atomtypes/decorators.smarts` or `parm@frosst/atomtypes/decorators.smarts`:
```
% bond order
$([*]=[*])     double-bonded
$([*]#[*])     triple-bonded
$([*]:[*])     aromatic-bonded
% bonded to atoms
$(*~[#1])      hydrogen-adjacent
$(*~[#6])      carbon-adjacent
$(*~[#7])      nitrogen-adjacent
$(*~[#8])      oxygen-adjacent
$(*~[#9])      fluorine-adjacent
$(*~[#15])     phosphorous-adjacent
$(*~[#16])     sulfur-adjacent
$(*~[#17])     chlorine-adjacent
$(*~[#35])     bromine-adjacent
$(*~[#53])     iodine-adjacent
% degree
D1             degree-1
D2             degree-2
D3             degree-3
D4             degree-4
D5             degree-5
D6             degree-6
% valence
v1             valence-1
v2             valence-2
v3             valence-3
v4             valence-4
v5             valence-5
v6             valence-6
% total-h-count
H1             total-h-count-1
H2             total-h-count-2
H3             total-h-count-3
% aromatic/aliphatic
a              atomatic
A              aliphatic
```
Each decorator has a corresponding string token (no spaces allowed!) that is used to create human-readable versions of the corresponding atom types.

For example, we may find the atom type ```[#6]&H3``` which is `carbon total-h-count-3` for a C atom bonded to three hydrogens.

The second option (combinatorial-decorator) attempt to create the new atomtype adding an Alpha or Beta substituent to a basetype or an atomtype.
This decorators are different from the simple-decorator option and do not have atom types or bond information on it.
The new decorators are listed in `AlkEtOH/atomtypes/new-decorators.smarts` and `parm@frosst/atomtypes/new-decorators.smarts`:

 ```
 % total connectivity
 X1             connections-1
 X2             connections-2
 X3             connections-3
 X4             connections-4
 % total-h-count
 H0             total-h-count-0
 H1             total-h-count-1
 H2             total-h-count-2
 H3             total-h-count-3
 % formal charge
 +0             neutral
 +1             cationic+1
 -1             anionic-1
 % aromatic/aliphatic
 a              aromatic
 A              aliphatic
 ```
This option also has the corresponding string token.

Example: `smarty --basetypes=examples/AlkEtOH/atomtypes/basetypes.smarts --initialtypes=examples/AlkEtOH/atomtypes/basetypes.smarts --decorators=examples/AlkEtOH/atomtypes/new-decorators.smarts  --molecules=examples/AlkEtOH/molecules/test_filt1_tripos.mol2 --reference=examples/AlkEtOH/molecules/test_filt1_ff.mol2 --iterations 1000 --temperature=0.00001`

Newly proposed atom types are added to the end of the list.
After a new atom type is proposed, all molecules are reparameterized using the new set of atom types.
Atom type matching proceeds by trying to see if each SMARTS match can be applied working from top to bottom of the list.
This means the atom type list is hierarchical, with more general types appearing at the top of the list and more specific subtypes appearing at the bottom.

If a proposed type matches zero atoms, the RJMCMC move is rejected.

Currently, the acceptance criteria does not include the full Metropolis-Hastings acceptance criteria that would include the reverse probability.  This needs to be added in.

### Elemental sampler

Command line example: `smarty --samplertype elemental --element=8 --basetypes=examples/AlkEtOH/atomtypes/basetypes.smarts --initialtypes=examples/AlkEtOH/atomtypes/basetypes.smarts --decorators=examples/AlkEtOH/atomtypes/new-decorators.smarts  --molecules=examples/AlkEtOH/molecules/test_filt1_tripos.mol2 --reference=examples/AlkEtOH/molecules/test_filt1_ff.mol2 --iterations 100 --temperature=0`

The elemental sampler has the same principles as the original sampler. However, the sampler will sample only a single element (such as Oxygen, Carbon, Hydrogen, etc), which needs to be specified on the command line.

The element number needs to be specified by atomic number (--element=8 for oxygen).

=======
##smirky

Check out examples in `examples/smirky/`:

This tool can sample any chemical environment type relevant to SMIRFFs, that is atoms, bonds, angles, and proper and improper torsions, one at a time
Scoring is analous to smarty (explained above), but uses a SMIRFF with existing parameters as a reference insteady of atomtyped molecules.

Input for this tool can require up to four different file types
* MOLECULES - any file that are readable in openeye, mol2, sdf, oeb, etc.
* ODDSFILES - File with the form "smarts     odds" for the different decorator or bond options
* SMARTS - .smarts file type with the form "smarts/smirks      label/typename"
* REFERENCE - a SMIRFF file with reference atoms, bonts, angles, torsions, and impropers

```
Usage:     Sample over fragment types (atoms, bonds, angles, torsions, or impropers)
    optionally attempting to match created types to an established SMIRFF.
    For all files left blank, they will be taken from this module's
    data/odds_files/ subdirectory.

    usage smirky --molecules molfile --typetag fragmentType
            [--atomORbases AtomORbaseFile --atomORdecors AtomORdecorFile
            --atomANDdecors AtomANDdecorFile --bondORbase BondORbaseFile
            --bondANDdecors BondANDdecorFile --atomIndexOdds AtomIndexFile
            --bondIndexOdds BondIndexFile --replacements substitutions
            --initialFragments initialFragments --SMIRFF referenceSMIRFF
            --temperature float --verbose verbose
            --iterations iterations --output outputFile]

    example:
    smirky -molecules AlkEthOH_test_filt1_ff.mol2 --typetag Angle



Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -m MOLECULES, --molecules=MOLECULES
                        Small molecule set (in any OpenEye compatible file
                        format) containing 'dG(exp)' fields with experimental
                        hydration free energies. This filename can also be an
                        option in this module's data/molecules sub-directory
  -T TYPETAG, --typetag=TYPETAG
                        type of fragment being sampled, options are 'VdW',
                        'Bond', 'Angle', 'Torsion', 'Improper'
  -e ODDFILES, --atomORbases=ODDFILES
                        Filename defining atom OR bases and associated
                        probabilities. These are combined with atom OR
                        decorators in SMIRKS, for example in
                        '[#6X4,#7X3;R2:2]' '#6' and '#7' are atom OR bases.
                        (OPTIONAL)
  -O ODDFILES, --atomORdecors=ODDFILES
                        Filename defining atom OR decorators and associated
                        probabilities. These are combined with atom bases in
                        SMIRKS, for example in '[#6X4,#7X3;R2:2]' 'X4' and
                        'X3' are ORdecorators. (OPTIONAL)
  -A ODDFILES, --atomANDdecors=ODDFILES
                        Filename defining atom AND decorators and associated
                        probabilities. These are added to the end of an atom's
                        SMIRKS, for example in '[#6X4,#7X3;R2:2]' 'R2' is an
                        AND decorator. (OPTIONAL)
  -o ODDFILES, --bondORbase=ODDFILES
                        Filename defining bond OR bases and their associated
                        probabilities. These are OR'd together to describe a
                        bond, for example in '[#6]-,=;@[#6]' '-' and '=' are
                        OR bases. (OPTIONAL)
  -a ODDFILES, --bondANDdecors=ODDFILES
                        Filename defining bond AND decorators and their
                        associated probabilities. These are AND'd to the end
                        of a bond, for example in '[#6]-,=;@[#7]' '@' is an
                        AND decorator.(OPTIONAL)
  -D ODDSFILE, --atomOddsFile=ODDSFILE
                        Filename defining atom descriptors and probabilities
                        with making changes to that kind of atom. Options for
                        descriptors are integers corresponding to that indexed
                        atom, 'Indexed', 'Unindexed', 'Alpha', 'Beta', 'All'.
                        (OPTIONAL)
  -d ODDSFILE, --bondOddsFile=ODDSFILE
                        Filename defining bond descriptors and probabilities
                        with making changes to that kind of bond. Options for
                        descriptors are integers corresponding to that indexed
                        bond, 'Indexed', 'Unindexed', 'Alpha', 'Beta', 'All'.
                        (OPTIONAL)
  -s SMARTS, --substitutions=SMARTS
                        Filename defining substitution definitions for SMARTS
                        atom matches. (OPTIONAL).
  -f SMARTS, --initialtypes=SMARTS
                        Filename defining initial (first) fragment types as
                        'SMIRKS    typename'. If this is left blank the
                        initial type will be a generic form of the given
                        fragment, for example '[*:1]~[*:2]' for a bond
                        (OPTIONAL)
  -r REFERENCE, --smirff=REFERENCE
                        Filename defining a SMIRFF force fielce used to
                        determine reference fragment types in provided set of
                        molecules. It may be an absolute file path, a path
                        relative to the current working directory, or a path
                        relative to this module's data subdirectory (for built
                        in force fields). (OPTIONAL)
  -i ITERATIONS, --iterations=ITERATIONS
                        MCMC iterations.
  -t TEMPERATURE, --temperature=TEMPERATURE
                        Effective temperature for Monte Carlo acceptance,
                        indicating fractional tolerance of mismatched atoms
                        (default: 0.1). If 0 is specified, will behave in a
                        greedy manner.
  -p OUTPUT, --output=OUTPUT
                        Filename base for output information. This same base
                        will be used for all output files created. If None
                        provided then it is set to 'typetag_temperature'
                        (OPTIONAL).
  -v VERBOSE, --verbose=VERBOSE
                        If True prints minimal information to the commandline
                        during iterations. (OPTIONAL)
```

## SMIRFF

The SMIRFF forcefield format is available in sample form under data/forcefield, and is handled by `forcefield.py`.
 An example comparing SMIRFF versus AMBER energies for the parm@frosst forcefield is provided under
examples/SMIRFF_comparison, where two scripts can compare energies for a single molecule or for the entire AlkEthOH set.
Note that two forcefields are currently available in this format, `Fross_AlkEtOH.ffxml`,
the parm@frosst forcefield as it should have been for this set, and `Frosst_AlkEtOH_parmAtFrosst.ffxml`,
the forcefield as it was actually implemented (containing several bugs as noted in the file itself).

It can also be of interest to know what SMIRFF parameters would be applied to particular molecules. Utility functionality for this is provided under `forcefield_labeler.py`, which has generally similar structure to `forcefield.py` but instead of providing OpenMM systems with parameters, it can be applied to specific molecules and returns information about what parameters would be applied.

## References

[1] Green PJ. Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. Biometrika 82:711, 1995.
http://dx.doi.org/10.1093/biomet/82.4.711
