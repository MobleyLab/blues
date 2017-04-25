# Diagram of classes in BLUES
![Class diagram](class-diagram.png)

## Legend
- ![#97D017](https://placehold.it/15/97D077/000000?text=+) `User Input`
- ![#B3B3B3](https://placehold.it/15/B3B3B3/000000?text=+) `Core BLUES Objects`
- ![#6C8EBF](https://placehold.it/15/6C8EBF/000000?text=+) `Simulation() functions`

### ![#97D017](https://placehold.it/15/97D077/000000?text=+) `User Input`
Before running the BLUES simulation, the user is expected to have a forcefield
parameterized `parmed.Structure` of the solvated protein:ligand system.
For example, in the `blues/example.py` script we have:

```python    
#Generate the ParmEd Structure
prmtop = utils.get_data_filename('blues', 'tests/data/eqToluene.prmtop')#
inpcrd = utils.get_data_filename('blues', 'tests/data/eqToluene.inpcrd')
struct = parmed.load_file(prmtop, xyz=inpcrd)
```

3 other inputs are required to generate the 3 core BLUES objects (described in more detail below):
- `Model()`
- `MoveProposal()`
- `SimulationFactory()`

In the `Model()` class, we specify the residue name of the _model_ object that we will
be rotating in the BLUES simulation.  In the example, we are rotating the toluene ligand
in T4 lysozyme.
