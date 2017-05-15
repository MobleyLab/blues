"""
ncmc.py: Provides the core class objects (Model, MoveProposal, SimulationFactory)
required to run the BLUES engine

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley
"""

from __future__ import print_function
from openmmtools import alchemy
import numpy as np

from simtk import unit, openmm
from simtk.openmm import app

from blues import utils
from blues.ncmc_switching import NCMCVVAlchemicalIntegrator

import sys, parmed, math, copy
import numpy as np
import mdtraj
from mdtraj.reporters import HDF5Reporter

class Model(object):
    """Model provides methods for calculating properties on the
    object 'model' (i.e ligand) being perturbed in the NCMC simulation.

    Current methods calculate the object's atomic masses and center of masss.
    Calculating the object's center of mass will get the positions and total mass.
    Ex.
        from blues.ncmc import Model
        ligand = Model(structure, 'LIG')
        ligand.calculateProperties()

    Attributes
    ----------
    ligand.resname : string specifying the residue name of the ligand
    ligand.topology : openmm.topology of ligand
    ligand.atom_indices : list of atom indicies of the ligand.
    ligand.masses : list of particle masses of the ligand with units.
    ligand.totalmass : integer of the total mass of the ligand.

    #Dynamic attributes that must be updated with each iteration
    ligand.center_of_mass : np.array of calculated center of mass of the ligand
    ligand.positions : np.array of ligands positions
    """

    def __init__(self, structure, resname='LIG'):
        """Initialize the model.

        Parameters
        ----------
        resname : str
            String specifying the resiue name of the ligand.
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        """

        self.resname = resname
        self.atom_indices = self.getAtomIndices(structure, self.resname)
        self.topology = structure[self.atom_indices].topology
        self.totalmass = 0
        self.masses = []

        self.center_of_mass = None
        self.positions = structure[self.atom_indices].positions

    def getAtomIndices(self, structure, resname):
        """
        Get atom indices of a ligand from ParmEd Structure.
        Arguments
        ---------
        resname : str
            String specifying the resiue name of the ligand.
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        Returns
        -------
        atom_indices : list of ints
            list of atoms in the coordinate file matching lig_resname
        """
        ###TODO: Add option for resnum to better select residue names
        atom_indices = []
        topology = structure.topology
        for atom in topology.atoms():
            if str(resname) in atom.residue.name:
                atom_indices.append(atom.index)
        return atom_indices

    def getMasses(self, topology):
        """Returns a list of masses of the atoms in the model.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        """
        masses = unit.Quantity(np.zeros([int(topology.getNumAtoms()),1],np.float32), unit.dalton)
        for idx,atom in enumerate(topology.atoms()):
            masses[idx] = atom.element._mass
        totalmass = masses.sum()
        return masses, totalmass

    def getCenterOfMass(self, positions, masses):
        """Returns the calculated center of mass of the ligand as a np.array

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the model to be moved.
        masses : numpy.array
            np.array of particle masses
        """
        coordinates = np.asarray(positions._value, np.float32)
        center_of_mass = parmed.geometry.center_of_mass(coordinates, masses) * positions.unit
        return center_of_mass

    def pDB2OEMol(pdbfile):
        '''This function takes in a pdbfile as a string (e.g. 'protein.pdb') and
            reads it into and OEGraphMol'''

        # check if file exists
        if os.path.isfile(pdbfile):
            # read file into an input stream
            ifs = oemolistream(pdbfile)
            # set the format of the input stream to pdb (other wise SMI default)
            ifs.SetFormat(OEFormat_PDB)
            # create OEMol destination
            pdb_OEMol = OEGraphMol()
            # assign input stream to OEMol
            OEReadMolecule(ifs, pdb_OEMol)
            return pdb_OEMol
        else:
            print('PDB filename not found.')

    def getBackboneAtoms(molecule):
        '''This function takes a OEGraphMol PDB structure and returns a list of backbone atoms'''
        backbone_atoms = []
        # Call this function to find atoms and bonds
        OEFindRingAtomsAndBonds(molecule)
        #OEPerceiveResidues(pdb_OEMol)
        if not OEHasResidues(molecule):
                OEPerceiveResidues(molecule, OEPreserveResInfo_All)
        aiter = molecule.GetAtoms(OEIsBackboneAtom())
        for atom in aiter:
            bb_atom_idx = atom.GetIdx()
            backbone_atoms.append(bb_atom_idx)
        return backbone_atoms

    def getTargetAtoms(molecule, residue_list):
        '''This function takes a OEGraphMol PDB structure and a list of residue numbers and
            generates a dictionary containing all the non-backbone, heavy atom locations and indicies for those residues.
            Note: The atom indicies start at 0 and are thus -1 from the PDB file indicies'''

        # Call this function to find atoms and bonds
        OEFindRingAtomsAndBonds(molecule)

        # create and clear dictionary to store atoms that make up residue list
        qry_atoms = {}
        qry_atoms.clear()

        # call backbone function and create backbone atom list
        backbone = getBackboneAtoms(molecule)
        # loop through all the atoms in the PDB OEGraphMol structure
        for atom in molecule.GetAtoms():
            # check if the atom is a heavy atom
            if atom.GetAtomicNum() > 1 and atom.GetIdx() not in backbone:
                # if heavy, find what residue it is associated with
                myres = OEAtomGetResidue(atom)
                # check if the residue number is amongst the list of residues
                if myres.GetResidueNumber() in residue_list:
                    # store the atom location in a query atom dict keyed by its atom index
                    qry_atoms.update({atom : atom.GetIdx()})

        #return dictionary of residue atoms and indicies
        return qry_atoms

    def findHeavyRotBonds(pdb_OEMol, qry_atoms):
        '''This function takes in an OEGraphMol PDB structure as well as a dictionary of atom locations (keys)
            and atom indicies.  It loops over the query atoms and identifies the bonds associated with each atom.
            It stores and returns the bond indicies (keys) and the two atom indicies for each bond in a dictionary
            **Note: atom indicies start at 0, so are offset by 1 compared to pdb)'''

        # Call this function to find atoms and bonds
        OEFindRingAtomsAndBonds(pdb_OEMol)
        # create and clear dictionary to store bond and atom indicies that are rotatable + heavy
        rot_atoms = {}
        rot_atoms.clear()

        for atom in qry_atoms:
            for bond in atom.GetBonds():
                # retrieve the begnning and ending atoms
                begatom = bond.GetBgn()
                endatom = bond.GetEnd()
                # if begnnning and ending atoms are not Hydrogen, and the bond is rotatable
                if endatom.GetAtomicNum() >1 and begatom.GetAtomicNum() >1 and OEIsRotor:
                    # if the bond has not been added to dictionary already..
                    # (as would happen if one of the atom pairs was previously looped over)
                    if bond.GetIdx() not in rot_atoms:
                        # print the bond index
                        print('Bond number',bond, 'is rotatable and contains only heavy atoms')
                        # store bond index number (key) and atom indicies in dictionary if not already there
                        rot_atoms.update({bond.GetIdx() : {'AtomIdx_1' : bond.GetBgnIdx(), 'AtomIdx_2': bond.GetEndIdx()}})
        # Return dictionary with bond atom indicies keyed by bond index
        return rot_atoms

    def getRotBonds(pdbfile,residue_list):
        '''This function takes in a PDB filename (as a string) and list of residue numbers.  It returns
            a dictionary of rotatable bonds (containing only heavy atoms), that are keyed by bond index
            and containing the associated atom indicies
            **Note: The atom indicies start at 0, and are offset by -1 from the PDB file indicies'''
        # read .pdb file into OEGraphMol
        structure = pDB2OEMol(pdbfile)
        # Generate dictionary containing locations and indicies of heavy residue atoms
        qry_atoms = getTargetAtoms(structure, residue_list)
        # Identify bonds containing query atoms and return dictionary of indicies
        rot_atoms = findHeavyRotBonds(structure,qry_atoms)
        print('\nRotable bond and atom index dictionary generated with format: {Bond Index: {Atom1Index:, Atom2Index:}}')
        print('\n')
        print(rot_atoms)

    def calculateProperties(self):
        """Function to quickly calculate available properties."""
        self.masses, self.totalmass = self.getMasses(self.topology)
        self.center_of_mass = self.getCenterOfMass(self.positions, self.masses)


class MoveProposal(object):
    """MoveProposal provides perturbation functions for the model during the NCMC
    simulation. Current supported methods: 'random_rotation.''

    Ex.
        from blues.ncmc import MoveProposal
        mover = MoveProposal(model, 'random_rotation', nstepsNC)

        #Get the dictionary of proposed moves
        mover.moves
    """
    def __init__(self, model, method, nstepsNC):
        """Initialize the MovePropsal object that contains functions to perturb
        the model in the NCMC simulation.

        Parameters
        ----------
        model : blues.ncmc.Model object
            The object to be perturbed in the NCMC simulation.
        method : str
            A string of the perturbation function name (i.e 'random_rotation')
        nstepsNC : int
            An integer value for the number of NCMC steps performed.
        """
        supported_methods = ['random_rotation']
        if method not in supported_methods:
            raise Exception("Method %s not implemented" % method)
        else:
            self.moves = {'model': model, 'method' : None, 'step' : 0}
            self.setMove(method, nstepsNC)

    @staticmethod
    def random_rotation(model, nc_context):
        """Function that performs a random rotation about the center of mass of the model during the NCMC simulation.
        """
        #TODO check if we can remove deepcopy
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

    def setMove(self, method, step):
        """Returns the dictionary that defines the perturbation methods to be
        performed on the model object and the step number to perform it at."""
        self.moves['method']  = getattr(MoveProposal, method)
        self.moves['step'] = int(step) / 2 - 1

class SimulationFactory(object):
    """SimulationFactory is used to generate the 3 required OpenMM Simulation
    objects (MD, NCMC, ALCH) required for the BLUES run.
    Ex.
        from blues.ncmc import SimulationFactory
        sims = SimulationFactory(structure, model, **opt)
        sims.createSimulationSet()
    """
    def __init__(self, structure, model, **opt):
        """Requires a parmed.Structure of the entire system and the ncmc.Model
        object being perturbed.

        Options is expected to be a dict of values. Ex:
        nIter=5, nstepsNC=50, nstepsMD=10000,
        temperature=300, friction=1, dt=0.002,
        nonbondedMethod='PME', nonbondedCutoff=10, constraints='HBonds',
        trajectory_interval=1000, reporter_interval=1000, platform=None,
        verbose=False"""

        #Structure of entire system
        self.structure = structure
        #Atom indicies for model
        self.atom_indices = model.atom_indices

        self.system = None
        self.alch_system = None
        self.md = None
        self.alch  = None
        self.nc  = None

        self.opt = opt
    def generateAlchSystem(self, system, atom_indices):
        """Returns the OpenMM System for alchemical perturbations.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list
            Atom indicies of the model.
        """
        import logging
        logging.getLogger("openmmtools.alchemy").setLevel(logging.ERROR)
        factory = alchemy.AlchemicalFactory()
        alch_region = alchemy.AlchemicalRegion(alchemical_atoms=atom_indices)
        alch_system = factory.create_alchemical_system(system, alch_region)
        return alch_system

    def generateSystem(self, structure, nonbondedMethod='PME', nonbondedCutoff=10,
                       constraints='HBonds', **opt):
        """Returns the OpenMM System for the reference system.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        opt : optional parameters (i.e. cutoffs/constraints)
        """
        system = structure.createSystem(nonbondedMethod=eval("app.%s" % nonbondedMethod),
                            nonbondedCutoff=nonbondedCutoff*unit.angstroms,
                            constraints=eval("app.%s" % constraints) )
        return system

    def generateSimFromStruct(self, structure, system, nIter, nstepsNC, nstepsMD,
                             temperature=300, dt=0.002, friction=1,
                             reporter_interval=1000,
                             ncmc=False, platform=None,
                             verbose=False, printfile=sys.stdout,  **opt):
        """Used to generate the OpenMM Simulation objects given a ParmEd Structure.

        Parameters
        ----------
        structure: parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system :
        opt : optional parameters (i.e. cutoffs/constraints)
        atom_indices : list
            Atom indicies of the model.
        """
        if ncmc:
            #During NCMC simulation, lambda parameters are controlled by function dict below
            # Keys correspond to parameter type (i.e 'lambda_sterics', 'lambda_electrostatics')
            # 'lambda' = step/totalsteps where step corresponds to current NCMC step,
            functions = { 'lambda_sterics' : 'min(1, (1/0.3)*abs(lambda-0.5))',
                          'lambda_electrostatics' : 'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)' }

            integrator = NCMCVVAlchemicalIntegrator(temperature*unit.kelvin,
                                                    system,
                                                    functions,
                                                    nsteps=nstepsNC,
                                                    direction='insert',
                                                    timestep=0.001*unit.picoseconds,
                                                    steps_per_propagation=1)
        else:
            integrator = openmm.LangevinIntegrator(temperature*unit.kelvin,
                                                   friction/unit.picosecond,
                                                   dt*unit.picoseconds)
        #TODO SIMPLIFY TO 1 LINE.
        #Specifying platform properties here used for local development.
        if platform is None:
            #Use the fastest available platform
            simulation = app.Simulation(structure.topology, system, integrator)
        else:
            platform = openmm.Platform.getPlatformByName(platform)
            #prop = dict(DeviceIndex='2') # For local testing with multi-GPU Mac.
            simulation = app.Simulation(structure.topology, system, integrator, platform) #, prop)

        if verbose:
            # OpenMM platform information
            mmver = openmm.version.version
            mmplat = simulation.context.getPlatform()
            print('OpenMM({}) simulation generated for {} platform'.format(mmver, mmplat.getName()), file=printfile)

            # Host information
            # ._asdict() is incompatible with py2.7
            #from platform import uname
            #for k,v in uname()._asdict().items():
            #    print(k, ':', v, file=printfile)

            # Platform properties
            for prop in mmplat.getPropertyNames():
                val = mmplat.getPropertyValue(simulation.context, prop)
                print(prop, ':', val, file=printfile)

        # Set initial positions/velocities
        # Will get overwritten from saved State.
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)

        #TODO MOVE SIMULATION REPORTERS TO OWN FUNCTION.
        simulation.reporters.append(app.StateDataReporter(sys.stdout, separator="\t",
                                    reportInterval=reporter_interval,
                                    step=True, totalSteps=nIter*nstepsMD,
                                    time=True, speed=True, progress=True,
                                    elapsedTime=True, remainingTime=True))

        return simulation

    def createSimulationSet(self):
        """Function used to generate the 3 OpenMM Simulation objects."""
        self.system = self.generateSystem(self.structure, **self.opt)
        self.alch_system = self.generateAlchSystem(self.system, self.atom_indices)

        self.md = self.generateSimFromStruct(self.structure, self.system, **self.opt)
        self.alch = self.generateSimFromStruct(self.structure, self.system,  **self.opt)
        self.nc = self.generateSimFromStruct(self.structure, self.alch_system,
                                            ncmc=True, **self.opt)
