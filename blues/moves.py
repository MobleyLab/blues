"""
Provides the main Move class which allows definition of moves
which alter the positions of subsets of atoms in a context during a BLUES
simulation, in order to increase sampling.
Also provides functionality for CombinationMove definitions which consist of
a combination of other pre-defined moves such as via instances of Move.

Authors: Samuel C. Gill

Contributors: Nathan M. Lim, Kalistyn Burley, David L. Mobley
"""
from simtk import openmm, unit
from openmmtools import alchemy
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from blues.integrators import AlchemicalExternalLangevinIntegrator
from openmmtools.mcmc import LangevinDynamicsMove, MetropolizedMove
from blues.simulation import SystemFactory
from blues import utils

import parmed
import mdtraj
import numpy
import abc
import copy
import math
import logging
import os
import time

import copy
import math
import random
import sys
import traceback

import mdtraj
import numpy
import parmed
from simtk import unit

try:
    import openeye.oechem as oechem
    if not oechem.OEChemIsLicensed():
        print('ImportError: Need License for OEChem! SideChainMove class will be unavailable.')
    try:
        import oeommtools.utils as oeommtools
    except ImportError:
        print('ImportError: Could not import oeommtools. SideChainMove class will be unavailable.')
except ImportError:
    print('ImportError: Could not import openeye-toolkits. SideChainMove class will be unavailable.')


class MetropolizedNCMCMove(MetropolizedMove):
    """This is the base Move class. Move provides methods for calculating properties
    and applying the move on the set of atoms being perturbed in the NCMC simulation.
    """

    def __init__(self, timestep, n_steps, atom_subset=None, context_cache=None):
        self.timestep = timestep
        self.n_steps = n_steps
        self.n_accepted = 0
        self.n_proposed = 0
        self.logp_accept = 0
        self.initial_energy = None
        self.initial_positions = None
        self.final_energy = None
        self.final_positions = None
        self.proposed_positions = None
        self.atom_subset = atom_subset
        self.context_cache = context_cache

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(
            n_accepted=self.n_accepted,
            n_proposed=self.n_proposed,
            initial_energy=self.initial_energy,
            initial_positions=self.initial_positions,
            final_energy=self.final_energy,
            proposed_positions=self.proposed_positions,
            final_positions=self.final_positions,
            logp_accept=self.logp_accept)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value['n_accepted']
        self.n_proposed = value['n_proposed']
        self.initial_energy = value['initial_energy']
        self.initial_positions = value['initial_positions']
        self.final_energy = value['final_energy']
        self.proposed_positions = value['proposed_positions']
        self.final_positions = value['final_positions']
        self.logp_accept = value['logp_accept']

    def _before_integration(self, context, thermodynamic_state):
        """Execute code after Context creation and before integration."""
        context_state = context.getState(getPositions=True,
                                         getVelocities=True,
                                         getEnergy=True,
                                         enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.initial_positions = context_state.getPositions()
        #self.initial_energy = context_state.getPotentialEnergy()
        self.initial_energy = thermodynamic_state.reduced_potential(context)

    def _after_integration(self, context, thermodynamic_state):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        context_state = context.getState(getPositions=True,
                                         getVelocities=True,
                                         getEnergy=True,
                                         enforcePeriodicBox=thermodynamic_state.is_periodic)


        self.final_positions = context_state.getPositions()
        #self.final_energy = context_state.getPotentialEnergy()
        self.final_energy = thermodynamic_state.reduced_potential(context)
        self.logp_accept = context._integrator.getLogAcceptanceProbability(context)

    def _get_integrator(self, thermodynamic_state):
        return AlchemicalExternalLangevinIntegrator(
            alchemical_functions={
                'lambda_sterics':
                'min(1, (1/0.3)*abs(lambda-0.5))',
                'lambda_electrostatics':
                'step(0.2-lambda) - 1/0.2*lambda*step(0.2-lambda) + 1/0.2*(lambda-0.8)*step(lambda-0.8)'
            },
            splitting="H V R O R V H",
            temperature=thermodynamic_state.temperature,
            nsteps_neq=self.n_steps,
            timestep=1.0 * unit.femtoseconds,
            nprop=1,
            prop_lambda=0.3)

    def apply(self, thermodynamic_state, sampler_state, reporters):
        """Apply a metropolized move to the sampler state.

        Total number of acceptances and proposed move are updated.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to apply the move.
        sampler_state : openmmtools.states.SamplerState
           The initial sampler state to apply the move to. This is modified.

        """
        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create integrator
        integrator = self._get_integrator(thermodynamic_state)

        # Create context
        context, integrator = context_cache.get_context(thermodynamic_state, integrator)


        # Compute initial energy. We don't need to set velocities to compute the potential.
        # TODO assume sampler_state.potential_energy is the correct potential if not None?
        sampler_state.apply_to_context(context, ignore_velocities=True)
        context.setVelocitiesToTemperature(thermodynamic_state.temperature)

        self._before_integration(context, thermodynamic_state)


        try:
            # Run dynamics
            #NML: Do in 1 steps for debugging
            rotation_step = int(self.n_steps / 2)
            for n in range(1, self.n_steps + 1):
                integrator.step(1)

                if n == rotation_step:
                    # Propose perturbed positions. Modifying the reference changes the sampler state.
                    sampler_state.update_from_context(context)
                    proposed_positions = self._propose_positions(sampler_state.positions[self.atom_subset])
                    # Compute the energy of the proposed positions.
                    sampler_state.positions[self.atom_subset] = proposed_positions
                    sampler_state.apply_to_context(context)


        except Exception as e:
            print(e)
            # Catches particle positions becoming nan during integration.
            restart = True
        else:
            context_state = context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=thermodynamic_state.is_periodic)
            # Update everything but the collective variables from the State object
            sampler_state.update_from_context(
                context_state, ignore_collective_variables=True, ignore_velocities=True)
            # Update only the collective variables from the Context
            sampler_state.update_from_context(
                context, ignore_positions=True, ignore_velocities=True, ignore_collective_variables=False)


        self._after_integration(context, thermodynamic_state)


    @abc.abstractmethod
    def _propose_positions(self, positions):
        """Return new proposed positions.

        These method must be implemented in subclasses.

        Parameters
        ----------
        positions : nx3 numpy.ndarray
            The original positions of the subset of atoms that these move
            applied to.

        Returns
        -------
        proposed_positions : nx3 numpy.ndarray
            The new proposed positions.

        """
        pass

class ReportLangevinDynamicsMove(LangevinDynamicsMove):
    def _before_integration(self, context, thermodynamic_state):
        """Execute code after Context creation and before integration."""
        context_state = context.getState(getPositions=True,
                                         getVelocities=True,
                                         getEnergy=True,
                                         enforcePeriodicBox=thermodynamic_state.is_periodic)

        self.initial_positions = context_state.getPositions()
        #self.initial_energy = context_state.getPotentialEnergy()
        self.initial_energy = thermodynamic_state.reduced_potential(context)

    def _after_integration(self, context, thermodynamic_state):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        context_state = context.getState(getPositions=True,
                                         getVelocities=True,
                                         getEnergy=True,
                                         enforcePeriodicBox=thermodynamic_state.is_periodic)


        self.final_positions = context_state.getPositions()
        #self.final_energy = context_state.getPotentialEnergy()
        self.final_energy = thermodynamic_state.reduced_potential(context)

    def apply(self, thermodynamic_state, sampler_state, reporters):
        """Propagate the state through the integrator.

        This updates the SamplerState after the integration. It also logs
        benchmarking information through the utils.Timer class.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        See Also
        --------
        openmmtools.utils.Timer

        """
        move_name = self.__class__.__name__

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create integrator.
        integrator = self._get_integrator(thermodynamic_state)

        # Create context.
        context, integrator = context_cache.get_context(thermodynamic_state, integrator)
        thermodynamic_state.apply_to_context(context)

        # If we reassign velocities, we can ignore the ones in sampler_state.
        sampler_state.apply_to_context(context, ignore_velocities=self.reassign_velocities)
        if self.reassign_velocities:
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)

        # Subclasses may implement _before_integration().
        self._before_integration(context, thermodynamic_state)

        try:
            # Run dynamics.
            for n in range(1, self.n_steps + 1):
                integrator.step(1)

                if n % reporters[0]._reportInterval == 0:
                    context_state = context.getState(
                        getPositions=True,
                        getVelocities=True,
                        getEnergy=True,
                        enforcePeriodicBox=thermodynamic_state.is_periodic)
                    context_state.currentStep = n
                    context_state.system = thermodynamic_state.system
                    for reporter in reporters:
                        reporter.report(context_state, integrator)

        except Exception as e:
            print(e)
            # Catches particle positions becoming nan during integration.
        else:


            # We get also velocities here even if we don't need them because we
            # will recycle this State to update the sampler state object. This
            # way we won't need a second call to Context.getState().
            context_state = context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=thermodynamic_state.is_periodic)

            # Check for NaNs in energies.
            potential_energy = context_state.getPotentialEnergy()

        # Updated sampler state.
        # This is an optimization around the fact that Collective Variables are not a part of the State,
        # but are a part of the Context. We do this call twice to minimize duplicating information fetched from
        # the State.
        # Update everything but the collective variables from the State object
        sampler_state.update_from_context(context_state,
                                          ignore_collective_variables=True)
        # Update only the collective variables from the Context
        sampler_state.update_from_context(context,
                                          ignore_positions=True,
                                          ignore_velocities=True,
                                          ignore_collective_variables=False)

        # Subclasses can read here info from the context to update internal statistics.
        self._after_integration(context, thermodynamic_state)

class RandomLigandRotationMove(MetropolizedNCMCMove):
    def _before_integration(self, context, thermodynamic_state):
        super(RandomLigandRotationMove, self)._before_integration(context, thermodynamic_state)
        masses, totalmass = utils.getMasses(self.atom_subset, thermodynamic_state.topology)
        self.masses = masses

    def _propose_positions(self, positions):
        """Return new proposed positions.

        These method must be implemented in subclasses.

        Parameters
        ----------
        positions : nx3 numpy.ndarray
            The original positions of the subset of atoms that these move
            applied to.

        Returns
        -------
        proposed_positions : nx3 numpy.ndarray
            The new proposed positions.

        """

        #print('Proposing positions...')
        # Calculate the center of mass

        center_of_mass = utils.getCenterOfMass(positions, self.masses)
        reduced_pos = positions - center_of_mass
        # Define random rotational move on the ligand
        rand_quat = mdtraj.utils.uniform_quaternion(size=None)
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
        #multiply lig coordinates by rot matrix and add back COM translation from origin
        proposed_positions = numpy.dot(reduced_pos, rand_rotation_matrix) * positions.unit + center_of_mass

        return proposed_positions

########################
## UNDER DEVELOPMENT ###
########################


# class SideChainMove(Move):
#     """**NOTE:** Usage of this class requires a valid OpenEye license.
#
#     SideChainMove provides methods for calculating properties needed to
#     rotate a sidechain residue given a parmed.Structure. Calculated properties
#     include: backbone atom indicies, atom pointers and indicies of the residue
#     sidechain, bond pointers and indices for rotatable heavy bonds in
#     the sidechain, and atom indices upstream of selected bond.
#
#     The class contains functions to randomly select a bond and angle to be rotated
#     and applies a rotation matrix to the target atoms to update their coordinates on the
#     object 'model' (i.e sidechain) being perturbed in the NCMC simulation.
#
#     Parameters
#     ----------
#     structure : parmed.Structure
#         The structure of the entire system to be simulated.
#     residue_list : list of int
#         List of the residue numbers of the sidechains to be rotated.
#     verbose : bool, default=False
#         Enable verbosity to print out detailed information of the rotation.
#     write_move : bool, default=False
#         If True, writes a PDB of the system after rotation.
#
#     Attributes
#     ----------
#     structure : parmed.Structure
#         The structure of the entire system to be simulated.
#     molecule : oechem.OEMolecule
#         The OEMolecule containing the sidechain(s) to be rotated.
#     residue_list : list of int
#         List containing the residue numbers of the sidechains to be rotated.
#     all_atoms : list of int
#         List containing the atom indicies of the sidechains to be rotated.
#     rot_atoms : dict
#         Dictionary of residues, bonds and atoms to be rotated
#     rot_bonds : dict of oechem.OEBondBase
#         Dictionary containing the bond pointers of the rotatable bonds.
#     qry_atoms : dict of oechem.OEAtomBase
#         Dictionary containing all the atom pointers (as OpenEye objects) that
#         make up the given residues.
#
#
#     Examples
#     --------
#     >>> from blues.move import SideChainMove
#     >>> sidechain = SideChainMove(structure, [1])
#
#     """
#
#     def __init__(self, structure, residue_list, verbose=False, write_move=False):
#         self.structure = structure
#         self.molecule = self._pmdStructureToOEMol()
#         self.residue_list = residue_list
#         self.all_atoms = [atom.index for atom in self.structure.topology.atoms()]
#         self.rot_atoms, self.rot_bonds, self.qry_atoms = self.getRotBondAtoms()
#         self.atom_indices = self.rot_atoms
#         self.verbose = verbose
#         self.write_move = write_move
#
#     def _pmdStructureToOEMol(self):
#         """Helper function for converting the parmed structure into an OEMolecule."""
#         top = self.structure.topology
#         pos = self.structure.positions
#         molecule = oeommtools.openmmTop_to_oemol(top, pos, verbose=False)
#         oechem.OEPerceiveResidues(molecule)
#         oechem.OEFindRingAtomsAndBonds(molecule)
#
#         return molecule
#
#     def getBackboneAtoms(self, molecule):
#         """Takes an OpenEye Molecule, finds the backbone atoms and
#         returns the indicies of the backbone atoms.
#
#         Parameters
#         ----------
#         molecule : oechem.OEMolecule
#             The OEmolecule of the simulated system.
#
#         Returns
#         -------
#         backbone_atoms : list of int
#             List containing the atom indices of the backbone atoms.
#
#         """
#
#         backbone_atoms = []
#         pred = oechem.OEIsBackboneAtom()
#         for atom in molecule.GetAtoms(pred):
#             bb_atom_idx = atom.GetIdx()
#             backbone_atoms.append(bb_atom_idx)
#
#         return backbone_atoms
#
#     def getTargetAtoms(self, molecule, backbone_atoms, residue_list):
#         """Takes an OpenEye molecule and a list of residue numbers then
#         generates a dictionary containing all the atom pointers and indicies for the
#         non-backbone, atoms of those target residues, as well as a list of backbone atoms.
#         Note: The atom indicies start at 0 and are thus -1 from the PDB file indicies
#
#         Parameters
#         ----------
#         molecule : oechem.OEMolecule
#             The OEmolecule of the simulated system.
#         backbone_atoms : list of int
#             List containing the atom indices of the backbone atoms.
#         residue_list : list of int
#             List containing the residue numbers of the sidechains to be rotated.
#
#         Returns
#         -------
#         backbone_atoms : list of int
#             List containing the atom indices of the backbone atoms to be rotated.
#         qry_atoms : dict of oechem.OEAtomBase
#             Dictionary containing all the atom pointers (as OpenEye objects) that
#             make up the given residues.
#
#         """
#
#         # create and clear dictionary to store atoms that make up residue list
#         qry_atoms = {}
#         qry_atoms.clear()
#
#         reslib = []
#
#         #print('Searching residue list for atoms...')
#         # loop through all the atoms in the PDB OEGraphMol structure
#         for atom in molecule.GetAtoms():
#             # check if the atom is in backbone
#             if atom.GetIdx() not in backbone_atoms:
#                 # if heavy, find what residue it is associated with
#                 myres = oechem.OEAtomGetResidue(atom)
#                 # check if the residue number is amongst the list of residues
#                 if myres.GetResidueNumber() in residue_list and myres.GetName() != "HOH":
#                     # store the atom location in a query atom dict keyed by its atom index
#                     qry_atoms.update({atom: atom.GetIdx()})
#                     #print('Found atom %s in residue number %i %s'%(atom,myres.GetResidueNumber(),myres.GetName()))
#                     if myres not in reslib:
#                         reslib.append(myres)
#
#         return qry_atoms, backbone_atoms
#
#     def findHeavyRotBonds(self, pdb_OEMol, qry_atoms):
#         """Takes in an OpenEye molecule as well as a dictionary of atom locations (keys)
#         and atom indicies.  It loops over the query atoms and identifies any heavy bonds associated with each atom.
#         It stores and returns the bond indicies (keys) and the two atom indicies for each bond in a dictionary
#         Note: atom indicies start at 0, so are offset by 1 compared to pdb)
#
#         Parameters
#         ----------
#         pdb_OEMol : oechem.OEMolecule
#             The OEmolecule of the simulated system generated from a PDB file.
#         qry_atoms : dict of oechem.OEAtomBase
#             Dictionary containing all the atom pointers (as OpenEye objects) that
#             make up the given residues.
#
#         Returns
#         -------
#         rot_bonds : dict of oechem.OEBondBase
#             Dictionary containing the bond pointers of the rotatable bonds.
#
#
#         """
#         # create and clear dictionary to store bond and atom indicies that are rotatable + heavy
#         rot_bonds = {}
#         rot_bonds.clear()
#
#         for atom in qry_atoms.keys():
#             myres = oechem.OEAtomGetResidue(atom)
#             for bond in atom.GetBonds():
#                 # retrieve the begnning and ending atoms
#                 begatom = bond.GetBgn()
#                 endatom = bond.GetEnd()
#                 # if begnnning and ending atoms are not Hydrogen, and the bond is rotatable
#                 if endatom.GetAtomicNum() > 1 and begatom.GetAtomicNum() > 1 and bond.IsRotor():
#                     # if the bond has not been added to dictionary already..
#                     # (as would happen if one of the atom pairs was previously looped over)
#                     if bond not in rot_bonds:
#                         #print('Bond number',bond, 'is rotatable, non-terminal, and contains only heavy atoms')
#                         # store bond pointer (key) and atom indicies in dictionary if not already there
#                         #rot_bonds.update({bond : {'AtomIdx_1' : bond.GetBgnIdx(), 'AtomIdx_2': bond.GetEndIdx()}})
#                         rot_bonds.update({bond: myres.GetResidueNumber()})
#
#         return rot_bonds
#
#     def getRotAtoms(self, rotbonds, molecule, backbone_atoms):
#         """Function identifies and stores neighboring, upstream atoms for a given sidechain bond.
#
#         Parameters
#         ----------
#         rot_bonds : dict of oechem.OEBondBase
#             Dictionary containing the bond pointers of the rotatable bonds.
#         molecule : oechem.OEMolecule
#             The OEmolecule of the simulated system.
#         backbone_atoms : list of int
#             List containing the atom indices of the backbone atoms.
#
#
#         Returns
#         -------
#         rot_atom_dict : dict of oechem.OEAtomBase
#             Dictionary containing the atom pointers for a given sidechain bond.
#
#         """
#         backbone = backbone_atoms
#         query_list = []
#         idx_list = []
#         rot_atom_dict = {}
#         rot_atom_dict.clear()
#
#         for bond in rotbonds.keys():
#             idx_list.clear()
#             query_list.clear()
#             resnum = (rotbonds[bond])
#             thisbond = bond
#             ax1 = bond.GetBgn()
#             ax2 = bond.GetEnd()
#
#             if resnum in rot_atom_dict.keys():
#                 rot_atom_dict[resnum].update({thisbond: []})
#             else:
#                 rot_atom_dict.update({resnum: {thisbond: []}})
#
#             idx_list.append(ax1.GetIdx())
#             idx_list.append(ax2.GetIdx())
#
#             if ax1 not in query_list and ax1.GetIdx() not in backbone_atoms:
#                 query_list.append(ax1)
#             if ax2 not in query_list and ax2.GetIdx() not in backbone_atoms:
#                 query_list.append(ax2)
#
#             for atom in query_list:
#                 checklist = atom.GetAtoms()
#                 for candidate in checklist:
#                     if candidate not in query_list and candidate.GetIdx() not in backbone and candidate != ax2:
#                         query_list.append(candidate)
#                         if candidate.GetAtomicNum() > 1:
#                             can_nbors = candidate.GetAtoms()
#                             for can_nbor in can_nbors:
#                                 if can_nbor not in query_list and candidate.GetIdx(
#                                 ) not in backbone and candidate != ax2:
#                                     query_list.append(can_nbor)
#
#             for atm in query_list:
#                 y = atm.GetIdx()
#                 if y not in idx_list:
#                     idx_list.append(y)
#
#             rot_atom_dict[resnum].update({thisbond: list(idx_list)})
#             #print("Moving these atoms:", idx_list)
#
#         return rot_atom_dict
#
#     def getRotBondAtoms(self):
#         """This function is called on class initialization.
#
#         Takes in a PDB filename (as a string) and list of residue numbers.  Returns
#         a nested dictionary of rotatable bonds (containing only heavy atoms), that are keyed by residue number,
#         then keyed by bond pointer, containing values of atom indicies [axis1, axis2, atoms to be rotated]
#         Note: The atom indicies start at 0, and are offset by -1 from the PDB file indicies
#
#         Returns
#         -------
#         rot_atoms : dict
#             Dictionary of residues, bonds and atoms to be rotated
#         rot_bonds : dict of oechem.OEBondBase
#             Dictionary containing the bond pointers of the rotatable bonds.
#         qry_atoms : dict of oechem.OEAtomBase
#             Dictionary containing all the atom pointers (as OpenEye objects) that
#             make up the given residues.
#
#         """
#         backbone_atoms = self.getBackboneAtoms(self.molecule)
#
#         # Generate dictionary containing locations and indicies of heavy residue atoms
#         #print('Dictionary of all query atoms generated from residue list\n')
#         qry_atoms, backbone_atoms = self.getTargetAtoms(self.molecule, backbone_atoms, self.residue_list)
#
#         # Identify bonds containing query atoms and return dictionary of indicies
#         rot_bonds = self.findHeavyRotBonds(self.molecule, qry_atoms)
#
#         # Generate dictionary of residues, bonds and atoms to be rotated
#         rot_atoms = self.getRotAtoms(rot_bonds, self.molecule, backbone_atoms)
#         return rot_atoms, rot_bonds, qry_atoms
#
#     def chooseBondandTheta(self):
#         """This function is called on class initialization.
#
#         Takes a dictionary containing nested dictionary, keyed by res#,
#         then keyed by bond_ptrs, containing a list of atoms to move, randomly selects a bond,
#         and generates a random angle (radians).  It returns the atoms associated with the
#         the selected bond, the pointer for the selected bond and the randomly generated angle
#
#
#         Returns
#         -------
#         theta_ran :
#
#         targetatoms :
#
#         res_choice :
#
#         bond_choice :
#
#         """
#
#         res_choice = random.choice(list(self.rot_atoms.keys()))
#         bond_choice = random.choice(list(self.rot_atoms[res_choice].keys()))
#         targetatoms = self.rot_atoms[res_choice][bond_choice]
#         theta_ran = random.random() * 2 * math.pi
#
#         return theta_ran, targetatoms, res_choice, bond_choice
#
#     def rotation_matrix(self, axis, theta):
#         """Function returns the rotation matrix associated with counterclockwise rotation
#         about the given axis by theta radians.
#
#         Parameters
#         ----------
#         axis :
#
#         theta : float
#             The angle of rotation in radians.
#         """
#         axis = numpy.asarray(axis)
#         axis = axis / math.sqrt(numpy.dot(axis, axis))
#         a = math.cos(theta / 2.0)
#         b, c, d = -axis * math.sin(theta / 2.0)
#         aa, bb, cc, dd = a * a, b * b, c * c, d * d
#         bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#         return numpy.array([[aa + bb - cc - dd, 2 * (bc + ad),
#                              2 * (bd - ac)], [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                             [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
#
#     def move(self, context, verbose=False):
#         """Rotates the target atoms around a selected bond by angle theta and updates
#         the atom coordinates in the parmed structure as well as the ncmc context object
#
#
#         Parameters
#         ----------
#         context: simtk.openmm.Context object
#             Context containing the positions to be moved.
#         verbose : bool, default=False
#             Enable verbosity to print out detailed information of the rotation.
#
#         Returns
#         -------
#         context: simtk.openmm.Context object
#             The same input context, but whose positions were changed by this function.
#
#         """
#
#         # determine the axis, theta, residue, and bond + atoms to be rotated
#         theta, target_atoms, res, bond = self.chooseBondandTheta()
#         print('Rotating bond: %s in resnum: %s by %.2f radians' % (bond, res, theta))
#
#         #retrieve the current positions
#         initial_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
#         nc_positions = copy.deepcopy(initial_positions)
#
#         model = copy.copy(self.structure)
#
#         # set the parmed model to the same coordinates as the context
#         for idx, atom in enumerate(self.all_atoms):
#             if self.verbose:
#                 print('Before:')
#                 print(atom, idx)
#                 print(nc_positions[atom], model.positions[atom])
#
#             model.atoms[atom].xx = nc_positions[atom][0].value_in_unit(unit.angstroms)
#             model.atoms[atom].xy = nc_positions[atom][1].value_in_unit(unit.angstroms)
#             model.atoms[atom].xz = nc_positions[atom][2].value_in_unit(unit.angstroms)
#
#             if self.verbose:
#                 print('After:')
#                 print(nc_positions[atom], model.positions[atom])
#
#         positions = model.positions
#
#         # find the rotation axis using the updated positions
#         axis1 = target_atoms[0]
#         axis2 = target_atoms[1]
#         rot_axis = (positions[axis1] - positions[axis2]) / positions.unit
#
#         #calculate the rotation matrix
#         rot_matrix = self.rotation_matrix(rot_axis, theta)
#
#         # apply the rotation matrix to the target atoms
#         for idx, atom in enumerate(target_atoms):
#
#             my_position = positions[atom]
#
#             if self.verbose:
#                 print('The current position for %i is: %s' % (atom, my_position))
#
#             # find the reduced position (substract out axis)
#             red_position = (my_position - model.positions[axis2])._value
#             # find the new positions by multiplying by rot matrix
#             new_position = numpy.dot(rot_matrix, red_position) * positions.unit + positions[axis2]
#
#             if self.verbose: print("The new position should be:", new_position)
#
#             positions[atom] = new_position
#             # Update the parmed model with the new positions
#             model.atoms[atom].xx = new_position[0] / positions.unit
#             model.atoms[atom].xy = new_position[1] / positions.unit
#             model.atoms[atom].xz = new_position[2] / positions.unit
#
#             #update the copied ncmc context array with the new positions
#             nc_positions[atom][0] = model.atoms[atom].xx * nc_positions.unit / 10
#             nc_positions[atom][1] = model.atoms[atom].xy * nc_positions.unit / 10
#             nc_positions[atom][2] = model.atoms[atom].xz * nc_positions.unit / 10
#
#             if self.verbose:
#                 print('The updated position for this atom is:', model.positions[atom])
#
#         # update the actual ncmc context object with the new positions
#         context.setPositions(nc_positions)
#
#         # update the class structure positions
#         self.structure.positions = model.positions
#
#         if self.write_move:
#             filename = 'sc_move_%s_%s_%s.pdb' % (res, axis1, axis2)
#             mod_prot = model.save(filename, overwrite=True)
#         return context
#
#
# class SmartDartMove(RandomLigandRotationMove):
#     """**WARNING:** This class has not been completely tested. Use at your own risk.
#
#     Move object that allows center of mass smart darting moves to be performed on a ligand,
#     allowing translations of a ligand between pre-defined regions in space. The
#     `SmartDartMove.move()` method translates the ligand to the locations of the ligand
#     found in the coord_files. These locations are defined in terms of the basis_particles.
#     These locations are picked with a uniform probability. Based on Smart Darting Monte Carlo [smart-dart]_
#
#     Parameters
#     ----------
#     structure: parmed.Structure
#         ParmEd Structure object of the relevant system to be moved.
#     basis_particles: list of 3 ints
#         Specifies the 3 indices of the protein whose coordinates will be used
#         to define a new set of basis vectors.
#     coord_files: list of str
#         List containing paths to coordinate files of the whole system for smart darting.
#     topology: str, optional, default=None
#         A path specifying a topology file matching the files in coord_files. Not
#         necessary if the coord_files already contain topologies (ex. PDBs).
#     dart_radius: simtk.unit float object compatible with simtk.unit.nanometers unit,
#         optional, default=0.2*simtk.unit.nanometers
#         The radius of the darting region around each dart.
#     self_dart: boolean, optional, default='False'
#         When performing the center of mass darting in `SmartDartMove.move()`,this
#         specifies whether or not to include the darting region where the center
#         of mass currently resides as an option to dart to.
#     resname : str, optional, default='LIG'
#         String specifying the residue name of the ligand.
#
#     References
#     ----------
#     .. [smart-dart] I. Andricioaei, J. E. Straub, and A. F. Voter, J. Chem. Phys. 114, 6994 (2001).
#         https://doi.org/10.1063/1.1358861
#
#     """
#
#     def __init__(self,
#                  structure,
#                  basis_particles,
#                  coord_files,
#                  topology=None,
#                  dart_radius=0.2 * unit.nanometers,
#                  self_dart=False,
#                  resname='LIG'):
#
#         super(SmartDartMove, self).__init__(structure, resname=resname)
#
#         if len(coord_files) < 2:
#             raise ValueError('You should include at least two files in coord_files ' +
#                              'in order to benefit from smart darting')
#         self.dartboard = []
#         self.n_dartboard = []
#         self.particle_pairs = []
#         self.particle_weights = []
#         self.basis_particles = basis_particles
#         self.dart_radius = dart_radius
#         self.calculateProperties()
#         self.self_dart = self_dart
#         self.dartsFromParmEd(coord_files, topology)
#
#     def dartsFromParmEd(self, coord_files, topology=None):
#         """
#         Used to setup darts from a generic coordinate file, through MDtraj using the basis_particles to define
#         new basis vectors, which allows dart centers to remain consistant through a simulation.
#         This adds to the self.n_dartboard, which defines the centers used for smart darting.
#
#         Parameters
#         ----------
#         coord_files: list of str
#             List containing coordinate files of the whole system for smart darting.
#         topology: str, optional, default=None
#             A path specifying a topology file matching the files in coord_files. Not
#             necessary if the coord_files already contain topologies.
#
#         """
#
#         n_dartboard = []
#         dartboard = []
#         #loop over specified files and generate parmed structures from each
#         #then the center of masses of the ligand in each structureare found
#         #finally those center of masses are added to the `self.dartboard`s to
#         #be used in the actual smart darting move to define darting regions
#         for coord_file in coord_files:
#             if topology == None:
#                 #if coord_file contains topology info, just load coord file
#                 temp_md = parmed.load_file(coord_file)
#             else:
#                 #otherwise load file specified in topology
#                 temp_md = parmed.load_file(topology, xyz=coord_file)
#             #get position values in terms of nanometers
#             context_pos = temp_md.positions.in_units_of(unit.nanometers)
#             lig_pos = numpy.asarray(context_pos._value)[self.atom_indices] * unit.nanometers
#             particle_pos = numpy.asarray(context_pos._value)[self.basis_particles] * unit.nanometers
#             #calculate center of mass of ligand
#             self.calculateProperties()
#             center_of_mass = self.getCenterOfMass(lig_pos, self.masses)
#             #get particle positions
#             new_coord = self._findNewCoord(particle_pos[0], particle_pos[1], particle_pos[2], center_of_mass)
#             #old_coord should be equal to com
#             old_coord = self._findOldCoord(particle_pos[0], particle_pos[1], particle_pos[2], new_coord)
#             numpy.testing.assert_almost_equal(old_coord._value, center_of_mass._value, decimal=1)
#             #add the center of mass in euclidian and new basis set (defined by the basis_particles)
#             n_dartboard.append(new_coord)
#             dartboard.append(old_coord)
#         self.n_dartboard = n_dartboard
#         self.dartboard = dartboard
#
#     def move(self, context):
#         """
#         Function for performing smart darting move with darts that
#         depend on particle positions in the system.
#
#         Parameters
#         ----------
#         context: simtk.openmm.Context object
#             Context containing the positions to be moved.
#
#         Returns
#         -------
#         context: simtk.openmm.Context object
#             The same input context, but whose positions were changed by this function.
#
#         """
#
#         atom_indices = self.atom_indices
#         if len(self.n_dartboard) == 0:
#             raise ValueError('No darts are specified. Make sure you use ' +
#                              'SmartDartMove.dartsFromParmed() before using the move() function')
#
#         #get state info from context
#         stateinfo = context.getState(True, True, False, True, True, False)
#         oldDartPos = stateinfo.getPositions(asNumpy=True)
#         #get the ligand positions
#         lig_pos = numpy.asarray(oldDartPos._value)[self.atom_indices] * unit.nanometers
#         #updates the darting regions based on the current position of the basis particles
#         self._findDart(context)
#         #find the ligand's current center of mass position
#         center = self.getCenterOfMass(lig_pos, self.masses)
#         #calculate the distance of the center of mass to the center of each darting region
#         selected_dart, changevec = self._calc_from_center(com=center)
#         #selected_dart is the selected darting region
#
#         #if the center of mass was within one darting region, move the ligand to another region
#         if selected_dart != None:
#             newDartPos = numpy.copy(oldDartPos)
#             #find the center of mass in the new darting region
#             dart_switch = self._reDart(selected_dart, changevec)
#             #find the vector that will translate the ligand to the new darting region
#             vecMove = dart_switch - center
#             #apply that vector to the ligand to actually translate the coordinates
#             for atom in atom_indices:
#                 newDartPos[atom] = newDartPos[atom] + vecMove._value
#             #set the positions after darting
#             context.setPositions(newDartPos)
#
#             return context
#
#     def _calc_from_center(self, com):
#         """
#         Helper function that finds the distance of the current center of
#         mass to each dart center in self.dartboard
#
#         Parameters
#         --------
#         com: 1x3 numpy.array*simtk.unit.nanometers
#             Current center of mass coordinates of the ligand.
#
#         Returns
#         -------
#         selected_dart: simtk.unit.nanometers, or None
#             The distance of a dart to a center. Returns
#             None if the distance is greater than the darting region.
#         changevec: 1x3 numpy.array*simtk.unit.nanometers,
#             The vector from the ligand center of mass
#             to the center of a darting region.
#
#         """
#
#         distList = []
#         diffList = []
#         indexList = []
#         #Find the distances of the COM to each dart, appending
#         #the results to distList
#         for dart in self.dartboard:
#             diff = com - dart
#             dist = numpy.sqrt(numpy.sum((diff) * (diff))) * unit.nanometers
#             distList.append(dist)
#             diffList.append(diff)
#         selected_dart = []
#         #Find the dart(s) less than self.dart_radius
#         for index, entry in enumerate(distList):
#             if entry <= self.dart_radius:
#                 selected_dart.append(index)
#                 diff = diffList[index]
#                 indexList.append(index)
#         #Dart error checking
#         #to ensure reversibility the COM should only be
#         #within self.dart_radius of one dart
#         if len(selected_dart) == 1:
#             return selected_dart[0], diffList[indexList[0]]
#         elif len(selected_dart) == 0:
#             return None, diff
#         elif len(selected_dart) >= 2:
#             #COM should never be within two different darts
#             raise ValueError(' The spheres defining two darting regions have overlapped, ' +
#                              'which results in potential problems with detailed balance. ' +
#                              'We are terminating the simulation. Please check the size and ' +
#                              'identity of your darting regions defined by dart_radius.')
#             #TODO can treat cases using appropriate probablility correction
#             #see https://doi.org/10.1016/j.patcog.2011.02.006
#
#     def _findDart(self, context):
#         """
#         Helper function to dynamically update dart positions based on the current positions
#         of the basis particles.
#
#         Parameters
#         ---------
#         context: Context object from simtk.openmm
#             Context from the ncmc simulation.
#
#         Returns
#         -------
#         dart_list list of 1x3 numpy.arrays in units.nm
#             new dart positions calculated from the particle_pairs
#             and particle_weights.
#
#         """
#
#         basis_particles = self.basis_particles
#         #make sure there's an equal number of particle pair lists
#         #and particle weight lists
#         dart_list = []
#         state_info = context.getState(True, True, False, True, True, False)
#         temp_pos = state_info.getPositions(asNumpy=True)
#         part1 = temp_pos[basis_particles[0]]
#         part2 = temp_pos[basis_particles[1]]
#         part3 = temp_pos[basis_particles[2]]
#         for dart in self.n_dartboard:
#             old_center = self._findOldCoord(part1, part2, part3, dart)
#             dart_list.append(old_center)
#         self.dartboard = dart_list[:]
#         return dart_list
#
#     def _reDart(self, selected_dart, changevec):
#         """
#         Helper function to choose a random dart and determine the vector
#         that would translate the COM to that dart center + changevec.
#         This is called reDart in the sense that it helps to switch
#         the ligand to another darting region.
#
#         Parameters
#         ---------
#         selected_dart :
#         changevec: 1x3 numpy.array * simtk.unit.nanometers
#             The vector difference of the ligand center of mass
#             to the closest dart center (if within the dart region).
#
#
#         Returns
#         -------
#         dart_switch: 1x3 numpy.array * simtk.unit.nanometers
#
#         """
#         dartindex = list(range(len(self.dartboard)))
#         if self.self_dart == False:
#             dartindex.pop(selected_dart)
#         dartindex = numpy.random.choice(dartindex)
#         dvector = self.dartboard[dartindex]
#         dart_switch = dvector + changevec
#         return dart_switch
#
#     def _changeBasis(self, a, b):
#         """
#         Changes positions of a particle (b) in the regular basis set to
#         another basis set (a). Used to recalculate the center of mass
#         in terms of the local coordinates defined by self.basis_particles.
#         Used to change between the basis sets defined from the basis_particles
#         and the normal euclidian basis set.
#
#         Parameters
#         ----------
#         a: 3x3 numpy.array
#             Defines vectors that will create the new basis.
#         b: 1x3 numpy.array
#             Defines position of particle to be transformed into
#             new basis set.
#
#         Returns
#         -------
#         changed_coord: 1x3 numpy.array
#             Coordinates of b in new basis.
#
#         """
#
#         ainv = numpy.linalg.inv(a.T)
#         changed_coord = numpy.dot(ainv, b.T) * unit.nanometers
#         return changed_coord
#
#     def _undoBasis(self, a, b):
#         """
#         Transforms positions in a transformed basis (b) to the regular
#         basis set. Used to transform the dart positions in the local
#         coordinate basis set to the cartesian basis set.
#
#         Parameters
#         ----------
#         a: 3x3 numpy.array
#             Defines vectors that defined the new basis.
#         b: 1x3 numpy.array
#             Defines position of particle to be transformed into
#             regular basis set.
#
#         Returns
#         -------
#         changed_coord: 1x3 numpy.array
#             Coordinates of b in new basis.
#         """
#
#         a = a.T
#         changed_coord = numpy.dot(a, b.T) * unit.nanometers
#         return changed_coord
#
#     def _normalize(self, vector):
#         """Normalize a given vector
#
#         Parameters
#         ----------
#         vector: 1xn numpy.array
#             Vector to be normalized.
#
#         Returns
#         -------
#         unit_vec: 1xn numpy.array
#             Normalized vector.
#
#         """
#
#         magnitude = numpy.sqrt(numpy.sum(vector * vector))
#         unit_vec = vector / magnitude
#         return unit_vec
#
#     def _localCoord(self, particle1, particle2, particle3):
#         """
#         Defines a new coordinate system using 3 particles
#         returning the new basis set vectors
#
#         Parameters
#         ----------
#         particle1, particle2, particle3: 1x3 numpy.array
#             numpy.array corresponding to a given particle's positions
#
#         Returns
#         -------
#         vec1, vec2, vec3: 1x3 numpy.array
#             Basis vectors of the coordinate system defined
#             by particles1-3.
#
#         """
#
#         part2 = particle2 - particle1
#         part3 = particle3 - particle1
#         vec1 = part2
#         vec2 = part3
#         vec3 = numpy.cross(vec1, vec2) * unit.nanometers
#         return vec1, vec2, vec3
#
#     def _findNewCoord(self, particle1, particle2, particle3, center):
#         """
#         Finds the coordinates of a given center in the standard basis
#             in terms of a new basis defined by particles1-3
#
#         Parameters
#         ----------
#         particle1, particle2, particle3: 1x3 numpy.array
#             numpy.array corresponding to a given particle's positions
#         center: 1x3 numpy.array * simtk.unit compatible with simtk.unit.nanometers
#             Coordinate of the center of mass in the standard basis set.
#
#         Returns
#         -------
#         new_coord : numpy.array
#             Updated coordinates in terms of new basis.
#         """
#
#         #calculate new basis set
#         vec1, vec2, vec3 = self._localCoord(particle1, particle2, particle3)
#         basis_set = numpy.zeros((3, 3)) * unit.nanometers
#         basis_set[0] = vec1
#         basis_set[1] = vec2
#         basis_set[2] = vec3
#         #since the origin is centered at particle1 by convention
#         #subtract to account for this
#         recenter = center - particle1
#         #find coordinate in new coordinate system
#         new_coord = self._changeBasis(basis_set, recenter)
#         return new_coord
#
#     def _findOldCoord(self, particle1, particle2, particle3, center):
#         """
#         Finds the coordinates of a given center (defined by a different basis
#         given by particles1-3) back in the euclidian coordinates
#
#         Parameters
#         ----------
#         particle1, particle2, particle3: 1x3 numpy.array
#             numpy.array corresponding to a given particle's positions
#         center: 1x3 numpy.array * simtk.unit compatible with simtk.unit.nanometers
#             Coordinate of the center of mass in the non-standard basis set.
#
#         Returns
#         -------
#         adjusted_center : numpy.array
#             Corrected coordinates of new center in euclidian coordinates.
#
#         """
#
#         vec1, vec2, vec3 = self._localCoord(particle1, particle2, particle3)
#         basis_set = numpy.zeros((3, 3)) * unit.nanometers
#         basis_set[0] = vec1
#         basis_set[1] = vec2
#         basis_set[2] = vec3
#         #since the origin is centered at particle1 by convention
#         #subtract to account for this
#         old_coord = self._undoBasis(basis_set, center)
#         adjusted_center = old_coord + particle1
#         return adjusted_center


# class CombinationMove(Move):
#     """**WARNING:** This class has not been completely tested. Use at your own risk.
#
#     Move object that allows Move object moves to be performed according to
#     the order in move_list. To ensure detailed balance, the moves have an equal
#     chance to be performed in listed or reverse order.
#
#     Parameters
#     ----------
#     moves : list of blues.move.Move
#
#     """
#
#     def __init__(self, moves):
#         self.moves = moves
#
#     def move(self, context):
#         """Performs the move() functions of the Moves in move_list on
#         a context.
#
#         Parameters
#         ----------
#         context: simtk.openmm.Context object
#             Context containing the positions to be moved.
#
#         Returns
#         -------
#         context: simtk.openmm.Context object
#             The same input context, but whose positions were changed by this function.
#
#         """
#         rand = numpy.random.random()
#         #to maintain detailed balance this executes both
#         #the forward and reverse order moves with equal probability
#         if rand > 0.5:
#             for single_move in self.move_list:
#                 single_move.move(context)
#         else:
#             for single_move in reverse(self.move_list):
#                 single_move.move(context)
