from yank.restraints import Boresch, RMSD, _AtomSelector
import numpy as np
from simtk import openmm, unit
import parmed
from openmmtools.states import ThermodynamicState
from openmmtools.states import SamplerState
from yank.yank import Topography
import itertools
import mdtraj as md
import random

import logging
logger = logging.getLogger(__name__)

class BoreschBLUES(Boresch):
    """Inherits over the Yank Boresch restraint class
    to add boresch-style restaints to the system used in BLUES
    """
    def __init__(self, restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                 K_r=None, r_aA0=None,
                 K_thetaA=None, theta_A0=None,
                 K_thetaB=None, theta_B0=None,
                 K_phiA=None, phi_A0=None,
                 K_phiB=None, phi_B0=None,
                 K_phiC=None, phi_C0=None,
                 *args, **kwargs):

        super(BoreschBLUES, self).__init__(restrained_receptor_atoms=restrained_receptor_atoms, restrained_ligand_atoms=restrained_ligand_atoms,
                 K_r=K_r, r_aA0=r_aA0,
                 K_thetaA=K_thetaA, theta_A0=theta_A0,
                 K_thetaB=K_thetaB, theta_B0=theta_B0,
                 K_phiA=K_phiA, phi_A0=phi_A0,
                 K_phiB=K_phiB, phi_B0=phi_B0,
                 K_phiC=K_phiC, phi_C0=phi_C0,
                 *args, **kwargs)
        #super(Boresch, self).__init__(*args, **kwargs)


    def _pick_restrained_atoms(self, sampler_state, topography):
        """Select atoms to be used in restraint.
        Parameters
        ----------
        sampler_state : openmmtools.states.SamplerState, optional
            The sampler state holding the positions of all atoms.
        topography : yank.Topography, optional
            The topography with labeled receptor and ligand atoms.
        Returns
        -------
        restrained_atoms : list of int
            List of six atom indices used in the restraint.
            restrained_atoms[0:3] belong to the receptor,
            restrained_atoms[4:6] belong to the ligand.
        Notes
        -----
        The current algorithm simply selects random subsets of receptor
        and ligand atoms and rejects those that are too close to collinear.
        Future updates can further refine this algorithm.
        """
        # If receptor and ligand atoms are explicitly provided, use those.
        heavy_ligand_atoms = self.restrained_ligand_atoms
        heavy_receptor_atoms = self.restrained_receptor_atoms

        # Otherwise we restrain only heavy atoms.
        heavy_atoms = set(topography.topology.select('not element H').tolist())
        # Intersect heavy atoms with receptor/ligand atoms (s1&s2 is intersect).

        atom_selector = _AtomSelector(topography)

        heavy_ligand_atoms = atom_selector.compute_atom_intersect(heavy_ligand_atoms, 'ligand_atoms', heavy_atoms)
        heavy_receptor_atoms = atom_selector.compute_atom_intersect(heavy_receptor_atoms, 'receptor_atoms', heavy_atoms)

        if len(heavy_receptor_atoms) < 3 or len(heavy_ligand_atoms) < 3:
            raise ValueError('There must be at least three heavy atoms in receptor_atoms '
                             '(# heavy {}) and ligand_atoms (# heavy {}).'.format(
                                     len(heavy_receptor_atoms), len(heavy_ligand_atoms)))
        print('heavy_ligand_atoms', heavy_ligand_atoms)
        print('heavy_receptor_atoms', heavy_receptor_atoms)
        # If r3 or l1 atoms are given. We have to pick those.
        if isinstance(heavy_receptor_atoms, list):
            #r3_atoms = [heavy_receptor_atoms[2]]
            r3_atoms = heavy_receptor_atoms

        else:
            r3_atoms = heavy_receptor_atoms
        if isinstance(heavy_ligand_atoms, list):
            l1_atoms = [heavy_ligand_atoms[0]]
            #l1_atoms = heavy_ligand_atoms
        else:
            l1_atoms = heavy_ligand_atoms
        # TODO: Cast itertools generator to np array more efficiently
        r3_l1_pairs = np.array(list(itertools.product(r3_atoms, l1_atoms)))
        print('pairs', r3_l1_pairs)
        # Filter r3-l1 pairs that are too close/far away for the distance constraint.
        max_distance = 8 * unit.angstrom/unit.nanometer
        min_distance = 1 * unit.angstrom/unit.nanometer
        t = md.Trajectory(sampler_state.positions / unit.nanometers, topography.topology)
        print('distance_short', md.geometry.compute_distances(t, r3_l1_pairs))
        distances = md.geometry.compute_distances(t, r3_l1_pairs)[0]
        indices_of_in_range_pairs = np.where(np.logical_and(distances > min_distance, distances <= max_distance))[0]
        print('distances', distances)
        if len(indices_of_in_range_pairs) == 0:
            error_msg = ('There are no heavy ligand atoms within the range of [{},{}] nm heavy receptor atoms!\n'
                         'Please Check your input files or try another restraint class')
            raise ValueError(error_msg.format(min_distance, max_distance))
        r3_l1_pairs = r3_l1_pairs[indices_of_in_range_pairs].tolist()

        def find_bonded_to(input_atom_index, comparison_set):
            """
            Find bonded network between the atoms to create a selection with 1 angle to the reference
            Parameters
            ----------
            input_atom_index : int
                Reference atom index to try and create the selection from the bonds
            comparison_set : iterable of int
                Set of additional atoms to try and make the selection from. There should be at least
                one non-colinear set 3 atoms which are bonded together in R-B-C where R is the input_atom_index
                and B, C are atoms in the comparison_set bonded to each other.
                Can be inclusive of input_atom_index and C can be bound to R as well as B
            Returns
            -------
            bonded_atoms : list of int, length 3
                Returns the list of atoms in order of input_atom_index <- bonded atom <- bonded atom
            """
            # Probably could make this faster if we added a graph module like networkx dep, but not needed
            # Could also be done by iterating over OpenMM System angles
            # Get topology
            top = topography.topology
            bonds = np.zeros([top.n_atoms, top.n_atoms], dtype=bool)
            # Create bond graph
            for a1, a2 in top.bonds:
                a1 = a1.index
                a2 = a2.index
                bonds[a1, a2] = bonds[a2, a1] = True
            all_bond_options = []
            # Cycle through all bonds on the reference
            for a2, first_bond in enumerate(bonds[input_atom_index]):
                # Enumerate all secondary bonds from the reference but only if in comparison set
                if first_bond and a2 in comparison_set:
                    # Same as first
                    for a3, second_bond in enumerate(bonds[a2]):
                        if second_bond and a3 in comparison_set and a3 != input_atom_index:
                            all_bond_options.append([a2, a3])
            # This will raise a ValueError if nothing is found
            return random.sample(all_bond_options, 1)[0]

        # Iterate until we have found a set of non-collinear atoms.
        accepted = False
        max_attempts = 100
        attempts = 0
        while not accepted:
            logger.debug('Attempt {} / {} at automatically selecting atoms and '
                         'restraint parameters...'.format(attempts, max_attempts))
            # Select a receptor/ligand atom in range of each other for the distance constraint.
            r3_l1_atoms = random.sample(r3_l1_pairs, 1)[0]
            # Determine remaining receptor/ligand atoms.
            if isinstance(heavy_receptor_atoms, list):
                r1_r2_atoms = heavy_receptor_atoms[:2]
            else:
                try:
                    r1_r2_atoms = find_bonded_to(r3_l1_atoms[0], heavy_receptor_atoms)[::-1]
                except ValueError:
                    r1_r2_atoms = None
            if isinstance(heavy_ligand_atoms, list):
                l2_l3_atoms = heavy_ligand_atoms[1:]
            else:
                try:
                    l2_l3_atoms = find_bonded_to(r3_l1_atoms[-1], heavy_ligand_atoms)
                except ValueError:
                    l2_l3_atoms = None
            # Reject collinear sets of atoms.
            if r1_r2_atoms is None or l2_l3_atoms is None:
                accepted = False
            else:
                restrained_atoms = r1_r2_atoms + r3_l1_atoms + l2_l3_atoms
                accepted = not self._is_collinear(sampler_state.positions, restrained_atoms)
            if attempts > max_attempts:
                raise RuntimeError("Could not find any good sets of bonded atoms to make stable Boresch-like "
                                   "restraints from. There should be at least 1 real defined angle in the"
                                   "selected restrained ligand atoms and 1 in the selected restrained receptor atoms "
                                   "for good numerical stability")
            else:
                attempts += 1

        logger.debug('Selected atoms to restrain: {}'.format(restrained_atoms))
        return restrained_atoms


    def restrain_state(self, thermodynamic_state, pose_num=0, force_group=0):
        """Add the restraint force to the state's ``System``.
        Overwrites original restrain_state so that it's also controlled by a
        `restraint_pose_X` value, where X is an integer.
        This appears in the energy expression essentially as a boolean,
        so if it = 1 then that restraint is on, or if it = 0 then thaat
        restraint is turned off, allowing for dynamic restraints
        depending on the pose.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.
        pose_num: the integer added to the `restraint_pose_` global parameter
            which controls wheter the restraint is on or off.
        """
        # TODO replace dihedral restraints with negative log von Mises distribution?
        #       https://en.wikipedia.org/wiki/Von_Mises_distribution, the von Mises parameter
        #       kappa would be computed from the desired standard deviation (kappa ~ sigma**(-2))
        #       and the standard state correction would need to be modified.

        # Check if all parameters are defined.
        self._check_parameters_defined()

        energy_function = """
            restraint_pose_%i * lambda_restraints * E;
            E = (K_r/2)*(distance(p3,p4) - r_aA0)^2
            + (K_thetaA/2)*(angle(p2,p3,p4)-theta_A0)^2 + (K_thetaB/2)*(angle(p3,p4,p5)-theta_B0)^2
            + (K_phiA/2)*dphi_A^2 + (K_phiB/2)*dphi_B^2 + (K_phiC/2)*dphi_C^2;
            dphi_A = dA - floor(dA/(2*pi)+0.5)*(2*pi); dA = dihedral(p1,p2,p3,p4) - phi_A0;
            dphi_B = dB - floor(dB/(2*pi)+0.5)*(2*pi); dB = dihedral(p2,p3,p4,p5) - phi_B0;
            dphi_C = dC - floor(dC/(2*pi)+0.5)*(2*pi); dC = dihedral(p3,p4,p5,p6) - phi_C0;
            pi = %f;
            """ % (pose_num, np.pi)

        # Add constant definitions to the energy function
        for name, value in self._parameters.items():
            energy_function += '%s = %f; ' % (name, value.value_in_unit_system(unit.md_unit_system))

        # Create the force
        n_particles = 6  # number of particles involved in restraint: p1 ... p6
        restraint_force = openmm.CustomCompoundBondForce(n_particles, energy_function)
        restraint_force.addGlobalParameter('lambda_restraints', 0.0)
        restraint_force.addGlobalParameter('restraint_pose_'+str(pose_num), 0)
        restraint_force.addBond(self.restrained_receptor_atoms + self.restrained_ligand_atoms, [])
        restraint_force.setUsesPeriodicBoundaryConditions(thermodynamic_state.is_periodic)
        restraint_force.setForceGroup(force_group)
        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system

        system.addForce(restraint_force)
        thermodynamic_state.system = system
        new_sys = thermodynamic_state.get_system(remove_thermostat=True)
        return new_sys


class RMSDBlues(RMSD):
    def __init__(self, restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                K_RMSD=None, RMSD0=None, *args, **kwargs):

        super(RMSDBlues, self).__init__(restrained_receptor_atoms=restrained_receptor_atoms, restrained_ligand_atoms=restrained_ligand_atoms,
            K_RMSD=K_RMSD, RMSD0=RMSD0, *args, **kwargs)

    def restrain_state(self, thermodynamic_state, pose_num=0, force_group=0):
        """Add the restraint force to the state's ``System``.
        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state holding the system to modify.
        """

        # Check if all parameters are defined.
        self._check_parameters_defined()

        # Merge receptor and ligand atoms in a single array for easy manipulation.
        restrained_atoms = list(set(self.restrained_receptor_atoms + self.restrained_ligand_atoms))

        # Create RMSDForce CV for all restrained atoms
        rmsd_cv = openmm.RMSDForce(self.reference_sampler_state.positions, restrained_atoms)

        # Create an CustomCVForce
        energy_expression = 'restraint_pose_%i * lambda_restraints * step(dRMSD) * (K_RMSD/2)*dRMSD^2; dRMSD = (RMSD-RMSD0);' % (pose_num)
        energy_expression += 'K_RMSD = %f;' % self.K_RMSD.value_in_unit_system(unit.md_unit_system)
        energy_expression += 'RMSD0 = %f;' % self.RMSD0.value_in_unit_system(unit.md_unit_system)
        restraint_force = openmm.CustomCVForce(energy_expression)
        restraint_force.addCollectiveVariable('RMSD', rmsd_cv)
        restraint_force.addGlobalParameter('lambda_restraints', 1.0)
        restraint_force.addGlobalParameter('restraint_pose_'+str(pose_num), 0)
        restraint_force.setForceGroup(force_group)

        # Get a copy of the system of the ThermodynamicState, modify it and set it back.
        system = thermodynamic_state.system
        system.addForce(restraint_force)

        thermodynamic_state.system = system
        new_sys = thermodynamic_state.get_system(remove_thermostat=True)
        return new_sys

def add_rmsd_restraints(sys, struct, pos, ligand_atoms, pose_num=0, force_group=0,
                 restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                 K_RMSD=0.6, RMSD0=2.0, **kwargs):
    """Add Boresch-style restraints to a system by giving positions of the reference orientation.
    These restraints can be controlled with the `restraint_pose_"pose_num"` parameter.

    Parameters
    ----------
    system: openmm.System
        OpenMM system to add restraints to
    struct: parmed.Structure
        Contains all the relevant information about the system (topology, positions, etc.)
    pos: simtk.unit compatible with unit.nanometers.
        Positions of the ligand/receptor to be used to define the orientation for restraints.
    ligand_atoms: list
        List of the ligand atom indices.
    pose_num: integer, optional, default=0
        Defines what parameter will control if the restraint is on or off,
        specified by `restraint_pose_"pose_num"`
    restrained_receptor_atoms: list or None, optional, default=None
        List of three atom indices corresponding to the receptor atoms for
        use with the boresch restraints. If none is specified, then
        Yank wil automatically set them randomly.
    restrained_ligand_atoms: list or None, optional, default=None
        List of three atom indices corresponding to the 3 atoms of the ligand
        to be used for the boresch restraints. If None, chooses those atoms
        from the ligand randomly.
    K_r: float
        The value of the bond restraint portion of the boresch restraints
        (given in units of kcal/(mol*angstrom**2)).
    K_angle: flaot
        The value of the angle and dihedral restraint portion of the boresh restraints
        (given in units of kcal/(mol*rad**2)).

    Returns
    -------
    system: openmm.System
        OpenMM system provided as input, with the added boresch restraints.
    """

    #added thermostat force will be removed anyway, so temperature is arbitrary
    topology = struct.topology
    new_struct_pos = np.array(struct.positions.value_in_unit(unit.nanometers))*unit.nanometers
    num_atoms = np.shape(pos)[0]
    new_struct_pos[:num_atoms] = pos
    thermo = ThermodynamicState(sys, temperature=300*unit.kelvin)
    sampler = SamplerState(new_struct_pos, box_vectors=struct.box_vectors)

    topography = Topography(topology=topology, ligand_atoms=ligand_atoms)

    rmsd_restraints = RMSDBlues(restrained_receptor_atoms=restrained_receptor_atoms, restrained_ligand_atoms=restrained_ligand_atoms,
        K_RMSD=K_RMSD*unit.kilocalorie_per_mole/unit.angstrom**2, RMSD0=RMSD0*unit.angstrom)

    rmsd_restraints.determine_missing_parameters(thermo, sampler, topography)
    new_sys = rmsd_restraints.restrain_state(thermo, pose_num=pose_num, force_group=force_group)
    #del thermo
    #del sampler
    #del topography

    return new_sys


def add_boresch_restraints(sys, struct, pos, ligand_atoms, pose_num=0, force_group=0,
                 restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                 K_r=10, K_angle=10, **kwargs):
    """Add Boresch-style restraints to a system by giving positions of the reference orientation.
    These restraints can be controlled with the `restraint_pose_"pose_num"` parameter.

    Parameters
    ----------
    system: openmm.System
        OpenMM system to add restraints to
    struct: parmed.Structure
        Contains all the relevant information about the system (topology, positions, etc.)
    pos: simtk.unit compatible with unit.nanometers.
        Positions of the ligand/receptor to be used to define the orientation for restraints.
    ligand_atoms: list
        List of the ligand atom indices.
    pose_num: integer, optional, default=0
        Defines what parameter will control if the restraint is on or off,
        specified by `restraint_pose_"pose_num"`
    restrained_receptor_atoms: list or None, optional, default=None
        List of three atom indices corresponding to the receptor atoms for
        use with the boresch restraints. If none is specified, then
        Yank wil automatically set them randomly.
    restrained_ligand_atoms: list or None, optional, default=None
        List of three atom indices corresponding to the 3 atoms of the ligand
        to be used for the boresch restraints. If None, chooses those atoms
        from the ligand randomly.
    K_r: float
        The value of the bond restraint portion of the boresch restraints
        (given in units of kcal/(mol*angstrom**2)).
    K_angle: flaot
        The value of the angle and dihedral restraint portion of the boresh restraints
        (given in units of kcal/(mol*rad**2)).

    Returns
    -------
    system: openmm.System
        OpenMM system provided as input, with the added boresch restraints.
    """

    #added thermostat force will be removed anyway, so temperature is arbitrary
    topology = struct.topology
    new_struct_pos = np.array(struct.positions.value_in_unit(unit.nanometers))*unit.nanometers
    num_atoms = np.shape(pos)[0]
    new_struct_pos[:num_atoms] = pos
    thermo = ThermodynamicState(sys, temperature=300*unit.kelvin)
    sampler = SamplerState(new_struct_pos, box_vectors=struct.box_vectors)

    topography = Topography(topology=topology, ligand_atoms=ligand_atoms)

    boresch = BoreschBLUES(restrained_receptor_atoms=restrained_receptor_atoms, restrained_ligand_atoms=restrained_ligand_atoms,
        K_r=K_r*unit.kilocalorie_per_mole/unit.angstrom**2, K_thetaA=K_angle*unit.kilocalories_per_mole / unit.radian**2, K_thetaB=K_angle*unit.kilocalories_per_mole / unit.radian**2,
        K_phiA=K_angle*unit.kilocalories_per_mole / unit.radian**2, K_phiB=K_angle*unit.kilocalories_per_mole / unit.radian**2, K_phiC=K_angle*unit.kilocalories_per_mole / unit.radian**2)

    boresch.determine_missing_parameters(thermo, sampler, topography)
    new_sys = boresch.restrain_state(thermo, pose_num=pose_num, force_group=force_group)

    #del thermo
    #del sampler
    #del topography


    return new_sys


if __name__ == "__main__":
    struct = parmed.load_file('eqToluene.prmtop', xyz='posA.pdb')
    top = struct.topology
    pos = np.array([list(i) for i in struct.positions.value_in_unit(unit.nanometers)])*unit.nanometers

    lig_atoms = list(range(2635,2649))
    struct.positions = pos
    sys = struct.createSystem(nonbondedMethod=openmm.app.PME)

    thermo = ThermodynamicState(sys, temperature=300*unit.kelvin)

    sampler = SamplerState(pos, box_vectors=struct.box_vectors)
    topography = Topography(topology=top, ligand_atoms=list(range(2635,2649)))
    a = BoreschBLUES(thermo, sampler, topography)
    a.determine_missing_parameters(thermo, sampler, topography)
    new_sys = a.restrain_state(thermo)
    other_sys = add_restraints(sys, struct, lig_atoms)
    struct2 = parmed.load_file('eqToluene.prmtop', xyz='posB.pdb')
    other_sys = add_restraints(other_sys, struct2, lig_atoms)
    print('new_sys', new_sys.getNumForces())
    print('new_sys', new_sys.getForces())
    print('other_sys', other_sys.getNumForces())
    print('other_sys', other_sys.getForces())
