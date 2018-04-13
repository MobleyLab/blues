from yank.restraints import Boresch
import numpy as np
from simtk import openmm, unit
import parmed
from openmmtools.states import ThermodynamicState
from openmmtools.states import SamplerState
from yank.yank import Topography
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
                 standard_state_correction_method='analytical', *args, **kwargs):

        super(BoreschBLUES, self).__init__(restrained_receptor_atoms=restrained_receptor_atoms, restrained_ligand_atoms=restrained_ligand_atoms,
                 K_r=K_r, r_aA0=r_aA0,
                 K_thetaA=K_thetaA, theta_A0=theta_A0,
                 K_thetaB=K_thetaB, theta_B0=theta_B0,
                 K_phiA=K_phiA, phi_A0=phi_A0,
                 K_phiB=K_phiB, phi_B0=phi_B0,
                 K_phiC=K_phiC, phi_C0=phi_C0,
                 standard_state_correction_method=standard_state_correction_method, *args, **kwargs)
        #super(Boresch, self).__init__(*args, **kwargs)


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

def add_restraints(sys, struct, pos, ligand_atoms, pose_num=0, force_group=0,
                 restrained_receptor_atoms=None, restrained_ligand_atoms=None,
                 K_r=10, K_angle=10):
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
