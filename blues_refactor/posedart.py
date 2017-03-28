"""
posedart.py: Provides the class for performing smart darting moves
during an NCMC simulation.

Authors: Samuel C. Gill
Contributors: David L. Mobley
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as unit
import numpy as np
from blues.ncmc import SimNCMC, get_lig_residues
import mdtraj as md
def zero_masses( system, firstres, lastres):
    for index in range(firstres, lastres):
        system.setParticleMass(index, 0*daltons)

def beta(temperature):
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    beta = 1.0 / kT
    return beta



def forcegroupify(system):
    forcegroups = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
        forcegroups[force] = i
    return forcegroups

def getEnergyDecomposition(context, forcegroups):
    energies = {}
    for f, i in forcegroups.items():
        energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
    return energies

class PoseDart(SimNCMC):
    """
    Class for performing smart darting moves during an NCMC simulation.
    """
    def __init__(self, pdb_files, fit_atoms, dart_size, **kwds):
        super(PoseDart, self).__init__(**kwds)
        self.dartboard = []
        self.dart_size = []
        print('initizalizing dart', dart_size._value, type(dart_size._value))
        print(self.residueList)
        if type(dart_size._value) == list:
            if len(dart_size) != len(residueList):
                raise ValueError('mismatch between length of dart_size (%i) and residueList (%i)' % (len(dart_size), len(residueList)) )
            self.dart_size = dart_size
        elif type(dart_size._value) == int or type(dart_size._value) == float:
            print('adding the same size darts')
            for entry in self.residueList:
                print('appending dart')
                self.dart_size.append(dart_size.value_in_unit(unit.nanometers))
            self.dart_size = self.dart_size*unit.nanometers


        #self.dart_size = 0.2*unit.nanometers
        self.binding_mode_traj = []
        self.fit_atoms = fit_atoms
        self.ligand_pos = None
        for pdb_file in pdb_files:
            traj = md.load(pdb_file)[0]
            self.binding_mode_traj.append(copy.deepcopy(traj))
        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])






    def setDartUpdates(self, residueList):
        self.residueList = residueList

    def defineLigandAtomsFromFile(lig_resname, coord_file, top_file=None):
        self.residueList = get_lig_residues(lig_resname,
                                    coord_file,
                                    top_file)


    def add_dart(self, dart):
        self.dartboard.append(dart)

    def dist_from_dart_center(self, sim_atom_pos, binding_mode_atom_pos):

        num_lig_atoms = len(self.residueList)

        dist_list = np.zeros((num_lig_atoms, 1))
        diff_list = np.zeros((num_lig_atoms, 3))
        indexList = []
        #Find the distances of the center to each dart, appending
        #the results to dist_list
        #TODO change to handle np.arrays instead

        for index, dart in enumerate(binding_mode_atom_pos):
            diff = sim_atom_pos[index] - dart
            dist = np.sqrt(np.sum((diff)*(diff)))
#            dist = np.sqrt(np.sum((diff)*(diff)))*unit.nanometers
            print('binding_mode_atom_pos', binding_mode_atom_pos)
            print('sim_atom_pos', sim_atom_pos[index])
            print('dart', dart)
            print('diff', diff)
            diff_list[index] = diff
            dist_list[index] = dist
            print('diff_list', diff_list[index])
            print('dist_list', dist_list[index])


        return dist_list, diff_list


    def poseDart(self, context=None, residueList=None):
        """check whether molecule is within a pose, and
        if it is, return the dart vectors for it's atoms
        """
        if context == None:
            context = self.nc_context
        if residueList == None:
            residueList = self.residueList
        total_diff_list = []
        total_dist_list = []
        nc_pos = context.getState(getPositions=True).getPositions()
        #update sim_traj positions for superposing binding modes
        #might need to make self.sim_traj.xyz = nc_pos._value into
        #self.sim_traj.xyz = [nc_pos._value] or np.array
        self.sim_traj.xyz = nc_pos._value
        #make a temp_pos to specify dart centers to compare
        #distances between each dart and binding mode reference
        temp_pos = []
        num_lig_atoms = len(self.residueList)
        temp_pos = np.zeros((num_lig_atoms, 3))


        for index, atom in enumerate(residueList):
            print('temp_pos', temp_pos[index])
            print('nc_pos', nc_pos[atom])
            #keep track of units
            temp_pos[index] = nc_pos[atom]._value

        #fit different binding modes to current protein
        #to remove rotational changes
        for pose in self.binding_mode_traj:
            print('pose', pose.xyz)
            pose = pose.superpose(reference=self.sim_traj,
                            atom_indices=self.fit_atoms,
                            ref_atom_indices=self.fit_atoms)
#            pose.save('temp.pdb')
            pose_coord = pose.xyz[0]
            print('pose_coord', pose.xyz[0])
#            help(pose.superpose)
            binding_mode_pos = []
            #find the dart vectors and distances to each protein
            #append the list to a storage list
            temp_binding_mode_pos = np.zeros((num_lig_atoms, 3))
            temp_binding_mode_diff = np.zeros((num_lig_atoms, 3))
            temp_binding_mode_dist = np.zeros((num_lig_atoms, 1))

            for index, atom in enumerate(residueList):
                temp_binding_mode_pos[index] = pose_coord[atom]
            temp_dist, temp_diff = self.dist_from_dart_center(temp_pos, temp_binding_mode_pos)
            total_diff_list.append(temp_diff)
            total_dist_list.append(temp_dist)

        print('total_diff_list', total_diff_list)
        print('total_dist_list', total_dist_list)
        print('self.dart_size', self.dart_size)
        print('self.dart_size._value', self.dart_size._value)
        selected = []
        #check to see which poses fall within the dart size
        for index, single_pose in enumerate(total_dist_list):
            counter = 0
            for atomnumber,dist in enumerate(single_pose):
                print(self.dart_size)
                print(self.dart_size[0])
                if dist <= self.dart_size[atomnumber]._value:
                    counter += 1
                print('counter for pose', index, 'is ', counter)
            if counter == len(residueList):
                selected.append(index)
        if len(selected) == 1:
            #returns binding mode index, and the diff_list
            #diff_list will be used to dart
            return selected[0], total_diff_list[selected[0]]
        elif len(selected) == 0:
            return None, total_diff_list
        elif len(selected) >= 2:
            print(selected)
            #COM should never be within two different darts
            raise ValueError('sphere size overlap, check darts')

        #use diff list to redart

    def poseRedart(self, changevec, binding_mode_pos, binding_mode_index, nc_pos, residueList=None):
        """
        Helper function to choose a random pose and determine the vector
        that would translate the current particles to that dart center
        Arguments
        ----------
        changevec: list
            The change in vector that you want to apply,
            typically supplied by poseDart
        """
        if residueList == None:
            residueList = self.residueList
            changed_pos = copy.deepcopy(nc_pos)
        rand_index = np.random.randint(len(self.binding_mode_traj))
        ###temp to encourage going to other binding modes
        while rand_index == binding_mode_index:
            rand_index = np.random.randint(len(self.binding_mode_traj))
        ###

        print('total residues', residueList)
        for index, atom in enumerate(residueList):
            #index refers to where in list
            #atom refers to atom#
            print('fitting atom', atom)
            dartindex = binding_mode_index
            print('binding_mode_pos', binding_mode_pos)
            print('binding_mode_pos.xyz', (binding_mode_pos[dartindex].xyz))
            dart_origin = (binding_mode_pos[rand_index].xyz)[0][atom]
            print('dart_origin', dart_origin)
            print('changevec', changevec)
            print('changevec[index]', changevec[index])
            dart_change = dart_origin + changevec[index]
            changed_pos[atom] = dart_change*unit.nanometers
            print('dart_change', dart_change)
            print('dart_before', nc_pos[atom])
            print('dart_after', changed_pos[atom])


        return changed_pos



        #select another binding pose and then for each atom
        #use poseRedart() for each atom position



    def poseMove(self, context=None, residueList=None):
        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context
        stateinfo = context.getState(True, True, False, True, True, False)
        oldEnergy = stateinfo.getPotentialEnergy()
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        selected_pose, diff_list = self.poseDart()
        if selected_pose == None:
            print('no pose found')
        else:
            print('yes pose found')
            new_pos = self.poseRedart(changevec=diff_list,
                binding_mode_pos=self.binding_mode_traj,
                binding_mode_index=selected_pose,
                nc_pos=oldDartPos)
            context.setPositions(new_pos)
            stateinfo = context.getState(True, True, False, True, True, False)
            newEnergy = stateinfo.getPotentialEnergy()
            print('oldEnergy', oldEnergy)
            print('newEnergy', newEnergy)
            old_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_oldEnergy',old_md_state.getPotentialEnergy())
            self.md_simulation.context.setPositions(new_pos)
            new_md_state = self.md_simulation.context.getState(True, True, False, True, True, False)
            print('md_newEnergy',new_md_state.getPotentialEnergy())












    def findDart(self, particle_pairs=None, particle_weights=None):
        """
        For dynamically updating dart positions based on positions
        of other particles.
        This takes the weighted average of the specified particles
        and changes the dartboard of the object

        Arguments
        ---------
        particle_pairs: list of list of ints
            each list defines the pairs to define darts
        particle_weights: list of list of floats
            each list defines the weights assigned to each particle positions
        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights

        """
        if particle_pairs == None:
            particle_pairs = self.particle_pairs
        if particle_weights == None:
            particle_weights = self.particle_weights
        #make sure there's an equal number of particle pair lists
        #and particle weight lists
        assert len(particle_pairs) == len(particle_weights)

        dart_list = []
        state_info = self.nc_context.getState(True, True, False, True, True, False)
        temp_pos = state_info.getPositions(asNumpy=True)
        #find particles positions and multiply by weights
        for i, ppair in enumerate(particle_pairs):
            temp_array = np.array([0, 0, 0]) * unit.nanometers
            #weighted average
            temp_wavg = 0
            for j, particle in enumerate(ppair):
                print('temp_pos', particle, temp_pos[particle])
                temp_array += (temp_pos[particle] * float(particle_weights[i][j]))
                temp_wavg += float(particle_weights[i][j])
                print(temp_array)
            #divide by total number of particles in a list and append
            #calculated postion to dart_list
            dart_list.append(temp_array[:] / temp_wavg)
        self.dartboard = dart_list[:]
        return dart_list

    def virtualDart(self, virtual_particles=None):
        """
        For dynamically updating dart positions based on positions
        of other particles.
        This takes the weighted average of the specified particles
        and changes the dartboard of the object

        Arguments
        ---------
        virtual_particles: list of ints
            Each int in the list specifies a particle
        particle_weights: list of list of floats
            each list defines the weights assigned to each particle positions
        Returns
        -------
        dart_list list of 1x3 np.arrays in units.nm
            new dart positions calculated from the particle_pairs
            and particle_weights

        """
        if virtual_particles == None:
            virtual_particles = self.virtual_particles

        dart_list = []
        state_info = self.nc_context.getState(True, True, False, True, True, False)
        temp_pos = state_info.getPositions(asNumpy=True)
        #find virtual particles positions and add to dartboard
        for particle in virtual_particles:
            print('temp_pos', particle, temp_pos[particle])
            dart_list.append(temp_pos[particle])
        self.dartboard = dart_list[:]
        return dart_list




    def reDart(self, changevec):
        """
        Helper function to choose a random dart and determine the vector
        that would translate the COM to that dart center
        """
        dartindex = np.random.randint(len(self.dartboard))
        dart_origin = self.dartboard[dartindex]
        chboard = dvector + changevec
        print('chboard', chboard)
        return chboard

    def dartmove(self, context=None, residueList=None):
        """
        Obsolete function kept for reference.
        """
        if residueList == None:
            residueList = self.residueList
        if context == None:
            self.nc_context

        stateinfo = self.context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('changevec', changevec)
        if selectedboard != None:
        #notes
        #comMove is where the com ends up after accounting from where
        #it was from the original dart center
        #basically it's final displacement location
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            vecMove = comMove - center
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            context.setPositions(newDartPos)
            newDartInfo = context.getState(True, True, False, True, True, False)
            newDartPE = newDartInfo.getPotentialEnergy()
            logaccept = -1.0*(newDartPE - oldDartPE) * self.beta
            randnum = math.log(np.random.random())
            print('logaccept', logaccept, randnum)
            print('old/newPE', oldDartPE, newDartPE)
            if logaccept >= randnum:
                print('move accepted!')
                self.acceptance = self.acceptance+1
            else:
                print('rejected')
                context.setPositions(oldDartPos)
            dartInfo = context.getState(True, False, False, False, False, False)

            return newDartInfo.getPositions(asNumpy=True)

    def justdartmove(self, context=None, residueList=None):
        """
        Function for performing smart darting move with fixed coordinate darts
        """
        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context


        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('selectedboard', selectedboard)
        print('changevec', changevec)
        print('centermass', center)
        if selectedboard != None:
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            print('worked')
            print(newDartPos)
            context.setPositions(newDartPos)
            newDartInfo = context.getState(True, True, False, True, True, False)
#            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)

    def updateDartMove(self, context=None, residueList=None):
        """
        Function for performing smart darting move with darts that
        depend on particle positions in the system
        """

        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context


        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        self.findDart()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('selectedboard', selectedboard)
        print('changevec', changevec)
        print('centermass', center)
        if selectedboard != None:
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            print('worked')
            print(newDartPos)
            context.setPositions(newDartPos)
            newDartInfo = self.nc_context.getState(True, True, False, True, True, False)
#            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)

    def virtualDartMove(self, context=None, residueList=None):
        """
        Function for performing smart darting move with darts that
        depend on particle positions in the system
        """

        if residueList == None:
            residueList = self.residueList
        if context == None:
            context = self.nc_context


        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        self.virtualDart()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('selectedboard', selectedboard)
        print('changevec', changevec)
        print('centermass', center)
        if selectedboard != None:
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.reDart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            print('worked')
            print(newDartPos)
            context.setPositions(newDartPos)
            newDartInfo = self.nc_context.getState(True, True, False, True, True, False)
#            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)
