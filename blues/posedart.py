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
    def __init__(self, pdb_files, fit_atoms, **kwds):
        super(SmartDarting, self).__init__(**kwds)
        self.dartboard = []
        self.ligand_atoms = []
        self.dart_size = 0.2*unit.nanometers
        self.binding_mode_traj = []
        self.fit_atoms = fit_atoms
        for pdb_file in pdb_files:
            traj = md.load(pdb_file)[0]
            self.binding_mode_traj.append(copy.deepcopy(traj))
        self.sim_traj = copy.deepcopy(self.binding_mode_traj[0])






    def setDartUpdates(self, ligand_atoms):
        self.ligand_atoms = ligand_atoms

    def defineLigandAtomsFromFile(lig_resname, coord_file, top_file=None):
        self.ligand_atoms = get_lig_residues(lig_resname, 
                                    coord_file, 
                                    top_file)


    def add_dart(self, dart):
        self.dartboard.append(dart)

    def dist_from_dart_center(self, sim_atom_pos, binding_mode_atom_pos):


        distList = []
        diffList = []
        indexList = []
        #Find the distances of the center to each dart, appending 
        #the results to distList
        for index, dart in enumerate(binding_mode_atom_pos):
            diff = sim_atom_pos[index] - dart
            dist = np.sqrt(np.sum((diff)*(diff)))
#            dist = np.sqrt(np.sum((diff)*(diff)))*unit.nanometers

            distList.append(dist)
            diffList.append(diff)
        selected = []
        #Find the dart(s) less than self.dart_size
        #for index, entry in enumerate(distList):
        #    if entry <= self.dart_size:
        #        selected.append(entry)
        #        diff = diffList[index]
        #        indexList.append(index)
        return dist_list, diff_list
        #Dart error checking
        #to ensure reversibility the center should only be 
        #within self.dart_size of one dart
        #if len(selected) == 1:
        #    return selected[0], diffList[indexList[0]]
        #elif len(selected) == 0:
        #    return None, diff
        #elif len(selected) >= 2:
        #    print(selected)
            #COM should never be within two different darts
        #    raise ValueError('sphere size overlap, check darts')


    def poseDart(self, context=None, ligand_atoms=None):
        if context == None:
        context = self.nc_context
        if ligand_atoms == None:
            ligand_atoms = self.ligand_atoms
        total_diff_list = []
        total_dist_list = []
        nc_pos = context.getState(getPositions=True).getPositions()
        #update sim_traj positions for superposing binding modes
        self.sim_traj.xyz = nc_pos._value
        #make a temp_pos to specify dart centers to compare
        #distances between each dart and binding mode reference
        temp_pos = []

        for atom in ligand_atoms:
            temp_pos.append(nc_pos[atom])
        for pose in binding_mode_traj:
            pose = pose.superpose(reference=self.sim_traj,
                            atom_indices=fit_atoms)
            pose_coord = pose.xyz[0]
            binding_mode_pos = []
            for atom in lig_atoms:
                binding_mode_pos.append(pose_coord)
                temp_dist, temp_dif = dist_from_dart_center(temp_pos, binding_mode_pos)
            total_diff_list.append(temp_diff)
            total_dist_list.append(temp_dist)
        selected = []
        for index, single_pose in enumerate(total_dist_list):
            for dist in single_pose:
                counter = 0
                if dist <= self.dart_size._value:
                    counter += 1
                if counter = len(ligand_atoms):
                    selected.append(index)
        if len(selected) == 1:
            return selected[0], total_diff_list[selected[0]]
        elif len(selected) == 0:
            return None, diff
        elif len(selected) >= 2:
            print(selected)
            #COM should never be within two different darts
            raise ValueError('sphere size overlap, check darts')








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
        dvector = self.dartboard[dartindex]
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



