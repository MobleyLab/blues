from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as unit
import numpy as np
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

class SmartDarting(object):
    def __init__(self, temperature, residueList):
        self.dartboard = []
        self.dart_size = 0.2*unit.nanometers
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kT = kB * temperature
        beta = 1.0 / kT
        self.beta = beta
        self.residueList = residueList
#        self.firstres = ligList[0]
#        self.lastres = ligList[-1]
        self.total_mass = 0
        self.mass_list = None
        self.acceptance = 0


    def get_particle_masses(self, system, residueList=None):
        if residueList == None:
            residueList = self.residueList
        mass_list = []
        total_mass = 0*unit.dalton
        for index in residueList:
            mass = system.getParticleMass(index)
            total_mass = total_mass + mass
            print('mass', mass, 'total_mass', total_mass)
            mass_list.append([mass])
        total_mass = np.sum(mass_list)
        mass_list = np.asarray(mass_list)
        mass_list.reshape((-1,1))
        total_mass = np.array(total_mass)
        total_mass = np.sum(mass_list)
        temp_list = np.zeros((len(residueList), 1))
        for index in range(len(residueList)):
            mass_list[index] = (np.sum(mass_list[index])).value_in_unit(unit.daltons)
        mass_list =  mass_list*unit.daltons
        self.total_mass = total_mass
        self.mass_list = mass_list
        return total_mass, mass_list


    def calculate_com(self, pos_state, total_mass=None, mass_list=None, residueList=None):
        """
        This controls the ability to run a ncmc simulation with MD
        Arguments
        ---------
        total_mass: simtk.unit.quantity.Quantity in units daltons, contains the total masses of the particles for COM calculation
        mass_list:  nx1 np.array in units daltons, contains the masses of the particles for COM calculation
        pos_state:  nx3 np. array in units.nanometers, returned from state.getPositions
        firstres:   int, first residue of ligand
        lastres:    int, last residue of ligand
    
        Returns
        -------
        rotation : nx3 np.array in units.nm
            positions of ligand after random rotation
        """
        if residueList == None:
            residueList = self.residueList
        if mass_list == None:
            mass_list = self.mass_list
        if total_mass == None:
            total_mass = self.total_mass
        if mass_list == None:
            mass_list = self.mass_list

        #choose ligand indicies
        copy_orig = copy.deepcopy(pos_state)
        lig_coord = np.zeros((len(residueList), 3))
        for index, resnum in enumerate(residueList):
            lig_coord[index] = copy_orig[resnum]
            print lig_coord
        lig_coord = lig_coord*unit.nanometers
        copy_coord = copy.deepcopy(lig_coord)
        #mass corrected coordinates (to find COM)
#        print('mass_list', mass_list)
#        print('total_mass', total_mass)
#        print('copy_coord', copy_coord)
        mass_corrected = mass_list / total_mass * copy_coord
        sum_coord = mass_corrected.sum(axis=0).value_in_unit(unit.nanometers)
        com_coord = [0.0, 0.0, 0.0]*unit.nanometers
        #units are funky, so do this step to get them to behave right
        for index in range(3):
            com_coord[index] = sum_coord[index]*unit.nanometers
        #remove COM from ligand coordinates to then perform rotation
        return com_coord    



    def add_dart(self, dart):
        self.dartboard.append(dart)

    def calc_from_center(self, com):
        distList = []
        diffList = []
        indexList = []
        for dart in self.dartboard:
#            diff = dart - com
            diff = com - dart

            print 'diff, dart, com', diff, dart, com
            dist = np.sqrt(np.sum((diff)*(diff)))*unit.nanometers
#            print 'dist', dist
            distList.append(dist)
            diffList.append(diff)
        selected = []
        for index, entry in enumerate(distList):
#            print distList
#            print type(distList)
            if entry <= self.dart_size:
                selected.append(entry)
                diff = diffList[index]
                indexList.append(index)
#            if entry._value > 2.5:
#                print('bugged')
#                exit()
        if len(selected) == 1:
            return selected[0], diffList[indexList[0]]
        elif len(selected) == 0:
            return None, diff
        elif len(selected) >= 2:
            print ('sphere size overlap, check darts')
            exit()

    def redart(self, changevec):
        dartindex = np.random.randint(len(self.dartboard))
        dvector = self.dartboard[dartindex]
#        chboard = dvector + diff   #EDIT!!!!!!! should be dvector + diff
        #chboard is the new dart location () moved by the changevector
        chboard = dvector + changevec   
#        chboard = dvector
        print 'chboard', chboard
        return chboard

    def dartmove(self, context, residueList=None):
        if residueList == None:
            residueList = self.residueList

        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('changevec', changevec)
        if selectedboard != None:
        #notes
        #comMove is where the com ends up after accounting from where it was from the original dart center
        #basically where it's final displacement location
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.redart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
#            print('vecMove*2', vecMove/2)
#            print('tile', np.reshape(np.tile(vecMove, (lastres-firstres)),(-1,3) ) )
#            tiled_vector = np.reshape(np.tile(vecMove, (len(residueList))), (-1,3))
#            print('tiledvec', tiled_vector)
#            print('selectedres', newDartPos[firstres:lastres])
#            print('add', newDartPos[firstres:lastres] + tiled_vector)
#            print newDartPos._value 
            #newDartPos = newDartPos._value

            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
#            newDartPos[firstres:lastres] = (newDartPos[firstres:lastres] + tiled_vector)
#            print('newDartpos', newDartPos)
            print 'worked'

    #        print('dartmove', dartmove)
    #        print('changevec', changevec)
            #print dartmove
            print newDartPos
            #newDartPos[firstres:lastres] = dartmove
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

    def justdartmove(self, context, residueList=None):
        if residueList == None:
            residueList = self.residueList

        stateinfo = context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('changevec', changevec)
        if selectedboard != None:
        #notes
        #comMove is where the com ends up after accounting from where it was from the original dart center
        #basically where it's final displacement location
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.redart(changevec)
            print('comMove', comMove)
            print('center', center)
            vecMove = comMove - center
            print('vecMove', vecMove)
#            print('vecMove*2', vecMove/2)
#            print('tile', np.reshape(np.tile(vecMove, (lastres-firstres)),(-1,3) ) )
#            tiled_vector = np.reshape(np.tile(vecMove, (len(residueList))), (-1,3))
#            print('tiledvec', tiled_vector)
#            print('selectedres', newDartPos[firstres:lastres])
#            print('add', newDartPos[firstres:lastres] + tiled_vector)
#            print newDartPos._value 
            #newDartPos = newDartPos._value

            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
#            newDartPos[firstres:lastres] = (newDartPos[firstres:lastres] + tiled_vector)
#            print('newDartpos', newDartPos)
            print 'worked'

    #        print('dartmove', dartmove)
    #        print('changevec', changevec)
            #print dartmove
            print newDartPos
            #newDartPos[firstres:lastres] = dartmove
            context.setPositions(newDartPos)
            newDartInfo = context.getState(True, True, False, True, True, False)
            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)

