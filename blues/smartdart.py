from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import simtk.unit as unit
import numpy as np
from blues.ncmc import SimNCMC
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

class SmartDarting(SimNCMC):
    """
    Class for performing smart darting moves during an NCMC simulation.
    """
    def __init__(self, **kwds):
        super(SmartDarting, self).__init__(**kwds)
        self.dartboard = []
        self.dart_size = 0.2*unit.nanometers


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



    def add_dart(self, dart):
        self.dartboard.append(dart)

    def calc_from_center(self, com):
        distList = []
        diffList = []
        indexList = []
        for dart in self.dartboard:
            diff = com - dart

#            print('diff, dart, com', diff, dart, com)
            dist = np.sqrt(np.sum((diff)*(diff)))*unit.nanometers
            distList.append(dist)
            diffList.append(diff)
        selected = []
        for index, entry in enumerate(distList):
            if entry <= self.dart_size:
                selected.append(entry)
                diff = diffList[index]
                indexList.append(index)

        if len(selected) == 1:
            return selected[0], diffList[indexList[0]]
        elif len(selected) == 0:
            return None, diff
        elif len(selected) >= 2:
            print('sphere size overlap, check darts')
            exit()

    def redart(self, changevec):
        dartindex = np.random.randint(len(self.dartboard))
        dvector = self.dartboard[dartindex]
        chboard = dvector + changevec   
        print('chboard', chboard)
        return chboard

    def dartmove(self, context=None, residueList=None):
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
        #comMove is where the com ends up after accounting from where it was from the original dart center
        #basically where it's final displacement location
            newDartPos = copy.deepcopy(oldDartPos)
            comMove = self.redart(changevec)
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
        if residueList == None:
            residueList = self.residueList
        if context == None:
            self.nc_context


        stateinfo = self.nc_context.getState(True, True, False, True, True, False)
        oldDartPos = stateinfo.getPositions(asNumpy=True)
        oldDartPE = stateinfo.getPotentialEnergy()
        center = self.calculate_com(oldDartPos)
        selectedboard, changevec = self.calc_from_center(com=center)
        print('selectedboard', selectedboard)
        print('changevec', changevec)
        print('centermass', center)
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
            for residue in residueList:
                newDartPos[residue] = newDartPos[residue] + vecMove
            print('worked')
            print(newDartPos)
            self.nc_context.setPositions(newDartPos)
            newDartInfo = self.nc_context.getState(True, True, False, True, True, False)
            newDartPE = newDartInfo.getPotentialEnergy()

            return newDartInfo.getPositions(asNumpy=True)

