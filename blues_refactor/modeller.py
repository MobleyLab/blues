"""
ncmc.py: Provides the Simulation class for running the NCMC simulation.

Authors: Samuel C. Gill
Contributors: Nathan M. Lim, David L. Mobley

version: 0.0.2 (WIP-Refactor)
"""

from __future__ import print_function
import sys
from simtk.openmm.app import *
from simtk.openmm import *
from blues.ncmc_switching import *
import simtk.unit as unit
import mdtraj as md
import math
from alchemy import AbsoluteAlchemicalFactory, AlchemicalState
import numpy as np
from openmmtools import testsystems

import blues.utils as utils

class LigandModeller(object):
    def __init__(self, nc_sim, atom_indices):
        self.nc_sim = nc_sim
        self.atom_indices = atom_indices
        self.totalmass = 0
        self.masses = []
        self.positions = None
        self.com = None

    def getMasses(self, context, atom_indices):
        masses = unit.Quantity(np.zeros([len(atom_indices),1],np.float32), unit.dalton)
        system = context.getSystem()
        for ele, idx in enumerate(atom_indices):
            masses[ele] = system.getParticleMass(idx)
        self.totalmass = masses.sum()
        self.masses = masses
        return self.masses

    def getTotalMass(self, masses):
        self.totalmass = self.masses.sum()
        return self.totalmass

    def getPositions(self, context, atom_indices):
        state = context.getState(getPositions=True)
        coordinates = state.getPositions(asNumpy=True) / unit.nanometers
        positions = unit.Quantity( np.zeros([len(atom_indices),3],np.float32), unit.nanometers)
        for ele, idx in enumerate(atom_indices):
            positions[ele,:] = unit.Quantity(coordinates[idx], unit.nanometers)
        self.positions = positions
        return self.positions

    def calculateCOM(self):
        context = self.nc_sim.context
        atom_indices = self.atom_indices
        #Update masses for current context
        masses = self.getMasses(context, atom_indices)
        totalmass = self.getTotalMass(masses)
        positions = self.getPositions(context, atom_indices)
        com =  (masses / totalmass * positions).sum(0)
        self.com = com
        return self.com
