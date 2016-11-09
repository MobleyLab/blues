from ncmc import SimNCMC
from smartdart import SmartDarting
from simtk.unit import *

class tclass(SmartDarting, SimNCMC):
#    def __init__(self, residueList, temperature):
    def __init__(self, **kwds):

        super().__init__(**kwds)
    pass
#a = tclass(residueList=[100])
a = tclass(residueList=[100], temperature=300*kelvin)
help(a)
print(a.residueList)

    
