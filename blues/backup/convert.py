import mdtraj as md
a = md.load('out.h5')
a.save('out.dcd')
