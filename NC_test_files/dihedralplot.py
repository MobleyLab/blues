import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_float(x):
    return float(''.join(ele for ele in x if ele.isdigit() or ele == '.'))

n_bins = 180
dihedrals1 = []
dihedrals2 = []
dihedrals3 = []
filename1 = "dihedrals500NC_gp1_MD1000step.txt"
filename2 = "dihedrals500NC_gp2_MD1000step.txt"
filename3 = "dihedrals500NC_gp3_MD1000step.txt"

with open(filename1) as f:
    for line in f:
        x1 = float(line)
        deg_x1 = x1/math.pi*180
        print(deg_x1)
        dihedrals1.append(deg_x1)

with open(filename2) as g:
    for line in g:
        x2 = float(line)
        #make_pos = x + math.pi
        deg_x2 = x2/math.pi*180
        print(deg_x2)
        dihedrals2.append(deg_x2)

with open(filename3) as h:
    for line in h:
        x3 = float(line)
        #make_pos = x + math.pi
        deg_x3 = x3/math.pi*180
        print(deg_x3)
        dihedrals3.append(deg_x3)

fig, ax = plt.subplots( nrows=1, ncols=1)
ax.hist(dihedrals1, n_bins, normed=1, histtype='step', color = 'red', label = 'Acceptance Ratio = .07')
ax.hist(dihedrals2, n_bins, normed = 1, histtype = 'step', color = 'orange', label = 'Acceptance Ratio = .01')
ax.hist(dihedrals3, n_bins, normed = 1, histtype = 'step', color = 'blue', label = 'Acceptance Ratio = .01')
plt.title('Dihedral Angles - 1000MD Steps, 100 iterations, 750 Relaxation steps')
plt.legend()
plt.xlim(-180, 180)
#plt.show()
fig.savefig('Dihedral_1000MD_750NC_step.png')
