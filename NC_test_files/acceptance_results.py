import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy_indexed as npi
import numpy as np
import math

def get_float(x):
    return float(''.join(ele for ele in x if ele.isdigit() or ele == '.'))

def get_int(x):
    return re.findall('\d+',x)

accept_y = []
MDstep_x = []
NaN_ct_1 = []
NaN_ct_2 = []
NaN_ct_3 = []
ct = 1


with open("oldblues_NC_500.log") as g:
    for line in g:
        if "Acceptance Ratio" in line:
            accept_y.append(get_float(line))
            ct = ct + 1
        if "Running blues with" in line:
            y = get_int(line)
            idx = list(y)
            #if y[0] != '10000':
            MDstep_x.append(int(y[0]))
        if "Particle position is NaN" in line:
            if ct == 1:
                NaN_ct_1.append(idx)
            if ct == 2:
                NaN_ct_2.append(idx)
            if ct == 3:
                NaN_ct_3.append(idx)

print("Acceptance Ratios: ", accept_y)
print("Number of MD steps: ", MDstep_x)
print("NaNs in Repeat 1: ", len(NaN_ct_1))
print("NaNs in Repeat 2: ", len(NaN_ct_2))
print("NaNs in Repeat 3: ", len(NaN_ct_3))
NaN_ct = NaN_ct_1 + NaN_ct_2 + NaN_ct_3
x_unique, y_mean = npi.group_by(MDstep_x).mean(accept_y)
x_uni, y_std = npi.group_by(MDstep_x).std(accept_y)
print(x_unique)
print("The mean is: ", y_mean)
print("The standard deviation is: ", y_std)

y_err = y_std/math.sqrt(3)
print("The standard error is:", y_err)

for idx, value in enumerate(x_unique):
    y_NaN = NaN_ct.count([str(value)])
    print("For %i NC steps, there were %i NaN results among the 3 repeats of 5000 proposed moves" % (value, y_NaN))
    #print("%.2f percent of moves were NaN" % (y_NaN/5000*100))
    #print("For %i NC steps, there were %i NaN results" % (value, NaN_ct.count([str(value)])))


