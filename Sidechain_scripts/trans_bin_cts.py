import math

def dihed_range(angle):
    deg = float(angle)*180/math.pi
    if deg<0. and deg>-115.:
        bin = 0
    elif deg> 0. and deg<115.:
        bin = 1
    elif deg>115. or deg<-115.:
        bin = 2
    return bin;

def dihed_list(filename):
    dihed = []
    with open(filename) as file:
        data = file.readlines()
    for val in data:
        dihed.append(float(val))

    return dihed;

def get_bins(dihed_list):
    bins = []
    for val in dihed_list:
        bins.append(dihed_range(val))
    return bins;

def get_num_trans(bins_list):
    trans = 0
    prev = bins_list[0]
    for val in bins_list:
        if val != prev:
            trans +=1
        prev = val

    return(trans);

def get_bin_counts(bins_list):
    m60 = 0.
    p60 = 0.
    mp180 = 0.
    for val in bins_list:
        if val == 0:
            m60 +=1
        elif val ==1:
            p60 +=1
        elif val ==2:
            mp180 +=1

    return(m60, p60, mp180);

dihedrals = dihed_list("nc1000_20k_nobias/dihedrals-1000NC-dival.txt")
bins = get_bins(dihedrals)
trans_ct = get_num_trans(bins)
m60, p60, mp180 = get_bin_counts(bins)
total = m60 + p60 + mp180
p_m60 = float(m60/total)
p_p60 = p60/total
p_mp180 = mp180/total

# Track number of instances of each rotamer state vs time/length of trajectory
m60_timescale = []
p60_timescale = []
mp180_timescale = []
for i in range(0,100000,5000):
    m60, p60, mp180 = get_bin_counts(bins[:i])
    m60_timescale.append(m60)
    p60_timescale.append(p60)
    mp180_timescale.append(mp180)


print("\nThere are %i transitions\n" %(trans_ct))
print("There are %i instances of -60 with %.2f probability\n" %(m60, p_m60))
print("There are %i instances of +60 with %.2f probability\n" %(p60, p_p60))
print("There are %i instances of +180/-180 with %.2f probability\n" %(mp180, p_mp180))

# Prints bin numbers vs trajectory length
print("These are the timescales:\n")
print("m60:\n")
for val in m60_timescale:
    print(val)
print("p60:\n")
for val in p60_timescale:
    print(val)
print("mp180:\n")
for val in mp180_timescale:
    print(val)
