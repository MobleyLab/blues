file_name = 'stats_1000.txt'

with open(file_name, 'r') as f:
	print('going')
	lines = f.readlines()
	for line in lines:
		print(line)
		aline = line.split()
		print(aline)
		if aline[0] == 'Final':
			print(aline)
