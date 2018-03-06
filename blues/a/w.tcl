mol load pdb posA.pdb
animate read dcd ncmc_output.dcd waitfor all
mol load pdb posB.pdb
mol load pdb posA.pdb
mol modselect 0 0 "resname LIG"
mol modselect 0 1 "resname LIG"
mol modselect 0 2 "resname LIG"
mol top 0
