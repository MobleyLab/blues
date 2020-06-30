#!/bin/bash

#SBATCH -J "2wall-pt3"
#SBATCH -p titanx
#SBATCH -o 2wall-prod-1-500k.tex
#SBATCH -e 2wall-prod-1-500k.tex

# Specifying resources needed for run:
##SBATCH --array=0-54
##SBATCH -t 0-54
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1400mb
#SBATCH --time=700:00:00
##SBATCH --distribution=block:cyclic
#SBATCH --gres=gpu:titan:1

# Informational output
echo "=================================== SLURM JOB ==================================="
echo
echo "The job will be started on the following node(s):"
echo $SLURM_JOB_NODELIST
echo
echo "Slurm User:         $SLURM_JOB_USER"
echo "Run Directory:      $(pwd)"
echo "Job ID:             $SLURM_JOB_ID"
echo "Job Name:           $SLURM_JOB_NAME"
echo "Partition:          $SLURM_JOB_PARTITION"
echo "Number of nodes:    $SLURM_JOB_NUM_NODES"
echo "Number of tasks:    $SLURM_NTASKS"
echo "Submitted From:     $SLURM_SUBMIT_HOST"
echo "Submit directory:   $SLURM_SUBMIT_DIR"
echo "=================================== SLURM JOB ==================================="
echo
cd $SLURM_SUBMIT_DIR
echo 'Working Directory:'
pwd

#Activate conda environment.
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate water
conda list
#cd ${PBS_O_WORKDIR}
#echo "Job directory: ${PBS_O_WORKDIR}"
#module load cuda/9.1.85
#nvcc -V
python example_yaml.py
conda deactivate wat
