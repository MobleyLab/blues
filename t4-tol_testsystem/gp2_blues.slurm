#!/bin/bash

#SBATCH --job-name="t4tol-hmr"
#SBATCH --partition=titanx
#SBATCH --nodes=1
#SBATCH --mem=7Gb
#SBATCH --time=144:00:00
#SBATCH --distribution=block:cyclic
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --array=0-7
#SBATCH --output="slurm-%A_%a.out"
#SBATCH --mail-user=limn1+queue@uci.edu
#SBATCH --mail-type=END

#Set environment variables
copy_local="yes"
export JOBID="${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

#Activate conda environment
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate blues-fork
conda list
#########################################

slurm_startjob(){
#----------------- Actual calculation command goes here: ---------------------------#
echo "Submitting ${SLURM_JOB_NAME}"
echo "Job directory: ${SLURM_SUBMIT_DIR}"
pwd
ls -lht

#Set CUDA Path/version
export CUDA_HOME="/usr/local/cuda-9.2"
export PATH="${CUDA_HOME}/bin:$PATH"
export OPENMM_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
nvcc -V

echo 'PWD:' $( pwd )
ls -lht

python -u $HOME/beegfs-data/blues-fork/t4-tol_testsystem/v2-example.py -j $JOBID -n 2500 -s 10000 -r 2500

echo Job Done
#-----------------------------------------------------------------------------------
}

slurm_info_out(){
# Informational output
echo "=================================== SLURM JOB ==================================="
date
stats=$(sstat --format=NTasks,AveCPU,AveDiskRead,AveDiskWrite,AveRSS,AveVMSize,JobID -j $SLURM_JOB_ID)
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
echo "Submitted From:     $SLURM_SUBMIT_HOST:$SLURM_SUBMIT_DIR"
echo "Stats:              $stats"
echo "=================================== SLURM JOB ==================================="
echo
echo "--- SLURM job-script output ---"
}


# Copy data to a local work directory:
if [ "$copy_local" = "yes" ]; then
  echo $HOSTNAME > $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID
  if [ "$?" -ne "0" ]; then
    echo "Unable to write $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID"
    echo "$SLURM_JOB_ID on node $HOSTNAME failed to write $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID " >> $HOME/SURM_WARNINGS
    echo "$SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID should contain:" >> $HOME/SLURM_WARNINGS
    echo "$HOSTNAME" >> $HOME/SLURM_WARNINGS
  fi
  if (( $SLURM_JOB_NUM_NODES > 1 )); then
    work_dir="/DFS-B/SCRATCH/$(id -gn)/$USER/$SLURM_JOB_ID"
  else
    work_dir="/work/$USER/$JOBID"
    #work_dir="/work/$USER/$SLURM_JOB_ID"
  fi

  mkdir -p $work_dir

  if (( $? != 0)); then
    echo "FAIL: rsync to local execution directory had problems. Aborting job."
    exit 1
  else
    echo "$work_dir" > $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID
    if [ "$?" -ne "0" ]; then
      echo "Unable to write $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID"
      echo "$SLURM_JOB_ID on node $HOSTNAME failed to write $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID" >> $HOME/SLURM_WARNINGS
      echo "$SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID should contain:" >> $HOME/SLURM_WARNINGS
      echo "$work_dir" >> $HOME/SLURM_WARNINGS
    fi
  fi
  cd $work_dir
fi

slurm_info_out

slurm_startjob

# Copy data back to the submission directory:
if [ "$copy_local" = "yes" ]; then
  rsync -avhP $work_dir $SLURM_SUBMIT_DIR/
  if (( $? == 0)); then
    cd $SLURM_SUBMIT_DIR
    rm -rf $work_dir
    # Since the copyback worked, delete the file that triggers the post-execution script
    rm $SLURM_SUBMIT_DIR/SLURM_WORK_DIR-$SLURM_JOB_ID
    rm $SLURM_SUBMIT_DIR/SLURM_WORK_NODE-$SLURM_JOB_ID
  else
    echo "FAIL: rsync back to submission directory had problems. Execution directory not removed."
    echo "$SLURM_JOB_ID on node $HOSTNAME had problems on final rsync" >> $HOME/SLURM_WARNINGS
    cd $SLURM_SUBMIT_DIR
    exit 1
  fi
fi
