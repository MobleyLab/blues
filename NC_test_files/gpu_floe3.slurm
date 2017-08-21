#!/bin/bash

# Job name:
#----------------
#SBATCH --job-name="blues-10000NC"
#----------------

# Partition name:
#----------------
#SBATCH --partition=mf_titanx
#----------------

######################

# Specifying resources needed for run:
#
#--------------
#SBATCH --nodes=1
#SBATCH --mem=4gb
#SBATCH --time=36:00:00
#SBATCH --distribution=block:cyclic
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --threads-per-core=2
#SBATCH --output="slurm-%j.out"
#--------------

#Set environment variables
copy_local="yes"
export PATH=$HOME/anaconda3/bin:$PATH
export OE_LICENSE="/home/burleyk/anaconda3/lib/python3.6/site-packages/oe_license.txt"
source activate bluesold


#Equil Inputs
prmtop="vacDivaline.prmtop"
inpcrd="vacDivaline.inpcrd"
myscript='sc_test_3.py'

slurm_startjob(){
#----------------- Actual calculation command goes here: ---------------------------#

#Export Anaconda3 into PATH

echo "Submitting ${SLURM_JOB_NAME}"
echo "Job directory: ${SLURM_SUBMIT_DIR}"
pwd

ls -lht

#Equil the System
python -u sc_test_3.py > oldblues_NC_10000_3.log

echo Job Done
#-----------------------------------------------------------------------------------
}

slurm_info_out(){
# Informational output
echo "=================================== SLURM JOB ==================================="
date
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
    work_dir="/beegfs/SCRATCH/$(id -gn)/$USER/$SLURM_JOB_ID"
  else
    work_dir="/work/$USER/$SLURM_JOB_ID"
  fi

  mkdir -p $work_dir
  #Prep/Equil RSYNC command:
  rsync -avhP --include="$myscript" --include="$prmtop" --include="$inpcrd" --exclude="*" $SLURM_SUBMIT_DIR/ $work_dir/

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
  rsync -avhP --ignore-existing $work_dir/ $SLURM_SUBMIT_DIR/
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
