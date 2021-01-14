#!/bin/bash
#
#----------------------------------------------------------------
# running a multiple independent jobs
#----------------------------------------------------------------
#


#  Defining options for slurm how to run
#----------------------------------------------------------------
#
#SBATCH --job-name=vaccmut_rella
#SBATCH --output=array.log
#
#Number of CPU cores to use within one node
#SBATCH -c 1
#
#Define the number of hours the job should run. 
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=0:01:00
#
#Define the amount of RAM used by your job in GigaBytes
#In shared memory applications this is shared among multiple CPUs
#SBATCH --mem=0.5G
#
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# load the respective software module(s) you intend to use
#----------------------------------------------------------------
# none needed for this example

# define sequence of jobs to run as you would do in a BASH script
# use variable $SLURM_ARRAY_TASK_ID to address individual behaviour
# in different iteration of the script execution
#----------------------------------------------------------------

module load python

srun python main_cluster.py ${SLURM_ARRAY_TASK_ID}
