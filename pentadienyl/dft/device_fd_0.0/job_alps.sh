#!/bin/bash -l
#SBATCH --job-name=gpaw-example
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=s1276
#SBATCH --uenv=gpaw/25.1:1639708786
#SBATCH --view=gpaw
# #SBATCH --output=_scheduler-stdout.txt
# #SBATCH --error=_scheduler-stderr.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

MINICONDA_PATH=/users/ajayaraj/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

# mpirun -n 48 gpaw python scatt.py
mpirun -n 1 gpaw python dump.py
