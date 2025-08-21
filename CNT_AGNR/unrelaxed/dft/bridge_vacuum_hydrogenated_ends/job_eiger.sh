#!/bin/bash -l
#SBATCH --job-name=gpaw-example
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=mr34
#SBATCH --uenv=gpaw/25.1.0:1632782471
#SBATCH --view=gpaw

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPICH_GPU_SUPPORT_ENABLED=0
export GPAW_SETUP_PATH=${HOME}/gpaw-setups-24.11.0

srun gpaw python scatt.py
# srun gpaw python dump.py
