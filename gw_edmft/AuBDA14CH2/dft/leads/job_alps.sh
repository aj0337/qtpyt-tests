#!/bin/bash -l
#SBATCH --job-name=gpaw-example
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=lp83
#SBATCH --uenv=gpaw/25.1:1639708786
#SBATCH --view=gpaw

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPICH_GPU_SUPPORT_ENABLED=0
export GPAW_SETUP_PATH=${HOME}/gpaw-setups-24.11.0

srun -n 48 gpaw python leads.py
# mpirun -n 1 gpaw python dump.py
