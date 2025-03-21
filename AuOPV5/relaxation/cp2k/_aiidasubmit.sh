#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="aiida-3427"
#SBATCH --partition=debug
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=s1276
#SBATCH --uenv=cp2k/2024.3:v2
#SBATCH --view=cp2k

# set environment
export CP2K_DATA_DIR=/users/ajayaraj/src/cp2k/data
export CUDA_CACHE_PATH="/dev/shm/$RANDOM"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MALLOC_FALLBACK=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

'srun' '--cpu-bind=socket' '/users/ajayaraj/bin/mps-wrapper.sh' '/user-environment/env/cp2k/bin/cp2k.psmp' '-i' 'aiida.inp' >'aiida.out' 2>&1
