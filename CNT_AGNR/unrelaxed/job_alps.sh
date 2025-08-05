#!/bin/bash -l
#SBATCH --job-name=aubda13ch2-los
#SBATCH --time=1:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=s1276
#SBATCH --uenv=prgenv-gnu/24.7:v3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MINICONDA_PATH=/users/ajayaraj/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

mpirun -N 1 -n 1 python get_los_prerequisites.py
mpirun -N 1 -n 1 python get_gf_prerequisites.py
# mpirun -n 96 python get_dft_transmission.py
