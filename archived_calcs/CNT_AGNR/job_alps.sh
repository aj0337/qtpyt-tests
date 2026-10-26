#!/bin/bash -l
#SBATCH --job-name=aubda13ch2-los
#SBATCH --time=7:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=lp83
#SBATCH --uenv=prgenv-gnu/24.7:v3

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MINICONDA_PATH=/users/ajayaraj/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

# compute tridiagonal nodes using notebook before proceeding

# mpirun -n 1 python get_los_prerequisites.py
# mpirun -n 1 python get_cubefiles.py

# mpirun -n 1 python get_leads_self_energy.py
# mpirun -n 1 python get_tridiagonal_matrix.py

mpirun -n 96 python get_dft_dos.py
mpirun -n 96 python get_dft_transmission.py
