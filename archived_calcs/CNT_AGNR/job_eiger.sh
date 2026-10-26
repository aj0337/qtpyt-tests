#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=1000

MINICONDA_PATH=/users/ajayaraj/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt


# compute tridiagonal nodes using notebook before proceeding

# mpirun -n 1 python get_los_prerequisites.py
# mpirun -n 1 python get_cubefiles.py

# mpirun -n 1 python get_leads_self_energy.py
# mpirun -n 1 python get_tridiagonal_matrix.py

mpirun -n 8 python get_dft_dos.py
mpirun -n 8 python get_dft_transmission.py
