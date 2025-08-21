#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=4-12:00:00
#SBATCH --mem=2500

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPICH_GPU_SUPPORT_ENABLED=0
export GPAW_SETUP_PATH=${HOME}/gpaw-setups-24.11.0
ulimit -s unlimited

MINICONDA_PATH=/home/jayn/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

# mpirun -n 20 gpaw python scatt_restart.py

# mpirun -n 1 python get_los_prerequisites.py
# mpirun -n 1 python get_cubefiles.py

# mpirun -n 1 python get_leads_self_energy.py
mpirun -n 1 python get_tridiagonal_matrix.py

mpirun -n 8 python get_dft_dos.py

# mpirun -n 8 python get_dft_transmission.py
