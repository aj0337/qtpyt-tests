#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=1000

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPICH_GPU_SUPPORT_ENABLED=0
export GPAW_SETUP_PATH=${HOME}/gpaw-setups-24.11.0
ulimit -s unlimited

MINICONDA_PATH=/home/jayn/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt


# compute tridiagonal nodes using notebook before proceeding

# mpirun -n 1 python get_los_prerequisites.py
# mpirun -n 1 python get_cubefiles.py

# mpirun -n 1 python get_leads_self_energy.py
# mpirun -n 1 python get_tridiagonal_matrix.py

mpirun -n 8 python get_dft_dos.py
mpirun -n 8 python get_dft_transmission.py


# Following are DMFT steps. Ensure all previous steps have been run with lowdin = True

# mpirun -n 1 python get_dft_states.py
# mpirun -n 96 python get_active_embedding_hybridization.py
# mpirun -n 1 python get_dft_occupancies.py

# mpirun -n 1 python run_no_spin_dmft.py
# mpirun -n 96 python get_no_spin_dmft_transmission.py
