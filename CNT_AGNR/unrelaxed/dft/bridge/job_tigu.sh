#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --time=4-12:00:00
#SBATCH --mem=62500

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MPICH_GPU_SUPPORT_ENABLED=0
export GPAW_SETUP_PATH=${HOME}/gpaw-setups-24.11.0
ulimit -s unlimited

MINICONDA_PATH=/home/jayn/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

mpirun -n 20 gpaw python scatt_restart.py
# srun gpaw python dump.py
