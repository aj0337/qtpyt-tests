#!/bin/bash -l
#SBATCH --job-name=aubda13ch2-los
#SBATCH --time=00:20:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=s1276
# #SBATCH --uenv=gpaw/25.1.0
# #SBATCH --view=gpaw
#SBATCH --uenv=prgenv-gnu/24.7:v3
# #SBATCH --output=_scheduler-stdout.txt
# #SBATCH --error=_scheduler-stderr.txt

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

MINICONDA_PATH=/users/ajayaraj/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

# mpirun -N 1 -n 1 python get_los_prerequisites.py
# mpirun -N 1 -n 1 python get_gf_prerequisites.py
# mpirun -n 96 python get_dmft_transmission.py
mpirun -n 96 python get_dmft_transmission_btm.py
