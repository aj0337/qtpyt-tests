#!/bin/bash -l
#SBATCH --job-name=aubda13ch2-los
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

# mpirun -n 1 python run_no_spin_dmft.py
mpirun -n 1 python run_spin_dmft.py

# mpirun -n 24 python get_no_spin_dmft_transmission.py
# mpirun -n 24 python get_spin_dmft_transmission.py

# mpirun -n 1 python restart_dmft.py
# mpirun -n 24 python get_ed_transmission.py
