#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --get-user-env
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-12:00:00
#SBATCH --mem=2500

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

MINICONDA_PATH=/home/jayn/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

mpirun -n 1 python get_los_prerequisites.py
# mpirun -n 1 python get_cubefiles.py
mpirun -n 1 python get_gf_prerequisites.py
mpirun -n 1 python get_dft_states.py
mpirun -n 1 python get_dft_occupancies.py
mpirun -n 1 python get_dft_dos.py
# mpirun -n 1 python get_dft_occupancies_gfloc.py
# mpirun -n 1 python get_dft_dos_gfloc.py
# mpirun -n 8 python get_dft_transmission.py
mpirun -n 1 python get_active_embedding_hybridization.py
mpirun -n 1 python run_dmft.py
# mpirun -n 1 python restart_dmft.py
# mpirun -n 1 python run_dmft_and_transmission.py
# mpirun -n 1 python get_dmft_transmission.py
