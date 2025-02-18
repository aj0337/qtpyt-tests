#!/bin/bash -l
#SBATCH --job-name=aubda13ch2-los
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
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

# mpirun -n 1 python get_los_prerequisites.py
# mpirun -n 1 python get_cubefiles.py
# mpirun -n 1 python get_gf_prerequisites.py
# mpirun -n 24 python get_dft_transmission.py
# mpirun -n 1 python get_dft_states.py

# mpirun -n 1 python get_dft_occupancies.py
# mpirun -n 96 python get_dft_dos.py

# mpirun -n 24 python get_active_embedding_hybridization.py

# mpirun -n 1 python get_dft_occupancies_gfloc.py
# mpirun -n 1 python get_dft_dos_gfloc.py

# mpirun -n 1 python run_dmft.py
# mpirun -n 1 python restart_dmft.py

# mpirun -n 1 python run_dmft_and_transmission.py
mpirun -n 24 python get_dmft_transmission.py
