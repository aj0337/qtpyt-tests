#!/bin/bash -l
#SBATCH --job-name=aubda13ch2-los
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=200
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
#SBATCH --account=lp83
#SBATCH --uenv=prgenv-gnu/24.7:v3
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

MINICONDA_PATH=/users/ajayaraj/miniconda3

source $MINICONDA_PATH/etc/profile.d/conda.sh
conda activate qtpyt

# mpirun -n 1 python get_los_prerequisites.py
# mpirun -n 1 python get_gf_prerequisites.py

# mpirun -n 200 python get_active_embedding_hybridization.py
# mpirun -n 1 python get_dft_occupancies.py

# mpirun -n 1 python get_dft_states.py
# mpirun -n 200 python get_dft_transmission.py
# mpirun -n 200 python get_gw_edmft_transmission.py

# mpirun -n 1 python get_init_ed_self_energy.py
# mpirun -n 1 python get_ed_dc_corrections.py
# mpirun -n 1 python get_ed_self_energy.py

mpirun -n 200 python get_ed_transmission.py
