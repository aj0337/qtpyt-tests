#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --account="s1267"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anooja.jayaraj@empa.ch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

module use /apps/empa/apps/modules/all
module load daint-mc cray-python cray-fftw numpy libxc libvdwxc intel # to run qtpyt/gpaw use this
# module load daint-mc cray-python    # to run edpyt use this

source "/users/ajayaraj/software/gpaw/gpaw-env/bin/activate"

# srun -n 1 python get_los_prerequisites.py
# srun -n 1 python get_gf_prerequisites.py
# srun -n 1 python get_dft_states.py
srun -n 4 python get_dft_transmission.py
# srun -n 4 python get_dft_dos.py
