#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
##SBATCH --nodes=4
##SBATCH --ntasks-per-node=2
##SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --account="s1237"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anooja.jayaraj@empa.ch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

module use /apps/empa/apps/modules/all
module load daint-mc cray-python cray-fftw numpy libxc libvdwxc

source "/users/ajayaraj/software/gpaw/gpaw-env/bin/activate"

python /scratch/snx3000/ajayaraj/tests/AuBDA/localization_dzp/get_los_cube.py >los.out
