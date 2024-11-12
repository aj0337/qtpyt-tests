#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="dmft_nbath_4_U_4.0"
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --account="s1267"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anooja.jayaraj@empa.ch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ulimit -s unlimited

module use /apps/empa/apps/modules/all
module load daint-mc cray-python

source "/users/ajayaraj/software/gpaw/gpaw-env/bin/activate"

srun -n 9 python run_dmft_model_iterate.py 4 4.0
