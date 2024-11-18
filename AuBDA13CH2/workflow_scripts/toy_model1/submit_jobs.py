import os
import subprocess

# Define parameter values
nbath_values = [4, 8]
U_values = [4.0, 5.0, 6.0]

# Job script template
job_script_template = """#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="dmft_nbath_{nbath}_U_{U}"
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

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
ulimit -s unlimited

module use /apps/empa/apps/modules/all
module load daint-mc cray-python

source "/users/ajayaraj/software/gpaw/gpaw-env/bin/activate"

srun -n 9 python run_dmft_model_iterate.py {nbath} {U}
"""

# Output folder for job scripts
job_scripts_folder = "job_scripts"
os.makedirs(job_scripts_folder, exist_ok=True)

# Iterate over all parameter combinations and create job scripts
for nbath in nbath_values:
    for U in U_values:
        # Create a unique job script for each (nbath, U) combination
        job_script = job_script_template.format(nbath=nbath, U=U)
        job_script_filename = f"{job_scripts_folder}/job_nbath_{nbath}_U_{U}.sh"

        with open(job_script_filename, "w") as f:
            f.write(job_script)

        # Submit the job using sbatch
        print(f"Submitting job for nbath={nbath}, U={U}")
        subprocess.run(["sbatch", job_script_filename])
