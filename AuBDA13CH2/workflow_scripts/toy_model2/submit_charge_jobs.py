import os
import subprocess

# Define parameter values for nzp and R
nzp_values = [50, 100, 200, 500]
R_values = [1e8, 1e9, 1e10, 1e11]

# Job script template
job_script_template = """#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="charge_nzp_{nzp}_R_{R}"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
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

srun -n 1 python check_charge_convergence.py {nzp} {R}
"""

# Output folder for job scripts
job_scripts_folder = "job_scripts"
os.makedirs(job_scripts_folder, exist_ok=True)

# Iterate over all parameter combinations and create job scripts
for nzp in nzp_values:
    for R in R_values:
        # Create a unique job script for each combination
        job_script = job_script_template.format(
            nzp=nzp,
            R=int(R)  # Convert R to integer for clean naming
        )
        job_script_filename = f"{job_scripts_folder}/job_nzp_{nzp}_R_{int(R)}.sh"

        with open(job_script_filename, "w") as f:
            f.write(job_script)

        # Submit the job using sbatch
        print(f"Submitting job for nzp={nzp}, R={R}")
        subprocess.run(["sbatch", job_script_filename])
