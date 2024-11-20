import os
import subprocess

# Define parameter values
nbath_values = [4, 8]
U_values = [0.5]
adjust_mu_values = [True, False]
double_counting_values = [True, False]

# Job script template
job_script_template = """#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="dmft_nbath_{nbath}_adjustmu_{adjust_mu}_dc_{double_counting}"
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

srun -n 9 python run_dmft.py {nbath} {U} {adjust_mu} {double_counting}
srun -n 9 python get_transmission.py {nbath} {U} {adjust_mu} {double_counting}
"""

# Output folder for job scripts
job_scripts_folder = "job_scripts"
os.makedirs(job_scripts_folder, exist_ok=True)

# Iterate over all parameter combinations and create job scripts
for nbath in nbath_values:
    for U in U_values:
        for adjust_mu in adjust_mu_values:
            for double_counting in double_counting_values:
                # Create a unique job script for each combination
                job_script = job_script_template.format(
                    nbath=nbath,
                    U=U,
                    adjust_mu=str(adjust_mu),
                    double_counting=str(double_counting),
                )
                job_script_filename = f"{job_scripts_folder}/job_nbath_{nbath}_adjustmu_{adjust_mu}_dc_{double_counting}.sh"

                with open(job_script_filename, "w") as f:
                    f.write(job_script)

                # Submit the job using sbatch
                print(
                    f"Submitting job for nbath={nbath}, adjust_mu={adjust_mu}, double_counting={double_counting}"
                )
                subprocess.run(["sbatch", job_script_filename])
