import os
import subprocess
import numpy as np

# Define parameter values
nbath_values = [4]
U_values = [4]
adjust_mu_values = [False]
double_counting_values = [False, True]
mu_values = np.arange(0, 2.6, 0.5)  # Mu values from 0 to 2.5 in steps of 0.5

# Job script template
job_script_template = """#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="dmft_nbath_{nbath}_adjustmu_{adjust_mu}_dc_{double_counting}_mu_{mu:.1f}"
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

srun -n 9 python run_dmft.py {nbath} {U} {adjust_mu} {double_counting} {mu:.1f}
# srun -n 9 python get_transmission.py {nbath} {U} {adjust_mu} {double_counting} {mu:.1f}
"""

# Output folder for job scripts
job_scripts_folder = "job_scripts"
os.makedirs(job_scripts_folder, exist_ok=True)

# Iterate over all parameter combinations and create job scripts
for nbath in nbath_values:
    for U in U_values:
        for adjust_mu in adjust_mu_values:
            for double_counting in double_counting_values:
                for mu in mu_values:
                    # Create a unique job script for each combination
                    job_script = job_script_template.format(
                        nbath=nbath,
                        U=U,
                        adjust_mu=str(adjust_mu),
                        double_counting=str(double_counting),
                        mu=mu,
                    )
                    job_script_filename = (
                        f"{job_scripts_folder}/job_nbath_{nbath}_U_{U}_adjustmu_{adjust_mu}_"
                        f"dc_{double_counting}_mu_{mu:.1f}.sh"
                    )

                    with open(job_script_filename, "w") as f:
                        f.write(job_script)

                    # Submit the job using sbatch
                    print(
                        f"Submitting job for nbath={nbath}, U={U}, adjust_mu={adjust_mu}, "
                        f"double_counting={double_counting}, mu={mu:.1f}"
                    )
                    subprocess.run(["sbatch", job_script_filename])
