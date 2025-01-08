import sys
import os
import pickle
import numpy as np
from qtpyt.block_tridiag import greenfunction
from qtpyt.continued_fraction import get_ao_charge
from qtpyt.projector import ProjectedGreenFunction

# Parse command-line arguments for nzp and R
if len(sys.argv) != 3:
    raise ValueError("Please provide nzp and R as command-line arguments.")

nzp = int(sys.argv[1])
R = float(sys.argv[2])

# Set parameters
eta = 3e-2
data_folder = "../../output/compute_run"
output_folder = "../../output/compute_run/toy_model2/charge_convergence"
os.makedirs(output_folder, exist_ok=True)

# Load necessary input files
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
beta = 1000

# Load data
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

# Initialize Green's function and projected Green's function
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)
gfp = ProjectedGreenFunction(gf, index_active_region)

# Compute charge
charge_dft = get_ao_charge(gfp, mu=0, beta=beta, nzp=nzp, R=R)

# Save results
output_file = f"{output_folder}/charge_per_orbital_nzp_{nzp}_R_{int(R)}.npy"
np.save(output_file, charge_dft)
print(f"Charge computation completed for nzp={nzp}, R={R}. Results saved to {output_file}.")
