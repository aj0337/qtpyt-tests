from __future__ import annotations
import os
import pickle
import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.continued_fraction import get_ao_charge

# Data paths
data_folder = "./output/lowdin"
output_folder = "./output/lowdin/"

# Load data
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

# Parameters
mu = 0
beta = 1000

# Green's Function Setup
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
)
gfp = ProjectedGreenFunction(gf, index_active_region)

np.save(
    os.path.join(output_folder, f"occupancies_gfp_mu_{mu}.npy"),
    get_ao_charge(gfp, mu=mu, beta=beta),
)
