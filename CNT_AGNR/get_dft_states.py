from __future__ import annotations

import os
import pickle

import numpy as np
from qtpyt.block_tridiag import greenfunction
from qtpyt.hybridization import Hybridization
from qtpyt.projector import ProjectedGreenFunction
from scipy.linalg import eigvalsh

# Data paths
data_folder = f"./output/lowdin/device"

# Load data
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

# Parameters
eta = 1e-3

# Green's Function Setup
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)
gfp = ProjectedGreenFunction(gf, index_active_region)

hyb = Hybridization(gfp)
Heff = (hyb.H + hyb.retarded(0.0)).real

np.save(os.path.join(data_folder, "bare_hamiltonian.npy"), hyb.H)
np.save(os.path.join(data_folder, "eigvals_Hbare.npy"), eigvalsh(hyb.H, hyb.S))
np.save(os.path.join(data_folder, "effective_hamiltonian.npy"), Heff)
np.save(os.path.join(data_folder, "eigvals_Heff.npy"), eigvalsh(Heff, gfp.S))
