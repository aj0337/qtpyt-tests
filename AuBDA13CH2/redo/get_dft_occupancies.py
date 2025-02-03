from __future__ import annotations
import os
import pickle
import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.continued_fraction import get_ao_charge

dft_types = ["device_fd_0.0", "device_fd_1e-3"]
for dft_type in dft_types:

    # Data paths
    data_folder = f"./output/lowdin/{dft_type}"
    output_folder = f"./output/lowdin/{dft_type}/occupancies"
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    index_active_region = np.load(f"{data_folder}/index_active_region.npy")
    self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
    with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
        hs_list_ii = pickle.load(f)
    with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
        hs_list_ij = pickle.load(f)

    # Parameters
    mus = [0.0, 0.1, 0.5]
    beta = 1000

    for mu in mus:
        print(f"Calculating occupancy for mu = {mu}",flush=True)
        # Green's Function Setup
        gf = greenfunction.GreenFunction(
            hs_list_ii,
            hs_list_ij,
            [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
            solver="dyson",
        )
        gfp = ProjectedGreenFunction(gf, index_active_region)
        occupancies = get_ao_charge(gfp, mu=mu, beta=beta)
        print(f"Total occupancy for mu = {mu} using contour integration are: {np.sum(occupancies)}",flush=True)
        np.save(
            os.path.join(output_folder, f"occupancies_gfp_mu_{mu}.npy"),
            occupancies,
        )
