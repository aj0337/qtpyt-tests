from __future__ import annotations

import os
import pickle
import numpy as np
from edpyt.nano_dmft import Gfloc
from qtpyt.block_tridiag import greenfunction
from qtpyt.hybridization import Hybridization
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import ProjectedGreenFunction
from scipy.interpolate import interp1d

# =========================
# PARAMETERS
# =========================
DFT_TYPES = ["device_fd_1e-3"]
BETA = 1000
ETA = [1e-3, 5e-3, 1e-2]
DE = 0.01
ENERGY_RANGE = (-2, 2)
NMATS = 3000
MUS = [0.0]

for dft_type in DFT_TYPES:
    for eta in ETA:
        data_folder = f"./output/lowdin/{dft_type}"
        output_folder = os.path.join(data_folder, "occupancies")


        os.makedirs(output_folder, exist_ok=True)

        # =========================
        # LOAD DATA
        # =========================
        index_active_region = np.load(f"{data_folder}/index_active_region.npy")
        self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

        with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
            hs_list_ii = pickle.load(f)
        with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
            hs_list_ij = pickle.load(f)

        z_mats = np.load(f"{data_folder}/matsubara_energies.npy")
        H_active = np.load(f"{data_folder}/bare_hamiltonian.npy").real
        len_active = H_active.shape[0]

        # =========================
        # DEFINE ENERGY GRIDS
        # =========================
        energies = np.arange(ENERGY_RANGE[0], ENERGY_RANGE[1] + DE / 2.0, DE).round(7)
        z_ret = energies + 1.0j * eta
        matsubara_energies = 1.0j * (2 * np.arange(NMATS) + 1) * np.pi / BETA

        # =========================
        # GREEN'S FUNCTION SETUP
        # =========================
        gf = greenfunction.GreenFunction(
            hs_list_ii,
            hs_list_ij,
            [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
            solver="dyson",
            # eta=eta,
        )
        gfp = ProjectedGreenFunction(gf, index_active_region)
        hyb = Hybridization(gfp)

        gf.eta = 0.0
        gfp.eta = 0.0
        assert self_energy[0].eta == 0.0
        assert self_energy[1].eta == 0.0

        # =========================
        # COMPUTE MATSUBARA HYBRIDIZATION
        # =========================
        mat_gd = GridDesc(matsubara_energies, len_active, complex)
        HB_mat = mat_gd.empty_aligned_orbs()

        for e, energy in enumerate(mat_gd.energies):
            HB_mat[e] = hyb.retarded(energy)

        filename = os.path.join(data_folder, "matsubara_hybridization.bin")
        mat_gd.write(HB_mat, filename)
        del HB_mat

        if comm.rank == 0:
            np.save(os.path.join(data_folder, "matsubara_energies.npy"), matsubara_energies)

            # =========================
            # COMPUTE OCCUPANCIES
            # =========================
            hyb_mats = np.fromfile(filename, complex).reshape(z_mats.size, len_active, len_active)
            _HybMats = interp1d(matsubara_energies.imag, hyb_mats, axis=0, bounds_error=False, fill_value=0.0)
            HybMats = lambda z: _HybMats(z.imag)

            S_active = np.eye(len_active)
            idx_neq = np.arange(len_active)
            idx_inv = np.arange(len_active)

            Sigma = lambda z: np.zeros((len_active, z.size), complex)

            for mu in MUS:
                print(f"Calculating occupancy for mu = {mu}", flush=True)

                gfloc = Gfloc(H_active, S_active, HybMats, idx_neq, idx_inv, nmats=z_mats.size, beta=BETA)
                gfloc.update(mu=0.0)
                gfloc.set_local(Sigma)

                occupancies = gfloc.integrate(mu=mu)
                total_occupancy = np.sum(occupancies)

                print(f"Total occupancy for mu = {mu} using Matsubara summation for eta {eta} : {total_occupancy}", flush=True)
                np.save(os.path.join(output_folder, f'occupancies_gfloc_mu_{mu}.npy'), occupancies)
