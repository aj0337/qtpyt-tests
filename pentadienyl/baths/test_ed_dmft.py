from __future__ import annotations
import pickle

import numpy as np
from edpyt.dmft import DMFT, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from scipy.interpolate import interp1d
from edpyt.nano_dmft import Gfloc

import matplotlib.pyplot as plt
import os


iteration_counter = 0
nsites_list = [4, 5, 6, 7]
relative_tols = [1e-4]
max_iter = 1000
alpha = 0.0
nspin = 1
de = 0.01
energies = np.arange(-3, 3 + de / 2.0, de).round(7)
eta = 1e-3
z_ret = energies + 1.0j * eta
beta = 1000
mu = 0.0
adjust_mus = [True]
use_double_counting = True

data_folder = "../output/lowdin"
V = np.loadtxt(f"{data_folder}/U_matrix.txt")
H_active = np.load(f"{data_folder}/bare_hamiltonian.npy").real
# H_active = np.load(f"{data_folder}/effective_hamiltonian.npy").real
with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)

with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

temperature_data_folder = f"{data_folder}/beta_{beta}"
occupancy_goal = np.load(f"{temperature_data_folder}/occupancies.npy")
z_mats = np.load(f"{temperature_data_folder}/matsubara_energies.npy")

len_active = occupancy_goal.size
hyb_mats = np.fromfile(
    f"{temperature_data_folder}/matsubara_hybridization.bin", complex
).reshape(
    z_mats.size,
    len_active,
    len_active,
)
_HybMats = interp1d(z_mats.imag, hyb_mats, axis=0, bounds_error=False, fill_value=0.0)
HybMats = lambda z: _HybMats(z.imag)
# HybMats = lambda z: np.zeros((len_active, len_active), dtype=complex)

S_active = np.eye(len_active)
idx_neq = np.arange(len_active)
idx_inv = np.arange(len_active)

double_counting = (
    np.diag(V.diagonal() * (occupancy_goal - 0.5))
    if use_double_counting
    else np.zeros((len_active, len_active))
)


for relative_tol in relative_tols:
    for nsites in nsites_list:
        for adjust_mu in adjust_mus:
            print(
                f"Starting spin unresolved DMFT calculation with {nsites} site(s), relative tolerance {relative_tol},  and adjust mu {adjust_mu}.",
                flush=True,
            )

            dmft_output_folder = f"{temperature_data_folder}/dmft/eta_{eta}/rel_tol_{relative_tol}/no_spin/nsites_{nsites}/adjust_mu_{adjust_mu}"
            figure_folder = f"{dmft_output_folder}/figures"
            os.makedirs(dmft_output_folder, exist_ok=True)
            os.makedirs(figure_folder, exist_ok=True)

            gfloc_with_dccorrection = Gfloc(
                H_active - double_counting,
                S_active,
                HybMats,
                idx_neq,
                idx_inv,
                nmats=z_mats.size,
                beta=beta,
            )

            nimp = gfloc_with_dccorrection.idx_neq.size
            gfimp = [Gfimp(nsites, z_mats.size, V[i, i], beta) for i in range(nimp)]
            gfimp = nanoGfimp(gfimp)

            Sigma = lambda z: np.zeros((nimp, z.size), complex)

            gfloc_no_dccorrection = Gfloc(
                H_active,
                S_active,
                HybMats,
                idx_neq,
                idx_inv,
                nmats=z_mats.size,
                beta=beta,
            )
            gfloc_no_dccorrection.update(mu=mu)
            gfloc_no_dccorrection.set_local(Sigma)

            # Initialize DMFT with adjust_mu parameter

            bath_filename = f"{dmft_output_folder}/bath_iter.h5py"
            iter_filename = f"{dmft_output_folder}/iter.h5py"

            dmft = DMFT(
                gfimp,
                gfloc_with_dccorrection,
                occupancy_goal,
                max_iter=max_iter,
                tol=relative_tol,
                adjust_mu=adjust_mu,
                alpha=alpha,
                DC=double_counting,
                store_iterations=False,
                store_last_n=1,
                bath_filename=bath_filename,
                iter_filename=iter_filename,
            )

            delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=mu)
            delta_prev = delta.copy()
            dmft.delta = delta
            # dmft.solve(dmft.delta, alpha=1.0)

            try:
                dmft.solve(dmft.delta, alpha=1.0)
            except:
                pass

            _Sigma = (
                lambda z: -double_counting.diagonal()[:, None]
                - gfloc_with_dccorrection.mu
                + gfloc_with_dccorrection.Sigma(z)[idx_inv]
            )

            Sigma_array = _Sigma(z_ret)
            trace_sigma_real = np.real(Sigma_array).sum(axis=0)
            trace_sigma_imag = np.imag(Sigma_array).sum(axis=0)

            plt.figure()
            plt.plot(energies, trace_sigma_real, label="Re Tr[Σ]")
            plt.plot(energies, trace_sigma_imag, label="Im Tr[Σ]")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Tr[Σ]")
            plt.legend()
            plt.title(f"Impurity self-energy trace, nsites={nsites}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{dmft_output_folder}/sigma_trace_nsites_{nsites}.png")
            plt.close()

            print(f"DMFT calculation with {nsites} site(s) completed.", flush=True)
