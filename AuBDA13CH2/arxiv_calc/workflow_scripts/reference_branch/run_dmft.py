from __future__ import annotations

import os
import sys

import numpy as np
from edpyt.dmft import DMFT, Converged, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from edpyt.nano_dmft import Gfloc
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.projector import expand
from scipy.optimize import root

class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_molecule, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


def distance(delta):
    global delta_prev
    delta_prev[:] = delta
    return dmft.distance(delta)


def save_sigma(sigma_diag, outputfile, npsin):
    L, ne = sigma_diag.shape
    sigma = np.zeros((ne, L, L), complex)

    def save(spin):
        for diag, mat in zip(sigma_diag.T, sigma):
            mat.flat[:: (L + 1)] = diag
        np.save(outputfile, sigma)

    for spin in range(nspin):
        save(spin)


nbaths = 4
U = 4
adjust_mu = True
use_double_counting = True

tol = 1e-4
max_iter = 1000
alpha = 0.0
nspin = 1
eta = 3e-2
data_folder = "../../output/compute_run"
output_folder = "../../output/compute_run/reference_branch"
os.makedirs(output_folder, exist_ok=True)
occupancy_goal = np.load(f"{data_folder}/occupancies.npy")
len_active = 9
energies = np.arange(-10, 10, 0.01)
z_ret = energies + 1.0j * eta

H_active = np.load(f"{data_folder}/hamiltonian.npy").real
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
beta = 70
ne = 3000
z_mats = 1.0j * (2 * np.arange(ne) + 1) * np.pi / beta

HybMats = lambda z: 0.0
S_active = np.eye(len_active)
idx_neq = np.arange(len_active)
idx_inv = np.arange(len_active)

V = np.eye(len_active) * U

# Apply double counting correction if specified
double_counting = (
    np.diag(V.diagonal() * (occupancy_goal - 0.5))
    if use_double_counting
    else np.zeros((len_active, len_active))
)
gfloc = Gfloc(
    H_active - double_counting, np.eye(len_active), HybMats, idx_neq, idx_inv
)

nimp = gfloc.idx_neq.size
gfimp = [Gfimp(nbaths, z_mats.size, V[i, i], beta) for i in range(nimp)]
gfimp = nanoGfimp(gfimp)

occupancy_goal_ = occupancy_goal[gfloc.idx_neq]

# Initialize DMFT with adjust_mu parameter
dmft = DMFT(
    gfimp,
    gfloc,
    occupancy_goal_,
    max_iter=max_iter,
    tol=tol,
    adjust_mu=adjust_mu,
    alpha=alpha,
)

Sigma = lambda z: np.zeros((nimp, z.size), complex)
delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=0)
delta_prev = delta.copy()

try:
    root(distance, delta_prev, method="broyden1")
except Converged:
    pass




_Sigma = (
    lambda z: -double_counting.diagonal()[:, None]
    - gfloc.mu
    + gfloc.Sigma(z)[idx_inv]
)
dmft_sigma_file = f"{output_folder}/dmft_sigma.npy"
save_sigma(_Sigma(z_ret), dmft_sigma_file, nspin)

gfloc_data = gfloc(z_ret)
np.save(f"{output_folder}/dmft_gfloc.npy", gfloc_data)
