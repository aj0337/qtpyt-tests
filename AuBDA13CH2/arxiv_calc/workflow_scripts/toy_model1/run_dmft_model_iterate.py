from __future__ import annotations
import sys

import os
import pickle

import numpy as np
from mpi4py import MPI
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.block_tridiag import greenfunction
from qtpyt.continued_fraction import get_ao_charge
from qtpyt.projector import ProjectedGreenFunction, expand
from scipy.optimize import root

from edpyt.dmft import DMFT, Converged, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from edpyt.nano_dmft import Gfloc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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


# Check for command-line arguments for nbath and U
if len(sys.argv) < 3:
    raise ValueError("Please provide nbath and U as command-line arguments.")
nbaths = int(sys.argv[1])
U = float(sys.argv[2])

tol = 1e-4
max_iter = 1000
alpha = 0.0
nspin = 1
eta = 3e-2
data_folder = "../output/compute_run/"
output_folder = "../output/compute_run/model/"
os.makedirs(output_folder, exist_ok=True)

occupancy_goal = np.load(f"{data_folder}/occupancies.npy")
len_active = 9
energies = np.arange(-10, 10, 0.01)
z_ret = energies + 1.0j * eta

H_active = np.load(f"{data_folder}/hamiltonian.npy").real
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
beta = 1000
ne = 3000
z_mats = 1.0j * (2 * np.arange(ne) + 1) * np.pi / beta

HybMats = lambda z: 0.0
S_active = np.eye(len_active)
idx_neq = np.arange(len_active)
idx_inv = np.arange(len_active)


V = np.eye(len_active) * U
mu = U / 2

gfloc = Gfloc(H_active, np.eye(len_active), HybMats, idx_neq, idx_inv)
gfloc.mu = mu
nimp = gfloc.idx_neq.size
gfimp = [Gfimp(nbaths, z_mats.size, V[i, i], beta) for i in range(nimp)]
gfimp = nanoGfimp(gfimp)

occupancy_goal_ = occupancy_goal[gfloc.idx_neq]

# Define a unique output folder for each combination of nbaths and U
output_folder_combination = f"{output_folder}/nbaths_{nbaths}_U_{U}"
os.makedirs(output_folder_combination, exist_ok=True)

dmft = DMFT(
    gfimp,
    gfloc,
    occupancy_goal_,
    max_iter=max_iter,
    tol=tol,
    adjust_mu=False,
    alpha=alpha,
    egrid=z_ret,
    store_iterations=True,
    iter_filename=f"{output_folder_combination}/dmft_iterations.h5",
)

Sigma = lambda z: np.zeros((nimp, z.size), complex)
delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=mu)
delta_prev = delta.copy()

try:
    root(distance, delta_prev, method="broyden1")
except Converged:
    pass

if rank == 0:
    # Save results for this combination of nbaths and U
    np.save(f"{output_folder_combination}/dmft_delta.npy", delta_prev)
    with open(f"{output_folder_combination}/mu.txt", "w") as mu_file:
        mu_file.write(str(gfloc.mu))

    _Sigma = lambda z: -gfloc.mu + gfloc.Sigma(z)[idx_inv]

    dmft_sigma_file = f"{output_folder_combination}/dmft_sigma.npy"
    save_sigma(_Sigma(z_ret), dmft_sigma_file, nspin)

    gfloc_data = gfloc(z_ret)
    np.save(f"{output_folder_combination}/dmft_gfloc.npy", gfloc_data)

    # Compute orbital occupancy before and after including DMFT self-energy

    self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
    with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
        hs_list_ii = pickle.load(f)
    with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
        hs_list_ij = pickle.load(f)

    gf = greenfunction.GreenFunction(
        hs_list_ii,
        hs_list_ij,
        [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
        solver="dyson",
        eta=eta,
    )
    gfp = ProjectedGreenFunction(gf, index_active_region)
    charge_dft = get_ao_charge(gfp)

    nodes = [0, 810, 1116, 1278, 1584, 2394]
    imb = 2  # index of molecule block from the nodes list
    S_molecule = hs_list_ii[imb][1]  # overlap of molecule
    idx_molecule = (
        index_active_region - nodes[imb]
    )  # indices of active region w.r.t molecule

    np.save(f"{output_folder_combination}/charge_per_orbital_dft.npy", charge_dft)

    dmft_sigma = load(dmft_sigma_file)
    self_energy[2] = dmft_sigma
    gf.selfenergies.append((imb, self_energy[2]))

    gfp_dmft = ProjectedGreenFunction(gf, index_active_region)
    charge_dmft = get_ao_charge(gfp_dmft)
    np.save(f"{output_folder_combination}/charge_per_orbital_dmft.npy", charge_dmft)
