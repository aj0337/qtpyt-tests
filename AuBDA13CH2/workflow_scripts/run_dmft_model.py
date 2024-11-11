from __future__ import annotations
import numpy as np

import os
from scipy.optimize import root

from edpyt.nano_dmft import Gfloc, Gfimp as nanoGfimp
from edpyt.dmft import Gfimp, DMFT, Converged

import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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

U = 4.0  # Interaction
nbaths = 8
tol = 1e-4
max_iter = 1000
alpha = 0.0
nspin = 1
mu = U/2
eta = 3e-2
data_folder = "../output/compute_run/"
output_folder = "../output/compute_run/model_parallel_nbath8_minimize"
os.makedirs(output_folder, exist_ok=True)

occupancy_goal = np.load(f"{data_folder}/occupancies.npy")
len_active = 9
energies = np.arange(-10,10,0.01)
z_ret = energies + 1.j * eta

H_active = np.load(f"{data_folder}/hamiltonian.npy").real

beta = 1000
ne = 3000
z_mats = 1.0j * (2 * np.arange(ne) + 1) * np.pi / beta

HybMats = lambda z: 0.0

S_active = np.eye(len_active)

idx_neq = np.arange(len_active)
idx_inv = np.arange(len_active)

V = np.eye(len_active) * U

gfloc = Gfloc(H_active, np.eye(len_active), HybMats, idx_neq, idx_inv)
gfloc.mu = mu
nimp = gfloc.idx_neq.size
gfimp = []
for i in range(nimp):
    gfimp.append(Gfimp(nbaths, z_mats.size, V[i, i], beta))

gfimp = nanoGfimp(gfimp)

occupancy_goal = occupancy_goal[gfloc.idx_neq]

dmft = DMFT(
    gfimp,
    gfloc,
    occupancy_goal,
    max_iter=max_iter,
    tol=tol,
    adjust_mu=False,
    alpha=alpha,
    store_iterations=False,

)

Sigma = lambda z: np.zeros((nimp, z.size), complex)
delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=mu)
delta_prev = delta.copy()


try:
    root(distance, delta_prev, method="broyden1")
except Converged:
    pass

if rank == 0:
    np.save(f"{output_folder}/dmft_delta.npy", delta_prev)
    open(f"{output_folder}/mu.txt", "w").write(str(gfloc.mu))

    _Sigma = lambda z: -gfloc.mu + gfloc.Sigma(z)[idx_inv]

    dmft_sigma_file = f"{output_folder}/dmft_sigma.npy"
    save_sigma(_Sigma(z_ret), dmft_sigma_file, nspin)

    gfloc_data = gfloc(z_ret)
    np.save(f"{output_folder}/dmft_gfloc.npy", gfloc_data)
