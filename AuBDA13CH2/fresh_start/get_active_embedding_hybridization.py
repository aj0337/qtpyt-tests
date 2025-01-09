from __future__ import annotations

import os
import pickle

import numpy as np
from qtpyt.block_tridiag import greenfunction
from qtpyt.hybridization import Hybridization
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import ProjectedGreenFunction

# Data paths
data_folder = "./output"

# Load data
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

# Parameters
z_ret = np.load(f"{data_folder}/retarded_energies.npy")
eta = z_ret.imag[0]
energies = z_ret.real
beta = 1000

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

n_A = len(index_active_region)
gd = GridDesc(energies, n_A, complex)
HB = gd.empty_aligned_orbs()

for e, energy in enumerate(gd.energies):
    HB[e] = hyb.retarded(energy)

filename = os.path.join(data_folder, 'hybridization.bin')
gd.write(HB,filename)
del HB

# Define parameters for matsubara grid
ne = 3000
matsubara_energies = 1.j * (2 * np.arange(ne) + 1) * np.pi / beta

gf.eta = 0.
assert self_energy[0].eta == 0.
assert self_energy[1].eta == 0.

mat_gd = GridDesc(matsubara_energies, n_A, complex)
HB_mat = mat_gd.empty_aligned_orbs()

for e, energy in enumerate(mat_gd.energies):
    HB_mat[e] = hyb.retarded(energy)

# Save the Matsubara hybrid data
filename = os.path.join(data_folder, 'matsubara_hybridization.bin')
mat_gd.write(HB_mat, filename)
del HB_mat

if comm.rank == 0:
    np.save(os.path.join(data_folder, 'matsubara_energies.npy'), matsubara_energies)
