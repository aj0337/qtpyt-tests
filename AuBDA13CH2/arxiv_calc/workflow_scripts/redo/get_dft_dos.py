from __future__ import annotations
import os
import pickle
import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.projector import ProjectedGreenFunction

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data paths
data_folder = "../../output/redo"

# Load data
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

# Parameters
z_ret = np.load(f"{data_folder}/energies.npy")
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
    mu=0.0,
    kt=1 / beta,
)
gfp = ProjectedGreenFunction(gf, index_active_region)

# Parallel DOS Computation
local_indices = np.array_split(range(len(energies)), size)[rank]
local_energies = energies[local_indices]
local_dft_dos = np.array([gfp.get_dos(energy) for energy in local_energies])

# Prepare for MPI.Gatherv
local_sizes = np.array([len(local_dft_dos)], dtype=int)
all_sizes = None
if rank == 0:
    all_sizes = np.zeros(size, dtype=int)
comm.Gather(local_sizes, all_sizes, root=0)

# Calculate displacements for gathering
displacements = None
if rank == 0:
    displacements = np.cumsum(np.insert(all_sizes[:-1], 0, 0))

# Gather variable-sized arrays
all_dft_dos = None
if rank == 0:
    all_dft_dos = np.zeros(sum(all_sizes), dtype=np.float64)
comm.Gatherv(local_dft_dos, [all_dft_dos, all_sizes, displacements, MPI.DOUBLE], root=0)

# Save results on root
if rank == 0:
    filename = os.path.join(data_folder, "dft_dos.npy")
    np.save(filename, all_dft_dos)
