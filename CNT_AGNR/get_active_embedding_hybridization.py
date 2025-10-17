from __future__ import annotations
from mpi4py import MPI

import os
import pickle
import numpy as np
from ase.units import kB
from qtpyt.block_tridiag import greenfunction
from qtpyt.hybridization import Hybridization
from qtpyt.projector import ProjectedGreenFunction

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print(f"Running with {size} MPI ranks")


data_folder = "./unrelaxed/output/lowdin/device"

index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

if rank == 0:
    print("Loaded files")

de = 0.01
energies = np.arange(-1.5, 1.5 + de / 2.0, de).round(7)
eta = 1e-2
temperature = 9
beta = 1 / (kB * temperature)

output_folder = f"./unrelaxed/output/lowdin/device/T_{temperature}K/"
if rank == 0:
    os.makedirs(output_folder, exist_ok=True)

comm.Barrier()

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

ne = 3000
matsubara_energies = 1.0j * (2 * np.arange(ne) + 1) * np.pi / beta
gfp.eta = 0.0
assert self_energy[0].eta == 0.0
assert self_energy[1].eta == 0.0

energy_indices = np.array_split(np.arange(ne), size)[rank]
local_energies = matsubara_energies[energy_indices]

if rank == 0:
    print(f"Distributing {ne} energies across {size} ranks...")
    print(f"Each rank will handle ~{len(energy_indices)} energies.")

HB_mat_local = np.empty((len(local_energies), n_A, n_A), dtype=np.complex64)

for i, energy in enumerate(local_energies):
    print(f"[Rank {rank}] Computing energy {energy_indices[i]} / {ne}")
    HB_mat_local[i] = hyb.retarded(energy)

if rank == 0:
    HB_mat = np.empty((ne, n_A, n_A), dtype=np.complex64)
else:
    HB_mat = None

counts = np.array(comm.gather(len(local_energies), root=0))
if rank == 0:
    displs = np.insert(np.cumsum(counts[:-1]), 0, 0)
else:
    displs = None

comm.Gatherv(
    sendbuf=HB_mat_local,
    recvbuf=(HB_mat, counts * n_A * n_A, displs * n_A * n_A, MPI.COMPLEX),
    root=0,
)

if rank == 0:
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "matsubara_energies.npy"), matsubara_energies)
    np.save(os.path.join(output_folder, "matsubara_hybridization.npy"), HB_mat)
    print("Saved Matsubara hybridization and energies successfully.")
    print("All ranks finished successfully.")
