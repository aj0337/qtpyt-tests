from __future__ import annotations
import os
import pickle
import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.projector import ProjectedGreenFunction
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

lowdin = True
data_folder = f"./unrelaxed/output/lowdin" if lowdin else f"./unrelaxed/output/no_lowdin"
dft_data_folder = f"{data_folder}/dft"
os.makedirs(dft_data_folder, exist_ok=True)

index_active_region = np.load(f"{data_folder}/index_active_region.npy")
# self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

de = 0.01
energies = np.arange(-1.5, 1.5 + de / 2.0, de).round(7)
eta = 1e-2

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    # [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

gfp = ProjectedGreenFunction(gf, index_active_region)
filename = "Evdos_C2pz.npy"
outputfile = os.path.join(dft_data_folder, filename)

local_energies = np.array_split(energies, size)[rank]
local_dos = np.empty(local_energies.size, dtype=np.float64)
for i, energy in enumerate(local_energies):
    local_dos[i] = np.real(gfp.get_dos(energy))

sendcounts = np.array(comm.gather(local_dos.size, root=0))
if rank == 0:
    dos = np.empty(energies.size, dtype=np.float64)
    displs = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)
    recvbuf = [dos, (sendcounts, displs), MPI.DOUBLE]
else:
    recvbuf = None

comm.Gatherv(sendbuf=local_dos, recvbuf=recvbuf, root=0)

if rank == 0:
    np.save(outputfile, (energies, dos))
    plt.figure()
    plt.plot(energies, dos)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS")
    plt.tight_layout()
    plt.savefig(outputfile.replace(".npy", ".png"), dpi=300)
    plt.close()
