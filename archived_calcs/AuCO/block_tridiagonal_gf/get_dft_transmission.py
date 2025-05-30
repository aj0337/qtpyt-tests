from __future__ import annotations

import pickle

import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def run(outputfile):
    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies, T.real))


data_folder = "./output/no_lowdin"
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

de = 0.01
energies = np.arange(-3.0, 3.0 + de / 2.0, de)
eta = 1e-5

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

print(self_energy[0].retarded(0).shape)
exit()
# print(f"hs_list_ii shape: {np.array(hs_list_ii).shape}")
# print(f"hs_list_ij shape: {np.array(hs_list_ij).shape}")

print(f"hs_list_ii shapes: {[np.shape(x) for x in hs_list_ii]}")
print(f"hs_list_ij shapes: {[np.shape(x) for x in hs_list_ij]}")

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

# Transmission function for DFT
outputfile = f"{data_folder}/dft_transmission.npy"
run(outputfile)
