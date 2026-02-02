from __future__ import annotations

import os
import pickle

import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run(outputfile, gf, total_dim):
    pdos = np.empty((energies.size, total_dim))
    for e, energy in enumerate(energies):
        pdos[e,:] = gf.get_pdos(energy)

    if comm.rank == 0:
        np.savez(outputfile, energies=energies, pdos=pdos.real)


data_folder = "./output/lowdin"
dft_data_folder = f"{data_folder}/dft"
os.makedirs(dft_data_folder, exist_ok=True)

self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

de = 0.01
energies = np.arange(-3,3,de)
eta = 1e-4

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

filename = "Evpdos.npz"
outputfile = os.path.join(dft_data_folder, filename)

shapes = [np.shape(block) for block in hs_list_ii]
dims = [shape[1] for shape in shapes]
total_dim = sum(dims)
run(outputfile, gf,total_dim)
