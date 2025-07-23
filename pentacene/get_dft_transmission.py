from __future__ import annotations

import os
import pickle

import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm
import matplotlib.pyplot as plt
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
        plt.figure()
        plt.plot(energies, T)
        plt.yscale("log")
        plt.xlim(-3.0, 3.0)
        plt.ylim(1e-5, 1)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Transmission")
        plt.tight_layout()
        plt.savefig(f"{dft_data_folder}/ET.png", dpi=300)
        plt.close()


data_folder = "./output/lowdin"
dft_data_folder = "./output/lowdin/dft"
os.makedirs(dft_data_folder, exist_ok=True)
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

de = 0.01
energies = np.arange(-1, 1 + de / 2.0, de).round(7)
eta = 1e-2

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

# Transmission function for DFT
outputfile = f"{dft_data_folder}/ET.npy"
run(outputfile)
