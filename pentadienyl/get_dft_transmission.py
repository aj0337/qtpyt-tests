from __future__ import annotations

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data_folder = "./output/lowdin"
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

de = 0.01
energies = np.arange(-1, 1 + de / 2.0, de).round(7)

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

etas = [1e-3, 1e-2, 1, 10]
transmissions = {}

for eta in etas:
    gf = greenfunction.GreenFunction(
        hs_list_ii,
        hs_list_ij,
        [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
        solver="dyson",
        eta=eta,
    )

    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        transmissions[eta] = T.real
        np.save(f"{data_folder}/dft_transmission_eta{eta:.0e}.npy", (energies, T.real))

if comm.rank == 0:
    plt.figure()
    for eta, T in transmissions.items():
        plt.plot(energies, T, label=f"eta={eta:.0e}")
    plt.yscale("log")
    plt.xlim(-1., 1.)
    plt.ylim(1e-5, 1)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Transmission")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{data_folder}/transmission_vs_eta.png", dpi=300)
    plt.close()
