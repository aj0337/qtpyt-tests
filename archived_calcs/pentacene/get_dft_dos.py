from __future__ import annotations

import os
import pickle

import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm
import matplotlib.pyplot as plt
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import ProjectedGreenFunction

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run(outputfile, gf_object):
    gd = GridDesc(energies, 1, float)
    dos = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        dos[e] = gf_object.get_dos(energy)

    dos = gd.gather_energies(dos)

    if comm.rank == 0:
        np.save(outputfile, (energies, dos.real))
        plt.figure()
        plt.plot(energies, dos)
        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS")
        plt.tight_layout()
        plt.savefig(outputfile.replace(".npy", ".png"), dpi=300)
        plt.close()

# === Setup ===
data_folder = "./output/lowdin"
dft_data_folder = f"{data_folder}/dft"
os.makedirs(dft_data_folder, exist_ok=True)

index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

de = 0.01
energies = np.arange(-3, 3 + de / 2.0, de).round(7)
eta = 1e-4

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

# === Green's Function Setup ===
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

# === Choose full or projected DOS ===
use_projected_dos = True  # Set to False for total DOS

gf_object = ProjectedGreenFunction(gf, index_active_region) if use_projected_dos else gf
filename = "Evdos_pz.npy" if use_projected_dos else "Evdos_total.npy"
outputfile = os.path.join(dft_data_folder, filename)

run(outputfile, gf_object)
