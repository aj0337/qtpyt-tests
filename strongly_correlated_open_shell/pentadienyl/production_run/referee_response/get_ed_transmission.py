from __future__ import annotations

import os
import pickle

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import expand

# -----------------------------
# MPI
# -----------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# -----------------------------
# User controls
# -----------------------------
U_list = [2.7]   # eV

data_folder = "../output/lowdin"
base_ed_data_folder = "../output/lowdin/ed/referee_response"

de = 0.01
energies = np.arange(-3, 3 + de / 2.0, de).round(7)
eta = 1e-2


# -----------------------------
# Load U-independent inputs once
# -----------------------------
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

nodes = np.load(f"{data_folder}/nodes.npy")

# Initialize the Green's function solver with the tridiagonalized matrices and lead self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

# Transmission function calculation setup (U-independent)
imb = 2  # index of molecule block from the nodes list
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
S_molecule_identity = np.eye(S_molecule.shape[0])
idx_molecule = index_active_region - nodes[imb]  # indices of active region w.r.t molecule


for U_val in U_list:
    ed_data_folder = f"{base_ed_data_folder}/Uppp_{U_val:.3f}"
    if rank == 0:
        os.makedirs(ed_data_folder, exist_ok=True)

    # -----------------------------
    # All “global-dependent” helpers live inside the loop
    # -----------------------------
    class DataSelfEnergy(BaseDataSelfEnergy):
        """Wrapper"""

        def retarded(self, energy):
            return expand(S_molecule_identity, super().retarded(energy), idx_molecule)

    def load_sigma(filename: str) -> DataSelfEnergy:
        return DataSelfEnergy(energies, np.load(filename))

    def run(outputfile: str):
        gd = GridDesc(energies, 1, float)
        T = np.empty(gd.energies.size)
        for e, energy in enumerate(gd.energies):
            T[e] = gf.get_transmission(energy, ferretti=False)

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
            plt.savefig(f"{ed_data_folder}/ET_Uppp_{U_val:.3f}.png", dpi=300)
            plt.close()

    # -----------------------------
    # Load ED self-energy for this U
    # -----------------------------
    ed_self_energy_file = f"{ed_data_folder}/ed_sigma_ppp_{U_val:.3f}.npy"
    if rank == 0 and (not os.path.isfile(ed_self_energy_file)):
        raise FileNotFoundError(
            f"Missing ED self-energy for U={U_val:.3f} eV:\n  {ed_self_energy_file}"
        )

    if rank == 0:
        ed_sigma = load_sigma(ed_self_energy_file)
    else:
        ed_sigma = None

    ed_sigma = comm.bcast(ed_sigma, root=0)

    # -----------------------------
    # Attach ED sigma, compute T(E), detach
    # -----------------------------
    self_energy[2] = ed_sigma
    gf.selfenergies.append((imb, self_energy[2]))

    outputfile = f"{ed_data_folder}/ET_Uppp_{U_val:.3f}.npy"
    run(outputfile)

    gf.selfenergies.pop()

    if rank == 0:
        print(f"[Done] U={U_val:.3f} saved:")
        print(f"  {outputfile}")
        print(f"  {ed_data_folder}/ET_Uppp_{U_val:.3f}.png")
