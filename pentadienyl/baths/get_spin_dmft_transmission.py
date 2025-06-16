from __future__ import annotations

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import expand

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_molecule_identity, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


def run(outputfile):
    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies, T.real))


def plot_transmission(dmft_data_folder):
    E_up, T_up = np.load(f"{dmft_data_folder}/dmft_transmission_up.npy")
    E_dw, T_dw = np.load(f"{dmft_data_folder}/dmft_transmission_dw.npy")

    plt.figure()
    plt.plot(E_up, T_up, label="DMFT spin up", color="red")
    plt.plot(E_dw, T_dw, label="DMFT spin down", color="blue")
    plt.yscale("log")
    plt.legend()
    plt.xlim(-2.5, 2.5)
    plt.ylabel("Transmission")
    plt.xlabel("Energy (eV)")
    plt.ylim(bottom=1e-4)

    plot_path = f"{dmft_data_folder}/dmft_transmission.png"
    plt.savefig(plot_path)
    plt.close()


de = 0.01
energies = np.arange(-3, 3 + de / 2.0, de).round(7)
eta = 3e-2
# betas = [30, 40] + list(np.arange(50, 175, 25)) + list(np.arange(200, 2050, 50))
beta = 1000
field = 5e-1
adjust_mus = [True, False]

data_folder = "../output/lowdin"
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

nodes = [0, 810, 1116, 1252, 1558, 2368]

imb = 2
S_molecule = hs_list_ii[imb][1]
S_molecule_identity = np.eye(S_molecule.shape[0])
idx_molecule = index_active_region - nodes[imb]

gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

temperature_data_folder = f"{data_folder}/beta_{beta}"
for adjust_mu in adjust_mus:
    print(
        f"Starting spin resolved DMFT calculation for beta = {beta} and adjust mu {adjust_mu}",
        flush=True,
    )
    dmft_data_folder = f"{temperature_data_folder}/dmft/eta_{eta}/spin/field_{field}/adjust_mu_{adjust_mu}"

    for spin, spin_label in enumerate(["up", "dw"]):
        dmft_sigma_file = f"{dmft_data_folder}/sigma_dmft_{spin_label}.npy"
        if comm.rank == 0:
            dmft_sigma = load(dmft_sigma_file)
        else:
            dmft_sigma = [None]

        dmft_sigma = comm.bcast(dmft_sigma, root=0)
        self_energy[2] = dmft_sigma
        gf.selfenergies.append((imb, self_energy[2]))

        outputfile = f"{dmft_data_folder}/dmft_transmission_{spin_label}.npy"
        run(outputfile)
        gf.selfenergies.pop()

    if comm.rank == 0:
        plot_transmission(dmft_data_folder)
