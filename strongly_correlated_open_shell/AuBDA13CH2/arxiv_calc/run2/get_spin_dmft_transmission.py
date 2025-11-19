from __future__ import annotations

import pickle
import numpy as np
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


def load_spin_dmft_sigma(filename):
    """Load spin-resolved DMFT self-energy and return two separate DataSelfEnergy objects"""
    sigma = np.load(filename)
    sigma = np.transpose(sigma, (0, 3, 1, 2))
    return [
        DataSelfEnergy(energies, sigma[nspin, ...]) for nspin in range(sigma.shape[0])
    ]


def run(outputfile):
    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies, T.real))


data_folder = "./output/lowdin"
dmft_data_folder = "./output/lowdin/dmft/spin"
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
dmft_sigma_file = f"{dmft_data_folder}/dmft_sigma.npy"

de = 0.01
energies = np.arange(-3, 3 + de / 2.0, de).round(7)
eta = 1e-3
z_ret = energies + 1.0j * eta

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

nodes = [0, 810, 1116, 1278, 1584, 2394]

gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

imb = 2
S_molecule = hs_list_ii[imb][1]
S_molecule_identity = np.eye(S_molecule.shape[0])
idx_molecule = index_active_region - nodes[imb]

if comm.rank == 0:
    dmft_sigma = load_spin_dmft_sigma(dmft_sigma_file)
else:
    dmft_sigma = [None, None]

dmft_sigma = comm.bcast(dmft_sigma, root=0)

for spin, spin_label in enumerate(["up", "down"]):
    self_energy[2] = dmft_sigma[spin]
    gf.selfenergies.append((imb, self_energy[2]))

    outputfile = f"{dmft_data_folder}/dmft_transmission_{spin_label}.npy"
    run(outputfile)
    gf.selfenergies.pop()
