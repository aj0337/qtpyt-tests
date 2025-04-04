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


data_folder = "output/lowdin"
ed_data_folder = "output/lowdin/beta_38.68"
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
ed_sigma_file = f"{ed_data_folder}/ed_sigma_eta_3e-2.npy"

de = 1e-3
energies = np.arange(-1.0, 1.0, de)
eta = 3e-2
z_ret = energies + 1.0j * eta

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

nodes = [0, 810, 1116, 1278, 1584, 2394]

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

# Add the DMFT self-energy for transmission
if comm.rank == 0:
    ed_sigma = load(ed_sigma_file)
else:
    ed_sigma = None

# Transmission function calculation
imb = 2  # index of molecule block from the nodes list
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
S_molecule_identity = np.eye(S_molecule.shape[0])
idx_molecule = (
    index_active_region - nodes[imb]
)  # indices of active region w.r.t molecule

ed_sigma = comm.bcast(ed_sigma, root=0)
self_energy[2] = ed_sigma
gf.selfenergies.append((imb, self_energy[2]))

outputfile = f"{ed_data_folder}/ed_transmission_fixed_U.npy"
run(outputfile)
gf.selfenergies.pop()
