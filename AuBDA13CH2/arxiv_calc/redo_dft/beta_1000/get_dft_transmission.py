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
        return expand(S_molecule, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(z_ret, np.load(filename))


def run(outputfile):
    gd = GridDesc(z_ret, 1, float)
    T = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (z_ret, T.real))


data_folder = "./output"
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
z_ret = np.load(f"{data_folder}/retarded_energies.npy")
eta = z_ret.imag[0]
beta = 1000

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

nodes = [0, 810, 1116, 1278, 1584, 2394]
# Define energy range and broadening factor for the Green's function calculation

# Transmission function calculation
imb = 2  # index of molecule block from the nodes list
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
idx_molecule = (
    index_active_region - nodes[imb]
)  # indices of active region w.r.t molecule

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
    mu=0.0,
    kt=1 / beta,
)

# Transmission function for DFT
outputfile = f"{data_folder}/dft_transmission.npy"
run(outputfile)
