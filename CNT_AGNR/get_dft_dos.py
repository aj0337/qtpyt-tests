from __future__ import annotations

import pickle

import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def run(outputfile_pdos, outputfile_dos, run_dos=False):
    gd = GridDesc(energies, 1, float)
    pdos = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        pdos[e] = gf.get_pdos(energy)

    if run_dos:
        dos = np.empty(gd.energies.size)
        for e, energy in enumerate(gd.energies):
            dos[e] = gf.get_dos(energy)
        dos = gd.gather_energies(dos)
        if comm.rank == 0:
            np.save(outputfile_dos, (energies, dos.real))

    pdos = gd.gather_energies(pdos)

    if comm.rank == 0:
        np.save(outputfile_pdos, (energies, pdos.real))


data_folder = "./output/lowdin"
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

de = 0.01
energies = np.arange(-3, 3 + de, de)
eta = 1e-3

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

nodes = [0, 5616, 10192, 15296]

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="spectral",
    eta=eta,
)

outputfile_pdos = f"{data_folder}/dft_pdos.npy"
outputfile_dos = f"{data_folder}/dft_dos.npy"
run(outputfile_pdos, outputfile_dos, run_dos=True)
