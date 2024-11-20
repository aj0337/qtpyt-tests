from __future__ import annotations
import os
import sys
import pickle
import numpy as np
from qtpyt.block_tridiag import greenfunction
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.projector import expand
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper for self-energy."""

    def retarded(self, energy):
        return expand(S_molecule, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


def run(outputfile):
    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    print(gd.energies.shape, flush=True)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies, T.real))


# Parse command-line arguments
if len(sys.argv) != 5:
    raise ValueError("Please provide nbath, U, adjust_mu, and double_counting as command-line arguments.")
nbath = int(sys.argv[1])
U = float(sys.argv[2])
adjust_mu = sys.argv[3].lower() == 'true'
double_counting = sys.argv[4].lower() == 'true'

# Define paths and directories based on arguments
data_folder = "../../output/compute_run"
output_folder = f"../../output/compute_run/toy_model2/nbaths_{nbath}_U_{U}_DC_{'with_dc' if double_counting else 'without_dc'}_{'adjust_mu' if adjust_mu else 'no_adjust_mu'}"
sigma_data_folder = f"../../output/compute_run/toy_model2/nbaths_{nbath}_U_{U}_DC_{'with_dc' if double_counting else 'without_dc'}_{'adjust_mu' if adjust_mu else 'no_adjust_mu'}"
os.makedirs(output_folder, exist_ok=True)

index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
dmft_sigma_file = f"{sigma_data_folder}/dmft_sigma.npy"
energies = np.load(f"{data_folder}/energies.npy")

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

nodes = [0, 810, 1116, 1278, 1584, 2394]
eta = 3e-2  # Broadening factor for Green's function calculation

# Transmission function calculation
imb = 2  # index of molecule block from the nodes list
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
idx_molecule = index_active_region - nodes[imb]  # indices of active region w.r.t molecule

# Initialize the Green's function solver with tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

# Transmission function for DFT
outputfile_dft = f"{output_folder}/dft_transmission.npy"
run(outputfile_dft)

# Load and configure DMFT self-energy if enabled
if os.path.exists(dmft_sigma_file):
    dmft_sigma = load(dmft_sigma_file) if comm.rank == 0 else None
    dmft_sigma = comm.bcast(dmft_sigma, root=0)

    # Add DMFT self-energy to Green's function and calculate transmission
    self_energy[2] = dmft_sigma
    gf.selfenergies.append((imb, self_energy[2]))
    outputfile_dmft = f"{output_folder}/dmft_transmission.npy"
    run(outputfile_dmft)
    gf.selfenergies.pop()

else:
    if rank == 0:
        print(f"DMFT sigma file {dmft_sigma_file} not found. Skipping DMFT transmission calculation.")
