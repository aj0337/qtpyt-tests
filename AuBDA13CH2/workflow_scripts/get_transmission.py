from __future__ import annotations
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
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_molecule, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))

def run(outputfile):

    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    print(gd.energies.shape,flush=True)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies,T.real))

data_folder = '../output/compute_run'
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy",allow_pickle=True)
dmft_sigma_file = f"{data_folder}/dmft_sigma.npy"
energies = np.load(f'{data_folder}/energies.npy')

with open(f"{data_folder}/hs_list_ii.pkl", 'rb') as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", 'rb') as f:
    hs_list_ij = pickle.load(f)

nodes = [0,810,1116,1278,1584,2394]
# Define energy range and broadening factor for the Green's function calculation
eta = 1e-3

# Transmission function calculation
imb = 2  # index of molecule block from the nodes list
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
idx_molecule = index_active_region - nodes[imb]  # indices of active region w.r.t molecule

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(hs_list_ii,
                                hs_list_ij,
                                [(0, self_energy[0]),
                                (len(hs_list_ii) - 1, self_energy[1])],
                                solver='dyson',
                                eta=eta)


# # Transmission function for DFT
# outputfile = f"{data_folder}/dft_transmission.npy"
# run(outputfile)

# Add the DMFT self-energy for transmission
if comm.rank == 0:
    dmft_sigma = load(dmft_sigma_file)
else:
    dmft_sigma = None

dmft_sigma = comm.bcast(dmft_sigma, root=0)
self_energy[2] = dmft_sigma
gf.selfenergies.append((imb, self_energy[2]))

outputfile = f"{data_folder}/dmft_transmission.npy"
run(outputfile)
gf.selfenergies.pop()
