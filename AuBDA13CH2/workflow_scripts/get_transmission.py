from __future__ import annotations
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

    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies,T.real))

data_folder = '../output/compute_run'
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
hs_list_ii = np.load(f"{data_folder}/hs_list_ii.npy")
hs_list_ij = np.load(f"{data_folder}/hs_list_ij.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy")
dmft_sigma_file = f"{data_folder}/dmft_sigma.npy"

nodes = [0,810,1116,1278,1584,2394]
# Define energy range and broadening factor for the Green's function calculation
de = 0.2
energies = np.arange(-3., 3. + de / 2., de).round(7)
eta = 1e-3

# Transmission function calculation
imb = 2  # index of molecule block from the nodes list
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
idx_molecule = index_active_region - nodes[imb]  # indices of active region w.r.t molecule

# Transmission function for DFT
outputfile = f"{data_folder}/dft_transmission.npy"
run(outputfile)

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(hs_list_ii,
                                hs_list_ij,
                                [(0, self_energy[0]),
                                (len(hs_list_ii) - 1, self_energy[1])],
                                solver='dyson',
                                eta=eta)

# Add the DMFT self-energy for transmission
if comm.rank == 0:
    self_energy[2] = load(dmft_sigma_file)
    gf.selfenergies.append((imb, self_energy[2]))

    # Transmission function with DMFT
    outputfile = f"{data_folder}/dmft_transmission.npy"
    run(outputfile)
    gf.selfenergies.pop()
