from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
from gpaw.lcao.tools import remove_pbc
from qtpyt.basis import Basis
from ase.io import read
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.hybridization import Hybridization
from scipy.linalg import eigvalsh
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc


from ase.io import read

import os

data_folder = '../../output/compute_run'
output_folder = '../../output/compute_run/toy_model5'
# Create the folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

H_subdiagonalized, S_subdiagonalized = np.load(f"{data_folder}/hs_los_lowdin.npy")
H_subdiagonalized = H_subdiagonalized.astype(np.complex128)
S_subdiagonalized = S_subdiagonalized.astype(np.complex128)

GPWDEVICEDIR = '../../dft/device/'
GPWLEADSDIR = '../../dft/leads/'

cc_path = Path(GPWDEVICEDIR)
pl_path = Path(GPWLEADSDIR)

H_leads_lcao, S_leads_lcao = np.load(pl_path / 'hs_pl_k.npy')

basis_dict = {'Au': 9, 'H': 5, 'C': 13, 'N': 13}

leads_atoms = read(pl_path / 'leads.xyz')
leads_basis = Basis.from_dictionary(leads_atoms, basis_dict)

device_atoms = read(cc_path / 'scatt.xyz')
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

nodes = [0,810,1116,1278,1584,2394]

# Define energy range and broadening factor for the Green's function calculation
energies = np.arange(-10, 10, 0.01)
eta = 3e-2

# Define the number of repetitions (Nr) and unit cell repetition in the leads
Nr = (1, 5, 3)
unit_cell_rep_in_leads = (5, 5, 3)

# Define parameters for matsubara grid
ne = 3000
beta = 1000.
matsubara_energies = 1.j * (2 * np.arange(ne) + 1) * np.pi / beta

# Prepare the k-points and matrices for the leads (Hamiltonian and overlap matrices)
kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
    H_leads_lcao, S_leads_lcao, unit_cell_rep_in_leads, align=(0, H_subdiagonalized[0, 0, 0]))

# Remove periodic boundary conditions (PBC) from the device Hamiltonian and overlap matrices
remove_pbc(device_basis, H_subdiagonalized)
remove_pbc(device_basis, S_subdiagonalized)

# Initialize self-energy list for left and right leads
self_energy = [None, None, None]
self_energy[0] = PrincipalSelfEnergy(kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr)
self_energy[1] = PrincipalSelfEnergy(kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr, id='right')

# Rotate the couplings for the leads based on the specified basis and repetition Nr
rotate_couplings(leads_basis, self_energy[0], Nr)
rotate_couplings(leads_basis, self_energy[1], Nr)

# Tridiagonalize the device Hamiltonian and overlap matrices based on the partitioned nodes
hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(nodes, H_subdiagonalized[0], S_subdiagonalized[0])
del H_subdiagonalized, S_subdiagonalized

# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(hs_list_ii,
                                hs_list_ij,
                                [(0, self_energy[0]),
                                (len(hs_list_ii) - 1, self_energy[1])],
                                solver='dyson',
                                eta=eta)


# Define active region and the Green's function for the active region

index_active_region = np.load(f"{data_folder}/index_active_region.npy")

gfp = ProjectedGreenFunction(gf, index_active_region)
hyb = Hybridization(gfp)


n_A = len(index_active_region)
gd = GridDesc(energies, n_A, complex)
HB = gd.empty_aligned_orbs()

for e, energy in enumerate(gd.energies):
    HB[e] = hyb.retarded(energy)

filename = os.path.join(output_folder, 'hybridization.bin')
gd.write(HB,filename)
del HB

# Matsubara
gf.eta = 0.
assert self_energy[0].eta == 0.
assert self_energy[1].eta == 0.

mat_gd = GridDesc(matsubara_energies, n_A, complex)
HB_mat = mat_gd.empty_aligned_orbs()

for e, energy in enumerate(mat_gd.energies):
    HB_mat[e] = hyb.retarded(energy)

# Save the Matsubara hybrid data
filename = os.path.join(output_folder, 'matsubara_hybridization.bin')
mat_gd.write(HB_mat, filename)
del HB_mat

# if comm.rank == 0:

#     save_path = os.path.join(data_folder, 'hs_list_ii.pkl')
#     with open(save_path, 'wb') as f:
#         pickle.dump(hs_list_ii, f)
#     save_path = os.path.join(data_folder, 'hs_list_ij.pkl')
#     with open(save_path, 'wb') as f:
#         pickle.dump(hs_list_ij, f)

#     # np.save(os.path.join(data_folder, 'energies.npy'), energies + 1.j * eta)
#     # # Effective Hamiltonian
#     # np.save(os.path.join(data_folder, 'hamiltonian.npy'), hyb.H)
#     # np.save(os.path.join(data_folder, 'occupancies.npy'), get_ao_charge(gfp))
#     # np.save(os.path.join(data_folder, 'matsubara_energies.npy'), matsubara_energies)
#     np.save(os.path.join(data_folder, 'self_energy.npy'), self_energy)
#     Heff = (hyb.H + hyb.retarded(0.)).real
#     np.save(os.path.join(data_folder, 'effective_hamiltonian.npy'), Heff)
#     np.save(os.path.join(data_folder, 'eigvals_Heff.npy'), eigvalsh(Heff, gfp.S))
