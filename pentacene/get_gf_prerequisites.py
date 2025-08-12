from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from ase.io import read
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings

lowdin = True
data_folder = f"./output/lowdin" if lowdin else f"./output/no_lowdin"

H_subdiagonalized, S_subdiagonalized = np.load(f"{data_folder}/hs_los.npy")

H_subdiagonalized = H_subdiagonalized.astype(np.complex128)
S_subdiagonalized = S_subdiagonalized.astype(np.complex128)

GPWDEVICEDIR = "./dft/device"
GPWLEADSDIR = "./dft/leads/"

cc_path = Path(GPWDEVICEDIR)
pl_path = Path(GPWLEADSDIR)

H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_leads_k.npy")

basis_dict = {"C": 9, "H": 4}

leads_atoms = read(pl_path / "leads_sorted.xyz")
leads_basis = Basis.from_dictionary(leads_atoms, basis_dict)

device_atoms = read(cc_path / "scatt_sorted.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

nodes = np.load(f"{data_folder}/nodes.npy")

unit_cell_rep_in_leads = (3, 1, 1)

# Prepare the k-points and matrices for the leads (Hamiltonian and overlap matrices)
h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = map(
    lambda m: m[0],
    prepare_leads_matrices(
        H_leads_lcao,
        S_leads_lcao,
        unit_cell_rep_in_leads,
        align=(0, H_subdiagonalized[0, 0, 0]),
    )[1:],
)

# Remove periodic boundary conditions (PBC) from the device Hamiltonian and overlap matrices
remove_pbc(device_basis, H_subdiagonalized)
remove_pbc(device_basis, S_subdiagonalized)

# Initialize self-energy list for left and right leads
self_energy = [None, None, None]
self_energy[0] = LeadSelfEnergy((h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij))
self_energy[1] = LeadSelfEnergy(
    (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), id="right"
)

# Tridiagonalize the device Hamiltonian and overlap matrices based on the partitioned nodes
hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
    nodes, H_subdiagonalized[0], S_subdiagonalized[0]
)
del H_subdiagonalized, S_subdiagonalized

save_path = os.path.join(data_folder, "hs_list_ii.pkl")
with open(save_path, "wb") as f:
    pickle.dump(hs_list_ii, f)
save_path = os.path.join(data_folder, "hs_list_ij.pkl")
with open(save_path, "wb") as f:
    pickle.dump(hs_list_ij, f)
np.save(os.path.join(data_folder, "self_energy.npy"), self_energy)
