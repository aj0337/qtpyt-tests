from __future__ import annotations

import os
import pickle
import numpy as np
from ase.io import read
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition
from qtpyt.tools import remove_pbc

lowdin = False
data_folder = f"./unrelaxed/output/lowdin" if lowdin else f"./unrelaxed/output/no_lowdin"

# Load matrices
H_subdiagonalized, S_subdiagonalized = np.load(f"{data_folder}/hs_los.npy")
H_subdiagonalized = H_subdiagonalized.astype(np.complex128)
S_subdiagonalized = S_subdiagonalized.astype(np.complex128)

# Load node partitioning
nodes = np.load(f"{data_folder}/nodes.npy")

# Load device basis
basis_dict = {"C": 9, "H": 4}
device_atoms = read("./structures/unrelaxed/sorted/scatt.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

# Remove PBC
remove_pbc(device_basis, H_subdiagonalized)
remove_pbc(device_basis, S_subdiagonalized)

# Tridiagonalize
hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
    nodes, H_subdiagonalized[0], S_subdiagonalized[0]
)

# Save results
with open(os.path.join(data_folder, "hs_list_ii.pkl"), "wb") as f:
    pickle.dump(hs_list_ii, f)
with open(os.path.join(data_folder, "hs_list_ij.pkl"), "wb") as f:
    pickle.dump(hs_list_ij, f)
