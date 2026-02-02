from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase.units import Hartree
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.utilities import pack
from gpaw.utilities.blas import rk
from qtpyt.basis import Basis
from qtpyt.lo.tools import lowdin_rotation, rotate_matrix, subdiagonalize_atoms
from qttools.gpaw.los import LOs


def get_species_indices(atoms, species):
    indices = []
    for element in species:
        element_indices = atoms.symbols.search(element)
        indices.extend(element_indices)
    return sorted(indices)


lowdin = True
data_folder = f"./output/lowdin" if lowdin else f"./output/no_lowdin"
# Create the folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

GPWDEVICEDIR = f"./dft/device/"
GPWLEADSDIR = "./dft/leads/"
SUBDIAG_SPECIES = ("C", "H")
# Define the active region within the subdiagonalized species
active = {"C": [3]}

cc_path = Path(GPWDEVICEDIR)
gpwfile = f"{cc_path}/scatt.gpw"

atoms, calc = restart(gpwfile, txt=None)
fermi = calc.get_fermi_level()
nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)

lcao = LCAOwrap(calc)
H_lcao = lcao.get_hamiltonian()
S_lcao = lcao.get_overlap()
H_lcao -= fermi * S_lcao

# Perform subdiagonalization
# subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)

z = basis.atoms.positions[:, 2]
subdiag_indices = np.where(z > (z.min() + (z.max() - z.min()) / 2))[0]

basis_subdiag_region = basis[subdiag_indices]
index_subdiag_region = basis_subdiag_region.get_indices()

extract_active_region = basis_subdiag_region.extract().take(active)
index_active_region = index_subdiag_region[extract_active_region]

Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)

# Positive projection onto p-z AOs
for idx_lo in index_active_region:
    if Usub[idx_lo - 1, idx_lo] < 0.0:  # change sign
        Usub[:, idx_lo] *= -1

H_subdiagonalized = rotate_matrix(H_lcao, Usub)
S_subdiagonalized = rotate_matrix(S_lcao, Usub)

Ulow = lowdin_rotation(H_subdiagonalized, S_subdiagonalized,
                       index_active_region)

# Combine subdiagonalization and Löwdin orthonormalization
rotation_matrix = Usub.dot(Ulow)

# Construct localized orbitals object using the subspace rotation
localized_orbitals = LOs(Usub[:, index_subdiag_region].T, lcao)

# Select only the active (pz-like) orbitals from the rotation matrix
# orbital_coefficients = np.ascontiguousarray(rotation_matrix[:, index_active_region].T)

# Get real-space grid representation of the active orbitals
orbital_grid_functions = localized_orbitals.get_orbitals()

# Compute orbital densities for Coulomb integrals
orbital_densities = orbital_grid_functions ** 2

# Number of active orbitals
num_active_orbitals = index_active_region.shape[0]

# Compute all pair products of orbital densities on the grid
pair_density_products = calc.wfs.gd.zeros(n=num_active_orbitals ** 2)
for pair_index, (i, j) in enumerate(np.ndindex(num_active_orbitals, num_active_orbitals)):
    np.multiply(orbital_grid_functions[i], orbital_grid_functions[j], pair_density_products[pair_index])

# Grid part of the four-index Coulomb tensor (D_pp)
coulomb_tensor_grid = np.zeros((num_active_orbitals ** 2, num_active_orbitals ** 2))
rk(calc.wfs.gd.dv, pair_density_products, 0.0, coulomb_tensor_grid)

# Project real-space orbitals into PAW augmentation basis (P_awi)
projected_coefficients = localized_orbitals.get_projections()

# Construct compressed augmentation projections of all orbital pairs (P_app)
compressed_pair_projections = {}
for atom_index, P_wi in projected_coefficients.items():
    compressed_pair_projections[atom_index] = np.array([
        pack(np.outer(P_wi[i], P_wi[j]))
        for i, j in np.ndindex(num_active_orbitals, num_active_orbitals)
    ])

# Add atomic augmentation corrections to the Coulomb tensor
for atom_index, pair_projection_matrix in compressed_pair_projections.items():
    four_center_integrals = calc.setups[atom_index].four_phi_integrals()
    atomic_correction = pair_projection_matrix @ four_center_integrals @ pair_projection_matrix.T
    coulomb_tensor_grid += atomic_correction

# Convert Coulomb tensor to eV units and reshape to full 4-index tensor
coulomb_tensor = coulomb_tensor_grid * Hartree
coulomb_tensor = coulomb_tensor.reshape(
    num_active_orbitals, num_active_orbitals,
    num_active_orbitals, num_active_orbitals
)

# Extract Hubbard-like U_ij = <ij|V|ji> elements (density-density form)
density_density_U_matrix = np.empty((num_active_orbitals, num_active_orbitals))
for i, j, k, l in np.ndindex(num_active_orbitals, num_active_orbitals,
                             num_active_orbitals, num_active_orbitals):
    density_density_U_matrix[i, j] = coulomb_tensor[i, j, j, i]

# Save interaction matrix for use in DMFT or Hubbard models
np.save(f"{data_folder}/U.npy", density_density_U_matrix)
