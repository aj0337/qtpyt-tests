from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import rotate_matrix, subdiagonalize_atoms, lowdin_rotation
from qtpyt.basis import Basis

# Getting localized orbitals and other prerequisites calculation (runs serially)

def get_species_indices(atoms,species):
    indices = []
    for element in species:
        element_indices = atoms.symbols.search(element)
        indices.extend(element_indices)
    return sorted(indices)

data_folder = './output/lowdin'
# Create the folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

GPWDEVICEDIR = './dft/device/'
GPWLEADSDIR = './dft/leads/'
SUBDIAG_SPECIES = ("C", "N", "H")
# Define the active region within the subdiagonalized species
active = {'C': [3],'N': [3]}
lowdin = True

cc_path = Path(GPWDEVICEDIR)
pl_path = Path(GPWLEADSDIR)
gpwfile = f'{cc_path}/scatt.gpw'

atoms, calc = restart(gpwfile, txt=None)
# fermi = calc.get_fermi_level()
fermi = 4.265
nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)

lcao = LCAOwrap(calc)
H_lcao = lcao.get_hamiltonian()
S_lcao = lcao.get_overlap()
H_lcao -= fermi * S_lcao

# Perform subdiagonalization
subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)

basis_subdiag_region = basis[subdiag_indices]
index_subdiag_region = basis_subdiag_region.get_indices()

extract_active_region = basis_subdiag_region.extract().take(active)
index_active_region = index_subdiag_region[extract_active_region]

np.save(f"{data_folder}/index_active_region.npy",index_active_region)

Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)
H_subdiagonalized = rotate_matrix(H_lcao, Usub)
S_subdiagonalized = rotate_matrix(S_lcao, Usub)

if lowdin:
    Ulow = lowdin_rotation(H_subdiagonalized, S_subdiagonalized, index_active_region)

    H_subdiagonalized = rotate_matrix(H_subdiagonalized, Ulow)
    S_subdiagonalized = rotate_matrix(S_subdiagonalized, Ulow)

    # Rotate matrices
    H_subdiagonalized = H_subdiagonalized[None, ...]
    S_subdiagonalized = S_subdiagonalized[None, ...]

    np.save(f"{data_folder}/hs_los_lowdin.npy", (H_subdiagonalized,S_subdiagonalized))

else:
    H_subdiagonalized = H_subdiagonalized[None, ...]
    S_subdiagonalized = S_subdiagonalized[None, ...]

    np.save(f"{data_folder}/hs_los_no_lowdin.npy", (H_subdiagonalized,S_subdiagonalized))
