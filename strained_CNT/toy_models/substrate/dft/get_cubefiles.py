import os
import numpy as np
from ase.io import read, write
from gpaw import GPAW
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import subdiagonalize_atoms, lowdin_rotation, rotate_matrix
from qttools.gpaw.los import LOs


def get_species_indices(atoms, species):
    indices = []
    for element in species:
        element_indices = atoms.symbols.search(element)
        indices.extend(element_indices)
    return sorted(indices)


# Define output directory
output_dir = "output/cube_files/"
os.makedirs(output_dir, exist_ok=True)

input_dir = "./"

# Load atomic structure and calculator
atoms = read(f"{input_dir}/CNT_Pt.xyz")
calc = GPAW(f"{input_dir}/leads.gpw", txt=None)
lcao = LCAOwrap(calc)

# Flags to control output generation
los_cube = True
lowdin_cube = True
lcao_cube = True
ao_cube = True

active = {"Pt": [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]}

# Common data for los and lowdin_cube
if los_cube or lowdin_cube:
    E_fermi = calc.get_fermi_level()
    H_lcao = lcao.get_hamiltonian()
    S_lcao = lcao.get_overlap()
    H_lcao -= E_fermi * S_lcao
    nao_a = np.array([setup.nao for setup in calc.wfs.setups])
    basis = Basis(atoms, nao_a)
    SUBDIAG_SPECIES = ("C", "Pt")
    subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)
    basis_subdiag_region = basis[subdiag_indices]
    index_subdiag_region = basis_subdiag_region.get_indices()

# Generate los cube files
if los_cube:
    folder_path = os.path.join(output_dir, "los")
    os.makedirs(folder_path, exist_ok=True)
    Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)
    los = LOs(Usub[:, index_subdiag_region].T, lcao)

    # Generate cube files for specified orbitals in orbital_map
    for key, values in active.items():
        for value in values:
            orbital_indices = basis_subdiag_region.extract().take({key: value})
            for w, w_G in enumerate(los.get_orbitals(orbital_indices)):
                write(
                    f"{folder_path}/los_{key}{w}_orbital{value}.cube", atoms, data=w_G
                )
