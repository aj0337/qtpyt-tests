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
output_dir = "output/cube_files"
os.makedirs(output_dir, exist_ok=True)

input_dir = "dft/device"

# Load atomic structure and calculator
atoms = read(f'{input_dir}/scatt.xyz')
calc = GPAW(f'{input_dir}/scatt.gpw', txt=None)
lcao = LCAOwrap(calc)

# Flags to control output generation
los_cube = True
lowdin_cube = True
lcao_cube = True
ao_cube = True

active = {"C": [3]}

# Common data for los and lowdin_cube
if los_cube or lowdin_cube:
    E_fermi = calc.get_fermi_level()
    H_lcao = lcao.get_hamiltonian()
    S_lcao = lcao.get_overlap()
    H_lcao -= E_fermi * S_lcao
    nao_a = np.array([setup.nao for setup in calc.wfs.setups])
    basis = Basis(atoms, nao_a)
    SUBDIAG_SPECIES = ("C", "H", "N")
    subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)
    basis_subdiag_region = basis[subdiag_indices]
    index_subdiag_region = basis_subdiag_region.get_indices()

# Generate los cube files
if los_cube:
    folder_path = os.path.join(output_dir, 'los')
    os.makedirs(folder_path, exist_ok=True)
    Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)
    los = LOs(Usub[:, index_subdiag_region].T, lcao)

    # Generate cube files for specified orbitals in orbital_map
    for key, values in active.items():
        orbital_indices = basis_subdiag_region.extract().take({key: values})
        for w, w_G in enumerate(los.get_orbitals(orbital_indices)):
            write(f"{folder_path}/los_{key}{w}_orbital{values}.cube", atoms, data=w_G)

# Generate lowdin cube files
if lowdin_cube:
    folder_path = os.path.join(output_dir, 'lowdin')
    os.makedirs(folder_path, exist_ok=True)
    Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)
    active_indices = basis_subdiag_region.extract().take(active)
    index_active_region = index_subdiag_region[active_indices]
    Ulow = lowdin_rotation(rotate_matrix(H_lcao, Usub), rotate_matrix(S_lcao, Usub), index_active_region)
    U = Usub.dot(Ulow)
    los = LOs(U.T, lcao)

    # Generate cube files for specified orbitals in orbital_map
    for key, values in active.items():
        orbital_indices = basis_subdiag_region.extract().take({key: values})
        for w, w_G in enumerate(los.get_orbitals(orbital_indices)):
            write(f"{folder_path}/lowdin_{key}{w}_orbital{values}.cube", atoms, data=w_G)

# Generate lcao cube files for orbitals around Fermi level
if lcao_cube:
    folder_path = os.path.join(output_dir, 'lcao')
    os.makedirs(folder_path, exist_ok=True)

    # Get HOMO and LUMO energies and indices
    homo_energy, lumo_energy = calc.get_homo_lumo()
    band_energies = calc.get_eigenvalues()
    homo_band_index = np.argmin(np.abs(band_energies - homo_energy))
    lumo_band_index = np.argmin(np.abs(band_energies - lumo_energy))

    nbands_around_fermi = 1
    bands_around_fermi = range(homo_band_index - nbands_around_fermi,
                               lumo_band_index + nbands_around_fermi + 1)

    for band in bands_around_fermi:
        wf = calc.get_pseudo_wave_function(band=band)

        # Define filename based on band position relative to HOMO and LUMO
        if band == homo_band_index:
            fname = f"{folder_path}/band_homo.cube"
        elif band == lumo_band_index:
            fname = f"{folder_path}/band_lumo.cube"
        elif band < homo_band_index:
            diff = homo_band_index - band
            fname = f"{folder_path}/band_homo-{diff}.cube"
        else:
            diff = band - lumo_band_index
            fname = f"{folder_path}/band_lumo+{diff}.cube"

        write(fname, atoms, data=wf)

# Generate atomic orbital cube files
if ao_cube:
    folder_path = os.path.join(output_dir, 'ao')
    os.makedirs(folder_path, exist_ok=True)
    orbital_indices = range(4)
    w_wG = lcao.get_orbitals(indices=orbital_indices)
    for w, w_G in enumerate(w_wG):
        write(f"{folder_path}/ao_orbital{w}.cube", atoms, data=w_G)

print("Cube files generated successfully in the output directory.")
