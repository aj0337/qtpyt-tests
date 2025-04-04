import numpy as np
from ase.io import write
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import lowdin_rotation, rotate_matrix, subdiagonalize_atoms
from qttools.gpaw.los import LOs
import os


def create_map(atom_list, nao_a):
    mapping = {}
    for atom, nao in zip(atom_list, nao_a):
        mapping[atom] = nao
    return mapping

def validate_spherical_harmonic_index(mapping, active_space):
    for element, nao in active_space.items():
        if element not in mapping:
            raise ValueError(f"No mapping found for element {element}")
        else:
            max_nao = mapping[element]
            for index in nao:
                if index >= max_nao:
                    raise ValueError(f"Spherical harmonic index {index} can't be larger than the number of spherical harmonics {max_nao} for element {element}")

lowdin = True

root_dir_path = '/scratch/snx3000/ajayaraj/tests/AuCO/'
gpwfile = f'{root_dir_path}/dft/device/scatt.gpw'
hsfile = 'hs_lolw_k.npy'

atoms, calc = restart(gpwfile, txt=None)
lcao = LCAOwrap(calc)

fermi = calc.get_fermi_level()
H = lcao.get_hamiltonian()
S = lcao.get_overlap()
H -= fermi * S

nao_per_atom = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_per_atom)

y = basis.atoms.positions[:,1]
scattering_region = np.where(y>5)[0]
spherical_harmonic_index = [3]
active_space = [
    {'C':spherical_harmonic_index},
                ]

atom_list = atoms.get_chemical_symbols()
mapping = create_map(atom_list, nao_per_atom)
validate_spherical_harmonic_index(mapping, active_space)

basis_scattering_region = basis[scattering_region]
index_scattering_region = basis_scattering_region.get_indices()
Usub, eig = subdiagonalize_atoms(basis, H, S, a=scattering_region)
index_active_space = basis_scattering_region.extract().take(active_space)
# Positive projection onto p-z AOs
for idx_lo in index_scattering_region[index_active_space]:
    if Usub[idx_lo - 1, idx_lo] < 0.:  # change sign
        Usub[:, idx_lo] *= -1

H = rotate_matrix(H, Usub)
S = rotate_matrix(S, Usub)

if lowdin:
    Ulow = lowdin_rotation(H, S, index_scattering_region[index_active_space])

    H = rotate_matrix(H, Ulow)
    S = rotate_matrix(S, Ulow)

    U = Usub.dot(Ulow)

np.save('idx_los.npy', index_scattering_region[index_active_space])
np.save(hsfile, (H[None,...],S[None,...]))
