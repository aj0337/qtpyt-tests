import numpy as np
from ase.io import write
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import lowdin_rotation, rotate_matrix, subdiagonalize_atoms
from qttools.gpaw.los import LOs
from pathlib import Path

lowdin = True
cc_path = Path('../../dft/device/')
gpwfile = f'{cc_path}/scatt.gpw'
hsfile = 'hs_lolw_k.npy'

atoms, calc = restart(gpwfile, txt=None)
lcao = LCAOwrap(calc)

fermi = calc.get_fermi_level()
H = lcao.get_hamiltonian()
S = lcao.get_overlap()
H -= fermi * S

nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)

x = basis.atoms.positions[:, 0]
scatt = np.where((x > 12.9) & (x < 19))[0]

basis_p = basis[scatt]
index_p = basis_p.get_indices()
Usub, eig = subdiagonalize_atoms(basis, H, S, a=scatt)
for idx_lo in index_p:
    if Usub[idx_lo - 1, idx_lo] < 0.:  # change sign
        Usub[:, idx_lo] *= -1

H = rotate_matrix(H, Usub)
S = rotate_matrix(S, Usub)

if lowdin:
    Ulow = lowdin_rotation(H, S, index_p)

    H = rotate_matrix(H, Ulow)
    S = rotate_matrix(S, Ulow)

    U = Usub.dot(Ulow)

np.save('idx_los.npy', index_p)
np.save(hsfile, (H[None, ...], S[None, ...]))
