import numpy as np
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from qtpyt.basis import Basis
from qtpyt.lo.tools import lowdin_rotation, rotate_matrix, subdiagonalize_atoms

lowdin = True

root_dir_path = '/scratch/snx3000/ajayaraj/tests/AuBDA'
gpwfile = f'{root_dir_path}/dft/device_dzp/scatt.gpw'
hsfile = 'hs_lolw_k.npy'
hfile = 'h.npy'

atoms, calc = restart(gpwfile, txt=None)
lcao = LCAOwrap(calc)

fermi = calc.get_fermi_level()
H = lcao.get_hamiltonian()
S = lcao.get_overlap()
H -= fermi * S

nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)

x = basis.atoms.positions[:, 0]
scatt = np.where((x > 12.9) & (x < 22))[0]
active = {'C': [0,1,2,3]}
basis_p = basis[scatt]
index_p = basis_p.get_indices()
index_c = basis_p.extract().take(active)
Usub, eig = subdiagonalize_atoms(basis, H, S, a=scatt)

# Positive projection onto p-z AOs
for idx_lo in index_p[index_c]:
    if Usub[idx_lo - 1, idx_lo] < 0.:  # change sign
        Usub[:, idx_lo] *= -1

H = rotate_matrix(H, Usub)
S = rotate_matrix(S, Usub)

if lowdin:
    Ulow = lowdin_rotation(H, S, index_p[index_c])

    H = rotate_matrix(H, Ulow)
    S = rotate_matrix(S, Ulow)

    U = Usub.dot(Ulow)

np.save('idx_los.npy', index_p[index_c])
np.save(hsfile, (H[None, ...], S[None, ...]))
