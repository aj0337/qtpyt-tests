from ase.io import read
from gpaw import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.units import Hartree

from gpaw import GPAW
from gpaw.utilities.dos import RestartLCAODOS, fold

atoms = read('bridge.xyz')
calc = GPAW('bridge.gpw', txt=None)
e_f = calc.get_fermi_level()
num_spherical_harmonics_C = 13
num_spherical_harmonics_S = 13
num_spherical_harmonics_H = 5
num_C_atoms = 46
num_S_atoms = 2
num_H_atoms = 46

tot_basis_func = num_spherical_harmonics_C * num_C_atoms + num_spherical_harmonics_S * num_S_atoms + num_spherical_harmonics_H * num_H_atoms

dos = RestartLCAODOS(calc)
energies, weights = dos.get_subspace_pdos(range(tot_basis_func))
e, w = fold(energies * Hartree, weights, 5000, 0.1)

# e, m_s_pdos = dos.get_subspace_pdos([0, 1])
# e, m_s_pdos = fold(e * Hartree, m_s_pdos, 5000, 0.1)
# e, m_p_pdos = dos.get_subspace_pdos([2, 3, 4])
# e, m_p_pdos = fold(e * Hartree, m_p_pdos, 5000, 0.1)
# e, m_d_pdos = dos.get_subspace_pdos([5, 6, 7, 8, 9])
# e, m_d_pdos = fold(e * Hartree, m_d_pdos, 5000, 0.1)

# e, x_s_pdos = dos.get_subspace_pdos([25])
# e, x_s_pdos = fold(e * Hartree, x_s_pdos, 5000, 0.1)
# e, x_p_pdos = dos.get_subspace_pdos([26, 27, 28])
# e, x_p_pdos = fold(e * Hartree, x_p_pdos, 5000, 0.1)

w_max = []
for i in range(len(e)):
    if (-4.5 <= e[i] - e_f <= 4.5):
        w_max.append(w[i])

w_max = np.asarray(w_max)

plt.plot(e - e_f, w, label='Total', c='k', lw=2, alpha=0.7)
# plt.plot(e - e_f, x_s_pdos, label='X-s', c='g', lw=2, alpha=0.7)
# plt.plot(e - e_f, x_p_pdos, label='X-p', c='b', lw=2, alpha=0.7)
# plt.plot(e - e_f, m_s_pdos, label='M-s', c='y', lw=2, alpha=0.7)
# plt.plot(e - e_f, m_p_pdos, label='M-p', c='c', lw=2, alpha=0.7)
# plt.plot(e - e_f, m_d_pdos, label='M-d', c='r', lw=2, alpha=0.7)

plt.axis(ymin=0., ymax=np.max(w_max), xmin=-2, xmax=2, )
plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
plt.ylabel('DOS')
plt.legend(loc=1)
plt.savefig('lcaodos.png')
