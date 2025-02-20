#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np


from gpaw.coulomb import Coulomb
from gpaw import GPAW
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.utilities import pack
from gpaw.utilities.tools import tri2full
from gpaw.utilities.blas import rk, gemm

from ase.io import write, read
from ase.units import Hartree

from qtpyt._deprecated.analysis.tk_analysis import get_orbitals
from qtpyt.basis import Basis
from qtpyt.lo import tools as lot


calc = GPAW('scatt.gpw')
lcao = LCAOwrap(calc)


basis_dict = {'Au':9,'N':13,'H':5,'S':13,'C':13,'O':13}
atoms = calc.atoms
basis = Basis.from_dictionary(atoms, basis_dict)

h, s = map(lambda m: m[0], np.load('hs_scatt_k.npy'))

act = np.where(np.isin(atoms.symbols,['C','N','S','O']))[0]
iact = basis[act].get_indices().reshape(act.size,13)
ilopz = iact[:,3].copy()#[3::13]
iaopz = iact[:,2].copy()#[2::13]
iem = np.setdiff1d(basis.nao, ilopz)

Us, eig = lot.subdiagonalize_atoms(basis, h, s, a=act)
flip_sign = np.where(Us[iaopz,ilopz]<0.)[0]
for i in flip_sign:
    Us[np.ix_(iact[i],iact[i])] *= -1

h = lot.rotate_matrix(h, Us)
s = lot.rotate_matrix(s, Us)

Ul = lot.lowdin_rotation(h, s, ilopz)
U = Us.dot(Ul)

c_wM = np.ascontiguousarray(U[:,ilopz].T)

w_wG = get_orbitals(calc, c_wM)

write(f'scatt_pzs.cube', atoms, data=w_wG.sum(0))

n_wG = w_wG ** 2

def get_coulomb(coulomb, n_wG):
    """Get coulomb pair integral."""
    nw = len(n_wG)
    U = np.empty((nw, nw))
    for i, j in np.ndindex(nw, nw):
        U[i,j] = coulomb.coulomb(n_wG[i], n_wG[j])
    return U

#coulomb = Coulomb(calc.wfs.gd)
#U = get_coulomb(coulomb, n_wG)
#np.savetxt('U.txt',U)

def get_exchange(coulomb, w_wG):
    """Get exchange integral."""
    nw = len(n_wG)
    J = np.zeros((nw,nw))
    for i, j in np.ndindex(nw, nw):
        if i==j:
            continue
        n_G = w_wG[i]*w_wG[j]
        J[i,j] = coulomb.coulomb(n_G, n_G)
    return J


Nw = c_wM.shape[0]


f_pG = calc.wfs.gd.zeros(n=Nw**2)
for p, (w1, w2) in enumerate(np.ndindex(Nw, Nw)):
    np.multiply(w_wG[w1], w_wG[w2], f_pG[p])

D_pp = np.zeros((Nw**2, Nw**2))
rk(calc.wfs.gd.dv, f_pG, 0., D_pp)

P_awi = {}
for a, P_qMi in calc.wfs.P_aqMi.items():
    P_awi[a] = c_wM.dot(P_qMi[0])

P_app = {}
for a, P_wi in P_awi.items():
    P_app[a] = np.array([pack(np.outer(P_wi[w1], P_wi[w2]))
                         for w1, w2 in np.ndindex(Nw, Nw)])

for a, P_pp in P_app.items():
    I4_pp = calc.setups[a].four_phi_integrals()
    A = P_pp.dot(I4_pp).dot(P_pp.T)
    D_pp += A


U = D_pp * Hartree

U = U.reshape(Nw,Nw,Nw,Nw)

U_nn = np.empty((Nw, Nw))

for i, j, k, l in np.ndindex(Nw, Nw, Nw, Nw):
    U_nn[i,j] = U[i,j,j,i]

np.savetxt('U.txt',U_nn)
