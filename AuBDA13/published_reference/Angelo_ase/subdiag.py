import numpy as np
import pickle
from ase import Atoms
from ase.io import read
from gpaw import *
from gpaw.lcao.tools import *

# auxiliary functions

# subdiagonalize block
def subdiagonalize(h_ii, s_ii, index_j):
    nb = h_ii.shape[0]
    nb_sub = len(index_j)
    h_sub_jj = get_subspace(h_ii, index_j)
    s_sub_jj = get_subspace(s_ii, index_j)
    e_j, v_jj = np.linalg.eig(np.linalg.solve(s_sub_jj, h_sub_jj))
    normalize(v_jj, s_sub_jj) # normalize: <v_j|s|v_j> = 1
    permute_list = np.argsort(e_j.real)
    e_j = np.take(e_j, permute_list)
    v_jj = np.take(v_jj, permute_list, axis=1)
    #
    # setup transformation matrix
    c_ii = np.identity(nb, complex)
    for i in np.arange(nb_sub):
        for j in np.arange(nb_sub):
            c_ii[index_j[i], index_j[j]] = v_jj[i, j]
    #
    h1_ii = unitary_trans(c_ii, h_ii)
    s1_ii = unitary_trans(c_ii, s_ii)
    #
    return h1_ii, s1_ii, c_ii, e_j

# get subspace of matrix given index array
def get_subspace(matrix, index):
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    return matrix.take(index, 0).take(index, 1)

# normalizate each vetor in matrix 
def normalize(matrix, S=None):
    for col in matrix.T:
        if S is None:
            col /= np.linalg.norm(col)
        else:
            col /= np.sqrt(np.dot(col.conj(), np.dot(S, col)))

# apply unitary transformmation A^{\dagger} B A
def unitary_trans(amat,bmat):
    cmat = np.dot(np.transpose(np.conj(amat)),np.dot(bmat,amat))
    return cmat

# permute atoms according to a symbol sequence
def permute_atoms(atoms, symbol_sequence):
    p = []
    for symbol in symbol_sequence:
        for atom in atoms:
            if atom.symbol == symbol:
                p.append(atom.index)
    return p,atoms[p]

# extract orbitals according to a symbol sequence
def extract_orbitals(atoms, symbol_sequence, basis):
    p = []
    a = Atoms()
    for symbol in symbol_sequence:
        n = 0
        for atom in atoms:
            norb = basis[atom.symbol]
            if atom.symbol == symbol:
                a += atom
                for i in range(norb):
                    p.append(n+i)
            n = n + norb
    return p, a

# extract block of orbitals in ibfs from matrix
def extract_block(h, s, ibfs):
    nbfs = len(ibfs)
    h_extract = np.zeros((nbfs,nbfs), dtype='complex')
    s_extract = np.zeros((nbfs,nbfs), dtype='complex')
    for i,ib in enumerate(ibfs):
        for j,jb in enumerate(ibfs):
            h_extract[i,j] = h[ib,jb]
            s_extract[i,j] = s[ib,jb]
    return h_extract, s_extract

# perform cut-coupling of (H,S) pair for orbitals in ibfs
def cut_coupling(h, s, ibfs):
    for i in ibfs:
        # overlp matrix
        s[:,i] = 0.0
        s[i,:] = 0.0
        s[i,i] = 1.0
        # Hamiltonian
        ei = h[i, i]
        h[:,i] = 0.0
        h[i,:] = 0.0
        h[i,i] = ei





# structure
atoms = read('scatt.xyz')
# basis set
basis = {'Au':9,'H':5,'C':13,'N':13}

# get chemical symbols
symbol_list = atoms.get_chemical_symbols()

# get (H,S) scattering region 
h_scatt, s_scatt = pickle.load(open('scatt_hs_lcao.pckl', 'rb'))

nbfs = h_scatt.shape[0]



# subdiagonalize or read subdiagonalized (H,S) pair
subdiagonalization = False

if subdiagonalization:
    # create subdiagonalized (H,S) -- indentical to scattering pair
    h_sub = np.copy(h_scatt)
    s_sub = np.copy(s_scatt)
    # subdiagonalize block by block except 'Au'
    ia = 0
    for i, symbol in enumerate(symbol_list):
        if symbol == 'Au':
            ia += basis[symbol]
        else:
            ni = ia
            nf = ni + basis[symbol]
            h_sub, s_sub, c_ii, e_j = subdiagonalize(h_sub, s_sub, np.arange(ni,nf))
            ia += basis[symbol]
    # 
    # save subdiagonalized (H,S) to HDD
    pickle.dump((h_sub.astype(complex), s_sub.astype(complex)), open('scatt_hs_sub.pckl', 'wb'), 2)

else:
    # get subdiagonalized (H,S) scattering region 
    h_sub, s_sub = pickle.load(open('scatt_hs_sub.pckl', 'rb'))




# extract C bfs
p, _ = extract_orbitals(atoms, ['C'], basis)
# extract C pz bfs
cpz = p[3::basis['C']]
# complementary set (i.e., all C non-pz bfs)
npz = [i for i in p if i not in cpz]
# edit subdiagonalized (H,S) perform cut-coupling of C non-pz bfs
h_cut = np.copy(h_sub)
s_cut = np.copy(s_sub)
cut_coupling(h_cut, s_cut, npz)
pickle.dump((h_cut.astype(complex), s_cut.astype(complex)), open('scatt_hs_pz.pckl', 'wb'), 2)


# debug
print()
print('C-pz: ',cpz)
print()
print('%3s \t %3s \t %4s' %('p', 'pz', 'npz '))
for i in range(len(p)):
    print('%d \t' %(p[0]), end=" ")
    if len(cpz) >0 and cpz[0] == p[0]:
        print('%d \t' %(cpz[0]), end=" ")
        cpz.pop(0)
    else:
        print('--- \t', end=" ")
    if len(npz) > 0 and npz[0] == p[0]:
        print('%d' %(npz[0]))
        npz.pop(0)
    else:
        print('---')
    p.pop(0)
print()

# check if Hamiltonian is symmetric
assert np.allclose(h_cut, h_cut.T, rtol=1e-05, atol=1e-08)


       
# extract C sp2 minimal LOs set bfs // original sorting
#cpzd = [638, 641, 642, 645, 647, 661, 664, 668, 670, 674, 677, 681, 683, 687, 690, 694, 696, 700, 703, 704, 707, 709, 718, 721, 725, 727]

# extract C bfs
p, _ = extract_orbitals(atoms, ['C'], basis)
# extract C pz bfs
cpz = p[3::basis['C']]
# complementary set (i.e., all C non-pz bfs)
npz = [i for i in p if i not in cpz]
# extract all relevant LOs given a threshold
threshold = 0.075
ibfs = []
for i in cpz:
    for j in p:
        if abs(h_sub[i,j].real) > threshold:
            ibfs.append(j)
            print(i,j,abs(h_sub[i,j].real))
cpzd = list(np.unique(ibfs))
# complementary set (i.e., all C non-pd bfs)
npzd = [i for i in p if i not in cpzd]

# edit subdiagonalized (possibly sorted) (H,S) perform cut-coupling of C non-pz bfs
h_cut = np.copy(h_sub)
s_cut = np.copy(s_sub)
cut_coupling(h_cut, s_cut, npzd)
pickle.dump((h_cut.astype(complex), s_cut.astype(complex)), open('scatt_hs_pzd.pckl', 'wb'), 2)


# debug
print()
print('C-LOs:', cpzd)
print()
print('%3s \t %3s \t %4s' %('p', 'pzd', 'npzd'))
for i in range(len(p)):
    print('%d \t' %(p[0]), end=" ")
    if len(cpzd) >0 and cpzd[0] == p[0]:
        print('%d \t' %(cpzd[0]), end=" ")
        cpzd.pop(0)
    else:
        print('--- \t', end=" ")
    if len(npzd) > 0 and npzd[0] == p[0]:
        print('%d' %(npzd[0]))
        npzd.pop(0)
    else:
        print('---')
    p.pop(0)
print()
 
# check if Hamiltonian is symmetric
assert np.allclose(h_cut, h_cut.T, rtol=1e-05, atol=1e-08)





import matplotlib.pyplot as plt

h_scatt, s_scatt = pickle.load(open('scatt_hs_lcao.pckl', 'rb'))
h_sub,   s_sub   = pickle.load(open('scatt_hs_sub.pckl',  'rb'))
h_pz,    s_pz    = pickle.load(open('scatt_hs_pz.pckl',   'rb'))
h_pzd,   s_pzd   = pickle.load(open('scatt_hs_pzd.pckl',  'rb'))

nbfs = h_scatt.shape[0]
nm   = 6*13+2*13+8*5
nc   = 6*13 
ni   = 612
nf   = ni + nm

fig, axes = plt.subplots(nrows=2, ncols=4)

precision = 0.1

axes[0,0].title.set_text('LCAO')
axes[0,1].title.set_text('subdiagonalized')
axes[1,2].title.set_text('C-pz')
axes[1,3].title.set_text('C-pz+d')
axes[1,0].title.set_text('molecule LCAO')
axes[1,1].title.set_text('molecule subdiagonalized')
axes[1,2].title.set_text('molecule C-pz')
axes[1,3].title.set_text('molecule C-pz+d')
axes[0,0].spy(h_scatt,              precision = precision)
axes[0,1].spy(h_sub,                precision = precision)
axes[0,2].spy(h_pz,                 precision = precision)
axes[0,3].spy(h_pzd,                precision = precision)
axes[1,0].spy(h_scatt[ni:nf,ni:nf], precision = precision)
axes[1,1].spy(h_sub[ni:nf,ni:nf],   precision = precision)
axes[1,2].spy(h_pz[ni:nf,ni:nf],    precision = precision)
axes[1,3].spy(h_pzd[ni:nf,ni:nf],   precision = precision)

plt.show()
 


