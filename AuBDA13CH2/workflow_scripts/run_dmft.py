from __future__ import annotations
import numpy as np


from scipy.optimize import root
from scipy.interpolate import interp1d

from edpyt.nano_dmft import Gfloc, Gfimp as nanoGfimp
from edpyt.dmft import Gfimp, DMFT, Converged

# DMFT calculation (runs serially)

def distance(delta):
    global delta_prev
    delta_prev[:] = delta
    return dmft.distance(delta)

def save_sigma(sigma_diag,outputfile,npsin):
    L, ne = sigma_diag.shape
    sigma = np.zeros((ne, L, L), complex)

    def save(spin):
        for diag, mat in zip(sigma_diag.T, sigma):
            mat.flat[::(L + 1)] = diag
        np.save(outputfile, sigma)

    for spin in range(nspin):
        save(spin)

U = 4.  # Interaction
nbaths = 4
tol = 27
max_iter = 200
adjust_mu = True
alpha = 0.
nspin = 1

data_folder = '../output/compute_run'

occupancy_goal = np.load(f'{data_folder}/occupancies.npy')
len_active = occupancy_goal.size
z_ret = np.load(f'{data_folder}/energies.npy')
z_mats = np.load(f'{data_folder}/matsubara_energies.npy')
hyb_ret = np.fromfile(f'{data_folder}/hybridization.bin', complex).reshape(z_ret.size, len_active, len_active)
hyb_mats = np.fromfile(f'{data_folder}/matsubara_hybridization.bin',
                        complex).reshape(z_mats.size, len_active, len_active)
H_active = np.load(f'{data_folder}/hamiltonian.npy').real

eta = z_ret[0].imag
beta = np.pi / (z_mats[0].imag)

_HybRet = interp1d(z_ret.real,
                    hyb_ret,
                    axis=0,
                    bounds_error=False,
                    fill_value=0.)
HybRet = lambda z: _HybRet(z.real)

_HybMats = interp1d(z_mats.imag,
                    hyb_mats,
                    axis=0,
                    bounds_error=False,
                    fill_value=0.)
HybMats = lambda z: _HybMats(z.imag)
HybZro = lambda z: np.zeros((len_active, z.size), complex)


S_active = np.eye(len_active)

idx_neq = np.arange(len_active)
idx_inv = np.arange(len_active)

V = np.eye(len_active) * U

double_counting = np.diag(V.diagonal() * (occupancy_goal - 0.5))
gfloc = Gfloc(H_active - double_counting, np.eye(len_active), HybMats, idx_neq, idx_inv)

nimp = gfloc.idx_neq.size
gfimp = []
for i in range(nimp):
    gfimp.append(Gfimp(nbaths, z_mats.size, V[i, i], beta))

gfimp = nanoGfimp(gfimp)

occupancy_goal = occupancy_goal[gfloc.idx_neq]

dmft = DMFT(gfimp,
            gfloc,
            occupancy_goal,
            max_iter=max_iter,
            tol=tol,
            adjust_mu=adjust_mu,
            alpha=alpha)

Sigma = lambda z: np.zeros((nimp, z.size), complex)
delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=0.)
delta_prev = delta.copy()

try:
    root(distance, delta_prev, method='broyden1')
except Converged:
    pass

np.save(f'{data_folder}/dmft_delta.npy', delta_prev)
open(f'{data_folder}/mu.txt', 'w').write(str(gfloc.mu))

_Sigma = lambda z: -double_counting.diagonal()[:, None] - gfloc.mu + gfloc.Sigma(z)[idx_inv]

dmft_sigma_file = f"{data_folder}/dmft_sigma.npy"
save_sigma(_Sigma(z_ret), dmft_sigma_file,nspin)
