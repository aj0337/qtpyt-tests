from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from edpyt.nano_dmft import Gfloc

import numpy as np
import os

# Data paths
data_folder = f"./output/lowdin/beta_70"
output_folder = f"./output/lowdin/beta_70/occupancies"
H_active = np.load(f"{data_folder}/bare_hamiltonian.npy").real
z_mats = np.load(f"{data_folder}/matsubara_energies.npy")

os.makedirs(output_folder, exist_ok=True)

# Parameters
mu = 0.0
beta = 70.0

len_active = H_active.shape[0]
hyb_mats = np.fromfile(f"{data_folder}/matsubara_hybridization.bin", complex).reshape(
    z_mats.size,
    len_active,
    len_active,
)
_HybMats = interp1d(z_mats.imag, hyb_mats, axis=0, bounds_error=False, fill_value=0.0)
HybMats = lambda z: _HybMats(z.imag)

S_active = np.eye(len_active)
idx_neq = np.arange(len_active)
idx_inv = np.arange(len_active)

nimp = len_active

Sigma = lambda z: np.zeros((nimp, z.size), complex)

print(f"Calculating occupancy for mu = {mu}",flush=True)
gfloc = Gfloc(
    H_active, S_active, HybMats, idx_neq, idx_inv, nmats=z_mats.size, beta=beta
)
gfloc.update(mu=0.0)
gfloc.set_local(Sigma)
occupancies = gfloc.integrate(mu=mu)
print(f"Total occupancy for mu = {mu} using matsubara summation are: {np.sum(occupancies)}",flush=True)
np.save(os.path.join(output_folder, f'occupancies_gfloc_mu_{mu}.npy'), occupancies)
