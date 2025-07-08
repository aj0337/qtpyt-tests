from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from edpyt.nano_dmft import Gfloc
import matplotlib.pyplot as plt
import numpy as np
import os


beta = 100.0
data_folder = "output/lowdin/beta_70"
output_folder = f"output/lowdin/beta_70/occupancies"
H_active = np.load(f"{data_folder}/bare_hamiltonian.npy").real
z_mats = np.load(f"{data_folder}/matsubara_energies.npy")
z_ret = np.load(f"{data_folder}/retarded_energies.npy")

os.makedirs(output_folder, exist_ok=True)

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

gfloc = Gfloc(
    H_active, S_active, HybMats, idx_neq, idx_inv, nmats=z_mats.size, beta=beta
)
gfloc.update(mu=0.0)
gfloc.set_local(Sigma)
dos = -1 / np.pi * gfloc(z_ret).sum(axis=0).imag
dos_file = f"{output_folder}/dft_dos_gfloc.npy"
np.save(dos_file, dos)
plt.plot(z_ret.real, dos)
plt.savefig(f"{output_folder}/dft_dos_gfloc.png")
