import os

import numpy as np
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos
from edpyt.shared import params


class Sigma:
    def __init__(self, gf0, gf, H_eff, eta=1e-5):
        self.gf0 = gf0
        self.gf = gf
        self.eta = eta
        self.H_eff = H_eff

    def retarded(self, energy):
        energies = np.atleast_1d(energy)
        g0 = self.gf0(energies, self.eta)
        g = self.gf(energies, self.eta)
        sigma = np.empty((energies.size, self.gf.n, self.gf.n), complex)
        for e, energy in enumerate(energies):
            sigma[e] = np.linalg.inv(g0[..., e]) - np.linalg.inv(g[..., e])
        return sigma


U_val = 3.5
input_folder = "output/lowdin"
output_folder = "output/lowdin/ed"
os.makedirs(output_folder, exist_ok=True)

H_eff = np.load(f"{input_folder}/effective_hamiltonian.npy")
occupancy_goal = np.load(f"{input_folder}/occupancies.npy")

eta = 1e-2
beta = 1000
de = 0.01
energies = np.arange(-8, 8 + de / 2.0, de).round(7)
z_ret = energies + 1.0j * eta

nimp = H_eff.shape[0]
V = np.eye(H_eff.shape[0]) * U_val

print("\n" + "=" * 72)
print(f"Computing ED self-energy for U_onsite = {U_val:.1f} eV")
print("=" * 72)

dc_diag_path = f"{output_folder}/ed_dcc_diag.npy"

DC_diag = np.load(dc_diag_path)
if DC_diag.shape != (nimp,):
    raise ValueError(
        f"DC diag has shape {DC_diag.shape}, expected {(nimp,)} for {dc_diag_path}"
    )

DC = np.diag(DC_diag)


neig = np.ones((nimp + 1) * (nimp + 1), int) * 6
params["z"] = occupancy_goal


espace0, egs0 = build_espace(H_eff, np.zeros_like(H_eff), neig_sector=neig)
screen_espace(espace0, egs0, beta)
gf0 = build_gf2_lanczos(H_eff, np.zeros_like(H_eff), espace0, beta, egs0)


espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
screen_espace(espace, egs, beta)
gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)


sigma = Sigma(gf0, gf, H_eff, eta=eta)


sigma_ret = sigma.retarded(z_ret)

out_sigma = f"{output_folder}/self_energy_with_dcc.npy"
np.save(out_sigma, sigma_ret)

print(f"[Done] Saved Σ^R(ω) to:\n  {out_sigma}")
