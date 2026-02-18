import os

import numpy as np
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos
from edpyt.shared import params
from scipy.optimize import broyden1


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


# === Load inputs ===
input_folder = "output/lowdin"
output_folder = "output/lowdin/ed"
os.makedirs(output_folder, exist_ok=True)

H_eff = np.load(f"{input_folder}/effective_hamiltonian.npy")
occupancy_goal = np.load(f"{input_folder}/occupancies.npy")
V = np.eye(H_eff.shape[0]) * 3.5
V_diag = np.diag(V.diagonal())

# === Parameters ===
nimp = H_eff.shape[0]
eta = 1e-2
beta = 1000

# === Initial double counting ===
DC0 = np.diag(V.diagonal() * (occupancy_goal - 0.5))
dc0_diag = DC0.diagonal()
np.save(f"{output_folder}/initial_dcc_diag.npy", dc0_diag)

neig = np.ones((nimp + 1) * (nimp + 1), int) * 6

params["z"] = occupancy_goal


# Non-interacting Green's function

espace, egs = build_espace(H_eff, np.zeros_like(H_eff), neig_sector=neig)
screen_espace(espace, egs, beta)
gf0 = build_gf2_lanczos(H_eff, np.zeros_like(H_eff), espace, beta, egs)


def residual_function(dc_diag):
    dc_diag = np.clip(dc_diag, 0.0, np.inf)
    DC = np.diag(dc_diag)

    espace, egs = build_espace(H_eff - DC, V_diag, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_eff - DC, V_diag, espace, beta, egs)
    sigma = Sigma(gf0, gf, H_eff, eta=eta)

    energies = np.array([-60.0, 60])
    sig = sigma.retarded(energies)  # shape (2, nimp, nimp)
    sig_real_diag = sig.real.diagonal(axis1=1, axis2=2)  # shape (2, nimp)

    residual = residual = 0.5 * (sig_real_diag[0] + sig_real_diag[1])

    residual_norm = np.linalg.norm(residual)
    print(f"[Broyden] Residual norm: {residual_norm:.6e}, DC_diag: {dc_diag}")

    return residual


# Initial guess
x0 = dc0_diag.copy()

dc_diag_optimized = broyden1(
    residual_function,
    x0,
    f_tol=1e-3,
    maxiter=50,
    verbose=True,
)

# Save the optimized double counting
np.save(f"{output_folder}/ed_dcc_diag.npy", dc_diag_optimized)
