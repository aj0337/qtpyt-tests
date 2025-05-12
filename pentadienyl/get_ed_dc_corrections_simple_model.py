import os
import numpy as np

from scipy.optimize import broyden1

from edpyt.shared import params
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos


class Sigma:
    def __init__(self, gf, H_eff, eta=1e-5):
        self.gf = gf
        self.eta = eta
        self.H_eff = H_eff

    def retarded(self, energy):
        energies = np.atleast_1d(energy)
        g = self.gf(energies, self.eta)
        sigma = np.empty((energies.size, self.gf.n, self.gf.n), complex)
        for e, energy in enumerate(energies):
            sigma[e] = energy - (self.H_eff) - np.linalg.inv(g[..., e])
        return sigma


# === Load inputs ===
input_folder = "output/lowdin"
output_folder = "output/lowdin/beta_1000/ed"
os.makedirs(output_folder, exist_ok=True)

H_eff = np.load(f"{input_folder}/effective_hamiltonian.npy")
occupancy_goal = np.load(f"{input_folder}/beta_1000/occupancies.npy")
V = np.loadtxt(f"{input_folder}/U_matrix.txt")

# === Parameters ===
nimp = H_eff.shape[0]
eta = 1e-3
beta = 1000

# === Initial double counting ===
DC0 = np.diag(V.diagonal() * (occupancy_goal - 0.5))
neig = np.ones((nimp + 1) * (nimp + 1), int) * 6

params["z"] = occupancy_goal

penalty_weight = 1.0
delta_order = 0.01  # Minimum margin for ordering (eV)
dc_diag_clip_bounds = (0.0, 10.0)  # Prevent unphysical values

# Precompute reference ratio structure
dc0_diag = DC0.diagonal()
target_ratios = dc0_diag / np.max(dc0_diag)


def residual_function(dc_diag):
    # Clip only the lower bound
    dc_diag = np.clip(dc_diag, 0.0, np.inf)

    DC = np.diag(dc_diag)
    espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
    sigma = Sigma(gf, H_eff, eta=eta)

    # Get physical residual from Σ at large negative frequency
    energies = np.array([-1000.0])
    sig_cost = sigma.retarded(energies)
    sig_real_diag = sig_cost.real.diagonal(axis1=1, axis2=2)
    residual = np.mean(sig_real_diag, axis=0)

    # Logging
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
np.save(f"{output_folder}/ed_dcc_diag_test.npy", dc_diag_optimized)
