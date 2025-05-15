import os
import numpy as np

from scipy.optimize import broyden1

from edpyt.shared import params
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos

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

params['z'] = occupancy_goal

penalty_weight = 1.0
delta_order = 0.01        # Minimum margin for ordering (eV)
dc_diag_clip_bounds = (0.0, 10.0)  # Prevent unphysical values

# Precompute reference ratio structure
dc0_diag = DC0.diagonal()
target_ratios = dc0_diag / np.max(dc0_diag)
espace0, egs0 = build_espace(H_eff, np.zeros_like(H_eff), neig_sector=neig)
screen_espace(espace0, egs0, beta)
gf0 = build_gf2_lanczos(H_eff, np.zeros_like(H_eff), espace0, beta, egs0)
def residual_function(dc_diag):

    # Clip unphysical values
    dc_diag = np.clip(dc_diag, *dc_diag_clip_bounds)

    DC = np.diag(dc_diag)
    espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
    sigma = Sigma(gf0, gf, H_eff, eta=eta)

    # Get physical residual from Σ at large negative frequency
    energies = np.array([-10000.0])
    sig_cost = sigma.retarded(energies)  # (n_energies, nimp, nimp)
    sig_real_diag = sig_cost.real.diagonal(axis1=1, axis2=2)
    residual = np.mean(sig_real_diag, axis=0)

    # -----------------------------------
    # Penalty terms for structure control
    # -----------------------------------

    # 1. Ordering penalty
    order_penalty = np.zeros_like(residual)
    others = np.delete(dc_diag, [0, 6])
    if dc_diag[0] < np.max(others) + delta_order:
        order_penalty[0] += penalty_weight * (np.max(others) + delta_order - dc_diag[0])
    if dc_diag[6] < np.max(np.delete(dc_diag, [0])) + delta_order:
        order_penalty[6] += penalty_weight * (np.max(np.delete(dc_diag, [0])) + delta_order - dc_diag[6])

    # 2. Ratio penalty (safe log scale)
    dc_diag_max = np.max(dc_diag) if np.max(dc_diag) > 1e-8 else 1.0
    current_ratios = dc_diag / dc_diag_max
    ratio_penalty = penalty_weight * (np.log1p(current_ratios) - np.log1p(target_ratios))

    # Combine all into total residual
    total_residual = residual + order_penalty + ratio_penalty

    # -----------------------------------
    # Logging
    # -----------------------------------
    residual_norm = np.linalg.norm(total_residual)
    print(f"[Broyden] Residual norm: {residual_norm:.6e}, DC_diag: {dc_diag}")
    print(f"Penalty | Order: {np.linalg.norm(order_penalty):.3e}, Ratio: {np.linalg.norm(ratio_penalty):.3e}")

    return total_residual

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
