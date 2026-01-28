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


# === User controls ===
U_list = [0.1]         # eV
input_folder = "../output/lowdin"
base_output_folder = "../output/lowdin/ed/referee_response"

# === Load U-independent inputs once ===
H_eff_master = np.load(f"{input_folder}/effective_hamiltonian.npy")
occupancy_goal_master = np.load(f"{input_folder}/occupancies.npy")

# === Parameters (shared) ===
eta = 1e-2
beta = 1000


for U_val in U_list:
    print("\n" + "=" * 72)
    print(f"Running ED double counting for U_onsite = {U_val:.1f} eV")
    print("=" * 72)

    # --- Per-U working copies (avoid accidental mutation across loop iterations) ---
    H_eff = H_eff_master.copy()
    occupancy_goal = occupancy_goal_master.copy()

    nimp = H_eff.shape[0]

    # --- Load PPP matrix for this U ---
    # Must match your generator naming: U_matrix_PPP_U_{U_onsite:.3f}.txt
    ppp_path = f"{input_folder}/U_matrix_PPP_U_{U_val:.1f}.txt"
    if not os.path.isfile(ppp_path):
        raise FileNotFoundError(f"PPP matrix not found for U = {U_val:.1f} eV:\n  {ppp_path}")

    V = np.loadtxt(ppp_path)
    if V.shape != (nimp, nimp):
        raise ValueError(
            f"PPP matrix shape {V.shape} does not match H_eff shape {(nimp, nimp)} "
            f"for file:\n  {ppp_path}"
        )

    # --- Output folder tagged by U ---
    output_folder = f"{base_output_folder}/Uppp_{U_val:.1f}"
    os.makedirs(output_folder, exist_ok=True)

    # --- Initial double counting (per-U, since it uses V.diagonal()) ---
    DC0 = np.diag(V.diagonal() * (occupancy_goal - 0.5))
    dc0_diag = DC0.diagonal().copy()

    # --- ED configuration ---
    neig = np.ones((nimp + 1) * (nimp + 1), int) * 6
    params["z"] = occupancy_goal

    # --- Non-interacting Green's function gf0 (per-U identical here, but built inside loop for safety) ---
    espace0, egs0 = build_espace(H_eff, np.zeros_like(H_eff), neig_sector=neig)
    screen_espace(espace0, egs0, beta)
    gf0 = build_gf2_lanczos(H_eff, np.zeros_like(H_eff), espace0, beta, egs0)

    # --- Residual function bound to THIS loop's H_eff, V, gf0, etc. ---
    def residual_function(dc_diag):
        dc_diag = np.clip(dc_diag, 0.0, np.inf)
        DC = np.diag(dc_diag)

        espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
        screen_espace(espace, egs, beta)
        gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)

        sigma = Sigma(gf0, gf, H_eff, eta=eta)

        energies = np.array([-100.0])
        sig = sigma.retarded(energies)  # shape (1, nimp, nimp)
        sig_real_diag = sig.real.diagonal(axis1=1, axis2=2)  # shape (1, nimp)

        residual = sig_real_diag[0]  # use -100 eV only

        residual_norm = np.linalg.norm(residual)
        print(f"[Broyden][U={U_val:.1f}] Residual norm: {residual_norm:.6e}, DC_diag: {dc_diag}")

        return residual

    # --- Solve ---
    x0 = dc0_diag.copy()

    dc_diag_optimized = broyden1(
        residual_function,
        x0,
        f_tol=1e-3,
        maxiter=50,
        verbose=True,
    )

    # --- Save results with correct U tag ---
    out_name = f"{output_folder}/ed_dc_diag_Uppp_{U_val:.1f}.npy"
    np.save(out_name, dc_diag_optimized)
    print(f"[Done] Saved optimized DC diag to:\n  {out_name}")
