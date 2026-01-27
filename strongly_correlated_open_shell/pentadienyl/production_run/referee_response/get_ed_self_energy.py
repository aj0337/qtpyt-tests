import os
import numpy as np

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
U_list = [2.7]                 # eV
input_folder = "../output/lowdin"
base_output_folder = "../output/lowdin/ed/referee_response"

# === Load U-independent inputs once ===
H_eff_master = np.load(f"{input_folder}/effective_hamiltonian.npy")
occupancy_goal_master = np.load(f"{input_folder}/occupancies.npy")

# === Parameters (shared) ===
eta = 1e-2
beta = 1000
de = 0.01
energies = np.arange(-3, 3 + de / 2.0, de).round(7)  # real axis grid
z_ret = energies + 1.0j * eta                        # retarded frequencies


for U_val in U_list:
    print("\n" + "=" * 72)
    print(f"Computing ED self-energy for U_onsite = {U_val:.3f} eV")
    print("=" * 72)

    # --- Per-U working copies ---
    H_eff = H_eff_master.copy()
    occupancy_goal = occupancy_goal_master.copy()
    nimp = H_eff.shape[0]

    # --- PPP matrix for this U (match your generator naming) ---
    # If your file is lowercase like U_matrix_ppp_2.0.txt, change this format accordingly.
    ppp_path = f"{input_folder}/U_matrix_PPP_U_{U_val:.3f}.txt"
    if not os.path.isfile(ppp_path):
        raise FileNotFoundError(f"PPP matrix not found for U = {U_val:.3f} eV:\n  {ppp_path}")
    V = np.loadtxt(ppp_path)
    if V.shape != (nimp, nimp):
        raise ValueError(
            f"PPP matrix shape {V.shape} does not match H_eff shape {(nimp, nimp)} "
            f"for file:\n  {ppp_path}"
        )

    # --- Output folder tagged by U ---
    output_folder = f"{base_output_folder}/Uppp_{U_val:.3f}"
    os.makedirs(output_folder, exist_ok=True)

    # --- Load optimized DC diag for this U ---
    # (use the filename produced by your Broyden script)
    dc_diag_path = f"{output_folder}/ed_dc_diag_Uppp_{U_val:.3f}.npy"
    if not os.path.isfile(dc_diag_path):
        # fallback to your older naming if present
        old_path = f"{output_folder}/ed_dcc_diag_Uppp_{U_val:.3f}.npy"
        if os.path.isfile(old_path):
            dc_diag_path = old_path
        else:
            raise FileNotFoundError(
                f"DC diagonal file not found for U = {U_val:.3f} eV.\n"
                f"Tried:\n  {output_folder}/ed_dc_diag_Uppp_{U_val:.3f}.npy\n"
                f"  {output_folder}/ed_dcc_diag_Uppp_{U_val:.3f}.npy"
            )

    DC_diag = np.load(dc_diag_path)
    if DC_diag.shape != (nimp,):
        raise ValueError(f"DC diag has shape {DC_diag.shape}, expected {(nimp,)} for {dc_diag_path}")

    DC = np.diag(DC_diag)

    # --- ED config ---
    neig = np.ones((nimp + 1) * (nimp + 1), int) * 6
    params["z"] = occupancy_goal

    # --- Build non-interacting GF (gf0) ---
    espace0, egs0 = build_espace(H_eff, np.zeros_like(H_eff), neig_sector=neig)
    screen_espace(espace0, egs0, beta)
    gf0 = build_gf2_lanczos(H_eff, np.zeros_like(H_eff), espace0, beta, egs0)

    # --- Build interacting GF (gf) with DC applied ---
    espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)

    # --- Self-energy object (all per-U objects) ---
    sigma = Sigma(gf0, gf, H_eff, eta=eta)

    # --- Compute and save sigma(ω) ---
    sigma_ret = sigma.retarded(z_ret)

    out_sigma = f"{output_folder}/ed_sigma_ppp_{U_val:.3f}.npy"
    np.save(out_sigma, sigma_ret)

    print(f"[Done] Saved Σ^R(ω) to:\n  {out_sigma}")
