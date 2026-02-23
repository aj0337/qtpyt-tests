import csv
import os
import time

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
        for e, _en in enumerate(energies):
            sigma[e] = np.linalg.inv(g0[..., e]) - np.linalg.inv(g[..., e])
        return sigma


# === Load inputs ===
print("[Main] Loading input files", flush=True)

input_folder = "output/"
output_folder = "output/ed"
os.makedirs(output_folder, exist_ok=True)

H_wann = np.load(f"{input_folder}/hamiltonian.npy")
occupancy_goal = np.load(f"{input_folder}/occupancies.npy")
V = np.loadtxt(f"{input_folder}/U_matrix.txt")
V_diag = np.diag(V.diagonal())

print(f"[Main] H_wann shape: {H_wann.shape}", flush=True)

# === Parameters ===
nimp = H_wann.shape[0]
eta = 1e-2
beta = 1000

print(f"[Main] nimp = {nimp}, beta = {beta}, eta = {eta}", flush=True)

# === Initial double counting ===
DC0 = np.diag(V.diagonal() * (occupancy_goal - 0.5))
dc0_diag = DC0.diagonal()

print(f"[Main] Initial DC diagonal: {dc0_diag}", flush=True)

neig = np.ones((nimp + 1) * (nimp + 1), int) * 6
params["z"] = occupancy_goal
print("[Main] Parameters set", flush=True)

# === Checkpoint / history settings ===
checkpoint_path = os.path.join(output_folder, "dc_checkpoint.npz")
history_path = os.path.join(output_folder, "dc_history.csv")

SAVE_EVERY = 3  # save every residual call (set to e.g. 5 or 10 to reduce I/O)
RESID_TOL = 1e-3  # should match f_tol passed to broyden1

# === Non-interacting Green's function ===
print("[Main] Building non-interacting ED space", flush=True)

espace, egs = build_espace(H_wann, np.zeros_like(H_wann), neig_sector=neig)

print("[Main] Screening ED space (non-interacting)", flush=True)
screen_espace(espace, egs, beta)

print("[Main] Building non-interacting Green's function", flush=True)
gf0 = build_gf2_lanczos(H_wann, np.zeros_like(H_wann), espace, beta, egs)

print("[Main] Non-interacting GF ready", flush=True)


# === Restart if checkpoint exists ===
if os.path.exists(checkpoint_path):
    ck = np.load(checkpoint_path, allow_pickle=False)
    x0 = ck["dc_diag"].copy()
    last_call = int(ck["call"])
    last_norm = float(ck["res_norm"])
    print(f"[Main] Restarting from checkpoint: {checkpoint_path}", flush=True)
    print(
        f"[Main] Checkpoint call={last_call}, last_res_norm={last_norm:.6e}", flush=True
    )
else:
    x0 = dc0_diag.copy()
    print("[Main] No checkpoint found; starting from initial DC0", flush=True)

# init history file (only if it doesn't exist)
if not os.path.exists(history_path):
    with open(history_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["call", "timestamp", "res_norm", "dc_diag"])


call_counter = {"n": 0}  # mutable counter in closure


def residual_function(dc_diag):
    call_counter["n"] += 1
    ncall = call_counter["n"]

    dc_diag = np.clip(dc_diag, 0.0, np.inf)
    DC = np.diag(dc_diag)

    espace, egs = build_espace(H_wann - DC, V_diag, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_wann - DC, V_diag, espace, beta, egs)

    sigma = Sigma(gf0, gf, H_wann, eta=eta)

    energies = np.array([-2])
    sig = sigma.retarded(energies)
    residual = sig.real.diagonal(axis1=1, axis2=2)

    # energies = np.array([-8, 8])
    # sig_real_diag = sig.real.diagonal(axis1=1, axis2=2)
    # residual = 0.5 * (sig_real_diag[0] + sig_real_diag[1])

    res_norm = float(np.linalg.norm(residual))
    print(
        f"[Residual] call={ncall}  Residual norm = {res_norm:.6e}\n"
        f"[Residual] Residual vector = {residual}",
        flush=True,
    )

    # --- Save checkpoint (and history) ---
    if (ncall % SAVE_EVERY) == 0:
        np.savez(
            checkpoint_path,
            dc_diag=dc_diag,
            residual=residual,
            res_norm=res_norm,
            call=ncall,
            t=time.time(),
            beta=beta,
            eta=eta,
        )
        with open(history_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ncall, time.time(), res_norm, dc_diag.tolist()])
        print(f"[Checkpoint] Saved: {checkpoint_path}", flush=True)

    # --- Print explicit convergence statement when within tolerance ---
    if res_norm <= RESID_TOL:
        print(
            f"[Main] ✅ CONVERGED at call={ncall}: residual norm {res_norm:.6e} <= {RESID_TOL:.6e}",
            flush=True,
        )

    return residual


# === Broyden optimization ===
print("[Main] Starting Broyden solver", flush=True)

dc_diag_optimized = broyden1(
    residual_function,
    x0,
    f_tol=RESID_TOL,
    maxiter=50,
    verbose=True,
)

print("[Main] Broyden solver finished", flush=True)
print(f"[Main] Optimized DC diag: {dc_diag_optimized}", flush=True)

# === Save result ===
np.save(f"{output_folder}/ed_dcc_diag.npy", dc_diag_optimized)
print("[Main] Optimized DC saved to disk", flush=True)

# Optional: remove checkpoint on success so future runs don't restart unnecessarily
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("[Main] Removed checkpoint (run finished)", flush=True)
