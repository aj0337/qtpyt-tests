import os
import numpy as np
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
            sigma[e] = energy - self.H_eff - np.linalg.inv(g[..., e])
        return sigma


def plot_callback_factory(H_eff, V, beta, eta, output_folder):
    fig_dir = os.path.join(output_folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    energies_plot = np.arange(-20, 20.1, 0.1)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.ion()

    def callback(dc_diag):
        DC = np.diag(dc_diag)
        neig = np.ones((H_eff.shape[0] + 1) ** 2, int) * 4
        espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
        screen_espace(espace, egs, beta)
        gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
        sigma = Sigma(gf, H_eff, eta=eta)
        sig = sigma.retarded(energies_plot)

        trace_real = np.trace(sig.real, axis1=1, axis2=2)
        trace_imag = np.trace(sig.imag, axis1=1, axis2=2)

        ax.clear()
        ax.plot(energies_plot, trace_real, label="Re Tr Σ(ω)", linestyle="-")
        ax.plot(energies_plot, trace_imag, label="Im Tr Σ(ω)", linestyle="--")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Tr Σ(ω)")
        ax.set_title(f"Σ Trace – Iter {callback.counter}")
        ax.legend()
        ax.grid(True)

        fig_path = os.path.join(fig_dir, f"sigma_trace_iter_{callback.counter:03d}.png")
        fig.savefig(fig_path, dpi=100)
        plt.pause(0.1)
        callback.counter += 1

    callback.counter = 0
    return callback


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
energies = np.array([-1000.0, 1000.0])

# === Initial double counting ===
DC0 = np.diag(V.diagonal() * (occupancy_goal - 0.5))
neig = np.ones((nimp + 1) * (nimp + 1), int) * 4


# === Cost function ===
def F(double_counting_diag):
    DC = np.diag(double_counting_diag)
    espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
    sigma = Sigma(gf, H_eff, eta=eta)
    sig = sigma.retarded(energies)
    sig_real_diag = sig.real.diagonal(axis1=1, axis2=2)  # shape (2, nimp)
    cost = np.linalg.norm(sig_real_diag) / sig_real_diag.shape[1]
    print(f"[F] Cost: {cost:.6f}, DC_diag: {double_counting_diag}")
    return cost


# === Callback ===
callback = plot_callback_factory(H_eff, V, beta, eta, output_folder)

# === Minimize ===
res = minimize(
    F,
    DC0.diagonal(),
    method="BFGS",
    options={"disp": True, "eps": 0.5},
    callback=callback,
)

# === Save result ===
DC_optimized = np.diag(res.x)
espace, egs = build_espace(H_eff - DC_optimized, V, neig_sector=neig)
screen_espace(espace, egs, beta)
gf = build_gf2_lanczos(H_eff - DC_optimized, V, espace, beta, egs)
sigma = Sigma(gf, H_eff, eta=eta)

extended_energies = np.arange(-20, 20.1, 0.1)
sig = sigma.retarded(extended_energies)
np.save(f"{output_folder}/sigma_ed.npy", sig)
np.save(f"{output_folder}/DC_optimized.npy", DC_optimized)
