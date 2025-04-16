import os
import numpy as np
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos
from edpyt.shared import params
from scipy.optimize import minimize
from matplotlib import pyplot as plt


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


def plot_callback_factory(energies, H_eff, V, beta, eta, output_folder):
    fig_dir = os.path.join(output_folder, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.ion()

    def callback(dc_diag):
        DC = np.diag(dc_diag)
        neig = np.ones((H_eff.shape[0] + 1) ** 2, int) * 4
        espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
        screen_espace(espace, egs, beta)
        gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
        sigma = Sigma(gf, H_eff, eta=eta)
        sig = sigma.retarded(energies)

        trace_real = np.trace(sig.real, axis1=1, axis2=2)
        trace_imag = np.trace(sig.imag, axis1=1, axis2=2)

        ax.clear()
        ax.plot(energies, trace_real, label="Tr[Re Σ(ω)]", linestyle="-")
        ax.plot(energies, trace_imag, label="Tr[Im Σ(ω)]", linestyle="--")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Tr Σ(ω)")
        ax.set_title(f"Self-energy Trace – Iteration {callback.counter}")
        ax.legend(loc="upper right", fontsize="medium")
        ax.grid(True)

        fig_path = os.path.join(fig_dir, f"sigma_trace_iter_{callback.counter:03d}.png")
        fig.savefig(fig_path, dpi=150)
        plt.pause(0.1)
        callback.counter += 1

    callback.counter = 0
    return callback


input_folder = "output/lowdin"
H_eff = np.load(f"{input_folder}/effective_hamiltonian.npy")
occupancy_goal = np.load(f"{input_folder}/beta_1000/occupancies.npy")
output_folder = "output/lowdin/beta_1000/ed"
os.makedirs(output_folder, exist_ok=True)
de = 0.01
nimp = H_eff.shape[0]
energies = np.arange(-12, 12 + de / 2.0, de).round(7)
eta = 1e-3
beta = 1000

V = np.loadtxt(f"{input_folder}/U_matrix.txt")
DC0 = np.diag(V.diagonal() * (occupancy_goal - 0.5))
neig = np.ones((nimp + 1) * (nimp + 1), int) * 4


def F(double_counting_diag, high_energy_mask=np.abs(energies) > 10):
    DC = np.diag(double_counting_diag)
    espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
    sigma = Sigma(gf, H_eff, eta=eta)
    sig = sigma.retarded(energies)
    sig_real_diag = sig.real.diagonal(axis1=1, axis2=2)
    residue = sig_real_diag[high_energy_mask, :]
    cost = np.linalg.norm(residue) / residue.shape[1]

    # sig_real = sig.real[high_energy_mask, :, :]  #
    # traces = np.trace(sig_real, axis1=1, axis2=2)
    # cost = np.sum(np.abs(traces))

    print(f"[F] Cost: {cost:.6f}, DC_diag: {double_counting_diag}")
    return cost


high_energy_mask = np.abs(energies) > 11.9
callback = plot_callback_factory(
    energies=energies,
    H_eff=H_eff,
    V=V,
    beta=beta,
    eta=eta,
    output_folder=output_folder,
)

res = minimize(
    lambda x: F(x, high_energy_mask),
    DC0.diagonal(),
    method="BFGS",
    callback=callback,
    options={"disp": True, "eps": 5e-1},
)


DC_optimized = np.diag(res.x)
np.save(f"{output_folder}/DC_optimized.npy", DC_optimized)

# params['z'] = occupancy_goal

## Once the DC correction is found, check if something needs to be passed to params['z']
