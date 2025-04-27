import os
import numpy as np
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


class Sigma:
    def __init__(self, gf, H_eff, DC0, eta=1e-5):
        self.gf = gf
        self.eta = eta
        self.H_eff = H_eff
        self.DC0 = DC0

    def retarded(self, energy):
        energies = np.atleast_1d(energy)
        g = self.gf(energies, self.eta)
        sigma = np.empty((energies.size, self.gf.n, self.gf.n), complex)
        for e, energy in enumerate(energies):
            sigma[e] = energy - (self.H_eff + self.DC0) - np.linalg.inv(g[..., e])
        return sigma


def objective(x):
    global plot_counter
    DC = np.diag(x)
    espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
    sigma = Sigma(gf, H_eff, DC, eta=eta)

    # High-frequency residual
    energies = np.array([-1000])
    sig_cost = sigma.retarded(energies)
    sig_real_diag = sig_cost.real.diagonal(axis1=1, axis2=2)
    residual = np.mean(sig_real_diag, axis=0)
    res_norm = np.linalg.norm(residual)

    res_norm_scaled = res_norm / 10

    # --- Ratio-preserving penalty ---
    ratio = x / x0
    ratio_penalty = np.sum((ratio - 1.0) ** 2)

    # --- Smoothness penalty ---
    smoothness_penalty = np.sum(np.diff(x) ** 2)

    # Weights
    alpha = 5.0  # ratio preservation
    beta_s = 1.0  # smoothness

    total_cost = res_norm_scaled + alpha * ratio_penalty + beta_s * smoothness_penalty

    print(
        f"[Eval {plot_counter}] ||Σ(∞)|| = {res_norm:.4e}, "
        f"Ratio Penalty = {ratio_penalty:.4e}, "
        f"Smoothness = {smoothness_penalty:.4e}, "
        f"Total = {total_cost:.4e}"
    )
    print(f"DC_diag: {x}")

    # Optional plot
    sig_plot = sigma.retarded(energies_plot)
    trace_real = np.trace(sig_plot.real, axis1=1, axis2=2)
    trace_imag = np.trace(sig_plot.imag, axis1=1, axis2=2)

    plt.figure(figsize=(6, 4))
    plt.plot(energies_plot, trace_real, label="Re Tr Σ(ω)", linestyle="-")
    plt.plot(energies_plot, trace_imag, label="Im Tr Σ(ω)", linestyle="--")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Tr Σ(ω)")
    plt.title(f"Σ Trace – Eval {plot_counter}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"sigma_trace_eval_{plot_counter:03d}.png")
    plt.savefig(fig_path, dpi=100)
    plt.close()
    plot_counter += 1

    return total_cost


# --- Setup ---
plot_counter = 0
output_folder = "output/lowdin/beta_1000/ed"
fig_dir = os.path.join(output_folder, "figures_constrained")
os.makedirs(fig_dir, exist_ok=True)
energies_plot = np.arange(-20, 20.1, 0.1)


# --- Constraint 1: DCC[0] and DCC[6] ≥ all others ---
def ordering_constraints(x):
    cons = []
    for i in range(1, 6):  # skip 0 and 6
        cons.append({"type": "ineq", "fun": lambda x, i=i: x[0] - x[i]})
        cons.append({"type": "ineq", "fun": lambda x, i=i: x[6] - x[i]})
    return cons


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
x0 = DC0.diagonal().copy()

# Define bounds and constraints
bounds = Bounds(
    [1.0] * len(x0), [10.0] * len(x0)
)  # using Bounds object (required by trust-constr)
constraints = ordering_constraints(x0)  # still fine as a list of dicts

# Run optimization with trust-constr
res = minimize(
    objective,
    x0,
    method="trust-constr",
    bounds=bounds,
    constraints=constraints,
    options={
        "verbose": 3,
        "maxiter": 300,
        "gtol": 1e-6,
        "xtol": 1e-8,
        "barrier_tol": 1e-6,
    },
    jac="2-point",  # or 'cs' (complex-step) if objective is smooth enough
)

dcc_optimized = res.x
print("\nOptimized DCC:", dcc_optimized)


# # === Save result ===
# DC_optimized = np.diag(res.x)
# espace, egs = build_espace(H_eff - DC_optimized, V, neig_sector=neig)
# screen_espace(espace, egs, beta)
# gf = build_gf2_lanczos(H_eff - DC_optimized, V, espace, beta, egs)
# sigma = Sigma(gf, H_eff, DC0, eta=eta)

# extended_energies = np.arange(-20, 20.1, 0.1)
# sig = sigma.retarded(extended_energies)
# np.save(f"{output_folder}/sigma_ed.npy", sig)
# np.save(f"{output_folder}/DC_optimized.npy", DC_optimized)
