from __future__ import annotations
import numpy as np
from edpyt.dmft import DMFT, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from edpyt.nano_dmft import Gfloc

import numpy as np
import matplotlib.pyplot as plt
import os

def distance(delta):
    global delta_prev
    delta_prev[:] = delta
    return dmft.distance(delta)


def save_sigma(sigma_diag, outputfile, npsin):
    L, ne = sigma_diag.shape
    sigma = np.zeros((ne, L, L), complex)

    def save(spin):
        for diag, mat in zip(sigma_diag.T, sigma):
            mat.flat[:: (L + 1)] = diag
        np.save(outputfile, sigma)

    for spin in range(nspin):
        save(spin)


def plot(gf, sigma_func, semilogy=True, reference_gf=None, label_ref="DFT"):
    """Plot the Green's function DOS and Tr(Sigma) with an optional reference DOS."""

    fig, axes = plt.subplots(2, 1, sharex=True)
    ax1, ax2 = axes

    w = z_ret.real
    dos = -1 / np.pi * gf(z_ret).sum(0).imag
    if semilogy:
        ax1.semilogy(w, dos, label="DMFT") if dos.ndim == 1 else ax1.semilogy(
            w, dos[0], label=r"spin $\uparrow$"
        )
    else:
        ax1.plot(w, dos, label="DMFT") if dos.ndim == 1 else ax1.plot(
            w, dos[0], label=r"spin $\uparrow$"
        )

    if reference_gf is not None:
        reference_dos = -1 / np.pi * reference_gf(z_ret).sum(0).imag
        ax1.plot(
            w, reference_dos, linestyle="--", label=label_ref
        ) if reference_dos.ndim == 1 else ax1.plot(
            w,
            reference_dos[0],
            linestyle="--",
            label=label_ref,
        )

    ax1.set_ylabel("DOS [a.u.]")
    ax1.legend(loc="upper right")

    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    sigma = sigma_func(z_ret)
    trace_sigma = sigma.sum(axis=0)
    ax2.plot(w, trace_sigma.real, label="Re Tr(Sigma)", color="blue")
    ax2.plot(w, trace_sigma.imag, label="Im Tr(Sigma)", color="orange")

    ax2.set_xlabel("E-E$_F$ [eV]")
    ax2.set_ylabel("Tr(Sigma) [eV]")
    ax2.legend(loc="upper right")

    plt.subplots_adjust(hspace=0)
    return ax1


data_folder = "../../output/redo"
figure_folder = f"{data_folder}/figures"
os.makedirs(figure_folder, exist_ok=True)
iteration_counter = 0


def callback(*args, **kwargs):
    """Callback function for DMFT solver to plot and save results."""
    global iteration_counter


    ax1 = plot(
        gf=gfloc,
        sigma_func=gfloc.Sigma,
        reference_gf=gfloc0,
        label_ref="DFT",
        semilogy=kwargs.get("semilogy", True),
    )


    mu_value = gfloc.mu
    ax1.set_title(f"Iteration {iteration_counter} | $\mu$ = {mu_value:.4f} eV")


    figure_filename = os.path.join(
        figure_folder, f"iteration_{iteration_counter:03d}_mu_{mu_value:.4f}.png"
    )
    plt.xlim(-10, 10)
    plt.savefig(figure_filename, dpi=300, bbox_inches="tight")

    plt.close()
    iteration_counter += 1


nbaths = 4
U = 4.0
adjust_mu = True
use_double_counting = True

tol = 1e-4
max_iter = 1000
alpha = 0.0
nspin = 1

L = 9

occupancy_goal = np.load(f"{data_folder}/occupancies.npy")[:L]
H_active = np.load(f"{data_folder}/hamiltonian.npy").real[:L, :L]
index_active_region = np.load(f"{data_folder}/index_active_region.npy")

len_active = occupancy_goal.size
z_ret = np.load(f"{data_folder}/energies.npy")
eta = z_ret.imag[0]
energies = z_ret.real
beta = 1000
z_mats = np.load(f"{data_folder}/matsubara_energies.npy")

# hyb_mats = np.fromfile(f"{data_folder}/matsubara_hybridization.bin", complex).reshape(
#     z_mats.size,
#     len_active,
#     len_active,
# )
# _HybMats = interp1d(z_mats.imag, hyb_mats, axis=0, bounds_error=False, fill_value=0.0)
# HybMats = lambda z: _HybMats(z.imag)

# hyb_ret = np.fromfile(f"{data_folder}/hybridization.bin", complex).reshape(
#     z_ret.size,
#     len_active,
#     len_active,
# )
# _HybRet = interp1d(z_ret.real, hyb_ret, axis=0, bounds_error=False, fill_value=0.0)
# HybRet = lambda z: _HybRet(z.real)

HybMats = lambda z: 0.0
HybRet = lambda z: 0.0

S_active = np.eye(len_active)
idx_neq = np.arange(len_active)
idx_inv = np.arange(len_active)

V = np.eye(len_active) * U

# Apply double counting correction if specified
double_counting = (
    np.diag(V.diagonal() * (occupancy_goal - 0.5))
    if use_double_counting
    else np.zeros((len_active, len_active))
)
gfloc = Gfloc(
    H_active - double_counting,
    S_active,
    HybMats,
    idx_neq,
    idx_inv,
    nmats=z_mats.size,
    beta=beta,
)

nimp = gfloc.idx_neq.size
gfimp = [Gfimp(nbaths, z_mats.size, V[i, i], beta) for i in range(nimp)]
gfimp = nanoGfimp(gfimp)

Sigma = lambda z: np.zeros((nimp, z.size), complex)

gfloc0 = Gfloc(
    H_active, S_active, HybMats, idx_neq, idx_inv, nmats=z_mats.size, beta=beta
)
gfloc0.set_local(Sigma)
gfloc0.update(mu=0.0)

# occupancy_goal_ = occupancy_goal[gfloc.idx_neq]
occupancy_goal_ = gfloc0.integrate(0.0)

# Initialize DMFT with adjust_mu parameter
dmft = DMFT(
    gfimp,
    gfloc,
    occupancy_goal_,
    max_iter=max_iter,
    tol=tol,
    adjust_mu=adjust_mu,
    alpha=alpha,
    DC=double_counting,
)

delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=0.0)
delta_prev = delta.copy()
dmft.delta = delta

try:
    dmft.solve(dmft.delta, alpha=1.0, callback=callback)
except:
    pass

sigma_data = dmft.Sigma(z_ret)

_Sigma = (
    lambda z: -double_counting.diagonal()[:, None] - gfloc.mu + gfloc.Sigma(z)[idx_inv]
)
dmft_sigma_file = f"{data_folder}/dmft_sigma.npy"
save_sigma(_Sigma(z_ret), dmft_sigma_file, nspin)

gfloc_data = gfloc(z_ret)
np.save(f"{data_folder}/dmft_gfloc.npy", gfloc_data)

np.save(f"{data_folder}/opt_delta_dmft", delta_prev)
np.save(f"{data_folder}/opt_mu_dmft", gfloc.mu)
