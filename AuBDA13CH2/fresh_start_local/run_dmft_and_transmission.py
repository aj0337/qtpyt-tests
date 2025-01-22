from __future__ import annotations
import pickle
import numpy as np
from edpyt.dmft import DMFT, Gfimp
from edpyt.nano_dmft import Gfimp as nanoGfimp
from scipy.interpolate import interp1d
from qtpyt.projector import expand
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.block_tridiag import greenfunction
from edpyt.nano_dmft import Gfloc
from qtpyt.parallel.egrid import GridDesc

import numpy as np
import matplotlib.pyplot as plt
import os


class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_molecule_identity, super().retarded(energy), idx_molecule)


def distance(delta):
    global delta_prev
    delta_prev[:] = delta
    return dmft.distance(delta)


def get_sigma(sigma_diag):
    L, ne = sigma_diag.shape
    sigma = np.zeros((ne, L, L), complex)

    for diag, mat in zip(sigma_diag.T, sigma):
        mat.flat[:: (L + 1)] = diag
    return DataSelfEnergy(energies, sigma)


def plot(
    gf,
    sigma_func,
    transmission,
    semilogy=True,
    reference_gf=None,
    reference_T=None,
    label_ref="DFT",
):
    """Plot the Green's function DOS and Tr(Sigma) with an optional reference DOS."""

    fig, axes = plt.subplots(3, 1, sharex=True)
    ax1, ax2, ax3 = axes

    w = z_ret.real
    dos = -1 / np.pi * gf(z_ret).sum(axis=0).imag
    if semilogy:
        ax1.semilogy(w, dos, label="DMFT") if dos.ndim == 1 else ax1.semilogy(
            w, dos[0], label=r"spin $\uparrow$"
        )
    else:
        ax1.plot(w, dos, label="DMFT") if dos.ndim == 1 else ax1.plot(
            w, dos[0], label=r"spin $\uparrow$"
        )

    if reference_gf is not None:
        reference_dos = -1 / np.pi * reference_gf(z_ret).sum(axis=0).imag
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

    ax2.set_ylabel("Tr(Sigma) [eV]")
    ax2.legend(loc="lower right")

    if reference_T is not None:
        ax3.semilogy(reference_T[0, :], reference_T[1, :], label="Reference DMFT")

    if semilogy:
        ax3.semilogy(w, transmission, label="Computed")
    else:
        ax3.plot(w, transmission, label="Computed")
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(bottom=1e-5)
    ax3.legend(loc="lower right")
    ax3.set_xlabel("E-E$_F$ [eV]")
    ax3.set_ylabel("T(E)")

    plt.subplots_adjust(hspace=0)
    return ax1


iteration_counter = 0


def callback(*args, **kwargs):
    global iteration_counter

    def sigma_func(z):
        return (
            -double_counting.diagonal()[:, None]
            - gfloc_with_dccorrection.mu
            + gfloc_with_dccorrection.Sigma(z)
        )

    # compute transmission
    gf = greenfunction.GreenFunction(
        hs_list_ii,
        hs_list_ij,
        [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
        solver="dyson",
        eta=eta,
    )
    dmft_sigma = get_sigma(sigma_func(z_ret))
    self_energy[2] = dmft_sigma
    gf.selfenergies.append((imb, self_energy[2]))

    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)
    ###

    ax1 = plot(
        gf=gfloc_with_dccorrection,
        sigma_func=sigma_func,
        transmission=T,
        reference_gf=gfloc_no_dccorrection,
        reference_T=reference_T,
        label_ref="DFT",
        semilogy=kwargs.get("semilogy", True),
    )
    mu_value = gfloc_with_dccorrection.mu
    ax1.set_title(f"Callback Iteration {iteration_counter} | $\mu$ = {mu_value:.4f} eV")

    figure_filename = os.path.join(
        figure_folder,
        f"callback_iter_{iteration_counter:03d}_mu_{mu_value:.4f}_dos.png",
    )
    plt.xlim(-2, 2)
    plt.savefig(figure_filename, dpi=300, bbox_inches="tight")
    plt.close()

    dmft_occupancy = gfloc_with_dccorrection.integrate(gfloc_with_dccorrection.mu)

    fig, ax = plt.subplots(figsize=(8, 4))
    x_indices = np.arange(len(occupancy_goal))
    ax.bar(
        x_indices - 0.2,
        occupancy_goal,
        width=0.4,
        label="Occupancy Goal",
        color="blue",
        align="center",
    )
    ax.bar(
        x_indices + 0.2,
        dmft_occupancy,
        width=0.4,
        label="DMFT Occupancy",
        color="orange",
        align="center",
    )

    ax.set_xlabel("Impurity Index")
    ax.set_ylabel("Occupancy")
    ax.legend()
    ax.set_title(f"Occupancy Comparison | Iteration {iteration_counter}")

    barplot_filename = os.path.join(
        figure_folder, f"callback_iter_{iteration_counter:03d}_occupancy.png"
    )
    plt.savefig(barplot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    iteration_counter += 1


nbaths = 4
U = 4
tol = 1e-4
max_iter = 1000
alpha = 0.0
nspin = 1
de = 0.01
energies = np.arange(-2, 2 + de / 2.0, de).round(7)
eta = 1e-2
z_ret = energies + 1.0j * eta
beta = 1000
mu = 1e-3
adjust_mu = True
use_double_counting = True

data_folder = "output/lowdin"
output_folder = f"output/lowdin/U_{U}/occp_gfp/tol_1e-4/eta_{eta}"
figure_folder = f"{output_folder}/figures"
occupancy_goal = np.load(f"{data_folder}/occupancies_gfp.npy")
H_active = np.load(f"{data_folder}/bare_hamiltonian.npy").real
z_mats = np.load(f"{data_folder}/matsubara_energies.npy")
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
dft_dos = np.load(f"{data_folder}/dft_dos.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)


with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)

with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)


nodes = [0, 810, 1116, 1278, 1584, 2394]
imb = 2  # index of molecule block from the nodes list
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
S_molecule_identity = np.eye(S_molecule.shape[0])
idx_molecule = index_active_region - nodes[imb]

os.makedirs(output_folder, exist_ok=True)
os.makedirs(figure_folder, exist_ok=True)

len_active = occupancy_goal.size
hyb_mats = np.fromfile(f"{data_folder}/matsubara_hybridization.bin", complex).reshape(
    z_mats.size,
    len_active,
    len_active,
)
_HybMats = interp1d(z_mats.imag, hyb_mats, axis=0, bounds_error=False, fill_value=0.0)
HybMats = lambda z: _HybMats(z.imag)

hyb_ret = np.fromfile(f"{data_folder}/hybridization.bin", complex).reshape(
    z_ret.size,
    len_active,
    len_active,
)
_HybRet = interp1d(z_ret.real, hyb_ret, axis=0, bounds_error=False, fill_value=0.0)
HybRet = lambda z: _HybRet(z.real)

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
gfloc_with_dccorrection = Gfloc(
    H_active - double_counting,
    S_active,
    HybMats,
    idx_neq,
    idx_inv,
    nmats=z_mats.size,
    beta=beta,
)

nimp = gfloc_with_dccorrection.idx_neq.size
gfimp = [Gfimp(nbaths, z_mats.size, V[i, i], beta) for i in range(nimp)]
gfimp = nanoGfimp(gfimp)

Sigma = lambda z: np.zeros((nimp, z.size), complex)

gfloc_no_dccorrection = Gfloc(
    H_active, S_active, HybMats, idx_neq, idx_inv, nmats=z_mats.size, beta=beta
)
gfloc_no_dccorrection.update(mu=mu)
gfloc_no_dccorrection.set_local(Sigma)
reference_T = np.load(f"reference/ET_dmft.npy")
# Initialize DMFT with adjust_mu parameter
dmft = DMFT(
    gfimp,
    gfloc_with_dccorrection,
    occupancy_goal,
    max_iter=max_iter,
    tol=tol,
    adjust_mu=adjust_mu,
    alpha=alpha,
    DC=double_counting,
)

delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=mu)
delta_prev = delta.copy()
dmft.delta = delta
dmft.solve(dmft.delta, alpha=1.0, callback=callback)

# try:
#     dmft.solve(dmft.delta, alpha=1.0, callback=callback)
# except:
#     pass

gfloc_data = gfloc_with_dccorrection(z_ret)
np.save(f"{output_folder}/dmft_gfloc.npy", gfloc_data)

np.save(f"{output_folder}/opt_delta_dmft", delta_prev)
np.save(f"{output_folder}/opt_mu_dmft", gfloc_with_dccorrection.mu)
