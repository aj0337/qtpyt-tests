from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from qtpyt.basis import Basis
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings


def compute_gamma_from_sigma(sigma: np.ndarray) -> np.ndarray:
    """
    Compute the broadening matrix Γ(ω) = i [Σ^r(ω) - Σ^a(ω)] from one Σ^r(ω)

    Parameters
    ----------
    sigma : np.ndarray
        Retarded self-energy matrix Σ^r(ω)

    Returns
    -------
    gamma : np.ndarray
        Broadening matrix Γ(ω) = -2 Im[Σ^r(ω)]
    """
    return 1j * (sigma - sigma.conj().T)


def plot_matrix_symmetry_pair(
    A, B, title_A, title_B, flip=True, cmap="viridis", clip=None
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if flip:
        B_plot = B[::-1, ::-1].conj()
    else:
        B_plot = B

    A_plot = A
    if clip is not None:
        A_plot = A_plot[-clip:, -clip:]
        B_plot = B_plot[-clip:, -clip:]

    vmin = 0
    vmax = max(np.abs(A_plot.real).max(), np.abs(B_plot.real).max())

    im1 = axes[0].imshow(np.abs(A_plot.real), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(title_A)
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(np.abs(B_plot.real), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(title_B)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def compute_transmission(
    sigma_L: np.ndarray,
    sigma_R: np.ndarray,
    G_r: np.ndarray,
) -> float:
    """
    Compute the Landauer transmission T(E) at a given energy in the molecule subspace.

    Parameters
    ----------
    sigma_L : np.ndarray
        Projected self-energy of the left lead into the molecule space at energy E (n x n).
    sigma_R : np.ndarray
        Projected self-energy of the right lead into the molecule space at energy E (n x n).
    G_r : np.ndarray
        Retarded Green's function of the molecule region at energy E (n x n).
    Returns
    -------
    T_E : float
        Transmission function T(E) at the given energy.

    Notes
    -----
    The transmission is computed using:

        T(E) = Tr[ Γ_L(E) G^r(E) Γ_R(E) G^a(E) ]
    """
    G_a = G_r.conj().T
    gamma_L = compute_gamma_from_sigma(sigma_L)
    gamma_R = compute_gamma_from_sigma(sigma_R)

    T_E = np.real(np.trace(gamma_L @ G_r @ gamma_R @ G_a))
    return T_E


def combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, side):
    """
    Merge the lead bulk and tip blocks into a single diagonal block.
    Return the merged diagonal block and the corresponding off-diagonal block
    that connects the merged lead to the molecule.

    Parameters
    ----------
    hs_list_ii : list of tuple
        Full diagonal blocks (Hii, Sii) of the entire system.
    hs_list_ij : list of tuple
        Full off-diagonal blocks (Hij, Sij) of the entire system.
    side : str
        "left" or "right"

    Returns
    -------
    hs_list_ii_combined : list of tuple
        One-element list with merged (H, S) block for the selected lead.

    hs_list_ij_combined : list of tuple
        One-element list with (H, S) block coupling merged lead to molecule.
        For left lead: shape = (1116, 136)
        For right lead: shape = (136, 1116)
    """
    if side == "left":
        # Blocks
        H_bulk, S_bulk = hs_list_ii[0]
        H_tip, S_tip = hs_list_ii[1]
        H_coup, S_coup = hs_list_ij[0]  # bulk–tip: (810, 306)

        # Merge diagonal
        H_merge = np.block([[H_bulk, H_coup], [H_coup.T.conj(), H_tip]])
        S_merge = np.block([[S_bulk, S_coup], [S_coup.T.conj(), S_tip]])

        # Coupling to molecule (block 1–2): shape (306, 136)
        H_lm, S_lm = hs_list_ij[1]
        H_lm_merge = np.vstack([H_lm, np.zeros((810, 136))])  # (1116, 136)
        S_lm_merge = np.vstack([S_lm, np.zeros((810, 136))])

        hs_list_ii_combined = [(H_merge, S_merge)]
        hs_list_ij_combined = [(H_lm_merge, S_lm_merge)]

    elif side == "right":
        # Blocks
        H_tip, S_tip = hs_list_ii[3]
        H_bulk, S_bulk = hs_list_ii[4]
        H_coup, S_coup = hs_list_ij[3]  # tip–bulk: (306, 810)

        # Merge diagonal
        H_merge = np.block([[H_tip, H_coup], [H_coup.T.conj(), H_bulk]])
        S_merge = np.block([[S_tip, S_coup], [S_coup.T.conj(), S_bulk]])

        # Coupling from molecule to tip (block 2–3): shape (136, 306)
        H_mr, S_mr = hs_list_ij[2]
        H_mr_merge = np.hstack([H_mr, np.zeros((136, 810))])  # (136, 1116)
        S_mr_merge = np.hstack([S_mr, np.zeros((136, 810))])

        hs_list_ii_combined = [(H_merge, S_merge)]
        hs_list_ij_combined = [(H_mr_merge, S_mr_merge)]

    else:
        raise ValueError("side must be 'left' or 'right'")

    return hs_list_ii_combined, hs_list_ij_combined


def compute_projected_self_energy(
    hs_list_ii: list,
    hs_list_ij: list,
    sigma_lead: np.ndarray,
    omega: float,
    direction: str = "left",
) -> np.ndarray:
    """
    Compute the projected self-energy Σ̃^{Mol}_{L/R}(ω) using the full lead self-energy.

    Parameters
    ----------
    hs_list_ii : list of (Hii, Sii)
        List of diagonal Hamiltonian and overlap blocks.
        Expected order: [(lead+tip), mol, (lead+tip)]
    hs_list_ij : list of (Hij, Sij)
        List of off-diagonal Hamiltonian and overlap blocks.
        Expected order: [(lead+tip)-mol, mol-(lead+tip)]
    sigma_lead : np.ndarray
        Full self-energy Σ_L(ω) or Σ_R(ω) of the lead at energy ω (same size as the lead block).
    omega : float
        Energy value ω at which to evaluate Σ.
    direction : str
        'left' or 'right', indicating which lead to project from.

    Returns
    -------
    Sigma_proj : np.ndarray
        Projected self-energy matrix Σ̃^{Mol}_{L/R}(ω)

    Notes
    -----
    Implements the expression:

        Σ̃_L(E) = [E S_ml - H_ml] [E S_l - H_l - Σ_L(E)]⁻¹ [E S_lm - H_lm]

    where:
        S_ml, H_ml : Overlap and Hamiltonian coupling blocks between molecule and lead.
        S_l, H_l   : Overlap and Hamiltonian of the lead.
        Σ_L(E)     : Self-energy of the lead at energy E.

    This equation has been described in https://doi.org/10.1063/1.4897448
    """
    z = omega + 1e-12j

    if direction == "left":
        H_l, S_l = hs_list_ii[0]
        H_lm, S_lm = hs_list_ij[0]  # lead–molecule coupling
        H_ml = H_lm.T.conj()  # molecule–lead coupling
        S_ml = S_lm.T.conj()
    elif direction == "right":
        H_l, S_l = hs_list_ii[-1]
        H_ml, S_ml = hs_list_ij[-1]  # molecule–lead coupling
        H_lm = H_ml.T.conj()
        S_lm = S_ml.T.conj()
    else:
        raise ValueError("Direction must be 'left' or 'right'.")

    # A: (n_mol, n_lead)
    A = z * S_ml - H_ml
    # B_inv: (n_lead, n_lead)
    B_inv = np.linalg.inv(z * S_l - H_l - sigma_lead)
    # C: (n_lead, n_mol)
    C = z * S_lm - H_lm
    D = A @ B_inv @ C
    return D


def pad_self_energy_to_full_space(
    sigma_lead: np.ndarray, n_full: int, direction: str = "left"
) -> np.ndarray:
    """
    Pad a self-energy defined in the lead subspace to the full lead+tip subspace.

    Parameters
    ----------
    sigma_lead : np.ndarray
        Square self-energy matrix defined in the lead subspace, shape (n_lead, n_lead).
    n_full : int
        Dimension of the full lead+tip subspace.
    direction : str
        'left' or 'right'. If 'left', pads into top-left corner.
        If 'right', pads into bottom-right corner.

    Returns
    -------
    sigma_padded : np.ndarray
        Full self-energy matrix of shape (n_full, n_full) with zeros elsewhere.

    Notes
    -----
    If direction == 'left':
        Σ_full = [ Σ_lead     0      ]
                  [   0     0_pad    ]

    If direction == 'right':
        Σ_full = [ 0_pad     0       ]
                  [   0     Σ_lead   ]
    """
    n_lead = sigma_lead.shape[0]
    sigma_padded = np.zeros((n_full, n_full), dtype=sigma_lead.dtype)

    if direction == "left":
        sigma_padded[:n_lead, :n_lead] = sigma_lead
    elif direction == "right":
        sigma_padded[-n_lead:, -n_lead:] = sigma_lead
    else:
        raise ValueError("Direction must be 'left' or 'right'.")

    return sigma_padded


def compute_greens_function_mol(
    H_mol: np.ndarray,
    S_mol: np.ndarray,
    sigma_L: np.ndarray,
    sigma_R: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Compute the retarded Green's function G_mol(E) in the molecule subspace.

    Parameters
    ----------
    H_mol : np.ndarray
        Hamiltonian matrix of the molecule region, shape (n_mol, n_mol).
    S_mol : np.ndarray
        Overlap matrix of the molecule region, shape (n_mol, n_mol).
    sigma_L : np.ndarray
        Projected self-energy from the left lead at energy E, shape (n_mol, n_mol).
    sigma_R : np.ndarray
        Projected self-energy from the right lead at energy E, shape (n_mol, n_mol).
    omega : float
        Energy value E (in eV) at which to evaluate the Green's function.

    Returns
    -------
    G_mol : np.ndarray
        Retarded Green's function G_mol(E), shape (n_mol, n_mol).

    Notes
    -----
    Implements the expression:

        G_mol(E) = [E S_mol - H_mol - Σ̃_L(E) - Σ̃_R(E)]⁻¹

    with E = ω + iη, where η is a small positive imaginary part for causality.
    """
    z = omega + 1e-2j  # small imaginary shift for retarded GF
    M = z * S_mol - H_mol - sigma_L - sigma_R
    return np.linalg.inv(M)


GPWDEVICEDIR = f"./dft/device"
GPWLEADSDIR = "./dft/leads/"

cc_path = Path(GPWDEVICEDIR)
pl_path = Path(GPWLEADSDIR)

H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")

basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}

leads_atoms = read(pl_path / "leads.xyz")
leads_basis = Basis.from_dictionary(leads_atoms, basis_dict)

device_atoms = read(cc_path / "scatt.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

Nr = (1, 5, 3)
unit_cell_rep_in_leads = (5, 5, 3)

nodes = np.load("output/lowdin/nodes.npy")

data_folder = "./output/lowdin"

E, T_dft_ref = np.load(f"{data_folder}/dft/ET.npy")

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)

with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

leads_se = np.load("./output/lowdin/self_energy.npy", allow_pickle=True)
de = 0.01
E_sampler = np.arange(-3, 3 + de / 2.0, de).round(7)
se_left = np.array([leads_se[0].retarded(e) for e in E_sampler])
se_right = np.array([leads_se[1].retarded(e) for e in E_sampler])

gamma_left = compute_gamma_from_sigma(se_left[0])
gamma_right = compute_gamma_from_sigma(se_right[0])

data_folder = f"./output/lowdin"
H_subdiagonalized, S_subdiagonalized = np.load(f"{data_folder}/hs_los_lowdin.npy")

H_subdiagonalized = H_subdiagonalized.astype(np.complex128)
S_subdiagonalized = S_subdiagonalized.astype(np.complex128)

kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
    H_leads_lcao,
    S_leads_lcao,
    unit_cell_rep_in_leads,
    align=(0, H_subdiagonalized[0, 0, 0]),
)

remove_pbc(device_basis, H_subdiagonalized)
remove_pbc(device_basis, S_subdiagonalized)

self_energy = [None, None, None]
self_energy[0] = PrincipalSelfEnergy(
    kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr
)
self_energy[1] = PrincipalSelfEnergy(
    kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr, id="right"
)

rotate_couplings(leads_basis, self_energy[0], Nr)
rotate_couplings(leads_basis, self_energy[1], Nr)

hs_ii_left, hs_ij_left = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "left")

hs_ii_right, hs_ij_right = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "right")

T_energy = []
H_mol, S_mol = hs_list_ii[2]
for i, omega in enumerate(E_sampler):
    sigma_L = self_energy[0].retarded(omega)
    sigma_R = self_energy[1].retarded(omega)

    sigma_L_padded = pad_self_energy_to_full_space(
        sigma_L, n_full=hs_ii_left[0][0].shape[0], direction="left"
    )
    sigma_L = compute_projected_self_energy(
        hs_list_ii=hs_ii_left,
        hs_list_ij=hs_ij_left,
        sigma_lead=sigma_L_padded,
        omega=omega,
        direction="left",
    )

    sigma_R_padded = pad_self_energy_to_full_space(
        sigma_R, n_full=hs_ii_right[0][0].shape[0], direction="right"
    )
    sigma_R = compute_projected_self_energy(
        hs_list_ii=hs_ii_right,
        hs_list_ij=hs_ij_right,
        sigma_lead=sigma_R_padded,
        omega=omega,
        direction="right",
    )

    G_mol = compute_greens_function_mol(
        H_mol=H_mol, S_mol=S_mol, sigma_L=sigma_L, sigma_R=sigma_R, omega=omega
    )
    T = compute_transmission(sigma_R=sigma_R, sigma_L=sigma_L, G_r=G_mol)

    T_energy.append(T)

plt.plot(E, T_dft_ref)
plt.plot(E_sampler, T_energy, "-.", label="T(E) from projected SE")
plt.ylabel("T (E)")
plt.xlabel("E (eV)")
plt.yscale("log")
plt.legend()
plt.savefig(f"{data_folder}/dft/projected_ET.png")
