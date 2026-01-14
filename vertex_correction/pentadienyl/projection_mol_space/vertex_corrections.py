from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mpi4py import MPI
from qtpyt.basis import Basis
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compute_gamma_from_sigma(sigma: np.ndarray) -> np.ndarray:
    """Γ(ω) = i [Σ^r(ω) - Σ^a(ω)] = i [Σ^r - (Σ^r)†]."""
    return 1j * (sigma - sigma.conj().T)


def compute_transmission(
    sigma_L: np.ndarray, sigma_R: np.ndarray, G_r: np.ndarray
) -> float:
    """T(E) = Tr[ Γ_L G^r Γ_R G^a ]."""
    G_a = G_r.conj().T
    gamma_L = compute_gamma_from_sigma(sigma_L)
    gamma_R = compute_gamma_from_sigma(sigma_R)
    return float(np.real(np.trace(gamma_L @ G_r @ gamma_R @ G_a)))


def combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, side: str):
    """Merge bulk+tip blocks into a single (lead+tip) block and return couplings to molecule."""
    if side == "left":
        H_bulk, S_bulk = hs_list_ii[0]
        H_tip, S_tip = hs_list_ii[1]
        H_coup, S_coup = hs_list_ij[0]

        H_merge = np.block([[H_bulk, H_coup], [H_coup.T.conj(), H_tip]])
        S_merge = np.block([[S_bulk, S_coup], [S_coup.T.conj(), S_tip]])

        H_lm, S_lm = hs_list_ij[1]
        H_lm_merge = np.vstack([H_lm, np.zeros((810, 136))])
        S_lm_merge = np.vstack([S_lm, np.zeros((810, 136))])

        return [(H_merge, S_merge)], [(H_lm_merge, S_lm_merge)]

    if side == "right":
        H_tip, S_tip = hs_list_ii[3]
        H_bulk, S_bulk = hs_list_ii[4]
        H_coup, S_coup = hs_list_ij[3]

        H_merge = np.block([[H_tip, H_coup], [H_coup.T.conj(), H_bulk]])
        S_merge = np.block([[S_tip, S_coup], [S_coup.T.conj(), S_bulk]])

        H_mr, S_mr = hs_list_ij[2]
        H_mr_merge = np.hstack([H_mr, np.zeros((136, 810))])
        S_mr_merge = np.hstack([S_mr, np.zeros((136, 810))])

        return [(H_merge, S_merge)], [(H_mr_merge, S_mr_merge)]

    raise ValueError("side must be 'left' or 'right'")


def pad_self_energy_to_full_space(
    sigma_lead: np.ndarray, n_full: int, direction: str
) -> np.ndarray:
    """Pad Σ_lead into Σ_full (lead+tip space)."""
    n_lead = sigma_lead.shape[0]
    sigma_padded = np.zeros((n_full, n_full), dtype=sigma_lead.dtype)
    if direction == "left":
        sigma_padded[:n_lead, :n_lead] = sigma_lead
    elif direction == "right":
        sigma_padded[-n_lead:, -n_lead:] = sigma_lead
    else:
        raise ValueError("direction must be 'left' or 'right'")
    return sigma_padded


def compute_projected_self_energy(
    hs_list_ii: list,
    hs_list_ij: list,
    sigma_lead: np.ndarray,
    energy: float,
    eta: complex,
    direction: str,
) -> np.ndarray:
    """Projected Σ̃ onto molecule: A [zS_l - H_l - Σ_l]^{-1} C."""

    z = energy  # + eta

    if direction == "left":
        H_l, S_l = hs_list_ii[0]
        H_lm, S_lm = hs_list_ij[0]
        H_ml = H_lm.T.conj()
        S_ml = S_lm.T.conj()
        S_lm_use, H_lm_use = S_lm, H_lm

    elif direction == "right":
        H_l, S_l = hs_list_ii[-1]
        H_ml, S_ml = hs_list_ij[-1]
        H_lm_use = H_ml.T.conj()
        S_lm_use = S_ml.T.conj()

    else:
        raise ValueError("direction must be 'left' or 'right'")

    A = z * S_ml - H_ml
    B_inv = np.linalg.inv(z * S_l - H_l - sigma_lead)
    C = z * S_lm_use - H_lm_use
    return A @ B_inv @ C


def compute_greens_function_mol(
    H_mol: np.ndarray,
    S_mol: np.ndarray,
    sigma_L: np.ndarray,
    sigma_R: np.ndarray,
    energy: float,
    eta: complex,
) -> np.ndarray:
    """G_mol(E) = [z S - H - Σ_L - Σ_R]^{-1}."""
    z = energy + eta
    return np.linalg.inv(z * S_mol - H_mol - sigma_L - sigma_R)


def split_indices(n: int, size: int, rank: int):
    """Contiguous block split for MPI Gatherv."""
    counts = np.full(size, n // size, dtype=int)
    counts[: n % size] += 1
    displs = np.concatenate(([0], np.cumsum(counts[:-1])))
    start = int(displs[rank])
    end = int(start + counts[rank])
    return start, end, counts, displs


GPWDEVICEDIR = "./dft/device"
GPWLEADSDIR = "./dft/leads/"
cc_path = Path(GPWDEVICEDIR)
pl_path = Path(GPWLEADSDIR)

data_folder = "./output/lowdin"


H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")

basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}
leads_atoms = read(pl_path / "leads.xyz")
leads_basis = Basis.from_dictionary(leads_atoms, basis_dict)
device_atoms = read(cc_path / "scatt.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

Nr = (1, 5, 3)
unit_cell_rep_in_leads = (5, 5, 3)


E_ref, T_dft_ref = np.load(f"{data_folder}/dft/ET.npy")

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)


de = 0.01
E_sampler = np.arange(-3, 3 + de / 2.0, de).round(7)
nE = len(E_sampler)
i_start, i_end, counts, displs = split_indices(nE, size, rank)
E_local = E_sampler[i_start:i_end]


H_sub, S_sub = np.load(f"{data_folder}/hs_los_lowdin.npy")
H_sub = H_sub.astype(np.complex128)
S_sub = S_sub.astype(np.complex128)

kpts_t, h_kii, s_kii, h_kij, s_kij = prepare_leads_matrices(
    H_leads_lcao,
    S_leads_lcao,
    unit_cell_rep_in_leads,
    align=(0, H_sub[0, 0, 0]),
)

remove_pbc(device_basis, H_sub)
remove_pbc(device_basis, S_sub)

se_left = PrincipalSelfEnergy(kpts_t, (h_kii, s_kii), (h_kij, s_kij), Nr=Nr)
se_right = PrincipalSelfEnergy(
    kpts_t, (h_kii, s_kii), (h_kij, s_kij), Nr=Nr, id="right"
)

rotate_couplings(leads_basis, se_left, Nr)
rotate_couplings(leads_basis, se_right, Nr)


hs_ii_left, hs_ij_left = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "left")
hs_ii_right, hs_ij_right = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "right")

H_mol, S_mol = hs_list_ii[2]


eta_list = [0.0]

for eta_val in eta_list:
    eta = 1j * eta_val

    T_local = np.zeros(len(E_local), dtype=float)

    for i, energy in enumerate(E_local):
        sigma_L_lead = se_left.retarded(energy)
        sigma_R_lead = se_right.retarded(energy)

        sigma_L_pad = pad_self_energy_to_full_space(
            sigma_L_lead, n_full=hs_ii_left[0][0].shape[0], direction="left"
        )
        sigma_R_pad = pad_self_energy_to_full_space(
            sigma_R_lead, n_full=hs_ii_right[0][0].shape[0], direction="right"
        )

        sigma_L_proj = compute_projected_self_energy(
            hs_list_ii=hs_ii_left,
            hs_list_ij=hs_ij_left,
            sigma_lead=sigma_L_pad,
            energy=energy,
            eta=eta,
            direction="left",
        )
        sigma_R_proj = compute_projected_self_energy(
            hs_list_ii=hs_ii_right,
            hs_list_ij=hs_ij_right,
            sigma_lead=sigma_R_pad,
            energy=energy,
            eta=eta,
            direction="right",
        )

        G_mol = compute_greens_function_mol(
            H_mol=H_mol,
            S_mol=S_mol,
            sigma_L=sigma_L_proj,
            sigma_R=sigma_R_proj,
            energy=energy,
            eta=eta,
        )

        T_local[i] = compute_transmission(
            sigma_L=sigma_L_proj, sigma_R=sigma_R_proj, G_r=G_mol
        )

    T_energy = None
    if rank == 0:
        T_energy = np.zeros(nE, dtype=float)

    comm.Gatherv(
        sendbuf=T_local,
        recvbuf=(T_energy, counts, displs, MPI.DOUBLE),
        root=0,
    )

    if rank == 0:
        plt.figure()
        plt.plot(E_ref, T_dft_ref, label="DFT reference")
        plt.plot(E_sampler, T_energy, "-.", label=f"Projected SE (η={eta_val:.0e})")
        plt.ylabel("T (E)")
        plt.xlabel("E (eV)")
        plt.yscale("log")
        plt.legend()

        out_png = f"{data_folder}/dft/projected_ET_eta_{eta_val:.0e}.png"
        plt.savefig(out_png)
        plt.close()
