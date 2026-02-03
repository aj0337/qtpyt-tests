from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mpi4py import MPI
from qtpyt.basis import Basis
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.projector import expand

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_molecule_identity, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


def compute_gamma_from_sigma(sigma: np.ndarray) -> np.ndarray:
    """
    Compute the broadening matrix Γ from a retarded self-energy Σ.
    """
    return 1j * (sigma - sigma.conj().T)


def compute_transmission(
    sigma_L: np.ndarray,
    sigma_R: np.ndarray,
    sigma_corr: Optional[np.ndarray],
    G_r: np.ndarray,
    eta: complex,
    ferretti: bool = False,
    brazilian: bool = False,
) -> tuple[float, float, float]:
    """
    Compute (T_elastic, T_inelastic, T_total).
    """
    G_a = G_r.conj().T
    gamma_L = compute_gamma_from_sigma(sigma_L)
    gamma_R = compute_gamma_from_sigma(sigma_R)

    T_elastic = float(np.real(np.trace(gamma_L @ G_r @ gamma_R @ G_a)))

    if ferretti:
        gamma_D = compute_gamma_from_sigma(sigma_corr)
        lambda_corr = compute_ferretti_correction(gamma_D, gamma_L, gamma_R, eta)
    elif brazilian:
        gamma_D = compute_gamma_from_sigma(sigma_corr)
        lambda_corr = gamma_D
    else:
        lambda_corr = np.zeros_like(G_r, dtype=G_r.dtype)

    T_inelastic = float(np.real(np.trace(gamma_L @ G_r @ gamma_R @ lambda_corr @ G_a)))
    T_total = T_elastic + T_inelastic
    return T_elastic, T_inelastic, T_total


def compute_ferretti_correction(
    gamma_D: np.ndarray,
    gamma_L: np.ndarray,
    gamma_R: np.ndarray,
    eta: complex,
) -> np.ndarray:
    """
    Ferretti-like correction piece.
    """
    lambda_corr_inv = gamma_L + gamma_R + 2 * eta * np.eye(gamma_L.shape[0])
    lambda_corr = np.linalg.solve(lambda_corr_inv, gamma_D)
    return lambda_corr


def combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, side: str):
    """
    Merge selected lead and tip blocks -> (lead+tip) and return coupling to molecule.
    """
    if side == "left":
        H_bulk, S_bulk = hs_list_ii[0]
        H_tip, S_tip = hs_list_ii[1]
        H_coup, S_coup = hs_list_ij[0]

        H_merge = np.block([[H_bulk, H_coup], [H_coup.T.conj(), H_tip]])
        S_merge = np.block([[S_bulk, S_coup], [S_coup.T.conj(), S_tip]])

        H_lm, S_lm = hs_list_ij[1]

        H_lm_merge = np.vstack([np.zeros((810, 136)), H_lm])
        S_lm_merge = np.vstack([np.zeros((810, 136)), S_lm])

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
    """
    Zero-pad Σ_lead into full (lead+tip) space.
    """
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
    """
    Compute Σ projected onto molecule by eliminating (lead+tip).
    """
    z = energy + eta * 1j

    if direction == "left":
        H_l, S_l = hs_list_ii[0]
        H_lm, S_lm = hs_list_ij[0]
        H_ml = H_lm.T.conj()
        S_ml = S_lm.T.conj()
        H_lm_use, S_lm_use = H_lm, S_lm

    elif direction == "right":
        H_l, S_l = hs_list_ii[-1]
        H_ml, S_ml = hs_list_ij[-1]
        H_lm_use = H_ml.T.conj()
        S_lm_use = S_ml.T.conj()

    else:
        raise ValueError("direction must be 'left' or 'right'")

    A = z * S_ml - H_ml
    C = z * S_lm_use - H_lm_use
    M = z * S_l - H_l - sigma_lead

    X = np.linalg.solve(M, C)
    return A @ X


def compute_greens_function_mol(
    H_mol: np.ndarray,
    S_mol: np.ndarray,
    sigma_L: np.ndarray,
    sigma_R: np.ndarray,
    sigma_corr: np.ndarray,
    energy: float,
    eta: complex,
) -> np.ndarray:
    """
    G_mol(E) = [ (E+iη)S - H - ΣL - ΣR - Σcorr ]^{-1}
    """
    z = energy + eta * 1j
    M = z * S_mol - H_mol - sigma_L - sigma_R - sigma_corr
    I = np.eye(M.shape[0], dtype=M.dtype)
    return np.linalg.solve(M, I)


def split_indices(n: int, size: int, rank: int):
    """
    Split [0,n) into contiguous chunks for MPI ranks; return (start,end,counts,displs).
    """
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
ed_data_folder = "./output/lowdin/ed"

H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")

basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}
leads_atoms = read(pl_path / "leads.xyz")
leads_basis = Basis.from_dictionary(leads_atoms, basis_dict)
device_atoms = read(cc_path / "scatt.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

E_ref, T_ed_ref = np.load(f"{ed_data_folder}/ET.npy")

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

de = 0.01
E_sampler = np.arange(-3, 3 + de / 2.0, de).round(7)
energies = np.arange(-3, 3 + de / 2.0, de).round(7)
nE = len(E_sampler)
i_start, i_end, counts, displs = split_indices(nE, size, rank)
E_local = E_sampler[i_start:i_end]

leads_self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
se_left = leads_self_energy[0]
se_right = leads_self_energy[1]

ed_self_energy_file = f"{ed_data_folder}/self_energy_with_dcc.npy"
nodes = np.load(f"{data_folder}/nodes.npy")
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
imb = 2
S_molecule = hs_list_ii[imb][1]
S_molecule_identity = np.eye(S_molecule.shape[0])
idx_molecule = index_active_region - nodes[imb]

hs_ii_left, hs_ij_left = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "left")
hs_ii_right, hs_ij_right = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "right")

H_mol, S_mol = hs_list_ii[2]

eta = 1e-2

FERRETTI = True
BRAZILIAN = False

if comm.rank == 0:
    ed_sigma = load(ed_self_energy_file)
else:
    ed_sigma = None
ed_sigma = comm.bcast(ed_sigma, root=0)


T_elastic_local = np.zeros(len(E_local), dtype=float)
T_inelastic_local = np.zeros(len(E_local), dtype=float)
T_total_local = np.zeros(len(E_local), dtype=float)

for i, energy in enumerate(E_local):
    sigma_L_lead = se_left.retarded(energy)
    sigma_R_lead = se_right.retarded(energy)
    sigma_corr = ed_sigma.retarded(energy)

    sigma_L_pad = pad_self_energy_to_full_space(
        sigma_L_lead,
        n_full=hs_ii_left[0][0].shape[0],
        direction="left",
    )
    sigma_R_pad = pad_self_energy_to_full_space(
        sigma_R_lead,
        n_full=hs_ii_right[0][0].shape[0],
        direction="right",
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
        sigma_corr=sigma_corr,
        energy=energy,
        eta=eta,
    )

    Tel, Tin, Ttot = compute_transmission(
        sigma_L=sigma_L_proj,
        sigma_R=sigma_R_proj,
        sigma_corr=sigma_corr,
        G_r=G_mol,
        eta=eta,
        ferretti=FERRETTI,
        brazilian=BRAZILIAN,
    )

    T_elastic_local[i] = Tel
    T_inelastic_local[i] = Tin
    T_total_local[i] = Ttot

T_elastic = None
T_inelastic = None
T_total = None
if rank == 0:
    T_elastic = np.zeros(nE, dtype=float)
    T_inelastic = np.zeros(nE, dtype=float)
    T_total = np.zeros(nE, dtype=float)

comm.Gatherv(
    sendbuf=T_elastic_local,
    recvbuf=(T_elastic, counts, displs, MPI.DOUBLE),
    root=0,
)
comm.Gatherv(
    sendbuf=T_inelastic_local,
    recvbuf=(T_inelastic, counts, displs, MPI.DOUBLE),
    root=0,
)
comm.Gatherv(
    sendbuf=T_total_local,
    recvbuf=(T_total, counts, displs, MPI.DOUBLE),
    root=0,
)

if rank == 0:
    if FERRETTI and not BRAZILIAN:
        scheme = "ferretti"
    elif BRAZILIAN and not FERRETTI:
        scheme = "brazilian"
    else:
        scheme = "none"

    out_dir = Path(ed_data_folder) / scheme
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / "ET_components.npz"
    np.savez(
        out_npz,
        E=E_sampler,
        T_elastic=T_elastic,
        T_inelastic=T_inelastic,
        T_total=T_total,
        eta_gf=float(eta),
        eta_se=float(eta),
    )

    plt.figure()
    plt.plot(E_ref, T_ed_ref, label="ED reference (total)")
    plt.plot(E_sampler, T_elastic, label="T_elastic")
    plt.plot(E_sampler, T_inelastic, label="T_inelastic")
    plt.plot(E_sampler, T_total, "-.", label="T_total")

    plt.ylabel("T(E)")
    plt.xlabel("E (eV)")
    plt.yscale("log")
    plt.legend()

    out_png = out_dir / "ET_components.png"
    plt.savefig(out_png)
    plt.close()
