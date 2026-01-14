from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction
from qtpyt.parallel.egrid import GridDesc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def run(outputfile):
    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)
    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies, T.real))
        plt.figure()
        plt.plot(energies, T)
        plt.yscale("log")
        plt.xlim(-3.0, 3.0)
        plt.ylim(1e-5, 1)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Transmission")
        plt.tight_layout()
        plt.savefig(f"{dft_data_folder}/ET_dft.png", dpi=300)
        plt.close()


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
    z = energy + eta

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
E_ref, T_dft_ref = np.load(f"{data_folder}/dft/ET.npy")
dft_data_folder = "./output/lowdin/dft"
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)


de = 0.01
E_sampler = np.arange(-3, 3 + de / 2.0, de).round(7)
eta = 1e-2
nE = len(E_sampler)
i_start, i_end, counts, displs = split_indices(nE, size, rank)
E_local = E_sampler[i_start:i_end]

hs_ii_left, hs_ij_left = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "left")
hs_ii_right, hs_ij_right = combine_HS_leads_tip_blocks(hs_list_ii, hs_list_ij, "right")

H_mol, S_mol = hs_list_ii[2]


# Initialize the Green's function solver with the tridiagonalized matrices and self-energies
gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

# Transmission function for DFT
outputfile = f"{dft_data_folder}/projected_ET_qtpyt.npy"
run(outputfile)
