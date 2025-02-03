from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from ase.io import read
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings


dft_types = ["device_fd_0.0", "device_fd_1e-3"]
for dft_type in dft_types:
    lowdin = True
    data_folder = f"./output/lowdin/{dft_type}"
    # Create the folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if lowdin:
        H_subdiagonalized, S_subdiagonalized = np.load(f"{data_folder}/hs_los_lowdin.npy")

    else:
        H_subdiagonalized, S_subdiagonalized = np.load(
            f"{data_folder}/hs_los_no_lowdin.npy"
        )
    H_subdiagonalized = H_subdiagonalized.astype(np.complex128)
    S_subdiagonalized = S_subdiagonalized.astype(np.complex128)

    GPWDEVICEDIR = f"./dft/{dft_type}"
    GPWLEADSDIR = "./dft/leads/"

    cc_path = Path(GPWDEVICEDIR)
    pl_path = Path(GPWLEADSDIR)

    H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")

    basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}

    leads_atoms = read(pl_path / "leads.xyz")
    leads_basis = Basis.from_dictionary(leads_atoms, basis_dict)

    device_atoms = read(cc_path / "scatt.xyz")
    device_basis = Basis.from_dictionary(device_atoms, basis_dict)

    nodes = [0, 810, 1116, 1278, 1584, 2394]

    # Define energy range and broadening factor for the Green's function calculation
    de = 0.01
    energies = np.arange(-2.0, 2.0 + de / 2.0, de).round(7)
    eta = 5e-3

    # Define the number of repetitions (Nr) and unit cell repetition in the leads
    Nr = (1, 5, 3)
    unit_cell_rep_in_leads = (5, 5, 3)

    # Prepare the k-points and matrices for the leads (Hamiltonian and overlap matrices)
    kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
        H_leads_lcao,
        S_leads_lcao,
        unit_cell_rep_in_leads,
        align=(0, H_subdiagonalized[0, 0, 0]),
    )

    # Remove periodic boundary conditions (PBC) from the device Hamiltonian and overlap matrices
    remove_pbc(device_basis, H_subdiagonalized)
    remove_pbc(device_basis, S_subdiagonalized)

    # Initialize self-energy list for left and right leads
    self_energy = [None, None, None]
    self_energy[0] = PrincipalSelfEnergy(
        kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr
    )
    self_energy[1] = PrincipalSelfEnergy(
        kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr, id="right"
    )

    # Rotate the couplings for the leads based on the specified basis and repetition Nr
    rotate_couplings(leads_basis, self_energy[0], Nr)
    rotate_couplings(leads_basis, self_energy[1], Nr)

    # Tridiagonalize the device Hamiltonian and overlap matrices based on the partitioned nodes
    hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
        nodes, H_subdiagonalized[0], S_subdiagonalized[0]
    )
    del H_subdiagonalized, S_subdiagonalized

    save_path = os.path.join(data_folder, "hs_list_ii.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(hs_list_ii, f)
    save_path = os.path.join(data_folder, "hs_list_ij.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(hs_list_ij, f)
    np.save(os.path.join(data_folder, "self_energy.npy"), self_energy)
    np.save(os.path.join(data_folder, "retarded_energies.npy"), energies + 1.0j * eta)
