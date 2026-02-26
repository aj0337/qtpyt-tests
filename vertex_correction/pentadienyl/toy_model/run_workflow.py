from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    # When running `python run_workflow.py` from this folder.
    from workflow_core import (
        build_hamiltonian,
        build_ppp_matrix,
        compute_ed_self_energy,
        compute_G_retarded,
        compute_transmission,
        compute_vertex_correction,
    )
except ModuleNotFoundError:
    # Fallback when importing via package path (e.g. from repo root).
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from vertex_correction.pentadienyl.toy_model.workflow_core import (
        build_hamiltonian,
        build_ppp_matrix,
        compute_ed_self_energy,
        compute_G_retarded,
        compute_transmission,
        compute_vertex_correction,
    )

# =====================
# USER CONFIG (edit me)
# =====================
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SPECIALIZED_OUTPUT_DIR = (
    OUTPUT_DIR / "ed/no_vertex"
)  # for calculation-specific outputs (e.g. self-energy, transmission)

# Model
ONSITE_PARAMS = [-0.5, 0.25, 0.5]
NEAREST_NEIGHBOR_T = 0.5
SECOND_NEAREST_NEIGHBOR_T = 0.0

# Leads (broadening on first/last site)
GAMMA_L = 0.05
GAMMA_R = 0.05

# PPP geometry + parameters
U_ONSITE = 4.62
COULOMB_EV_A = 14.397
PPP_SYMBOLS = ("C", "N")
XYZ_FILENAME = "scatt.xyz"  # expected inside OUTPUT_DIR

# ED parameters
OCCUPANCIES = [1.0, 1.0, 1.0]  # set to None to default to ones
ETA = 1e-2
BETA = 1000.0
DE = 0.01
EMIN = -4.0
EMAX = 4.0
NEIG_VALUE = 8

# Vertex correction scheme for inelastic contribution
# One of: "none", "ferretti", "brazilian"
VERTEX_SCHEME = "none"


def main() -> None:
    onsite_params = [float(value) for value in ONSITE_PARAMS]
    nsites = len(onsite_params)
    if nsites < 2:
        raise ValueError("ONSITE_PARAMS must contain at least 2 values.")

    if OCCUPANCIES is None:
        occupancies = np.ones(nsites, dtype=float)
    else:
        occupancies = np.array([float(value) for value in OCCUPANCIES], dtype=float)

    if occupancies.shape != (nsites,):
        raise ValueError(
            "OCCUPANCIES must be a list with the same length as ONSITE_PARAMS. "
            f"Got len(OCCUPANCIES)={occupancies.shape[0]} and len(ONSITE_PARAMS)={nsites}."
        )

    output_dir = OUTPUT_DIR
    specialized_output_dir = SPECIALIZED_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(specialized_output_dir, exist_ok=True)

    H = build_hamiltonian(onsite_params, NEAREST_NEIGHBOR_T, SECOND_NEAREST_NEIGHBOR_T)
    np.save(os.path.join(output_dir, "hamiltonian.npy"), H)

    Gamma_L = np.zeros((nsites, nsites), dtype=complex)
    Gamma_R = np.zeros((nsites, nsites), dtype=complex)
    Gamma_L[0, 0] = float(GAMMA_L)
    Gamma_R[-1, -1] = float(GAMMA_R)
    np.save(os.path.join(output_dir, "gamma_L.npy"), Gamma_L)
    np.save(os.path.join(output_dir, "gamma_R.npy"), Gamma_R)

    np.save(os.path.join(output_dir, "occupancies.npy"), occupancies)

    U_ppp = build_ppp_matrix(
        system_root=str(output_dir),
        num_sites=nsites,
        U_onsite=float(U_ONSITE),
        coulomb_evA=float(COULOMB_EV_A),
        symbols=tuple(PPP_SYMBOLS),
        xyz_path=str(XYZ_FILENAME),
    )
    np.savetxt(os.path.join(output_dir, "U_ppp.txt"), U_ppp)

    energies, sigma_correlated = compute_ed_self_energy(
        input_folder=str(output_dir),
        output_folder=str(specialized_output_dir),
        eta=float(ETA),
        beta=float(BETA),
        de=float(DE),
        e_min=float(EMIN),
        e_max=float(EMAX),
        neig_value=int(NEIG_VALUE),
    )

    # Save the full retarded Green's function G(E) on the same grid.
    # Shape: (nE, n, n)
    G_ret = np.empty((energies.size, nsites, nsites), dtype=np.complex128)
    for eidx, E in enumerate(energies):
        G_ret[eidx] = compute_G_retarded(
            float(E),
            H,
            Gamma_L,
            Gamma_R,
            sigma_correlated=sigma_correlated[eidx],
            eta=float(ETA),
        )
    np.savez_compressed(
        os.path.join(specialized_output_dir, "G_retarded.npz"),
        energies=energies,
        G=G_ret,
    )

    T = compute_transmission(
        energies,
        H,
        Gamma_L,
        Gamma_R,
        sigma_correlated=sigma_correlated,
        eta=float(ETA),
        scheme=str(VERTEX_SCHEME),
    )
    np.save(
        os.path.join(specialized_output_dir, "ET.npy"),
        np.array([energies, T["elastic"], T["inelastic"], T["total"]]),
    )

    # Save the vertex-correction matrix Lambda(E) so it can be analyzed directly.
    # Shape: (nE, n, n)
    Lambda = np.empty((energies.size, nsites, nsites), dtype=np.complex128)
    for eidx in range(energies.size):
        Lambda[eidx] = compute_vertex_correction(
            gamma_L=Gamma_L,
            gamma_R=Gamma_R,
            sigma_correlated=sigma_correlated[eidx],
            eta=float(ETA),
            scheme=str(VERTEX_SCHEME),
        )
    lambda_trace = np.trace(Lambda, axis1=1, axis2=2)
    np.savez_compressed(
        os.path.join(specialized_output_dir, "vertex_correction.npz"),
        energies=energies,
        Lambda=Lambda,
        trace=lambda_trace,
        scheme=str(VERTEX_SCHEME),
        eta=float(ETA),
    )

    print("Workflow completed.")
    print(f"Output directory: {output_dir}")
    print(f"Saved self-energy: {specialized_output_dir / 'self_energy.npy'}")
    print(f"Saved Green's function: {specialized_output_dir / 'G_retarded.npz'}")
    print(
        "Saved transmission components (E, T_elastic, T_inelastic, T_total): "
        f"{specialized_output_dir / 'ET.npy'}"
    )
    print(
        "Saved vertex correction (energies, Lambda(E), trace(Lambda)): "
        f"{specialized_output_dir / 'vertex_correction.npz'}"
    )


if __name__ == "__main__":
    main()
