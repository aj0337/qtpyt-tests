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

# Workflow method:
#   - "ed": compute correlated self-energy via ED (edpyt)
#   - "dft": no correlated self-energy (Sigma_correlated=None)
#   - "dmft": load a precomputed correlated self-energy from disk
METHOD = "ed"
# Vertex correction scheme for inelastic contribution
# One of: "none", "ferretti", "brazilian"
VERTEX_SCHEME = "ferretti"

# Model
ONSITE_PARAMS = [-1.5, 0.25, 1.5]
NEAREST_NEIGHBOR_T = 0.5
SECOND_NEAREST_NEIGHBOR_T = 0.0

# Leads (broadening on first/last site)
GAMMA_L = 0.5
GAMMA_R = 0.5

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


def _normalize_method(method: str) -> str:
    method_norm = str(method).strip().lower()
    allowed = {"ed", "dft", "dmft"}
    if method_norm not in allowed:
        raise ValueError(
            f"Unknown METHOD={method!r}. Expected one of: {sorted(allowed)}"
        )
    return method_norm


def _normalize_scheme(scheme: str) -> str:
    scheme_norm = str(scheme).strip().lower()
    allowed = {"none", "ferretti", "brazilian"}
    if scheme_norm not in allowed:
        raise ValueError(
            f"Unknown VERTEX_SCHEME={scheme!r}. Expected one of: {sorted(allowed)}"
        )
    return scheme_norm


def _write_summary(path: Path, *, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    method = _normalize_method(METHOD)
    scheme = _normalize_scheme(VERTEX_SCHEME)

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
    os.makedirs(output_dir, exist_ok=True)

    # Method-specific outputs (self-energy, G_retarded, etc.)
    method_output_dir = output_dir / method
    os.makedirs(method_output_dir, exist_ok=True)

    # Vertex-scheme-specific outputs (transmission, vertex correction)
    scheme_output_dir = method_output_dir / scheme
    os.makedirs(scheme_output_dir, exist_ok=True)

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

    sigma_correlated: np.ndarray | None
    if method == "ed":
        energies, sigma_correlated = compute_ed_self_energy(
            input_folder=str(output_dir),
            output_folder=str(method_output_dir),
            eta=float(ETA),
            beta=float(BETA),
            de=float(DE),
            e_min=float(EMIN),
            e_max=float(EMAX),
            neig_value=int(NEIG_VALUE),
        )
    elif method == "dft":
        # Correlated self-energies are disabled for DFT.
        energies = np.arange(
            float(EMIN), float(EMAX) + float(DE) / 2.0, float(DE)
        ).round(7)
        sigma_correlated = None
    elif method == "dmft":
        raise NotImplementedError("METHOD='dmft' is not implemented yet.")
    else:
        raise ValueError(f"Unexpected METHOD={method!r}.")

    # Persist the energy grid used throughout this run.
    np.save(os.path.join(output_dir, "energies.npy"), energies)

    # Save the full retarded Green's function G(E) on the same grid.
    # Shape: (nE, n, n)
    G_ret = np.empty((energies.size, nsites, nsites), dtype=np.complex128)
    for eidx, E in enumerate(energies):
        G_ret[eidx] = compute_G_retarded(
            float(E),
            H,
            Gamma_L,
            Gamma_R,
            sigma_correlated=None
            if sigma_correlated is None
            else sigma_correlated[eidx],
            eta=float(ETA),
        )
    np.savez_compressed(
        os.path.join(method_output_dir, "G_retarded.npz"),
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
        scheme=str(scheme),
    )
    et_path = scheme_output_dir / "ET.npy"
    save_total_only = (method == "dft") or (scheme == "none")
    if save_total_only:
        np.save(str(et_path), np.array([energies, T["total"]]))
    else:
        np.save(
            str(et_path),
            np.array([energies, T["elastic"], T["inelastic"], T["total"]]),
        )

    vertex_path = scheme_output_dir / "vertex_correction.npz"
    if scheme != "none":
        # Save the vertex-correction matrix Lambda(E) so it can be analyzed directly.
        # Shape: (nE, n, n)
        Lambda = np.empty((energies.size, nsites, nsites), dtype=np.complex128)
        for eidx in range(energies.size):
            Lambda[eidx] = compute_vertex_correction(
                gamma_L=Gamma_L,
                gamma_R=Gamma_R,
                sigma_correlated=None
                if sigma_correlated is None
                else sigma_correlated[eidx],
                eta=float(ETA),
                scheme=str(scheme),
            )
        lambda_trace = np.trace(Lambda, axis1=1, axis2=2)
        np.savez_compressed(
            str(vertex_path),
            energies=energies,
            Lambda=Lambda,
            trace=lambda_trace,
            scheme=str(scheme),
            eta=float(ETA),
        )

    # Write summaries to the method + scheme directories.
    common_lines = [
        "Toy-model workflow summary",
        "========================",
        "",
        f"method: {method}",
        f"vertex_scheme: {scheme}",
        "",
        "Model",
        f"  nsites: {nsites}",
        f"  onsite_params: {onsite_params}",
        f"  nearest_neighbor_t: {float(NEAREST_NEIGHBOR_T)}",
        f"  second_nearest_neighbor_t: {float(SECOND_NEAREST_NEIGHBOR_T)}",
        f"  occupancies: {occupancies.tolist()}",
        "",
        "Leads",
        f"  GAMMA_L: {float(GAMMA_L)}",
        f"  GAMMA_R: {float(GAMMA_R)}",
        "",
        "PPP",
        f"  U_onsite: {float(U_ONSITE)}",
        f"  coulomb_evA: {float(COULOMB_EV_A)}",
        f"  symbols: {tuple(PPP_SYMBOLS)}",
        f"  xyz: {XYZ_FILENAME}",
        "",
        "Energy grid",
        f"  eta: {float(ETA)}",
        f"  EMIN: {float(EMIN)}",
        f"  EMAX: {float(EMAX)}",
        f"  DE: {float(DE)}",
        f"  nE: {energies.size}",
        "",
        "ED parameters (only used for METHOD=ed)",
        f"  beta: {float(BETA)}",
        f"  neig_value: {int(NEIG_VALUE)}",
        "",
        "Inputs written under output/",
        "  hamiltonian.npy, gamma_L.npy, gamma_R.npy, occupancies.npy, U_ppp.txt, energies.npy",
        "",
        "Method outputs",
        f"  method_dir: {method_output_dir}",
        f"  scheme_dir: {scheme_output_dir}",
        "",
        "Key outputs",
        f"  G_retarded: {method_output_dir / 'G_retarded.npz'}",
        f"  transmission: {scheme_output_dir / 'ET.npy'}",
        f"  vertex_correction: {vertex_path if scheme != 'none' else 'skipped (scheme=none)'}",
    ]
    if save_total_only:
        common_lines.append(
            "  ET.npy format: 2 rows (E, T_total) because inelastic contribution is 0"
        )
    else:
        common_lines.append(
            "  ET.npy format: 4 rows (E, T_elastic, T_inelastic, T_total)"
        )
    if method == "ed":
        common_lines.append(f"  self_energy: {method_output_dir / 'self_energy.npy'}")
    elif method == "dft":
        common_lines.append("  self_energy: disabled (Sigma_correlated=None)")
    else:
        common_lines.append(
            f"  self_energy: loaded from {DMFT_SIGMA_PATH or str(method_output_dir / 'self_energy.npy')}"
        )

    _write_summary(method_output_dir / "summary.out", lines=common_lines)
    _write_summary(scheme_output_dir / "summary.out", lines=common_lines)

    print("Workflow completed.")
    print(f"Output directory: {output_dir}")
    if method == "ed":
        print(f"Saved self-energy: {method_output_dir / 'self_energy.npy'}")
    elif method == "dft":
        print("Correlated self-energy: disabled (METHOD='dft')")
    elif method == "dmft":
        print(
            "Not implemented: correlated self-energy loading for METHOD='dmft' is not implemented yet."
        )
    else:
        print(
            f"Saved self-energy: loaded from {DMFT_SIGMA_PATH or str(method_output_dir / 'self_energy.npy')}"
        )

    print(f"Saved Green's function: {method_output_dir / 'G_retarded.npz'}")
    print(f"Saved transmission: {scheme_output_dir / 'ET.npy'}")
    if scheme != "none":
        print(
            "Saved vertex correction (energies, Lambda(E), trace(Lambda)): "
            f"{vertex_path}"
        )
    else:
        print("Vertex correction output: skipped (VERTEX_SCHEME='none')")
    print(
        f"Saved summaries: {method_output_dir / 'summary.out'} and {scheme_output_dir / 'summary.out'}"
    )


if __name__ == "__main__":
    main()
