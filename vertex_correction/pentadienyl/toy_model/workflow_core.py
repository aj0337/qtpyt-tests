"""Core implementation for the toy-model workflow.

This module is intentionally kept free of user configuration.
Edit parameters in `run_workflow.py`.
"""

from __future__ import annotations

import os

import numpy as np
from ase.io import read


class Sigma:
    def __init__(self, gf0, gf, H_eff, eta: float = 1e-5):
        """Construct a self-energy helper from two Green's function callables.

        Parameters
        ----------
        gf0
            Callable returning the non-interacting (reference) Green's function.
            Expected signature:
                gf0(energies, eta) -> ndarray
            The result must be indexable as g0[..., e] for each energy index e and
            return square matrices compatible with H_eff.shape[0].

        gf
            Callable returning the interacting Green's function.
            Same calling convention as gf0.
            Must expose attribute gf.n giving the matrix dimension.

        H_eff
            Effective one-particle Hamiltonian. Only its dimension is used.

        eta
            Positive imaginary broadening.

        Notes
        -----
        The retarded self-energy is defined via the Dyson equation:

            Sigma(z) = inverse(G0(z)) - inverse(G(z))

        where z is a complex energy (typically z = E + i*eta).
        """

        self.gf0 = gf0
        self.gf = gf
        self.eta = eta
        self.H_eff = H_eff

    def retarded(self, energy):
        """Evaluate the retarded self-energy at one or more energies.

        Parameters
        ----------
        energy
            Complex energy or array of complex energies, typically E + i*eta.

        Returns
        -------
        numpy.ndarray
            Array of shape (nE, n, n) containing the self-energy matrices.

        Notes
        -----
        For each energy z, the self-energy is computed as:

            Sigma(z) = inverse(G0(z)) - inverse(G(z))

        The matrix inverses are computed numerically by solving linear systems
        of the form:

            G(z) * X = I

        which is more stable than explicitly forming inverse(G).
        """

        energies = np.atleast_1d(energy)
        g0 = self.gf0(energies, self.eta)
        g = self.gf(energies, self.eta)
        sigma = np.empty((energies.size, self.gf.n, self.gf.n), complex)
        for e, _energy in enumerate(energies):
            g0_inv = np.linalg.solve(g0[..., e], np.eye(self.H_eff.shape[0]))
            g_inv = np.linalg.solve(g[..., e], np.eye(self.H_eff.shape[0]))
            sigma[e] = g0_inv - g_inv
        return sigma


def compute_ed_self_energy(
    input_folder: str,
    output_folder: str,
    *,
    eta: float,
    beta: float,
    de: float,
    e_min: float,
    e_max: float,
    neig_value: int,
):
    """Compute the ED (exact diagonalization) self-energy on an energy grid.

    This function builds interacting and non-interacting Green's functions
    using edpyt, then computes the retarded self-energy and writes:

        energies.npy
        self_energy.npy

    Parameters
    ----------
    input_folder
        Folder containing:
            hamiltonian.npy
            occupancies.npy
            U_ppp.txt

    output_folder
        Folder where self_energy.npy is written.

    eta
        Imaginary broadening used in z = E + i*eta.

    beta
        Inverse temperature.

    de
        Energy spacing.

    e_min, e_max
        Energy window.

    neig_value
        Number of eigenstates per particle-number sector.

    Returns
    -------
    energies : ndarray
        1D array of real energies.

    sigma_ret : ndarray
        Array of shape (nE, n, n) with the retarded self-energy.

    Notes
    -----
    The self-energy is computed from two ED Green's functions:

        Sigma(E) = inverse(G0(E + i*eta)) - inverse(G(E + i*eta))

    The energy grid is:

        E_k = e_min + k * de

    with k chosen so that E_k <= e_max (up to rounding).
    """

    try:
        from edpyt.espace import (
            build_espace,
            screen_espace,
        )
        from edpyt.gf2_lanczos import (
            build_gf2_lanczos,
        )
        from edpyt.shared import params
    except ImportError as exc:
        raise ImportError(
            "edpyt is required to compute ED self-energy. "
            "Install it in your active Python environment (e.g. `pip install edpyt`)."
        ) from exc

    os.makedirs(output_folder, exist_ok=True)

    H = np.load(f"{input_folder}/hamiltonian.npy")
    occupancy_goal = np.load(f"{input_folder}/occupancies.npy")
    V = np.loadtxt(f"{input_folder}/U_ppp.txt")
    DC0 = np.diag(V.diagonal() * (occupancy_goal - 0.5))

    nimp = H.shape[0]
    neig = np.ones((nimp + 1) * (nimp + 1), int) * neig_value
    params["z"] = occupancy_goal

    espace0, egs0 = build_espace(H, np.zeros_like(H), neig_sector=neig)
    screen_espace(espace0, egs0, beta)
    gf0 = build_gf2_lanczos(H, np.zeros_like(H), espace0, beta, egs0)

    espace, egs = build_espace(H - DC0, V, neig_sector=neig)
    screen_espace(espace, egs, beta)
    gf = build_gf2_lanczos(H - DC0, V, espace, beta, egs)

    sigma = Sigma(gf0, gf, H, eta=eta)

    energies = np.arange(e_min, e_max + de / 2.0, de).round(7)
    z_ret = energies + 1.0j * eta
    sigma_ret = sigma.retarded(z_ret)

    np.save(f"{input_folder}/energies.npy", energies)
    np.save(f"{output_folder}/self_energy.npy", sigma_ret)
    return energies, sigma_ret


def build_hamiltonian(
    onsite_params: list[float],
    nearest_neighbor_t: float,
    second_nearest_neighbor_t: float,
) -> np.ndarray:
    """Construct a 1D tight-binding Hamiltonian with up to second neighbors.

    Parameters
    ----------
    onsite_params
        On-site energies epsilon_i for each site.

    nearest_neighbor_t
        Nearest-neighbor hopping amplitude t1.

    second_nearest_neighbor_t
        Second-nearest-neighbor hopping amplitude t2.

    Returns
    -------
    numpy.ndarray
        Hamiltonian matrix of shape (n, n).
    """

    nsites = len(onsite_params)
    if nsites == 0:
        raise ValueError("onsite_params must not be empty.")

    H = np.zeros((nsites, nsites), dtype=float)
    np.fill_diagonal(H, onsite_params)

    for i in range(nsites - 1):
        H[i, i + 1] = nearest_neighbor_t
        H[i + 1, i] = np.conj(nearest_neighbor_t)

    for i in range(nsites - 2):
        H[i, i + 2] = second_nearest_neighbor_t
        H[i + 2, i] = np.conj(second_nearest_neighbor_t)

    return H


def build_ppp_matrix(
    system_root: str,
    num_sites: int,
    *,
    U_onsite: float = 4.62,
    coulomb_evA: float = 14.397,
    symbols: tuple[str, ...] = ("C", "N"),
    xyz_path: str = "scatt.xyz",
) -> np.ndarray:
    """Build a PPP (Pariser–Parr–Pople) interaction matrix from geometry.

    Parameters
    ----------
    system_root
        Root directory containing the XYZ file.

    num_sites
        Number of interacting sites.

    U_onsite
        On-site Coulomb interaction U.

    coulomb_evA
        Coulomb prefactor (in eV·Å).

    symbols
        Chemical symbols defining valid PPP sites.

    xyz_path
        Path to the XYZ file.

    Returns
    -------
    numpy.ndarray
        Symmetric interaction matrix U_ij.

    Notes
    -----
    The interaction is computed using the Ohno/PPP form:

        U_ij = U / sqrt(1 + alpha * r_ij^2)

    where:

        alpha = (U / coulomb_evA)^2
        r_ij  = distance between sites i and j (in Å)

    The diagonal elements are set explicitly to:

        U_ii = U

    and the matrix is symmetrized as (U + U.T) / 2 to suppress numerical noise.
    """

    if num_sites <= 0:
        raise ValueError(f"num_sites must be positive; got {num_sites}")

    alpha_ppp = (U_onsite / coulomb_evA) ** 2

    xyz_full_path = os.path.join(system_root, xyz_path)
    if not os.path.isfile(xyz_full_path):
        raise FileNotFoundError(f"Missing geometry file: {xyz_full_path}")

    atoms_or_list = read(xyz_full_path)
    atoms = atoms_or_list[0] if isinstance(atoms_or_list, list) else atoms_or_list
    site_indices = [
        i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol in symbols
    ]

    if len(site_indices) < num_sites:
        raise ValueError(
            f"Found only {len(site_indices)} {symbols} atoms, but num_sites={num_sites} was requested."
        )

    positions = atoms.get_positions()[site_indices[:num_sites]]
    disp = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(disp, axis=-1)

    U = U_onsite / np.sqrt(1.0 + alpha_ppp * r**2)
    np.fill_diagonal(U, U_onsite)
    U = 0.5 * (U + U.T)
    return U


def compute_G_retarded(
    E: float,
    H: np.ndarray,
    gamma_L: np.ndarray,
    gamma_R: np.ndarray,
    sigma_correlated: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Compute the retarded Green's function.

    Parameters
    ----------
    E
        Real energy.

    H
        Device Hamiltonian.

    gamma_L, gamma_R
        Left and right broadening matrices.

    sigma_correlated
        Additional device self-energy at energy E.

    eta
        Positive imaginary broadening.

    Returns
    -------
    numpy.ndarray
        The Green's function matrix G(E).

    Notes
    -----
    The retarded Green's function is defined as:

        G(E) = inverse(
                   (E + i*eta) * I
                 - H
                 + (i/2) * gamma_L
                 + (i/2) * gamma_R
                 - sigma_correlated(E)
               )

    This function returns the Green's function matrix:

    """

    n = H.shape[0]
    identity = np.eye(n, dtype=complex)
    z = E + 1j * eta
    if sigma_correlated is None:
        sigma_correlated = np.zeros_like(H)
    A = (
        (z * identity)
        - H.astype(complex)
        + 0.5j * gamma_L.astype(complex)
        + 0.5j * gamma_R.astype(complex)
        - sigma_correlated.astype(complex)
    )
    GD = np.linalg.inv(A)
    return GD


def compute_vertex_correction(
    gamma_L: np.ndarray,
    gamma_R: np.ndarray,
    sigma_correlated: np.ndarray,
    eta: float,
    scheme: str = "none",
):
    """Compute the vertex-correction matrix for one energy.

    Parameters
    ----------
    gamma_L, gamma_R
        Left and right broadening matrices (Gamma_L and Gamma_R).

    sigma_correlated
        Retarded device/correlation self-energy matrix Sigma_correlated(E) at this energy.

    eta
        Broadening used in z = E + i*eta. This eta is also used in the Ferretti
        denominator.

    scheme
        Vertex correction scheme. Supported values:
            - "none": no correction (returns zeros)
            - "ferretti": Ferretti-like Lambda
            - "brazilian": Lambda = Gamma_correlated

    Returns
    -------
    numpy.ndarray
        Vertex-correction matrix Lambda(E) of shape (n, n).

    Notes
    -----
    The device broadening associated with the correlation self-energy is:

        Gamma_correlated(E) = i * ( Sigma_correlated(E) - Sigma_correlated(E)^{dagger} )

    where "dagger" denotes conjugate transpose.

    Implemented schemes:

    - Brazilian:

        Lambda(E) = Gamma_correlated(E)

    - Ferretti-like:

        Lambda(E) = [ Gamma_L + Gamma_R + 2 * eta * I ]^{-1} * Gamma_correlated(E)

    This function only constructs Lambda(E). The inelastic transmission term is
    assembled in `compute_transmission`.
    """

    if eta < 0:
        raise ValueError(f"eta must be non-negative; got {eta}")

    scheme_norm = scheme.strip().lower()
    n = gamma_L.shape[0]
    if scheme_norm == "none":
        return np.zeros((n, n), dtype=complex)

    gamma_correlated = compute_gamma_from_sigma(sigma_correlated)

    if scheme_norm == "brazilian":
        return gamma_correlated

    if scheme_norm == "ferretti":
        return compute_ferretti_correction(gamma_correlated, gamma_L, gamma_R, eta)

    raise ValueError(
        f"Unknown scheme={scheme!r}. Expected one of: 'none', 'ferretti', 'brazilian'."
    )


def compute_gamma_from_sigma(sigma: np.ndarray) -> np.ndarray:
    """Compute a broadening matrix Gamma from a retarded self-energy Sigma.

    Parameters
    ----------
    sigma
        Retarded self-energy matrix Sigma(E) at one energy.

    Returns
    -------
    numpy.ndarray
        Broadening matrix Gamma(E).

    Notes
    -----
    The broadening is defined as:

        Gamma(E) = i * ( Sigma(E) - Sigma(E)^{dagger} )

    where "dagger" denotes conjugate transpose.
    """

    return 1j * (sigma - sigma.conj().T)


def compute_ferretti_correction(
    gamma_correlated: np.ndarray,
    gamma_L: np.ndarray,
    gamma_R: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Compute the Ferretti-like correction matrix Lambda.

    Parameters
    ----------
    gamma_correlated
        Device broadening matrix Gamma_correlated(E).

    gamma_L, gamma_R
        Left and right broadening matrices (Gamma_L and Gamma_R).

    eta
        Broadening used in z = E + i*eta.

    Returns
    -------
    numpy.ndarray
        The matrix Lambda(E) used in the Ferretti-like inelastic correction.

    Notes
    -----
    The Ferretti-like correction used here is:

        Lambda(E) = [ Gamma_L + Gamma_R + 2 * eta * I ]^{-1} * Gamma_correlated(E)

    The matrix inverse is computed via a linear solve.
    """

    n = gamma_L.shape[0]
    denom = gamma_L + gamma_R + 2.0 * float(eta) * np.eye(n, dtype=complex)
    return np.linalg.solve(denom, gamma_correlated)


def compute_transmission(
    energies: np.ndarray,
    H: np.ndarray,
    gamma_L: np.ndarray,
    gamma_R: np.ndarray,
    sigma_correlated: np.ndarray | None,
    *,
    eta: float = 1e-2,
    left_index: int = 0,
    right_index: int = -1,
    scheme: str = "none",
) -> dict[str, np.ndarray]:
    """Compute the transmission function on an energy grid.

    Parameters
    ----------
    energies
        Array of real energies.

    H
        Device Hamiltonian.

    gamma_L, gamma_R
        Left and right broadening matrices.

    sigma_correlated
        Energy-dependent device self-energy.
        If None, it is treated as zero.

    eta
        Broadening used in E + i*eta.

    left_index, right_index
        Indices of the left and right contact sites.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'elastic': elastic transmission
            - 'inelastic': inelastic/vertex correction contribution
            - 'total': elastic + inelastic

    Notes
    -----
    Elastic transmission (toy-model contact-element form):

        T_elastic(E) = Gamma_L(l,l) * Gamma_R(r,r) * | G(l,r; E) |^2

    where l = left_index and r = right_index.

    The retarded Green's function is:

        G(E) = [ (E + i*eta) * I
                 - H
                 + (i/2) * Gamma_L
                 + (i/2) * Gamma_R
                 - Sigma_correlated(E) ]^{-1}

    Inelastic (vertex-correction) contribution (trace form):

        T_inelastic(E) = Re Tr[ Gamma_L * G(E) * Gamma_R * Lambda(E) * G(E)^{dagger} ]

    where Lambda(E) is produced by `compute_vertex_correction` according to the
    selected scheme.
    """

    n = H.shape[0]
    nE = energies.size

    if sigma_correlated is None:
        sigma_correlated = np.zeros((nE, n, n), dtype=complex)

    if sigma_correlated.shape != (nE, n, n):
        raise ValueError(
            f"sigma_correlated must be {(nE, n, n)}; got {sigma_correlated.shape}"
        )

    left_idx = left_index if left_index >= 0 else n + left_index
    right_idx = right_index if right_index >= 0 else n + right_index
    print(f"Using left contact index: {left_idx}")
    print(f"Using right contact index: {right_idx}")

    T_elastic = np.empty(nE, dtype=float)
    T_inelastic = np.empty(nE, dtype=float)
    T_total = np.empty(nE, dtype=float)
    for eidx, E in enumerate(energies):
        G = compute_G_retarded(E, H, gamma_L, gamma_R, sigma_correlated[eidx], eta)
        Glr = G[left_idx, right_idx]
        gamma_sq = (gamma_L[left_idx, left_idx] * gamma_R[right_idx, right_idx]).real
        tel = float(gamma_sq * (abs(Glr) ** 2))
        tel = max(0.0, tel)  # clamp tiny negative from rounding

        lam = compute_vertex_correction(
            gamma_L=gamma_L,
            gamma_R=gamma_R,
            sigma_correlated=sigma_correlated[eidx],
            eta=float(eta),
            scheme=scheme,
        )
        ga = G.conj().T
        tin = float(np.real(np.trace(gamma_L @ G @ gamma_R @ lam @ ga)))

        T_elastic[eidx] = tel
        T_inelastic[eidx] = tin
        T_total[eidx] = tel + tin

    return {"elastic": T_elastic, "inelastic": T_inelastic, "total": T_total}
