import os
import numpy as np
from ase.io import read

# ============================================================
# USER INPUT
# ============================================================
SYSTEM_ROOT = "../"

U_VALUES = [0.1]  # eV
COULOMB_EV_ANG = 14.397  # e^2 / (4*pi*eps0) in eV·Å


# --------------------------------------------------------
# Infer number of sites from existing U matrix
# --------------------------------------------------------
U_path = f"{SYSTEM_ROOT}/output/lowdin/U_matrix_crpa.txt"
if not os.path.isfile(U_path):
    raise FileNotFoundError(f"Missing file: {U_path}")

U_matrix_in = np.loadtxt(U_path)

if U_matrix_in.ndim != 2 or U_matrix_in.shape[0] != U_matrix_in.shape[1]:
    raise ValueError(
        f"U_matrix.txt must be square; got shape {U_matrix_in.shape}"
    )

num_sites = U_matrix_in.shape[0]

# --------------------------------------------------------
# Load geometry (C and N atoms only)
# --------------------------------------------------------
xyz_path = f"{SYSTEM_ROOT}/dft/device/scatt.xyz"
if not os.path.isfile(xyz_path):
    raise FileNotFoundError(f"Missing file: {xyz_path}")

all_atoms = read(xyz_path)
symbols = np.array(all_atoms.get_chemical_symbols())

is_CN = np.isin(symbols, ["C", "N"])
mol_atoms = all_atoms[is_CN]

if len(mol_atoms) < num_sites:
    raise ValueError(
        f"Found {len(mol_atoms)} C/N atoms, but U is {num_sites}x{num_sites}"
    )

positions = mol_atoms.get_positions()[:num_sites]  # Å

# --------------------------------------------------------
# Distance matrix r_ij
# --------------------------------------------------------
disp = positions[:, None, :] - positions[None, :, :]
r_ij = np.linalg.norm(disp, axis=-1)  # Å

# --------------------------------------------------------
# Output directory
# --------------------------------------------------------
out_dir = f"{SYSTEM_ROOT}/output/lowdin"
os.makedirs(out_dir, exist_ok=True)

# --------------------------------------------------------
# Build and write PPP matrices
# --------------------------------------------------------
for U in U_VALUES:
    alpha = (U / COULOMB_EV_ANG) ** 2  # 1/Å^2

    U_ppp = U / np.sqrt(1.0 + alpha * r_ij**2)
    np.fill_diagonal(U_ppp, U)

    # Symmetrize for numerical safety
    U_ppp = 0.5 * (U_ppp + U_ppp.T)

    out_path = f"{out_dir}/U_matrix_PPP_U_{U:.1f}.txt"
    np.savetxt(out_path, U_ppp, fmt="%.10f")

    print(f"Wrote {out_path}")

print("\nDone.")
