from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings


lowdin = True
data_folder = f"./unrelaxed/output/lowdin" if lowdin else f"./unrelaxed/output/no_lowdin"
GPWLEADSDIR = "./unrelaxed/dft/leads1/"

unit_cell_rep_in_leads = (6, 1, 1)

pl_path = Path(GPWLEADSDIR)
H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")

H_subdiagonalized, _ = np.load(f"{data_folder}/hs_los.npy")

H_subdiagonalized = H_subdiagonalized.astype(np.complex128)

# Prepare the k-points and matrices for the leads (Hamiltonian and overlap matrices)
h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = map(
    lambda m: m[0],
    prepare_leads_matrices(
        H_leads_lcao,
        S_leads_lcao,
        unit_cell_rep_in_leads,
        align=(0, H_subdiagonalized[0, 0, 0]),
    )[1:],
)

# Compute self-energy
self_energy = [None, None, None]
self_energy[0] = LeadSelfEnergy((h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij))
self_energy[1] = LeadSelfEnergy((h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), id="right")

# Save self-energy
np.save(os.path.join(data_folder, "self_energy.npy"), self_energy)
