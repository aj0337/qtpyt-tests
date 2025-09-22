import os
from pathlib import Path

import numpy as np
from ase.io import read
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.basis import Basis

from qtpyt.parallel import comm
from mpi4py import MPI

from qtpyt.parallel.egrid import GridDesc
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings, expand_coupling
from qtpyt.projector import expand

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DataSelfEnergy(BaseDataSelfEnergy):
    def retarded(self, energy):
        return expand(S_device, super().retarded(energy), index_active_region)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


# === Paths and Input Files ===
pl_path = Path("../dft/leads/")
cc_path = Path("../dft/device/")
data_folder = "../output/lowdin"
ed_data_folder = "../output/lowdin/ed"
output_folder = "../output/lowdin/ed/vertex_tests"
os.makedirs(output_folder, exist_ok=True)

# === Parameters ===
de = 0.01
energies = np.arange(-3.0, 3.0 + de / 2.0, de).round(7)
eta = 1e-2
Nr = (1, 5, 3)
unit_cell_rep_in_leads = (5, 5, 3)
basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}

# === Load Structures and Basis ===
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")
H_subdiagonalized, S_subdiagonalized = map(
    lambda m: m.astype(complex),
    np.load(f"{data_folder}/hs_los_lowdin.npy"),
)
leads_atom = read(pl_path / "leads.xyz")
leads_basis = Basis.from_dictionary(leads_atom, basis_dict)
device_atoms = read(cc_path / "scatt.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

# === Prepare Leads Matrices ===
kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
    H_leads_lcao,
    S_leads_lcao,
    unit_cell_rep_in_leads,
    align=(0, H_subdiagonalized[0, 0, 0]),
)
remove_pbc(device_basis, H_subdiagonalized)
remove_pbc(device_basis, S_subdiagonalized)
S_device = S_subdiagonalized[0].copy()

# === Build Self-Energies ===
self_energy = [None, None, None]
self_energy[0] = PrincipalSelfEnergy(
    kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr
)
self_energy[1] = PrincipalSelfEnergy(
    kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr, id="right"
)
rotate_couplings(leads_basis, self_energy[0], Nr)
rotate_couplings(leads_basis, self_energy[1], Nr)
expand_coupling(self_energy[0], len(H_subdiagonalized[0]))
expand_coupling(self_energy[1], len(H_subdiagonalized[0]), id="right")

# === Initialize Green's Function ===
gf = GreenFunction(
    H_subdiagonalized[0],
    S_subdiagonalized[0],
    selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
    eta=eta,
)

# === Add Correlated Self-Energy ===
ndim_device = len(H_subdiagonalized[0])
ed_self_energy_file = f"{ed_data_folder}/self_energy_with_dcc.npy"

if comm.rank == 0:
    ed_sigma = load(ed_self_energy_file)
else:
    ed_sigma = None

ed_sigma = comm.bcast(ed_sigma, root=0)
self_energy[2] = ed_sigma
gf.selfenergies.append((slice(None), self_energy[2]))

gd = GridDesc(energies, 1, float)
T_total = np.empty(gd.energies.size)
T_elastic = np.empty(gd.energies.size)
T_inelastic = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T_total[e], T_elastic[e], T_inelastic[e] = gf.get_transmission(
        energy, ferretti=False, brazilian=True
    )

T_total = gd.gather_energies(T_total)
T_elastic = gd.gather_energies(T_elastic)
T_inelastic = gd.gather_energies(T_inelastic)

if comm.rank == 0:
    np.save(
        f"{output_folder}/transmission_data_brazilian.npy",
        (energies, T_total.real, T_elastic.real, T_inelastic.real),
    )

# # === Allocate Outputs ===
# T = np.empty(energies.size)
# gamma_L_list = []
# gamma_R_list = []
# delta_list = []

# # === Main Loop ===
# for e, energy in enumerate(energies):
#     T[e] = gf.get_transmission(energy, ferretti=True)
#     gamma_L_list.append(gf.gammas[0])
#     gamma_R_list.append(gf.gammas[1])
#     delta_list.append(gf.delta)

# np.save(f"{output_folder}/ET_non_btm_with_correction.npy", (energies, T.real))
# np.savez_compressed(
#     f"{output_folder}/gamma_L_vs_energy.npz",
#     energies=energies,
#     gamma_L=gamma_L_list,
# )
# np.savez_compressed(
#     f"{output_folder}/gamma_R_vs_energy.npz",
#     energies=energies,
#     gamma_R=gamma_R_list,
# )
# np.savez_compressed(
#     f"{output_folder}/delta_vs_energy.npz",
#     energies=energies,
#     delta=delta_list,
# )

# === Cleanup ===
gf.selfenergies.pop()
