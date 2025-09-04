import os
from pathlib import Path

import numpy as np
from ase.io import read
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.basis import Basis
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings, expand_coupling
from qtpyt.projector import expand


# class DataSelfEnergy(BaseDataSelfEnergy):
#     def __init__(self, energies, sigma, idx_active, ndim_device):
#         super().__init__(energies, sigma)
#         self.idx_active = idx_active
#         self.ndim_device = ndim_device

#     def retarded(self, energy):
#         sigma_active = super().retarded(energy)
#         print("energy.shape",energy.shape)
#         print("sigma_active.shape",sigma_active.shape)
#         print("self.idx_active",self.idx_active)
#         sigma_full = np.zeros((self.ndim_device, self.ndim_device), dtype=complex)
#         ia = self.idx_active
#         sigma_full[np.ix_(ia, ia)] = sigma_active
#         return sigma_full


# def load(filename, idx_active, ndim_device):
#     sigma = np.load(filename)  # shape: (len(energies), n_active, n_active)
#     return DataSelfEnergy(energies, sigma, idx_active, ndim_device)

class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_device, super().retarded(energy), index_active_region)

def load(filename):
    return DataSelfEnergy(energies, np.load(filename))

pl_path = Path("../dft/leads/")
cc_path = Path("../dft/device/")
data_folder = "../output/lowdin"
ed_data_folder = "../output/lowdin/ed"
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
ed_self_energy_file = f"{ed_data_folder}/self_energy_with_dcc.npy"
output_folder = "../output/lowdin/ed/vertex_tests"
os.makedirs(output_folder, exist_ok=True)

H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")
H_subdiagonalized, S_subdiagonalized = map(
    lambda m: m.astype(complex), np.load(cc_path / "hs_cc_k.npy")
)
S_device = np.eye(len(S_subdiagonalized[0]))

basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}

leads_atom = read(pl_path / "leads.xyz")
leads_basis = Basis.from_dictionary(leads_atom, basis_dict)

device_atoms = read(cc_path / "scatt.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

de = 0.01
energies = np.arange(-3.0, 3.0 + de / 2.0, de).round(7)
eta = 1e-2

Nr = (1, 5, 3)
unit_cell_rep_in_leads = (5, 5, 3)

kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
    H_leads_lcao,
    S_leads_lcao,
    unit_cell_rep_in_leads,
    align=(0, H_subdiagonalized[0, 0, 0]),
)

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

# expand to dimension of scattering
expand_coupling(self_energy[0], len(H_subdiagonalized[0]))
expand_coupling(self_energy[1], len(H_subdiagonalized[0]), id="right")

gf = GreenFunction(
    H_subdiagonalized[0],
    S_subdiagonalized[0],
    selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
    eta=eta,
)

ndim_device = len(H_subdiagonalized[0])
if comm.rank == 0:
    ed_sigma = load(ed_self_energy_file)
else:
    ed_sigma = None

ed_sigma = comm.bcast(ed_sigma, root=0)
self_energy[2] = ed_sigma
gf.selfenergies.append((slice(None), self_energy[2]))

gd = GridDesc(energies, 1)
T = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T[e] = gf.get_transmission(energy, ferretti=False)

T = gd.gather_energies(T)
if comm.rank == 0:
    np.save(
        f"{output_folder}/ET_non_btm_no_correction.npy",
        (energies, T.real),
    )
gf.selfenergies.pop()
