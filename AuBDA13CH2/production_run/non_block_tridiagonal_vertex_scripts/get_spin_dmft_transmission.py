import os
from pathlib import Path
import pickle

import numpy as np
from ase.io import read
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.basis import Basis
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import expand
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import expand_coupling, remove_pbc, rotate_couplings

rank = comm.Get_rank()


class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_device, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


def run(outputfile):
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
        np.save(outputfile, (energies, T_total.real, T_elastic.real, T_inelastic.real))


pl_path = Path("../dft/leads/")
cc_path = Path("../dft/device/")
output_folder = "../output/lowdin/dmft/spin/vertex_tests"
os.makedirs(output_folder, exist_ok=True)

data_folder = "../output/lowdin"
dmft_data_folder = "../output/lowdin/dmft/spin"
index_active_region = np.load(f"{data_folder}/index_active_region.npy")
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)

H_leads_lcao, S_leads_lcao = np.load(pl_path / "hs_pl_k.npy")
H_subdiagonalized, S_subdiagonalized = map(
    lambda m: m.astype(complex),
    np.load("../output/lowdin/hs_los_lowdin.npy"),
)

basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}

leads_atom = read(pl_path / "leads.xyz")
leads_basis = Basis.from_dictionary(leads_atom, basis_dict)

device_atoms = read(cc_path / "scatt.xyz")
device_basis = Basis.from_dictionary(device_atoms, basis_dict)

# de = 0.01
# energies = np.arange(-3.0, 3.0 + de / 2.0, de).round(7)
energies = np.load(f"{dmft_data_folder}/energies.npy")
eta = 1e-3

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

S_device = S_subdiagonalized[0]
idx_molecule = index_active_region

# Add the DMFT self-energy for transmission
for spin, spin_label in enumerate(["up", "dw"]):
    dmft_sigma_file = f"{dmft_data_folder}/self_energy_{spin_label}.npy"
    if comm.rank == 0:
        dmft_sigma = load(dmft_sigma_file)
    else:
        dmft_sigma = [None]

    dmft_sigma = comm.bcast(dmft_sigma, root=0)
    self_energy[2] = dmft_sigma
    gf.selfenergies.append((slice(None), self_energy[2]))

    outputfile = f"{dmft_data_folder}/transmission_data_brazilian_{spin_label}.npy"
    run(outputfile)
    gf.selfenergies.pop()
