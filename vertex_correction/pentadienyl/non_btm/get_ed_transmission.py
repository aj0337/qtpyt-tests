from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mpi4py import MPI
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.basis import Basis
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import expand
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import expand_coupling, remove_pbc, rotate_couplings

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class DataSelfEnergy(BaseDataSelfEnergy):
    def retarded(self, energy):
        return expand(S_device, super().retarded(energy), index_active_region)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


pl_path = Path("./dft/leads/")
cc_path = Path("./dft/device/")
data_folder = "./output/lowdin"
ed_data_folder = "./output/lowdin/ed"


de = 0.01
energies = np.arange(-3.0, 3.0 + de / 2.0, de).round(7)
eta = 1e-2
Nr = (1, 5, 3)
unit_cell_rep_in_leads = (5, 5, 3)
basis_dict = {"Au": 9, "H": 5, "C": 13, "N": 13}

FERRETTI = True
BRAZILIAN = False


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


kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
    H_leads_lcao,
    S_leads_lcao,
    unit_cell_rep_in_leads,
    align=(0, H_subdiagonalized[0, 0, 0]),
)
remove_pbc(device_basis, H_subdiagonalized)
remove_pbc(device_basis, S_subdiagonalized)
S_device = S_subdiagonalized[0].copy()


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


gf = GreenFunction(
    H_subdiagonalized[0],
    S_subdiagonalized[0],
    selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
    eta=eta,
    index_active_region=index_active_region,
)


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
        energy,
        ferretti=FERRETTI,
        brazilian=BRAZILIAN,
    )

T_total = gd.gather_energies(T_total)
T_elastic = gd.gather_energies(T_elastic)
T_inelastic = gd.gather_energies(T_inelastic)

if rank == 0:
    if FERRETTI and not BRAZILIAN:
        scheme = "ferretti"
    elif BRAZILIAN and not FERRETTI:
        scheme = "brazilian"
    else:
        scheme = "none"

    out_dir = Path(ed_data_folder) / scheme
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / "ET_components.npz"
    np.savez(
        out_npz,
        E=energies,
        T_elastic=T_elastic.real,
        T_inelastic=T_inelastic.real,
        T_total=T_total.real,
        eta_gf=float(eta),
        eta_se=float(eta),
    )

    plt.figure()
    plt.plot(energies, T_elastic.real, label="T_elastic")
    plt.plot(energies, T_inelastic.real, label="T_inelastic")
    plt.plot(energies, T_total.real, "-.", label="T_total")
    plt.ylabel("T(E)")
    plt.xlabel("E (eV)")
    plt.yscale("log")
    plt.legend()

    out_png = out_dir / "ET_components.png"
    plt.savefig(out_png)
    plt.close()


gf.selfenergies.pop()
