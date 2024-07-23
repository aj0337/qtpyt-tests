from pathlib import Path

import numpy as np
from ase.io import read
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings


GPWDEVICEDIR = '../dft/device/'
GPWLEADSDIR = '../dft/leads_bulk/'

leads_path = Path(GPWLEADSDIR)
device_path = Path(GPWDEVICEDIR)
basis = {'Au': 9, 'H': 5, 'C': 13, 'N': 13}
leads_atoms = read(leads_path / 'leads.traj')
leads_basis = Basis.from_dictionary(leads_atoms, basis)
device_atoms = read(device_path / 'scatt.xyz')
device_basis = Basis.from_dictionary(device_atoms, basis)
H_leads_dft, S_leads_dft = np.load(leads_path / 'hs_leads_k.npy')


H_device_dft, S_device_dft = np.load(device_path/'hs_cc_k.npy')
kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
    H_leads_dft, S_leads_dft, (5, 1, 1), align=(0, H_device_dft[0, 0, 0]))
remove_pbc(device_basis, H_device_dft)
remove_pbc(device_basis, S_device_dft)

Nr = (1, 1, 1)

self_energy = [None, None]
self_energy[0] = PrincipalSelfEnergy(kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij),
                            Nr=Nr)
self_energy[1] = PrincipalSelfEnergy(kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij),
                            Nr=Nr,
                            id='right')

rotate_couplings(leads_basis, self_energy[0], Nr)
rotate_couplings(leads_basis, self_energy[1], Nr)

nodes = graph_partition.get_tridiagonal_nodes(device_basis, H_device_dft[0],
                                              len(leads_atoms.repeat(Nr)))


hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
    nodes, H_device_dft[0], S_device_dft[0])

de = 0.01
energies = np.arange(-4., 4. + de / 2., de).round(7)
gd = GridDesc(energies, 1)
eta = 1e-3

gf = greenfunction.GreenFunction(hs_list_ii,
                                    hs_list_ij,
                                    [(0, self_energy[0]),
                                    (len(hs_list_ii) - 1, self_energy[1])],
                                    #  solver='coupling',
                                    solver='dyson',
                                    eta=eta)

T = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T[e] = gf.get_transmission(energy)

T = gd.gather_energies(T)
if comm.rank == 0:
    np.save(f'lcao_bulk_leads/ET_eta{eta}.npy', (energies, T))
