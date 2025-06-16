from pathlib import Path

import numpy as np
from ase.io import read
from gpaw import restart
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc

dft_leads_path = Path('../../../dft/leads/')
dft_device_path = Path('../../../dft/device/')
local_orbital_path = Path('../../../localized_orbitals/BDA_active/')
gpwfile = f'{dft_device_path}/scatt.gpw'

h_pl_k, s_pl_k = np.load(dft_leads_path / 'hs_pl_k.npy')
h_cc_k, s_cc_k = map(lambda m: m.astype(complex), np.load(local_orbital_path/f'hs_lolw_k.npy'))

atoms, calc = restart(gpwfile, txt=None)
nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)
x = basis.atoms.positions[:, 0]
scatt = np.where((x > 12.9) & (x < 19))[0]

basis_p = basis[scatt]
index_p = basis_p.get_indices()
ibf_los = np.load(local_orbital_path / 'idx_los.npy')

basis = {'Au':9,'H': 5, 'C': 13, 'N': 13}

atoms_pl = read(dft_leads_path / 'leads.xyz')
basis_pl = Basis.from_dictionary(atoms_pl, basis)

atoms_cc = read(dft_device_path / 'scatt.xyz')
basis_cc = Basis.from_dictionary(atoms_cc, basis)

h_pl_ii, s_pl_ii, h_pl_ij, s_pl_ij = map(
    lambda m: m[0],
    prepare_leads_matrices(h_pl_k,
                           s_pl_k, (6, 1, 1),
                           align=(0, h_cc_k[0, 0, 0]))[1:])
del h_pl_k, s_pl_k
remove_pbc(basis_cc, h_cc_k)
remove_pbc(basis_cc, s_cc_k)

se = [None, None]
se[0] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij))
se[1] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), id='right')

nodes = [0, basis_pl.nao, basis_cc.nao - basis_pl.nao, basis_cc.nao]

hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
    nodes, h_cc_k[0], s_cc_k[0])
del h_cc_k, s_cc_k

de = 0.01
energies = np.arange(-3, 3 + de / 2., de).round(7)

etas = [2.5e-2, 1e-1]
for eta in etas:

    gf = greenfunction.GreenFunction(hs_list_ii,
                                     hs_list_ij,
                                     [(0, se[0]),
                                      (len(hs_list_ii) - 1, se[1])],
                                     solver='dyson',
                                     eta=eta)

    gfp = ProjectedGreenFunction(gf, ibf_los)

    no = len(ibf_los)
    gd = GridDesc(energies, no, complex)
    D = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        D[e] = gfp.get_dos(energy)
    D = gd.gather_energies(D)

    if comm.rank == 0:
        np.save(f'pdos_eta_{eta}.npy', D.real)
