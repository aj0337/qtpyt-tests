from pathlib import Path

import numpy as np
from ase.io import read
from gpaw import restart
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings

pl_path = Path('../../dft/leads/')
cc_path = Path('../../dft/device/')
los_path = Path('../../localized_orbitals/')
gpwfile = f'{cc_path}/scatt.gpw'

h_pl_k, s_pl_k = np.load(pl_path / 'hs_pl_k.npy')
h_cc_k, s_cc_k = map(lambda m: m.astype(complex),
                     np.load(cc_path / f'hs_cc_k.npy'))

atoms, calc = restart(gpwfile, txt=None)
nao_a = np.array([setup.nao for setup in calc.wfs.setups])
basis = Basis(atoms, nao_a)
x = basis.atoms.positions[:, 0]
scatt = np.where((x > 12.9) & (x < 22))[0]

basis_p = basis[scatt]
index_p = basis_p.get_indices()

active = {'C': [3]}
index_c = basis_p.extract().take(active)
ibf_los = index_p[index_c]

basis = {'Au': 15, 'H': 5, 'C': 13, 'N': 13}

atoms_pl = read(pl_path / 'leads.xyz')
basis_pl = Basis.from_dictionary(atoms_pl, basis)

atoms_cc = read(cc_path / 'scatt.xyz')
basis_cc = Basis.from_dictionary(atoms_cc, basis)

kpts_t, h_pl_kii, s_pl_kii, h_pl_kij, s_pl_kij = prepare_leads_matrices(
    h_pl_k, s_pl_k, (5, 5, 3), align=(0, h_cc_k[0, 0, 0]))
del h_pl_k, s_pl_k
remove_pbc(basis_cc, h_cc_k)
remove_pbc(basis_cc, s_cc_k)

Nr = (1, 5, 3)

se = [None, None]
se[0] = PrincipalSelfEnergy(kpts_t, (h_pl_kii, s_pl_kii), (h_pl_kij, s_pl_kij),
                            Nr=Nr)
se[1] = PrincipalSelfEnergy(kpts_t, (h_pl_kii, s_pl_kii), (h_pl_kij, s_pl_kij),
                            Nr=Nr,
                            id='right')

rotate_couplings(basis_pl, se[0], Nr)
rotate_couplings(basis_pl, se[1], Nr)

# find block tridiagonal indices
nodes = graph_partition.get_tridiagonal_nodes(basis_cc, h_cc_k[0],
                                              len(atoms_pl.repeat(Nr)))

hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
    nodes, h_cc_k[0], s_cc_k[0])
del h_cc_k, s_cc_k

de = 0.01
energies = np.arange(-3, 3 + de / 2., de).round(7)

# if comm.rank == 0:
#     np.save('energies.npy', energies)

etas = [1e-1]
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
