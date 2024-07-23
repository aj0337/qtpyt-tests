from pathlib import Path
import numpy as np

from ase.io import read

from qtpyt.tools import remove_pbc
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.basis import Basis
from qtpyt.projector import expand
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc

pl_path = Path('../../dft/leads/')
cc_path = Path('../../dft/device/')
los_path = Path('../../localized_orbital/bda_active/')

h_pl_k, s_pl_k = np.load(pl_path / 'hs_pl_k.npy')
h_cc_k, s_cc_k = map(lambda m: m.astype(complex),
                     np.load(los_path / f'hs_lolw_k.npy'))

basis = {'Au': 9, 'H': 5, 'C': 13, 'N': 13}

atoms_pl = read(pl_path / 'leads.xyz')
basis_pl = Basis.from_dictionary(atoms_pl, basis)

atoms_cc = read(cc_path / 'scatt.xyz')
basis_cc = Basis.from_dictionary(atoms_cc, basis)

h_pl_ii, s_pl_ii, h_pl_ij, s_pl_ij = map(
    lambda m: m[0],
    prepare_leads_matrices(h_pl_k,
                           s_pl_k, (5, 1, 1),
                           align=(0, h_cc_k[0, 0, 0]))[1:])
del h_pl_k, s_pl_k
remove_pbc(basis_cc, h_cc_k)
remove_pbc(basis_cc, s_cc_k)

se = [None, None, None]
se[0] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij))
se[1] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), id='right')

nodes = [0, basis_pl.nao, basis_cc.nao - basis_pl.nao, basis_cc.nao]

hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(
    nodes, h_cc_k[0], s_cc_k[0])
del h_cc_k, s_cc_k

de = 0.01
energies = np.arange(-4., 4. + de / 2., de).round(7)
gd = GridDesc(energies, 1)
eta = 1e-2


gf = greenfunction.GreenFunction(hs_list_ii,
                                 hs_list_ij, [(0, se[0]),
                                              (len(hs_list_ii) - 1, se[1])],
                                 solver='dyson',
                                 eta=eta)
i1 = np.load(los_path / 'idx_los.npy') - nodes[1]
s1 = hs_list_ii[1][1]

gd = GridDesc(energies, 1, float)
T = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T[e] = gf.get_transmission(energy)

T = gd.gather_energies(T)

if comm.rank == 0:
    np.save(f'ET_eta{eta}.npy', (energies,T.real))
