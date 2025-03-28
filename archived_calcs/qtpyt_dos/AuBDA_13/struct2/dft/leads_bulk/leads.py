import numpy as np
from ase.io import read
from ase.units import Hartree
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER
import pickle

atoms = read('leads.traj')
basis = 'dzp'

calc = GPAW(h=0.2,
            xc='PBE',
            basis='szp(dzp)',
            occupations=FermiDirac(width=0.1),
            kpts={
                'size': (5, 1, 1),
                'gamma': True
            },
            mode='lcao',
            txt='leads.txt',
            mixer=Mixer(0.1, 5, weight=100.0),
            symmetry={
                'point_group': False,
                'time_reversal': True
            })

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('leads.gpw', mode='all')

fermi = calc.get_fermi_level()
print(fermi, file=open('fermi_leads.txt', 'w'))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    # np.save('hs_leads_k.npy', (H_kMM, S_kMM))
    with open('hs_leads_k.pkl', 'wb') as file:
        pickle.dump((H_kMM, S_kMM), file)
