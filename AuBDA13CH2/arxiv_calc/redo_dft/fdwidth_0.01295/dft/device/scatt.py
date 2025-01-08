from __future__ import print_function

from ase import *
from ase.io import read
from gpaw import *

atoms = read('scatt.xyz')
basis = {'Au':'szp(dzp)','H':'dzp','C':'dzp','N':'dzp'}


kbt = 0.01295
calc = GPAW(h=0.2,
            xc='PBE',
            basis=basis,
            occupations=FermiDirac(width=kbt),
            kpts=(1, 1, 1),
            mode='lcao',
            txt='scatt.txt',
            mixer=Mixer(0.02, 5, weight=100.0),
            symmetry={'point_group': False, 'time_reversal': True})

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('scatt.gpw',mode='all')

fermi = calc.get_fermi_level()
print(repr(fermi), file=open('fermi.txt', 'w'))
