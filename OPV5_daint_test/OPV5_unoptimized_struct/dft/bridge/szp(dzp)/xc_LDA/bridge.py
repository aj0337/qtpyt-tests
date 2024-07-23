from ase import *
from ase.io import read, write
from gpaw import *
import pickle

atoms = read('./bridge.xyz')

calc = GPAW(
    h=0.2,
    xc='LDA',
    nbands='nao',
    convergence={'bands': 'all'},
    basis='szp(dzp)',
    occupations=FermiDirac(width=0.01),
    kpts=(1, 1, 1),
    mode='lcao',
    txt='bridge.txt',
    mixer=Mixer(0.02, 5, 100),
    parallel=dict(
        band=1,  # band parallelization
        augment_grids=True,  # use all cores for XC/Poisson
        sl_auto=True  # enable ScaLAPACK parallelization
    ))
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('bridge_111.gpw',mode='all')

fermi = calc.get_fermi_level()
print(repr(fermi), file=open('fermi.txt', 'w'))

# Generate cube-files of the orbitals:
# for spin in [0, 1]:
#     for n in range(calc.get_number_of_bands()):
#         wf = calc.get_pseudo_wave_function(band=n, spin=spin)
#         write('device.%d.%d.cube' % (spin, n), atoms, data=wf)
