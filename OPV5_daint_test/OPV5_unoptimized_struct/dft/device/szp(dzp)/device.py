from ase import *
from ase.io import read, write
from gpaw import *
import pickle
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

atoms = read('./device.xyz')

calc = GPAW(
    h=0.2,
    xc='LDA',
    nbands='nao',
    convergence={'bands': 'all'},
    basis='szp(dzp)',
    occupations=FermiDirac(width=0.01),
    kpts=(1, 1, 1),
    mode='lcao',
    txt='device.txt',
    mixer=Mixer(0.02, 5, 100),
    parallel=dict(
        band=1,  # band parallelization
        augment_grids=True,  # use all cores for XC/Poisson
        sl_auto=True  # enable ScaLAPACK parallelization
    ))
atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.write('device.gpw', mode='all')
fermi = calc.get_fermi_level()
print(repr(fermi), file=open('fermi.txt', 'w'))
