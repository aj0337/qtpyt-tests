from __future__ import print_function

from ase import *
from ase.io import read
from gpaw import *

atoms = read("scatt.xyz")
basis = {"Au": "szp(dzp)", "default": "dzp"}

calc = GPAW(
    h=0.2,
    xc="PBE",
    nbands="nao",
    convergence={"bands": "all"},
    basis=basis,
    occupations=FermiDirac(width=0.01),
    kpts=(1, 1, 1),
    mode="lcao",
    txt="scatt.txt",
    mixer=Mixer(0.02, 5, 100),
)
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write("scatt.gpw")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open("fermi_.txt", "w"))
