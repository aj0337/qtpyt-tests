from __future__ import print_function

from ase import *
from ase.io import read
from gpaw import *

atoms = read("scatt.xyz")
basis = {"Au": "szp(dzp)", "H": "dzp", "C": "dzp", "N": "dzp"}

calc = GPAW(
    h=0.2,
    xc="PBE",
    basis=basis,
    occupations=FermiDirac(width=0.2),
    kpts=(1, 1, 1),
    mode="lcao",
    txt="scatt.txt",
    mixer=Mixer(0.1, 5, weight=100.0),
    symmetry={"point_group": False, "time_reversal": True},
)

atoms.calc = calc
atoms.get_potential_energy()
calc.write("scatt.gpw", mode="all")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open("fermi_scatt.txt", "w"))
