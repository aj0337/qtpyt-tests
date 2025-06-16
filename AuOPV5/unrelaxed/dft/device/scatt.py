from __future__ import print_function

import os

from ase import *
from ase.io import read
from gpaw import *

input_folder = "../../../structures/unrelaxed"
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)

atoms = read(f"{input_folder}/scatt.xyz")
basis = {"Au": "szp(dzp)", "H": "dzp", "C": "dzp", "S": "dzp"}

temperature = 4.2
kbt = temperature * 8.617343e-5
calc = GPAW(
    h=0.2,
    xc="PBE",
    basis=basis,
    occupations=FermiDirac(width=kbt),
    kpts=(1, 1, 1),
    mode="lcao",
    txt=f"{output_folder}/scatt.txt",
    mixer=Mixer(0.02, 5, weight=100.0),
    symmetry={"point_group": False, "time_reversal": True},
)

atoms.calc = calc
atoms.get_potential_energy()
calc.write(f"{output_folder}/scatt.gpw", mode="all")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open(f"{output_folder}/fermi.txt", "w"))
