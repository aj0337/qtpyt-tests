from __future__ import print_function

import os

from ase import *
from ase.io import read
from gpaw import *

input_folder = "../../../structures/unrelaxed"
output_folder = "./"
os.makedirs(output_folder, exist_ok=True)

atoms = read(f"{input_folder}/scatt.xyz")
basis = {"H": "szp(dzp)", "C": "szp(dzp)"}

# temperature = 9
# kbt = temperature * 8.617343e-5
kbt = 0.1
calc = GPAW(
    h=0.2,
    xc="PBE",
    basis=basis,
    occupations=FermiDirac(width=kbt),
    convergence = {"energy": 1e-3},
    kpts=(1, 1, 1),
    mode="lcao",
    txt=f"{output_folder}/scatt.txt",
    # mixer=Mixer(0.1, 5, weight=100.0),
    mixer=MixerSum(beta=0.1, nmaxold=3, weight=50.0),
    symmetry={"point_group": False, "time_reversal": True},
)

atoms.calc = calc
atoms.get_potential_energy()
calc.write(f"{output_folder}/scatt.gpw", mode="all")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open(f"{output_folder}/fermi.txt", "w"))
