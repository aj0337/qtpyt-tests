from __future__ import print_function

import os
from ase.io import read
from gpaw import GPAW, FermiDirac, Mixer

input_folder = "../../../structures/unrelaxed/sorted"
output_folder = "./"
os.makedirs(output_folder, exist_ok=True)

atoms = read(f"{input_folder}/scatt.xyz")
basis = {"H": "szp(dzp)", "C": "szp(dzp)"}

kbt = 1e-3
calc = GPAW(
    h=0.2,
    xc="PBE",
    basis=basis,
    occupations=FermiDirac(width=kbt),
    convergence={"energy": 1e-2},
    kpts=(1, 1, 1),
    mode="lcao",
    txt=f"{output_folder}/scatt.txt",
    mixer=Mixer(0.05, 5, weight=100.0),
    maxiter=20,
    symmetry={"point_group": False, "time_reversal": True},
)

atoms.calc = calc

try:
    atoms.get_potential_energy()
except Exception as e:
    print("WARNING: Calculation did not converge.")
    print(f"Error: {e}")
finally:
    calc.write(f"{output_folder}/scatt.gpw", mode="all")
    try:
        fermi = calc.get_fermi_level()
        with open(f"{output_folder}/fermi.txt", "w") as f:
            print(repr(fermi), file=f)
    except Exception:
        print("Could not determine Fermi level due to failure in SCF convergence.")
