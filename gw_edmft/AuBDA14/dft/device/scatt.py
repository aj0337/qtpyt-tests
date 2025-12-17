from ase import *
from ase.io import read
from gpaw import *

atoms = read("scatt.xyz")
basis = {"Au": "szp(dzp)", "H": "szp(dzp)", "C": "szp(dzp)", "N": "szp(dzp)"}

kbt = 1e-2
calc = GPAW(
    h=0.2,
    xc="LDA",
    basis=basis,
    occupations=FermiDirac(width=kbt),
    kpts=(1, 1, 1),
    mode="lcao",
    txt="scatt.txt",
    mixer=Mixer(0.02, 5, 100),
)
atoms.calc = calc
atoms.get_potential_energy()
calc.write("scatt.gpw", mode="all")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open("fermi.txt", "w"))
