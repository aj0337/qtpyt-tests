import os
import numpy as np
from ase.io import read
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank


input_folder = "../../../structures/unrelaxed"
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)

atoms = read(f"{input_folder}/leads.xyz")
basis = {"H": "dzp", "C": "dzp"}

temperature = 9
kbt = temperature * 8.617343e-5
calc = GPAW(
    h=0.2,
    xc="PBE",
    basis=basis,
    occupations=FermiDirac(width=kbt),
    kpts={"size": (11, 1, 1), "gamma": True},
    mode="lcao",
    txt=f"{output_folder}/leads.txt",
    mixer=Mixer(0.02, 5, weight=100.0),
    symmetry={"point_group": False, "time_reversal": True},
)

atoms.calc = calc
atoms.get_potential_energy()
calc.write(f"{output_folder}/leads.gpw", mode="all")

fermi = calc.get_fermi_level()
print(fermi, file=open(f"{output_folder}/fermi.txt", "w"))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= fermi * S_kMM
    np.save(f"{output_folder}/hs_pl_k.npy", (H_kMM, S_kMM))
