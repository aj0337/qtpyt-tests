import numpy as np
from ase.io import read
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank

atoms = read("leads.xyz")
basis = "dzp"

calc = GPAW(
    h=0.2,
    xc="PBE",
    basis="szp(dzp)",
    occupations=FermiDirac(width=0.1),
    kpts={"size": (5, 5, 3), "gamma": True},
    mode="lcao",
    txt="leads.txt",
    mixer=Mixer(0.1, 5, weight=100.0),
    symmetry={"point_group": False, "time_reversal": True},
)

atoms.calc = calc
atoms.get_potential_energy()
calc.write("leads.gpw", mode="all")

fermi = calc.get_fermi_level()
print(fermi, file=open("fermi_leads.txt", "w"))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= fermi * S_kMM
    np.save("hs_pl_k.npy", (H_kMM, S_kMM))
