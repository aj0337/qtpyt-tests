import numpy as np
from ase.io import read
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank

output_folder = "./"
structure_folder = "../../../structures/unrelaxed"

print("Get paths")

atoms = read(f"{structure_folder}/scatt.xyz")

print("Get atoms")

calc = GPAW(restart=f"{output_folder}/scatt_restart3.gpw", txt=None)
atoms.calc = calc
calc.wfs.set_positions(calc.spos_ac)

fermi = calc.get_fermi_level()

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= fermi * S_kMM
    np.save(f"{output_folder}/hs_cc_k.npy", (H_kMM, S_kMM))
