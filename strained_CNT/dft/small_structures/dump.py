import pickle
import numpy as np
from ase.io import read
from ase.units import Hartree
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER

atoms = read("cnt_(6,0)_cells_3.vasp")

calc = GPAW(restart="scatt.gpw", txt=None)
atoms.set_calculator(calc)
calc.wfs.set_positions(calc.spos_ac)

fermi = calc.get_fermi_level()

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= fermi * S_kMM
    np.save("hs_cc_k.npy", (H_kMM, S_kMM))
    with open("hs_cc_k.pkl", "wb") as file:
        pickle.dump((H_kMM, S_kMM), file)
