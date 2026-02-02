import numpy as np
import sys

from ase.io import read
from ase.units import Hartree

from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER

atoms = read("scatt_sorted.xyz")

calc = GPAW(restart="scatt.gpw", txt=None)
atoms.calc = calc
calc.wfs.set_positions(calc.spos_ac)

fermi = calc.get_fermi_level()

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= calc.get_fermi_level() * S_kMM
    np.save(f"hs_cc_k.npy", (H_kMM, S_kMM))
