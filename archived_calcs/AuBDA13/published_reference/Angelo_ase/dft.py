from ase import Atoms
from ase.io import read
from gpaw import *
from gpaw.lcao.tools import *
import pickle as pickle


# basis set
basis = {'Au':'szp(dzp)','H':'dzp','C':'dzp','N':'dzp'}



# scattering region
# ---------------------------------------------------------------------
atoms = read('scatt.xyz')

# GPAW calculator
calc = GPAW(h=0.2,
            xc='PBE',
            basis=basis,
            occupations=FermiDirac(width=0.1),
            kpts=(1, 1, 1),
            mode='lcao',
            txt='scatt.txt',
            mixer=Mixer(0.1, 5, weight=100.0),
            symmetry={'point_group': False, 'time_reversal': False})
atoms.calc = calc

# DFT calculation
atoms.get_potential_energy()

# fetch Fermi level
Ef = atoms.calc.get_fermi_level()

# get LCAO Hamiltonian and overlap matrix
H_skMM, S_kMM = get_lcao_hamiltonian(calc)

# only use first kpt, spin (as there are no more)
H, S = H_skMM[0, 0], S_kMM[0]
H -= Ef * S
remove_pbc(atoms, H, S, 0)

# dump Hamiltonian and overlap matrix to a pickle file
pickle.dump((H.astype(complex), S.astype(complex)),
            open('scatt_hs_lcao.pickle', 'wb'), 2)



# left principal layer 
# ---------------------------------------------------------------------

atoms = read('leads_supercell.xyz')

# GPAW calculator
calc = GPAW(h=0.2,
            xc='PBE',
            basis=basis,
            occupations=FermiDirac(width=0.1),
            kpts=(5, 1, 1), # more kpts needed as the x-direction is shorter
            mode='lcao',
            txt='leads.txt',
            mixer=Mixer(0.1, 5, weight=100.0),
            symmetry={'point_group': False, 'time_reversal': False})
atoms.calc = calc

# DFT calculation
atoms.get_potential_energy()

# fetch Fermi level
Ef = atoms.calc.get_fermi_level()

# get LCAO Hamiltonian and overlap matrix
ibz2d_k, weight2d_k, H_skMM, S_kMM = get_lead_lcao_hamiltonian(calc)

# only use first kpt, spin (as there are no more)
H, S = H_skMM[0, 0], S_kMM[0]
H -= Ef * S

# dump Hamiltonian and overlap matrix to a pickle file
pickle.dump((H, S), open('lead1_hs_lcao.pickle', 'wb'), 2)


# right principal layer 
# ---------------------------------------------------------------------
# identical to left prinicpal layer so we don't have to do anything
# just dump same Hamiltonian and overlap matrix to a pickle file
pickle.dump((H, S), open('lead2_hs_lcao.pickle', 'wb'), 2)


