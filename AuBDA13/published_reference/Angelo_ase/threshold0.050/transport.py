from ase.transport.calculators import TransportCalculator
import pickle
import numpy as np

imode = 'pzd' # lcao, pz pzd

# leads:            Hamiltonian and overlap matrices
h1, s1 = pickle.load(open('lead1_hs_lcao.pckl', 'rb'))
h2, s2 = pickle.load(open('lead2_hs_lcao.pckl', 'rb'))

# scattering region Hamiltonian and overlap matrices
if imode == 'lcao':
    h, s  = pickle.load(open('scatt_hs_lcao.pckl', 'rb'))
elif imode == 'pz':
    h, s  = pickle.load(open('scatt_hs_pz.pckl',  'rb'))
elif imode == 'pzd':
    h, s  = pickle.load(open('scatt_hs_pzd.pckl',  'rb'))
else:
    print('error: imode {\'lcao\', \'pz\', \'pzd\'}')
    quit()

# energy mesh
energies = np.arange(-4.0,3.0,0.01)

# setup calculator and evaluate transmission function
tcalc = TransportCalculator(h=h, h1=h1, h2=h2, # hamiltonian matrices
                            s=s, s1=s1, s2=s2, # overlap matrices
                            energies=energies, # energy grid
                            align_bf=0)        # align the Fermi levels

Te = tcalc.get_transmission()

# save data on HDD
ofp = open('ET_'+str(imode)+'.dat','w')
for ie in range(len(energies)):
    ofp.write('%9.6f\t%21.16e\n' %(energies[ie],Te[ie]))
ofp.close()

