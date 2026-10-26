import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW

calc = GPAW('bridge.gpw', txt=None)
energies, dos_values = calc.get_dos(spin=0, npts=2001, width=0.01) # width in eV for Gaussian smearing
Efermi = calc.get_fermi_level()
energies -= Efermi

np.savetxt('dos.dat', np.column_stack([energies, dos_values]), header='Energy (eV)   DOS')

plt.plot(energies, dos_values)
plt.xlabel('Energy (eV)')
plt.ylabel('DOS')
plt.title('Density of States')
plt.savefig('dos.png')
plt.show()
