from ase.io import read, write
from ase.units import Bohr
from gpaw import *

# load binary file and get calculator
atoms = read('scatt.xyz')
calc = GPAW('scatt.gpw',txt=None)

# loop over all wfs and write their cube files
nbands = calc.get_number_of_bands()
print(nbands)
for band in range(nbands):
    print(band)
    wf = calc.get_pseudo_wave_function(band=band)
    fname = f'band_{band}.cube'
    print('writing wf', band, 'to file', fname)
    write(fname, atoms, data=wf * Bohr**1.5)
