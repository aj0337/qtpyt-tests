from ase.io import read, write
from ase.units import Bohr
from gpaw import *
import numpy as np
import os

atoms = read('scatt.xyz')
calc = GPAW('scatt.gpw', txt=None)

folder_path = 'lcao_cube_files'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

homo_energy, lumo_energy = calc.get_homo_lumo()
band_energies = calc.get_eigenvalues()

homo_band_index = np.argmin(np.abs(band_energies - homo_energy))
lumo_band_index = np.argmin(np.abs(band_energies - lumo_energy))

nbands_around_fermi = 4
bands_around_fermi = range(homo_band_index - nbands_around_fermi,
                           lumo_band_index + nbands_around_fermi + 1)

for band in bands_around_fermi:
    wf = calc.get_pseudo_wave_function(band=band)
    if band == homo_band_index:
        fname = f'{folder_path}/band_homo.cube'
    elif band == lumo_band_index:
        fname = f'{folder_path}/band_lumo.cube'
    elif band < homo_band_index:
        diff = homo_band_index - band
        fname = f'{folder_path}/band_homo-{diff}.cube'
    else:
        diff = band - lumo_band_index
        fname = f'{folder_path}/band_lumo+{diff}.cube'
    write(fname, atoms, data=wf * Bohr**1.5)
