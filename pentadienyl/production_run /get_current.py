from os import walk
import numpy as np
from matplotlib import pyplot as plt
from ase.units import _e, _hplanck, kB

G0 = 2. * _e**2 / _hplanck


def fermidistribution(energy, kt):
    # fermi level is fixed to zero
    # energy can be a single number or a list
    assert kt >= 0., 'Negative temperature encountered!'

    if kt == 0:
        if isinstance(energy, float):
            return int(energy / 2. <= 0)
        else:
            return (energy / 2. <= 0).astype(int)
    else:
        return 1. / (1. + np.exp(energy / kt))


def current(bias, energy, transmission, temperature=300, unit='uA'):
    """Get the current in uA."""
    if not isinstance(bias, (int, float)):
        bias = bias[np.newaxis]
        energy = energy[:, np.newaxis]
        transmission = transmission[:, np.newaxis]

    fl = fermidistribution(energy - bias / 2., kB * temperature)
    fr = fermidistribution(energy + bias / 2., kB * temperature)

    return G0 * np.trapz((fl - fr) * transmission, x=energy, axis=0) * 1e6  # uA

dV = 0.001
Vmin = 0
Vmax = 0.1
bias = np.arange(Vmin, Vmax + dV / 2., dV)
output_folder = "output/lowdin/beta_1000"
temperature = 11.6
E, T = np.load("reference/ET_dmft.npy")
I = current(bias, E, T, temperature=temperature)

np.save(f"{output_folder}/Reference_DMFT_IV.npy", (bias, I))

plt.figure(figsize=(6, 4))
plt.plot(bias, I, label='DMFT', color='blue')
plt.title('Current-Voltage Characteristic')
plt.legend()
plt.grid()
plt.xlim(Vmin, Vmax)

plt.xlabel('Voltage (V)')
plt.ylabel('Current (uA)')
plt.savefig(f'{output_folder}/Reference_DMFT_IV.png', bbox_inches='tight', dpi=300)
