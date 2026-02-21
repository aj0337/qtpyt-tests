from __future__ import annotations

import numpy as np
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.projector import expand


class DataSelfEnergy(BaseDataSelfEnergy):
    def retarded(self, energy):
        return expand(S_matrix, super().retarded(energy), index_active_region)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


data_folder = "output/ed"
hamiltonian = np.load(f"{data_folder}/hamiltonian.npy")
index_active_region = np.arange(
    hamiltonian.shape[0]
)  # Placeholder for the indices of the active region, replace with actual indices if available
S_matrix = np.eye(
    hamiltonian.shape[0]
)  # Placeholder for the overlap matrix, replace with actual S if available
ed_self_energy_file = f"{data_folder}/ed/self_energy_with_dcc.npy"
energies = np.array([-100, 100])
ed_sigma = load(ed_self_energy_file)
for e in energies:
    print(f"Sigma at energy {e:.3f} eV: {ed_sigma.retarded(e)}")
