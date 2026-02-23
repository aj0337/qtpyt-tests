from __future__ import annotations

import numpy as np
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.projector import expand


class DataSelfEnergy(BaseDataSelfEnergy):
    def retarded(self, energy):
        return expand(S_matrix, super().retarded(energy), index_active_region)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))


output_folder = "output"
data_folder = "output/ed"
hamiltonian = np.load(f"{output_folder}/hamiltonian.npy")
index_active_region = np.arange(
    hamiltonian.shape[0]
)  # Placeholder for the indices of the active region, replace with actual indices if available
S_matrix = np.eye(
    hamiltonian.shape[0]
)  # Placeholder for the overlap matrix, replace with actual S if available
ed_self_energy_file = f"{data_folder}/sigma_initial.npy"
energies = np.load(f"{output_folder}/energies.npy")
ed_sigma = load(ed_self_energy_file)
test_energies = np.array([-100, -60, -2, 60, 100])
for e in test_energies:
    retarded_se = ed_sigma.retarded(e)
    trace_retarded_se = np.trace(retarded_se)
    real_trace = np.real(trace_retarded_se)
    imag_trace = np.imag(trace_retarded_se)
    print(f"Real part of Trace of Sigma at energy {e:.3f} eV: {real_trace}")
    print(f"Imaginary part of Trace of Sigma at energy {e:.3f} eV: {imag_trace}")
