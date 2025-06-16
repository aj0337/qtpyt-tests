import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.units import Hartree
from gpaw import *
from gpaw.utilities.dos import LCAODOS, RestartLCAODOS, fold

gpaw_file = '../../dft/device/scatt.gpw'
calc = GPAW(gpaw_file, txt=None)
dos = RestartLCAODOS(calc)
pdos = LCAODOS(calc)

atoms = read('../../dft/device/scatt.xyz')
atoms_C_index = np.where(atoms.symbols == 'C')[0]
atoms_N_index = np.where(atoms.symbols == 'N')[0]
atoms_H_index = np.where(atoms.symbols == 'H')[0]
atoms_bridge_index = np.where((atoms.symbols == 'C')
                              | (atoms.symbols == 'N')
                              | (atoms.symbols == 'H'))[0]
atoms_Au_index = np.where(atoms.symbols == 'Au')[0]
num_C_atoms = len(atoms_C_index)
num_N_atoms = len(atoms_N_index)
num_H_atoms = len(atoms_H_index)
num_Au_atoms = len(atoms_Au_index)

C_orbitals = pdos.get_atom_indices(atoms_C_index)
N_orbitals = pdos.get_atom_indices(atoms_N_index)
H_orbitals = pdos.get_atom_indices(atoms_H_index)
Au_orbitals = pdos.get_atom_indices(atoms_Au_index)
bridge_orbitals = pdos.get_atom_indices(atoms_bridge_index)

num_spherical_harmonics_C = 13
num_spherical_harmonics_N = 13
num_spherical_harmonics_Au = 15
num_spherical_harmonics_H = 5

tot_basis_func = num_spherical_harmonics_C * num_C_atoms + num_spherical_harmonics_N * num_N_atoms + num_spherical_harmonics_H * num_H_atoms + num_spherical_harmonics_Au * num_Au_atoms

smearing = [0.05, 0.07]

for s in smearing:

    directory = f'smearing_{s}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    energies, weights = dos.get_subspace_pdos(range(tot_basis_func))
    energy, total_dos = fold(energies * Hartree, weights, 5000, s)
    np.save(os.path.join(directory, 'total_dos.npy'), (energy, total_dos))

    energy, C_pdos = dos.get_subspace_pdos(C_orbitals)
    energy, C_pdos = fold(energy * Hartree, C_pdos, 5000, s)
    np.save(os.path.join(directory, 'C_pdos.npy'), (energy, C_pdos))

    energy, N_pdos = dos.get_subspace_pdos(N_orbitals)
    energy, N_pdos = fold(energy * Hartree, N_pdos, 5000, s)
    np.save(os.path.join(directory, 'N_pdos.npy'), (energy, N_pdos))

    energy, H_pdos = dos.get_subspace_pdos(H_orbitals)
    energy, H_pdos = fold(energy * Hartree, H_pdos, 5000, s)
    np.save(os.path.join(directory, 'H_pdos.npy'), (energy, H_pdos))

    energy, Au_pdos = dos.get_subspace_pdos(Au_orbitals)
    energy, Au_pdos = fold(energy * Hartree, Au_pdos, 5000, s)
    np.save(os.path.join(directory, 'Au_pdos.npy'), (energy, Au_pdos))

    energy, bridge_pdos = dos.get_subspace_pdos(bridge_orbitals)
    energy, bridge_pdos = fold(energy * Hartree, bridge_pdos, 5000, s)
    np.save(os.path.join(directory, 'bridge_pdos.npy'), (energy, bridge_pdos))
