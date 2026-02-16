import matplotlib.pyplot as plt
import numpy as np
from qtpyt.tools import expand_coupling
from ase.build import nanotube
from ase.neighborlist import NeighborList
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.parallel.egrid import GridDesc


def build_real_space_device_hamiltonian(atoms, pairs, bond=1.42, beta=0.0, t0=-2.7):
    """Construct real-space Hamiltonian for a CNT device using known neighbor pairs.

    Args:
        atoms (ase.Atoms): CNT device structure.
        pairs (List[Tuple[int, int, Tuple[int, int, int]]]): Precomputed nearest-neighbor pairs.
        bond (float): Ideal bond length.
        beta (float): Decay factor for hopping integral.
        t0 (float): Hopping at ideal bond length.
        atol (float): Absolute tolerance for bond length deviation.

    Returns:
        np.ndarray: Hamiltonian matrix (shape: N_atoms x N_atoms).
    """

    N_atoms = len(atoms)
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    H = np.zeros((N_atoms, N_atoms), dtype=float)

    for i, j, offset in pairs:
        ri = positions[i]
        rj = positions[j] + np.dot(offset, cell)
        dist = np.linalg.norm(ri - rj)
        hopping = t0 * np.exp(-beta * (dist / bond - 1))
        H[i, j] = hopping
        H[j, i] = hopping

    return H


def get_nearest_neighbor_pairs(atoms, neighbor_cutoff=1.43, bond=1.42, atol=0.1):
    """
    Compute nearest-neighbor atom pairs using a distance cutoff and bond tolerance.

    Args:
        atoms (ase.Atoms): The atomic structure.
        neighbor_cutoff (float): Distance threshold for neighbors.
        bond (float): Ideal bond length.
        atol (float): Absolute tolerance for bond length deviation.

    Returns:
        List[Tuple[int, int, Tuple[int, int, int]]]: Valid neighbor pairs and cell offsets.
    """
    N = len(atoms)
    cutoffs = [neighbor_cutoff] * N
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    positions = atoms.get_positions()
    cell = atoms.get_cell()

    pairs = set()
    for i in range(N):
        for j, offset in zip(*nl.get_neighbors(i)):
            if j > i:
                ri = positions[i]
                rj = positions[j] + np.dot(offset, cell)
                dist = np.linalg.norm(ri - rj)
                if np.abs(dist - bond) <= atol:
                    pairs.add((i, j, tuple(offset)))
    return list(pairs)


# Parameters
n, m = 6, 0
ncells = 2
bond = 1.42
onsite = 0.0  # eV
eta = 1e-5
first_neighbor_hopping = -2.7  # eV
beta = 0.0
neighbor_cutoff = 1.43  # Å for nearest neighbors
d0 = bond
energies = np.linspace(-3, 3, 100)
self_energy = np.load("lead_self_energy.npy", allow_pickle=True)

device = build_device_structure(n=6, m=0, N_L=1, N_M=4, N_R=1, direction="x")
pairs = get_nearest_neighbor_pairs(device, neighbor_cutoff=neighbor_cutoff)


H_device = build_real_space_device_hamiltonian(
    device, pairs, bond=bond, beta=beta, t0=first_neighbor_hopping
)
S_device = np.eye(H_device.shape[0])
plt.imshow(H_device, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Real-space Hamiltonian")
plt.show()


expand_coupling(self_energy[0], len(H_device[0]))
expand_coupling(self_energy[1], len(H_device[0]), id="right")

# slice(None) means that we've already expanded the leads to the scattering region
gf = GreenFunction(
    H_device,
    S_device,
    selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
    eta=eta,
)

gd = GridDesc(energies, 1)
T = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T[e] = gf.get_transmission(energy, ferretti=False)

T = gd.gather_energies(T)

plt.figure(figsize=(10, 5))
plt.plot(energies, T)
plt.xlabel("Energy (eV)")
plt.ylabel("Transmission")
plt.title("Transmission vs Energy")
plt.xlim(-3, 3)
# plt.ylim(0, 1)
plt.grid()
plt.show()


# ### Strained

beta = 3.1451  # Value from WannierTools, https://pubs.acs.org/doi/10.1021/acs.nanolett.9b05117
strained_device = apply_strain(device, 4.0, axis=0)
H_strained_device = build_real_space_device_hamiltonian(
    strained_device, pairs, bond=bond, beta=beta, t0=first_neighbor_hopping
)
S_strained_device = np.eye(H_strained_device.shape[0])
plt.imshow(H_strained_device, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.title("Strained Real-space Hamiltonian")
plt.show()

strain_percentages = np.array([0, 1, 2, 3, 4])
plt.figure(figsize=(10, 5))
for strain in strain_percentages:
    strained_device = apply_strain(device, strain, axis=0)
    H_strained_device = build_real_space_device_hamiltonian(
        strained_device, pairs, bond=bond, beta=beta, t0=first_neighbor_hopping
    )
    S_strained_device = np.eye(H_strained_device.shape[0])

    gf = GreenFunction(
        H_strained_device,
        S_strained_device,
        selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
        eta=eta,
    )

    gd = GridDesc(energies, 1)
    T = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy, ferretti=False)

    T = gd.gather_energies(T)
    plt.plot(energies, T, label=f"Strain: {strain}%")
plt.xlabel("Energy (eV)")
plt.ylabel("Transmission")
plt.title("Transmission vs Energy")
plt.xlim(-3, 3)
# plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

# ### Increase number of unit cell that are strained

N_Ms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.figure(figsize=(10, 5))

for N_M in N_Ms:
    device = build_device_structure(n=6, m=0, N_L=1, N_M=N_M, N_R=1, direction="x")
    pairs = get_nearest_neighbor_pairs(device, neighbor_cutoff=neighbor_cutoff)

    H_device = build_real_space_device_hamiltonian(
        device, pairs, bond=bond, beta=beta, t0=first_neighbor_hopping
    )
    S_device = np.eye(H_device.shape[0])

    self_energy = np.load("lead_self_energy.npy", allow_pickle=True)

    expand_coupling(self_energy[0], len(H_device[0]))
    expand_coupling(self_energy[1], len(H_device[0]), id="right")

    gf = GreenFunction(
        H_device,
        S_device,
        selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
        eta=eta,
    )

    gd = GridDesc(energies, 1)
    T = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy, ferretti=False)

    T = gd.gather_energies(T)
    plt.plot(energies, T, label=f"N_M: {N_M}")

plt.xlabel("Energy (eV)")
plt.ylabel("Transmission")
plt.title("Transmission vs Energy")
plt.xlim(-3, 3)
# plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

N_Ms = [2, 4, 6, 8, 10]
strain = 4.0
plt.figure(figsize=(10, 5))

for N_M in N_Ms:
    device = build_device_structure(n=6, m=0, N_L=1, N_M=N_M, N_R=1, direction="x")
    pairs = get_nearest_neighbor_pairs(device, neighbor_cutoff=neighbor_cutoff)

    strained_device = apply_strain(device, strain, axis=0)
    H_strained_device = build_real_space_device_hamiltonian(
        strained_device, pairs, bond=bond, beta=beta, t0=first_neighbor_hopping
    )
    S_strained_device = np.eye(H_strained_device.shape[0])

    self_energy = np.load("lead_self_energy.npy", allow_pickle=True)
    expand_coupling(self_energy[0], len(H_strained_device[0]))
    expand_coupling(self_energy[1], len(H_strained_device[0]), id="right")

    gf = GreenFunction(
        H_strained_device,
        S_strained_device,
        selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
        eta=eta,
    )

    gd = GridDesc(energies, 1)
    T = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy, ferretti=False)

    T = gd.gather_energies(T)
    plt.plot(energies, T, label=f"N_M: {N_M}")

plt.xlabel("Energy (eV)")
plt.ylabel("Transmission")
plt.title("Transmission vs Energy of Strained CNT")
plt.xlim(-3, 3)
# plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

# ### Increase the number of unstrained unit cells that serve as connection to the leads
# - Fix the number of strained unit cells to 10 and the strain to 4 %

N_M = 10
N_Ls = [1, 2, 3, 4, 5]
strain = 4.0
plt.figure(figsize=(10, 5))

for N_L in N_Ls:
    device = build_device_structure(n=6, m=0, N_L=N_L, N_M=N_M, N_R=N_L, direction="x")
    pairs = get_nearest_neighbor_pairs(device, neighbor_cutoff=neighbor_cutoff)

    strained_device = apply_strain(device, strain, axis=0)
    H_strained_device = build_real_space_device_hamiltonian(
        strained_device, pairs, bond=bond, beta=beta, t0=first_neighbor_hopping
    )
    S_strained_device = np.eye(H_strained_device.shape[0])

    self_energy = np.load("lead_self_energy.npy", allow_pickle=True)
    expand_coupling(self_energy[0], len(H_strained_device[0]))
    expand_coupling(self_energy[1], len(H_strained_device[0]), id="right")

    gf = GreenFunction(
        H_strained_device,
        S_strained_device,
        selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
        eta=eta,
    )

    gd = GridDesc(energies, 1)
    T = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy, ferretti=False)

    T = gd.gather_energies(T)
    plt.plot(energies, T, label=f"N_L, N_R: {N_L}")

plt.xlabel("Energy (eV)")
plt.ylabel("Transmission")
plt.title("Transmission vs Energy of Strained CNT")
plt.xlim(-3, 3)
# plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()
