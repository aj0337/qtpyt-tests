import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList
from mpi4py import MPI
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.parallel.egrid import GridDesc
from qtpyt.tools import expand_coupling

rank = MPI.COMM_WORLD.Get_rank()


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
            if j > i:  # avoid double counting
                ri = positions[i]
                rj = positions[j] + np.dot(offset, cell)
                dist = np.linalg.norm(ri - rj)
                if np.abs(dist - bond) <= atol:
                    pairs.add((i, j, tuple(offset)))
    return list(pairs)


def build_real_space_device_hamiltonian(
    atoms, pairs, bond=1.42, beta=0.0, t0=-2.7, atol=0.1
):
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


# Parameters
bond = 1.42
onsite = 0.0  # eV
first_neighbor_hopping = -2.7  # eV
beta = 0  # 3.37
neighbor_cutoff = 1.43  # Å for nearest neighbors
energies = np.linspace(-3, 3, 300)

device = read("structure/init.xyz")
self_energy = np.load("output/lead_self_energy.npy", allow_pickle=True)

pairs = get_nearest_neighbor_pairs(device, neighbor_cutoff=neighbor_cutoff)
H_device = build_real_space_device_hamiltonian(
    device, pairs, bond=bond, beta=beta, t0=first_neighbor_hopping, atol=0.1
)
S_device = np.eye(H_device.shape[0])

# expand dimension of lead self energy to dimension of scattering region
expand_coupling(self_energy[0], len(H_device[0]))
expand_coupling(self_energy[1], len(H_device[0]), id="right")

eta = 1e-5
# slice(None) means that we've already expanded the leads to the scattering region
gf = GreenFunction(
    H_device,
    S_device,
    selfenergies=[(slice(None), self_energy[0]), (slice(None), self_energy[1])],
    eta=eta,
)

gd = GridDesc(energies, 1)
dos = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    dos[e] = gf.get_dos(energy)

dos = gd.gather_energies(dos)

if rank == 0:
    np.save("output/Edos_unstrained.npy", (energies, dos))
    plt.plot(energies, dos)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of States (states/eV)")
    plt.title("Density of States")
    plt.grid()
    plt.savefig("output/dos_unstrained.png")
