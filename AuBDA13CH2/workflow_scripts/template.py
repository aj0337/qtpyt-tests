from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.lcao.tools import remove_pbc
from qtpyt.basis import Basis
from qtpyt.lo.tools import rotate_matrix, subdiagonalize_atoms
from ase.io import read
from qtpyt.basis import Basis
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import remove_pbc, rotate_couplings
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.hybridization import Hybridization
from qtpyt.continued_fraction import get_ao_charge
from scipy.linalg import eigvalsh
from qtpyt.base.selfenergy import DataSelfEnergy as BaseDataSelfEnergy
from qtpyt.projector import expand
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc


from scipy.optimize import root
from scipy.interpolate import interp1d
from ase.io import read

from edpyt.nano_dmft import Gfloc, Gfimp as nanoGfimp
from edpyt.dmft import Gfimp, DMFT, Converged
import os
from mpi4py import MPI

# MPI communicators
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Split the communicator into two parts: one for parallel, one for serial
if rank == 0:
    # The master process will run the DMFT serial part
    comm_serial = comm.Split(color=1, key=rank)
else:
    # All other processes will run the parallel part
    comm_parallel = comm.Split(color=2, key=rank)

# ### Helper functions

def get_species_indices(atoms,species):
    indices = []
    for element in species:
        element_indices = atoms.symbols.search(element)
        indices.extend(element_indices)
    return sorted(indices)

def distance(delta):
    global delta_prev
    delta_prev[:] = delta
    return dmft.distance(delta)

def save_sigma(sigma_diag,outputfile):
    L, ne = sigma_diag.shape
    sigma = np.zeros((ne, L, L), complex)

    def save(spin):
        for diag, mat in zip(sigma_diag.T, sigma):
            mat.flat[::(L + 1)] = diag
        np.save(outputfile, sigma)

    for spin in range(1):
        save(spin)

class DataSelfEnergy(BaseDataSelfEnergy):
    """Wrapper"""

    def retarded(self, energy):
        return expand(S_molecule, super().retarded(energy), idx_molecule)


def load(filename):
    return DataSelfEnergy(energies, np.load(filename))

def run(outputfile):

    gd = GridDesc(energies, 1, float)
    T = np.empty(gd.energies.size)

    for e, energy in enumerate(gd.energies):
        T[e] = gf.get_transmission(energy)

    T = gd.gather_energies(T)

    if comm.rank == 0:
        np.save(outputfile, (energies,T.real))

data_folder = 'output/compute_run'

# Parallel part: run on all ranks except rank 0
if rank != 0:
    # ### Control parameters

    GPWDEVICEDIR = 'dft/device/'
    BRIDGE_SPECIES = ("N", "C", "H")
    GPWLEADSDIR = 'dft/leads/'

    lowdin = True
    cc_path = Path(GPWDEVICEDIR)
    pl_path = Path(GPWLEADSDIR)
    gpwfile = f'{cc_path}/scatt.gpw'

    atoms, calc = restart(gpwfile, txt=None)
    fermi = calc.get_fermi_level()
    nao_a = np.array([setup.nao for setup in calc.wfs.setups])
    basis = Basis(atoms, nao_a)

    lcao = LCAOwrap(calc)
    H_lcao = lcao.get_hamiltonian()
    S_lcao = lcao.get_overlap()
    H_lcao -= fermi * S_lcao


    H_leads_lcao, S_leads_lcao = np.load(pl_path / 'hs_pl_k.npy')

    basis_dict = {'Au': 9, 'H': 5, 'C': 13, 'N': 13}

    leads_atoms = read(pl_path / 'leads.xyz')
    leads_basis = Basis.from_dictionary(leads_atoms, basis_dict)

    device_atoms = read(cc_path / 'scatt.xyz')
    device_basis = Basis.from_dictionary(device_atoms, basis_dict)

    nodes = [0,810,1116,1278,1584,2394]

    # Define energy range and broadening factor for the Green's function calculation
    de = 0.2
    energies = np.arange(-3., 3. + de / 2., de).round(7)
    eta = 1e-3

    # Define the number of repetitions (Nr) and unit cell repetition in the leads
    Nr = (1, 5, 3)
    unit_cell_rep_in_leads = (5, 5, 3)

    # ### Subdiagonalize C, N and H of LCAO Hamiltonian (Hamiltonian obtained directly from gpaw)

    # Perform subdiagonalization
    SUBDIAG_SPECIES = ("C", "N", "H")
    subdiag_indices = get_species_indices(atoms, SUBDIAG_SPECIES)
    Usub, eig = subdiagonalize_atoms(basis, H_lcao, S_lcao, a=subdiag_indices)

    # Rotate matrices
    H_sudiagonalized = rotate_matrix(H_lcao, Usub)[None, ...]
    S_sudiagonalized = rotate_matrix(S_lcao, Usub)[None, ...]
    H_sudiagonalized = H_sudiagonalized.astype(np.complex128)
    S_sudiagonalized = S_sudiagonalized.astype(np.complex128)


    # Prepare the k-points and matrices for the leads (Hamiltonian and overlap matrices)
    kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = prepare_leads_matrices(
        H_leads_lcao, S_leads_lcao, unit_cell_rep_in_leads, align=(0, H_sudiagonalized[0, 0, 0]))

    # Remove periodic boundary conditions (PBC) from the device Hamiltonian and overlap matrices
    remove_pbc(device_basis, H_sudiagonalized)
    remove_pbc(device_basis, S_sudiagonalized)

    # Initialize self-energy list for left and right leads
    self_energy = [None, None, None]
    self_energy[0] = PrincipalSelfEnergy(kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr)
    self_energy[1] = PrincipalSelfEnergy(kpts_t, (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), Nr=Nr, id='right')

    # Rotate the couplings for the leads based on the specified basis and repetition Nr
    rotate_couplings(leads_basis, self_energy[0], Nr)
    rotate_couplings(leads_basis, self_energy[1], Nr)

    # Tridiagonalize the device Hamiltonian and overlap matrices based on the partitioned nodes
    hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(nodes, H_sudiagonalized[0], S_sudiagonalized[0])
    del H_sudiagonalized, S_sudiagonalized

    # Initialize the Green's function solver with the tridiagonalized matrices and self-energies
    gf = greenfunction.GreenFunction(hs_list_ii,
                                    hs_list_ij,
                                    [(0, self_energy[0]),
                                    (len(hs_list_ii) - 1, self_energy[1])],
                                    solver='dyson',
                                    eta=eta)


    # ### Define active region and the Green's function for the active region

    # Extract the basis for the subdiagonalized region and get their indices
    basis_subdiag_region = basis[subdiag_indices]
    index_subdiag_region = basis_subdiag_region.get_indices()

    # Define the active region within the subdiagonalized species
    active = {'C': [3],'N': [3]}
    extract_active_region = basis_subdiag_region.extract().take(active)
    index_active_region = index_subdiag_region[extract_active_region]

    gfp = ProjectedGreenFunction(gf, index_active_region)
    hyb = Hybridization(gfp)


    energies = np.arange(-3., 3. + de / 2., de).round(7)
    n_A = len(index_active_region)
    gd = GridDesc(energies, n_A, complex)
    HB = gd.empty_aligned_orbs()

    for e, energy in enumerate(gd.energies):
        HB[e] = hyb.retarded(energy)


    # Create the folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    filename = os.path.join(data_folder, 'hybridization.bin')
    gd.write(HB,filename)


    if comm.rank == 0:
        # Save other data inside the folder
        np.save(os.path.join(data_folder, 'energies.npy'), energies + 1.j * eta)

        # Effective Hamiltonian
        Heff = (hyb.H + hyb.retarded(0.)).real
        np.save(os.path.join(data_folder, 'hamiltonian.npy'), hyb.H)
        np.save(os.path.join(data_folder, 'effective_hamiltonian.npy'), Heff)
        np.save(os.path.join(data_folder, 'eigvals_Heff.npy'), eigvalsh(Heff, gfp.S))

    # Matsubara
    gf.eta = 0.
    assert self_energy[0].eta == 0.
    assert self_energy[1].eta == 0.
    ne = 30
    beta = 70.
    matsubara_energies = 1.j * (2 * np.arange(ne) + 1) * np.pi / beta

    mat_gd = GridDesc(matsubara_energies, n_A, complex)
    HB = mat_gd.empty_aligned_orbs()

    for e, energy in enumerate(mat_gd.energies):
        HB[e] = hyb.retarded(energy)


    # Save the Matsubara hybrid data
    filename = os.path.join(data_folder, 'matsubara_hybridization.bin')
    mat_gd.write(HB, filename)

    if comm.rank == 0:
        np.save(os.path.join(data_folder, 'occupancies.npy'), get_ao_charge(gfp))
        np.save(os.path.join(data_folder, 'matsubara_energies.npy'), matsubara_energies)

    comm_parallel.Barrier()

# Serial part: run only on rank 0
if rank == 0:
    # DMFT calculation (runs serially)
    occupancy_goal = np.load(f'{data_folder}/occupancies.npy')

    L = occupancy_goal.size

    z_ret = np.load(f'{data_folder}/energies.npy')
    z_mats = np.load(f'{data_folder}/matsubara_energies.npy')

    eta = z_ret[0].imag
    beta = np.pi / (z_mats[0].imag)

    hyb_ret = np.fromfile(f'{data_folder}/hybridization.bin', complex).reshape(z_ret.size, L, L)
    hyb_mats = np.fromfile(f'{data_folder}/matsubara_hybridization.bin',
                           complex).reshape(z_mats.size, L, L)

    _HybRet = interp1d(z_ret.real,
                       hyb_ret,
                       axis=0,
                       bounds_error=False,
                       fill_value=0.)
    HybRet = lambda z: _HybRet(z.real)

    _HybMats = interp1d(z_mats.imag,
                        hyb_mats,
                        axis=0,
                        bounds_error=False,
                        fill_value=0.)
    HybMats = lambda z: _HybMats(z.imag)
    HybZro = lambda z: np.zeros((L, z.size), complex)

    H = np.load(f'{data_folder}/hamiltonian.npy').real
    S = np.eye(L)

    idx_neq = np.arange(L)
    idx_inv = np.arange(L)

    U = 4.  # Interaction
    V = np.eye(L) * U

    double_counting = np.diag(V.diagonal() * (occupancy_goal - 0.5))
    gfloc = Gfloc(H - double_counting, np.eye(L), HybMats, idx_neq, idx_inv)

    nimp = gfloc.idx_neq.size
    gfimp = []
    nbaths = 4
    for i in range(nimp):
        gfimp.append(Gfimp(nbaths, z_mats.size, V[i, i], beta))

    gfimp = nanoGfimp(gfimp)

    occupancy_goal = occupancy_goal[gfloc.idx_neq]

    dmft = DMFT(gfimp,
                gfloc,
                occupancy_goal,
                max_iter=200,
                tol=27,
                adjust_mu=True,
                alpha=0.)

    Sigma = lambda z: np.zeros((nimp, z.size), complex)
    delta = dmft.initialize(V.diagonal().mean(), Sigma, mu=0.)
    delta_prev = delta.copy()

    try:
        root(distance, delta_prev, method='broyden1')
    except Converged:
        pass

    np.save(f'{data_folder}/dmft_delta.npy', delta_prev)
    open(f'{data_folder}/mu.txt', 'w').write(str(gfloc.mu))

    _Sigma = lambda z: -double_counting.diagonal()[:, None] - gfloc.mu + gfloc.Sigma(z)[idx_inv]

    dmft_sigma_file = f"{data_folder}/dmft_sigma.npy"
    save_sigma(_Sigma(z_ret), dmft_sigma_file)

    comm_serial.Barrier()

# Synchronize all ranks
comm.Barrier()

# Transmission function calculation
imb = 2  # index of molecule block
S_molecule = hs_list_ii[imb][1]  # overlap of molecule
idx_molecule = index_active_region - nodes[imb]  # indices of active region w.r.t molecule

# Transmission function for DFT
outputfile = f"{data_folder}/dft_transmission.npy"
run(outputfile)

# Add the DMFT self-energy for transmission
if rank == 0:
    self_energy[2] = load(dmft_sigma_file)
    gf.selfenergies.append((imb, self_energy[2]))

    # Transmission function with DMFT
    outputfile = f"{data_folder}/dmft_transmission.npy"
    run(outputfile)
    gf.selfenergies.pop()
