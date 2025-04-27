import matplotlib.pyplot as plt
import numpy as np
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.tools import expand_coupling

H_device = np.load("hamiltonian/H_R_device.npy")
H_device = H_device.astype(np.complex128)
S_device = np.eye(H_device.shape[0])
H_k_leads = np.load("hamiltonian/H_k_leads.npy")
nkpts = H_k_leads.shape[0]
dim = H_k_leads.shape[1]
S_k_leads = np.zeros((nkpts, dim, dim), dtype=np.complex128)
for i in range(nkpts):
    S_k_leads[i] = np.eye(dim, dtype=np.complex128)

nkpts_leads = (6, 1, 1)

# Prepare the k-points and matrices for the leads (Hamiltonian and overlap matrices)
kpts_t, h_leads_kii, s_leads_kii, h_leads_kij, s_leads_kij = map(lambda m: m[0], prepare_leads_matrices(
    H_k_leads,
    S_k_leads,
    nkpts_leads,
    # align=(0, H_device[0, 0]),
))

# Initialize self-energy list for left and right leads
self_energy = [None, None]

# Create LeadSelfEnergy objects for left and right leads
# Uses Sancho Rubio method to compute the surface Green's function
self_energy[0] = LeadSelfEnergy((h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij))
self_energy[1] = LeadSelfEnergy(
    (h_leads_kii, s_leads_kii), (h_leads_kij, s_leads_kij), id="right"
)

# expand dimension of lead self energy to dimension of scattering region
expand_coupling(self_energy[0], len(H_device[0]))
expand_coupling(self_energy[1], len(H_device[0]), id='right')

de = 0.01
energies = np.arange(-1.,1.+de/2.,de).round(7)
eta = 1e-3

# slice(None) means that we've already expanded the leads to the scattering region
gf = GreenFunction(H_device, S_device, selfenergies=[(slice(None),self_energy[0]),(slice(None),self_energy[1])], eta=eta)


gd = GridDesc(energies, 1)
T = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T[e] = gf.get_transmission(energy)

T = gd.gather_energies(T)
if comm.rank == 0:
    np.save(f'ET', (energies, T))

    plt.plot(energies, T)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Transmission")
    plt.title("Transmission vs Energy")
    plt.grid()
    plt.savefig("transmission.png")
