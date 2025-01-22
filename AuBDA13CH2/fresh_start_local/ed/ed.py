from pathlib import Path

import numpy as np
import tqdm
from edpyt.cotunneling import get_active_neig
from edpyt.espace import build_espace, screen_espace
from edpyt.gf2_lanczos import build_gf2_lanczos
from edpyt.shared import params
from matplotlib import pyplot as plt


class Sigma:
    def __init__(self, gf0, gf, eta=1e-5):
        self.gf0 = gf0
        self.gf = gf
        self.eta = eta

    def retarded(self, energy):
        energies = np.atleast_1d(energy)
        g0 = self.gf0(energies, self.eta)
        g = self.gf(energies, self.eta)
        sigma = np.empty((energies.size, gf.n, gf.n), complex)
        for e, energy in enumerate(tqdm.tqdm(energies)):
            sigma[e] = np.linalg.inv(g0[..., e]) - np.linalg.inv(g[..., e])
        return sigma


path = Path("../output/lowdin/")
H_eff = np.load(path / "effective_hamiltonian.npy")
nimp = H_eff.shape[0]
occupancy_goal = np.load("../output/lowdin/occupancy/occupancies_gfp.npy")
de = 0.01
energies = np.arange(-2, 2 + de / 2.0, de).round(7)
eta = 1e-2
beta = 1000.0
params["z"] = occupancy_goal
V = np.loadtxt("../output/lowdin/U_matrix.txt")
DC = np.diag(V.diagonal() * (occupancy_goal - 0.5))
neig = get_active_neig(nimp, [(nimp // 2, nimp // 2)], 3)

espace, egs = build_espace(H_eff, np.zeros_like(H_eff), neig_sector=neig)
screen_espace(espace, egs, beta)
gf0 = build_gf2_lanczos(H_eff, np.zeros_like(H_eff), espace, beta, egs)
# TODO Look at implementing build_gf_lanczos here instead of build_gf2_lanczos
DOS0 = -1 / np.pi * gf0(energies, eta).imag.trace(axis1=0, axis2=1)

espace, egs = build_espace(H_eff - DC, V, neig_sector=neig)
screen_espace(espace, egs, beta)
gf = build_gf2_lanczos(H_eff - DC, V, espace, beta, egs)
DOS = -1 / np.pi * gf(energies, eta).imag.trace(axis1=0, axis2=1)

plt.plot(energies, DOS0, label="U=0")
plt.plot(energies, DOS, label="U=matrix")
plt.legend()
plt.yscale("log")
plt.show()

sigma = Sigma(gf0, gf, eta=eta)
sig = sigma.retarded(energies)
np.save("../output/lowdin/ed/ed_sigma.npy", sig)
