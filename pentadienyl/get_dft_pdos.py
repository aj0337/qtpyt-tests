import pickle
import numpy as np
from mpi4py import MPI
from qtpyt.block_tridiag import greenfunction

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data_folder = "./output/lowdin"
self_energy = np.load(f"{data_folder}/self_energy.npy", allow_pickle=True)
H_subdiagonalized, _ = np.load(f"{data_folder}/hs_los_lowdin.npy")
de = 0.01
energies = np.arange(-15, 15 + de / 2.0, de).round(7)
eta = 1e-3

with open(f"{data_folder}/hs_list_ii.pkl", "rb") as f:
    hs_list_ii = pickle.load(f)
with open(f"{data_folder}/hs_list_ij.pkl", "rb") as f:
    hs_list_ij = pickle.load(f)

gf = greenfunction.GreenFunction(
    hs_list_ii,
    hs_list_ij,
    [],
    # [(0, self_energy[0]), (len(hs_list_ii) - 1, self_energy[1])],
    solver="dyson",
    eta=eta,
)

local_energies = np.array_split(energies, size)[rank]
local_pdos = np.zeros((len(local_energies), H_subdiagonalized.shape[-1]))

for i, energy in enumerate(local_energies):
    local_pdos[i] = gf.get_pdos(energy)

pdos = None
if rank == 0:
    pdos = np.empty(
        (len(energies), H_subdiagonalized.shape[-1]), dtype=local_pdos.dtype
    )

sendcounts = np.array(comm.gather(local_pdos.shape[0], root=0))
if rank == 0:
    displs = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)
    recvbuf = [
        pdos,
        (
            sendcounts * H_subdiagonalized.shape[-1],
            displs * H_subdiagonalized.shape[-1],
        ),
        MPI.DOUBLE,
    ]
else:
    recvbuf = None

comm.Gatherv(sendbuf=local_pdos, recvbuf=recvbuf, root=0)

if rank == 0:
    np.save(f"{data_folder}/pdos_dyson_no_leads_se.npy", pdos)
