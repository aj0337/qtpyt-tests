# Toy model workflow

Quick testing workflow:

```bash
python run_workflow.py
```

## Layout

- `run_workflow.py`: **USER CONFIG** + `main()` (edit parameters here)
- `workflow_core.py`: implementation (Hamiltonian/PPP/ED/transmission)

## Outputs

Writes under `output/`:

- `hamiltonian.npy`, `gamma_L.npy`, `gamma_R.npy`, `occupancies.npy`, `U_ppp.txt`, `energies.npy`
- `ed/self_energy.npy`, `ed/ET.npy`
- `ed/vertex_correction.npz`

`ed/ET.npy` contains 4 rows:

- `E` (energies)
- `T_elastic`
- `T_inelastic`
- `T_total = T_elastic + T_inelastic`

Load example:

```python
E, Tel, Tin, Ttot = np.load("output/ed/ET.npy")
```

The vertex-correction matrix is saved as a compressed NumPy archive:

```python
import numpy as np

data = np.load("output/ed/vertex_correction.npz")
E = data["energies"]
Lambda = data["Lambda"]  # shape: (nE, n, n)

tr = np.trace(Lambda, axis1=1, axis2=2)
tr_re = tr.real
tr_im = tr.imag
```

## Dependencies

- `numpy`
- `ase` (and typically `scipy` due to ASE imports)
- `edpyt` (for ED self-energy)

Example install:

```bash
pip install numpy ase scipy edpyt
```
