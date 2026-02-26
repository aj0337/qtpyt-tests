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

- Common inputs:
  - `hamiltonian.npy`, `gamma_L.npy`, `gamma_R.npy`, `occupancies.npy`, `U_ppp.txt`, `energies.npy`

- Method folder (`METHOD`): `output/ed/`, `output/dft/`, or `output/dmft/`
  - `self_energy.npy` (only for `METHOD="ed"`, or provided by you for `METHOD="dmft"`)
  - `G_retarded.npz`
  - `summary.out`

- Vertex-scheme subfolder (`VERTEX_SCHEME`) under the method folder, e.g. `output/ed/none/`:
  - `ET.npy`
  - `vertex_correction.npz` (not written for `VERTEX_SCHEME="none"`)
  - `summary.out`

`ET.npy` format depends on the run mode:

- If `METHOD="dft"` (correlated self-energy disabled) OR `VERTEX_SCHEME="none"`, the inelastic contribution is identically 0 and the workflow saves **only the total**:
  - 2 rows: `E`, `T_total`

- Otherwise it saves the full decomposition:
  - 4 rows: `E`, `T_elastic`, `T_inelastic`, `T_total = T_elastic + T_inelastic`

- `E` (energies)
- `T_elastic`
- `T_inelastic`
- `T_total = T_elastic + T_inelastic`

Load example:

```python
arr = np.load("output/ed/none/ET.npy")
if arr.shape[0] == 2:
	E, Ttot = arr
else:
	E, Tel, Tin, Ttot = arr
```

For `VERTEX_SCHEME != "none"`, the vertex-correction matrix is saved as a compressed NumPy archive:

```python
import numpy as np

data = np.load("output/ed/none/vertex_correction.npz")
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
