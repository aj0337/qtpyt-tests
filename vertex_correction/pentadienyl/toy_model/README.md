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

## Dependencies

- `numpy`
- `ase` (and typically `scipy` due to ASE imports)
- `edpyt` (for ED self-energy)

Example install:

```bash
pip install numpy ase scipy edpyt
```
