# Pentacene on pentacene Calculation Checklist

This file provides a checklist of essential steps to follow for the CNT_AGNR calculation.
Use this list to ensure all necessary tasks are completed in the correct order.

- Step 1: Structure Preparation

  - Refer to `sort_positions.ipynb` for the ordering process
  - [ ] Ensure that the atoms in the device/scattering region are ordered as left lead, scattering region, and right lead. Within each region, atoms should be ordered from left to right along the transport direction.
  - [ ] Ensure that the atoms in the leads and bridge structure file are ordered in the same way as the device.

- Step 2: DFT calculations

  - Refer to the python scripts in the `dft` folder
  - [ ] Perform DFT calculations on the device, leads and bridge structure to obtain the ground state electronic properties.

- Step 3:

  - Refer to `get_los_prerequisites.py`
  - [ ] Determine the active space
  - [ ] Get subdiagonalized Hamiltonian and overlap matrices (transform to localized orbital basis set)
  - [ ] If performing DMFT or ED calculations, ensure that the Hamiltonian and overlap matrices are Lowdin orthogonalized.

- Step 4: Nodes and Regions

  - Refer to `compute_tridiagonal_nodes.ipynb`
  - [ ] Compute the nodes of the tridiagonal matrix
  - [ ] Identify the regions in the subdiagonalized Hamiltonian matrix that correspond to the left lead, bridge, and right lead.
  <!-- - [ ] If the leads are reduced in dimensionality due to picking of the active space, the computed leads self-energy needs to be expanded to the original leads dimensions. -->

- Step 5: Green's Function Prerequisites

  - Refer to `get_gf_prerequisites.py`
  - [ ] Identify if leads self-energies need to be computed using `LeadSelfEnergy` or `PrincipalSelfEnergy` classes.
  - [ ] Compute and save the self-energies for the left and right leads.
  - [ ] Save the Hamiltonian and overlap matrices in a tridiagonalized format suitable for Green's function calculations

- Step 6: DFT Transmission
  - Refer to `get_dft_transmission.py`

<!-- - [ ] Compare and contrast DOS or orbitals or energies with and without lowdin orthogonalization
  - [ ] Analyze if Lowdin orthogonalization changes the DOS or orbitals significantly. -->
