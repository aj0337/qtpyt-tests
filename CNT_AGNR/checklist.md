# CNT_AGNR Calculation Checklist

This file provides a checklist of essential steps to follow for the CNT_AGNR calculation.
Use this list to ensure all necessary tasks are completed in the correct order.

- [ ] Structure building
  - [ ] Ensure that the atoms in the device/scattering region are ordered as left lead, scattering region, and right lead. Within each region, atoms should be ordered from left to right along the transport direction.
  - [ ] Ensure that the atoms in the leads and bridge structure file are ordered in the same way as the device.
- [ ] DFT calculations

  - [ ] Perform DFT calculations on the device, leads and bridge structure to obtain the ground state electronic properties.
  - [ ] Check bandgap of the bridge structure

- [ ] Compare and contrast DOS or orbitals or energies with and without lowdin orthogonalization
  - [ ] Analyze if Lowdin orthogonalization changes the DOS or orbitals significantly.
