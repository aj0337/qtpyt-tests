- Refer to `get_los_prerequisites.py`

  - Note that in this case, the flipping of the sign to create a positive projection onto the p-z AOs is the difference between getting the right and wrong transmission function. This needs to be examined as the reason and rationale for this is unclear to me.

- The current implementation of the code doesn't seem to allow for the leads to be cut based on the active region. The way `ProjectedGreenFunction` is set up assumes that the entries of the index of the active region fall within the same block of the Hamiltonian.
  - If we want a speed-up by cutting down the size of the leads block of the Hamiltonian, we need to implement the `ProjectedGreenFunction` in such a way that it can handle the leads being cut based on the active region.
