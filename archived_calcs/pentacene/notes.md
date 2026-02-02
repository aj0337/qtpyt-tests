- Refer to `get_los_prerequisites.py`

  - Note that in this case, the flipping of the sign to create a positive projection onto the p-z AOs is the difference between getting the right and wrong transmission function. This needs to be examined as the reason and rationale for this is unclear to me.

- The current implementation of the code doesn't seem to allow for the leads to be cut based on the active region. The way `ProjectedGreenFunction` is set up assumes that the entries of the index of the active region fall within the same block of the Hamiltonian.

  - If we want a speed-up by cutting down the size of the leads block of the Hamiltonian, we need to implement the `ProjectedGreenFunction` in such a way that it can handle the leads being cut based on the active region.

- Refer to `get_gf_prerequisites.py`

  - This script saves the leads self-energy terms to a file for later use. There are two methods to do this:
    - `LeadSelfEnergy`: Works for 1D leads without transverse periodicity, or when the transverse problem has already been reduced to real space.
    - `PrincipalSelfEnergy`: Use when your lead is periodic in directions perpendicular to transport and you want to treat that periodicity explicitly via k-point sampling.

- Normalization of the transmission function
  - The transmission function is normalized in some cases, but not in others. ~~My suspicion is that the normalization occurs when the leads self-energies are computed using the `PrincipalSelfEnergy` method, but not when using the `LeadSelfEnergy`method.~~ The above suspicion was tested and found to be false. It wasn't normalized in either cases. This raises the question why it is normalized automatically in the case of the pentadienyl and benzyl radical on Au leads. I have included a function that normalizes the transmission function using the number of open channels. There is a possibility that the number of open channels in the pentadienyl and benzyl case was 1 across all energies. (Something that needs to be verified)

- DOS plots suggest that one of the peaks around the Fermi energy can't be described by the C 2 pz of the the bridge molecule. This peak seems to have largest contributions from the C 2 pz of the leads.
