fd width = 1e-2 for scattering region and leads
eta = 1e-2
The above two values were what gave me an exact match of the DFT T(E) to the reference.

Need to set gfp.eta = 0.0 when computing matsubara hybridization.
When integrating for occupancy goal, mu needs to be set to 0.0 eV

If eta is 1e-2, should beta be set to 1/eta i.e., 100?
beta = 70
(It is possible Guido used beta = 70.)
