fd width = 1e-2 for scattering region and leads
beta = 38.68 (T = 300 K)
eta = 1e-3
Need to set gfp.eta = 0.0 when computing matsubara hybridization.
When integrating for occupancy goal, mu needs to be set to 0.0 eV
Use matsubara integration when computing occupancy goal to keep it consistent with the integration used in dmft runs.
