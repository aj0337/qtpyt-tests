fd width = 1e-2 for scattering region and leads
beta = 38.68 (T = 300 K)
The aim is to check if we are able to obtain convergence with this temperature and still obtain Fano peaks for spin resolved and unresolved dmft calculations since the spin resolved calculations didn't converge to an error tolerance of 1e-4 when beta was 100.
eta = 1e-3
Need to set gfp.eta = 0.0 when computing matsubara hybridization.
When integrating for occupancy goal, mu needs to be set to 0.0 eV
Use matsubara integration when computing occupancy goal to keep it consistent with the integration used in dmft runs.
