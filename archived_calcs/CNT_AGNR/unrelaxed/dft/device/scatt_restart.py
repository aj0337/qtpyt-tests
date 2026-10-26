from gpaw import GPAW, restart, FermiDirac, Mixer
import os

# === Settings ===
output_folder = "./"
gpw_in = f"{output_folder}/scatt.gpw"
gpw_out = f"{output_folder}/scatt_restart.gpw"
txt_out = f"{output_folder}/scatt_restart.txt"
fermi_out = f"{output_folder}/fermi_restart.txt"
kbt = 8e-3          # Desired new Fermi-Dirac smearing
force_scf = True    # Always rerun SCF with updated smearing

# === Load previous calculation ===
atoms, calc_old = restart(gpw_in, txt=txt_out)

# Build a brand-new calculator with the updated kBT
calc = GPAW(
    mode=calc_old.parameters.mode,
    basis=calc_old.parameters.basis,
    kpts=calc_old.parameters.kpts,
    xc=calc_old.parameters.xc,
    mixer=Mixer(0.02, 5, weight=100.0),
    occupations=FermiDirac(width=kbt),
    maxiter=1000,
    convergence={
        "energy": 1e-4,
        "density": 1e-4,
        "eigenstates": 1e-8,
    },
    symmetry={"point_group": False, "time_reversal": True},
    txt=txt_out,
)

# Attach the new calculator
atoms.calc = calc

try:
    if force_scf:
        # Discard old wavefunction/occupation data completely
        calc.wfs = None
        calc.density = None
        calc.initialize(atoms)
    atoms.get_potential_energy()
except Exception as e:
    print("WARNING: Refined calculation did not converge.")
    print(f"Error: {e}")
finally:
    # Always save the updated .gpw file even if not converged
    calc.write(gpw_out, mode="all")
    # Try to get Fermi level; skip if unavailable
    try:
        fermi = calc.get_fermi_level()
        with open(fermi_out, "w") as f:
            print(repr(fermi), file=f)
        print(f"Updated Fermi level = {fermi:.6f} eV")
    except Exception:
        print("Could not determine Fermi level after refinement.")

print(f"Restart complete. Results saved to: {gpw_out}")
