from gpaw import restart, FermiDirac, Mixer
import os

output_folder = "./"
atoms, calc = restart(f"{output_folder}/scatt_restart2.gpw", txt=f"{output_folder}/scatt_restart3.txt")

calc.set(
    occupations=FermiDirac(width=0.1),
    mixer=Mixer(0.1, 5, weight=100.0),
    maxiter=50,
    convergence={
        "energy": 1e-4,
        "density": 1e-4,
        "eigenstates": 1e-8,
    },
    symmetry={"point_group": False, "time_reversal": True},
)

atoms.calc = calc

try:
    atoms.get_potential_energy()
except Exception as e:
    print("WARNING: Refined calculation did not converge.")
    print(f"Error: {e}")
finally:
    calc.write(f"{output_folder}/scatt_restart3.gpw", mode="all")
    try:
        fermi = calc.get_fermi_level()
        with open(f"{output_folder}/fermi_restart3.txt", "w") as f:
            print(repr(fermi), file=f)
    except Exception:
        print("Could not determine Fermi level after refinement.")
