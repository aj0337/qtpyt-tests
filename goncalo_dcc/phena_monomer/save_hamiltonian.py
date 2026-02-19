import os
import numpy as np

filename = "output/wan_hr.dat"

rows = []

with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # We only care about lines with at least 7 columns
        if len(parts) < 7:
            continue

        try:
            # Try casting to float to ensure it's a data line
            float(parts[5])
            float(parts[6])
        except ValueError:
            continue

        rows.append(parts)

# Convert to numpy array for convenience
rows = np.array(rows)

# Extract indices and Hamiltonian elements
# Convert from 1-based indexing to 0-based indexing
i = rows[:, 3].astype(int) - 1
j = rows[:, 4].astype(int) - 1
re = rows[:, 5].astype(float)

# Determine matrix size
n = max(i.max(), j.max()) + 1

# Initialize Hamiltonian
H = np.zeros((n, n), dtype=np.float64)

# Fill Hamiltonian
H[i, j] = re

# Save Hamiltonian to a file
os.makedirs("output", exist_ok=True)
np.save("output/hamiltonian.npy", H)
print("Hamiltonian saved to output/hamiltonian.npy")
