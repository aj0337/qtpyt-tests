"""
This script sorts an extended XYZ file for a quantum transport device.
The sorting follows the device geometry: left lead → bridge (molecule) → right lead.
Within each block, atoms are sorted by increasing x, y, and z coordinates.

### Assumptions:
1. The left and right leads occupy roughly 25% each of the total x-span.
2. The bridge (molecule) occupies the middle 50% of the x-span.
3. The device geometry is approximately symmetric along the x-axis.
4. Leads and the bridge are separated based on x-coordinates, not atom type.

The output is written to a new extended XYZ file with consistent column alignment.
"""

# Load the original extended xyz file
file_path = "./unrelaxed/scatt_original.xyz"
with open(file_path, "r") as f:
    data = f.readlines()

# First two lines contain the number of atoms and a comment line
num_atoms = int(data[0].strip())
comment = data[1].strip()

# Read the atomic data
atoms = []
for line in data[2 : num_atoms + 2]:
    parts = line.split()
    element = parts[0]
    x, y, z = map(float, parts[1:4])
    atoms.append((element, x, y, z))

# Determine boundaries for left lead, bridge, and right lead based on x-position
x_positions = [atom[1] for atom in atoms]
x_min, x_max = min(x_positions), max(x_positions)

# Define thresholds based on geometry
left_threshold = x_min + (x_max - x_min) * 0.25
right_threshold = x_max - (x_max - x_min) * 0.25


# Define sorting keys based on position in the device
def sorting_key(atom):
    element, x, y, z = atom
    if x < left_threshold:  # Left lead
        block = 0
    elif x > right_threshold:  # Right lead
        block = 2
    else:  # Bridge (molecule)
        block = 1
    return (block, x, y, z)


# Sort the atoms based on the defined sorting order
sorted_atoms = sorted(atoms, key=sorting_key)

# Write the sorted structure to a new extended xyz file with better alignment
output_path_aligned = "./unrelaxed/scatt.xyz"
with open(output_path_aligned, "w") as f:
    f.write(f"{num_atoms}\n{comment}\n")
    for atom in sorted_atoms:
        element, x, y, z = atom
        f.write(f"{element:<3} {x:15.8f} {y:15.8f} {z:15.8f}\n")
