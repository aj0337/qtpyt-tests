"""
This script sorts an extended XYZ file from a format where all atoms of the same type are generally
combined together to a sorting that follows the device geometry: left Au lead → molecule → right Au lead.
Within each block, atoms are sorted by increasing x, y, and z coordinates.
The output is written to a new extended XYZ file with consistent column alignment.
"""

# Load the original extended xyz file
file_path = "relaxed/scatt_original.xyz"
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


# Define sorting keys based on position in the device
# Assuming Au atoms define the leads and non-Au atoms define the molecule
def sorting_key(atom):
    element, x, y, z = atom
    if element == "Au" and x < -5:  # Left lead
        block = 0
    elif element == "Au" and x > 5:  # Right lead
        block = 2
    else:  # Molecule
        block = 1
    return (block, x, y, z)


# Sort the atoms based on the defined sorting order
sorted_atoms = sorted(atoms, key=sorting_key)

# Write the sorted structure to a new extended xyz file with better alignment
output_path_aligned = "/mnt/data/scatt.xyz"
with open(output_path_aligned, "w") as f:
    f.write(f"{num_atoms}\n{comment}\n")
    for atom in sorted_atoms:
        element, x, y, z = atom
        f.write(f"{element:<3} {x:15.8f} {y:15.8f} {z:15.8f}\n")
