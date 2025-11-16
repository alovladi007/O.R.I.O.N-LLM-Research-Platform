# Structure Parser Module - Examples and Usage Guide

This document provides comprehensive examples for using the ORION structure parser module.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Parsing Structures](#parsing-structures)
- [Exporting Structures](#exporting-structures)
- [Format Conversion](#format-conversion)
- [Accessing Structure Properties](#accessing-structure-properties)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)

## Installation

Ensure pymatgen is installed:

```bash
pip install pymatgen>=2023.5.0
```

## Basic Usage

```python
from backend.common.structures import parse_structure, StructureFormat

# Read a structure file
with open('structure.cif', 'r') as f:
    structure = parse_structure(f.read(), StructureFormat.CIF)

# Access properties
print(f"Formula: {structure.formula}")
print(f"Number of atoms: {structure.num_atoms}")
print(f"Volume: {structure.volume:.2f} Å³")
```

## Parsing Structures

### Parsing CIF Files

```python
from backend.common.structures import parse_structure, StructureFormat

cif_content = """
data_silicon
_cell_length_a    5.46873
_cell_length_b    5.46873
_cell_length_c    5.46873
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'F d -3 m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.00000 0.00000 0.00000
Si2 0.25000 0.25000 0.25000
"""

structure = parse_structure(cif_content, StructureFormat.CIF)

print(f"Space group: {structure.space_group}")
print(f"Space group number: {structure.space_group_number}")
print(f"Dimensionality: {structure.dimensionality}D")
```

### Parsing POSCAR Files

```python
from backend.common.structures import parse_structure, StructureFormat

poscar_content = """Si2
1.0
   5.46873000   0.00000000   0.00000000
   0.00000000   5.46873000   0.00000000
   0.00000000   0.00000000   5.46873000
Si
2
direct
   0.00000000   0.00000000   0.00000000
   0.25000000   0.25000000   0.25000000
"""

structure = parse_structure(poscar_content, StructureFormat.POSCAR)

print(f"Lattice parameters:")
print(f"  a = {structure.a:.3f} Å")
print(f"  b = {structure.b:.3f} Å")
print(f"  c = {structure.c:.3f} Å")
```

### Parsing XYZ Files

```python
from backend.common.structures import parse_structure, StructureFormat

xyz_content = """3
Water molecule
O   0.000000   0.000000   0.119262
H   0.000000   0.763239  -0.477047
H   0.000000  -0.763239  -0.477047
"""

structure = parse_structure(xyz_content, StructureFormat.XYZ)

print(f"Formula: {structure.formula}")
print(f"Dimensionality: {structure.dimensionality}D (0 = molecule)")
print(f"Atoms: {structure.num_atoms}")
```

## Exporting Structures

### Export to CIF

```python
from backend.common.structures import parse_structure, to_cif, StructureFormat

# Parse from POSCAR
structure = parse_structure(poscar_content, StructureFormat.POSCAR)

# Export to CIF
cif_output = to_cif(structure)

# Save to file
with open('output.cif', 'w') as f:
    f.write(cif_output)
```

### Export to POSCAR

```python
from backend.common.structures import parse_structure, to_poscar, StructureFormat

# Parse from CIF
structure = parse_structure(cif_content, StructureFormat.CIF)

# Export to POSCAR
poscar_output = to_poscar(structure)

# Save to file
with open('POSCAR', 'w') as f:
    f.write(poscar_output)
```

### Export to XYZ

```python
from backend.common.structures import parse_structure, to_xyz, StructureFormat

# Parse from any format
structure = parse_structure(cif_content, StructureFormat.CIF)

# Export to XYZ
xyz_output = to_xyz(structure)

# Save to file
with open('structure.xyz', 'w') as f:
    f.write(xyz_output)
```

## Format Conversion

### Direct Format Conversion

```python
from backend.common.structures.parsers import convert_structure, StructureFormat

# Read POSCAR file
with open('POSCAR', 'r') as f:
    poscar_content = f.read()

# Convert to CIF in one step
cif_output = convert_structure(
    poscar_content,
    StructureFormat.POSCAR,
    StructureFormat.CIF
)

# Save CIF file
with open('output.cif', 'w') as f:
    f.write(cif_output)
```

### Batch Conversion

```python
from pathlib import Path
from backend.common.structures import parse_structure, to_cif, StructureFormat

# Convert all POSCAR files in a directory to CIF
input_dir = Path('structures/poscar/')
output_dir = Path('structures/cif/')
output_dir.mkdir(exist_ok=True)

for poscar_file in input_dir.glob('*.vasp'):
    # Parse POSCAR
    with open(poscar_file, 'r') as f:
        structure = parse_structure(f.read(), StructureFormat.POSCAR)

    # Export to CIF
    cif_content = to_cif(structure)

    # Save with same base name
    output_file = output_dir / f"{poscar_file.stem}.cif"
    with open(output_file, 'w') as f:
        f.write(cif_content)

    print(f"Converted {poscar_file.name} -> {output_file.name}")
```

## Accessing Structure Properties

### Lattice Information

```python
from backend.common.structures import parse_structure, StructureFormat

structure = parse_structure(cif_content, StructureFormat.CIF)

# Lattice parameters
print(f"a = {structure.a:.3f} Å")
print(f"b = {structure.b:.3f} Å")
print(f"c = {structure.c:.3f} Å")
print(f"α = {structure.alpha:.2f}°")
print(f"β = {structure.beta:.2f}°")
print(f"γ = {structure.gamma:.2f}°")
print(f"Volume = {structure.volume:.2f} Å³")

# Lattice vectors (3x3 matrix)
print("\nLattice vectors:")
for i, vec in enumerate(structure.lattice_vectors):
    print(f"  v{i+1} = [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")
```

### Atomic Information

```python
from backend.common.structures import parse_structure, StructureFormat

structure = parse_structure(cif_content, StructureFormat.CIF)

# Iterate over atoms
print(f"Atoms in {structure.formula}:")
for i, (species, pos) in enumerate(zip(structure.atomic_species, structure.atomic_positions)):
    print(f"  {i+1}. {species:2s} at ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# Count atoms by species
from collections import Counter
species_count = Counter(structure.atomic_species)
print("\nComposition:")
for element, count in species_count.items():
    print(f"  {element}: {count}")
```

### Dimensionality Detection

```python
from backend.common.structures import parse_structure, StructureFormat

structure = parse_structure(cif_content, StructureFormat.CIF)

dimensionality_labels = {
    0: "Molecule/Cluster (0D)",
    1: "Chain/Wire (1D)",
    2: "Layer/Slab (2D)",
    3: "Bulk Crystal (3D)"
}

print(f"Structure type: {dimensionality_labels[structure.dimensionality]}")
```

## Error Handling

### Catching Parse Errors

```python
from backend.common.structures import parse_structure, StructureFormat
from src.api.exceptions import ParsingError

try:
    structure = parse_structure(malformed_content, StructureFormat.CIF)
except ParsingError as e:
    print(f"Error: {e.message}")
    print(f"Format: {e.details['format']}")
    if 'hint' in e.details:
        print(f"Hint: {e.details['hint']}")
```

### Validation Errors

```python
from backend.common.structures import InternalStructureModel
from pydantic import ValidationError

try:
    # This will fail validation
    structure = InternalStructureModel(
        lattice_vectors=[[5.0, 0], [0, 5.0]],  # Invalid: not 3x3
        atomic_species=["Si"],
        atomic_positions=[[0.0, 0.0, 0.0]],
        dimensionality=3,
        formula="Si",
        a=5.0, b=5.0, c=5.0,
        alpha=90.0, beta=90.0, gamma=90.0,
        volume=125.0,
        num_atoms=1
    )
except ValidationError as e:
    print("Validation errors:")
    for error in e.errors():
        print(f"  - {error['loc']}: {error['msg']}")
```

## Advanced Usage

### Custom Structure Analysis

```python
from backend.common.structures import parse_structure, StructureFormat
import numpy as np

structure = parse_structure(cif_content, StructureFormat.CIF)

# Calculate density
from collections import Counter
species_count = Counter(structure.atomic_species)

# Atomic masses (simplified)
atomic_masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'Si': 28.085}

total_mass = sum(atomic_masses.get(s, 0) * species_count[s] for s in species_count)
volume_cm3 = structure.volume * 1e-24  # Å³ to cm³
density = total_mass / (6.022e23 * volume_cm3)  # g/cm³

print(f"Density: {density:.3f} g/cm³")
```

### Working with Pydantic Models

```python
from backend.common.structures import parse_structure, InternalStructureModel, StructureFormat

# Parse structure
structure = parse_structure(cif_content, StructureFormat.CIF)

# Export to JSON
json_data = structure.model_dump_json(indent=2)
print(json_data)

# Import from JSON
structure_from_json = InternalStructureModel.model_validate_json(json_data)

# Export to dict for database storage
structure_dict = structure.model_dump()
```

### Integration with FastAPI

```python
from fastapi import FastAPI, HTTPException, UploadFile
from backend.common.structures import parse_structure, to_cif, StructureFormat
from src.api.exceptions import ParsingError

app = FastAPI()

@app.post("/api/structures/convert/{output_format}")
async def convert_structure_endpoint(
    output_format: StructureFormat,
    file: UploadFile
):
    """Convert uploaded structure file to specified format."""

    # Read file content
    content = await file.read()
    text = content.decode('utf-8')

    # Detect input format from filename
    if file.filename.endswith('.cif'):
        input_format = StructureFormat.CIF
    elif file.filename.endswith(('.POSCAR', '.vasp')):
        input_format = StructureFormat.POSCAR
    elif file.filename.endswith('.xyz'):
        input_format = StructureFormat.XYZ
    else:
        raise HTTPException(400, "Unknown file format")

    try:
        # Parse structure
        structure = parse_structure(text, input_format)

        # Convert to output format
        if output_format == StructureFormat.CIF:
            output = to_cif(structure)
        elif output_format == StructureFormat.POSCAR:
            output = to_poscar(structure)
        else:
            output = to_xyz(structure)

        return {
            "structure": structure.model_dump(),
            "output": output
        }

    except ParsingError as e:
        raise HTTPException(422, str(e))
```

### Modifying Structures

```python
from backend.common.structures import parse_structure, to_poscar, StructureFormat

# Parse original structure
structure = parse_structure(cif_content, StructureFormat.CIF)

# Create a supercell by modifying the model
supercell_structure = structure.model_copy()
supercell_structure.lattice_vectors = [
    [v[0] * 2, v[1], v[2]] for v in structure.lattice_vectors
]
supercell_structure.a *= 2
supercell_structure.volume *= 2

# Note: This is simplified - for real supercells, use pymatgen's make_supercell
```

## Best Practices

1. **Always validate input**: Use try-except blocks to catch ParsingError
2. **Check dimensionality**: Verify the structure type matches expectations
3. **Preserve metadata**: When converting, important metadata might be lost (especially XYZ)
4. **Handle space group**: Not all structures have space group information
5. **Fractional coordinates**: All positions are in fractional coordinates (0-1 range)

## Common Pitfalls

### XYZ Format Limitations

```python
# XYZ format loses lattice information!
structure = parse_structure(cif_content, StructureFormat.CIF)
original_volume = structure.volume

xyz_output = to_xyz(structure)
reparsed = parse_structure(xyz_output, StructureFormat.XYZ)

# Volume will be different (XYZ creates a box around atoms)
print(f"Original volume: {original_volume:.2f} Å³")
print(f"After XYZ conversion: {reparsed.volume:.2f} Å³")
```

### Space Group Information

```python
structure = parse_structure(some_content, StructureFormat.POSCAR)

# Space group might be None for POSCAR or low-symmetry structures
if structure.space_group:
    print(f"Space group: {structure.space_group}")
else:
    print("Space group could not be determined")
```

## Troubleshooting

### PyMatGen Not Installed

```python
# This will raise a clear error if pymatgen is not installed
try:
    structure = parse_structure(content, StructureFormat.CIF)
except ParsingError as e:
    if "pymatgen" in e.message:
        print("Please install pymatgen: pip install pymatgen")
```

### Invalid Coordinates

```python
# Fractional coordinates should be in [0, 1) range
# Some software outputs coordinates outside this range
structure = parse_structure(content, StructureFormat.CIF)

for i, pos in enumerate(structure.atomic_positions):
    if any(x < 0 or x >= 1 for x in pos):
        print(f"Warning: Atom {i} has coordinates outside [0, 1): {pos}")
```

## Further Reading

- [PyMatGen Documentation](https://pymatgen.org/)
- [CIF Format Specification](https://www.iucr.org/resources/cif)
- [VASP POSCAR Format](https://www.vasp.at/wiki/index.php/POSCAR)
- [XYZ Format](https://en.wikipedia.org/wiki/XYZ_file_format)
