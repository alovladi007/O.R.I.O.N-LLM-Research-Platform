# ORION Structure Parser Module

A comprehensive, production-ready module for parsing, validating, and converting crystallographic structure files.

## Features

- **Multiple Format Support**: CIF, POSCAR, and XYZ formats
- **Type-Safe**: Full Pydantic validation with type hints
- **Robust Error Handling**: Detailed error messages with line numbers and hints
- **Format Conversion**: Seamless conversion between all supported formats
- **Rich Metadata**: Automatic extraction of lattice parameters, space groups, and dimensionality
- **Clean Abstraction**: Pymatgen-powered with a simple, intuitive API

## Quick Start

```python
from backend.common.structures import parse_structure, to_cif, StructureFormat

# Parse a structure file
with open('structure.cif', 'r') as f:
    structure = parse_structure(f.read(), StructureFormat.CIF)

# Access properties
print(f"Formula: {structure.formula}")
print(f"Volume: {structure.volume:.2f} Å³")
print(f"Dimensionality: {structure.dimensionality}D")

# Convert to POSCAR format
poscar_output = to_poscar(structure)
```

## Installation

Requires pymatgen:

```bash
pip install pymatgen>=2023.5.0
```

## Module Structure

```
backend/common/structures/
├── __init__.py          # Public API exports
├── parsers.py           # Core parsing and conversion logic
├── README.md           # This file
└── EXAMPLES.md         # Comprehensive usage examples
```

## API Reference

### Models

#### `InternalStructureModel`

Normalized internal representation of crystal structures.

**Fields:**
- `lattice_vectors`: 3x3 matrix of lattice vectors (Å)
- `atomic_species`: List of element symbols
- `atomic_positions`: Nx3 fractional coordinates (0-1)
- `dimensionality`: 0=molecule, 1=chain, 2=layer, 3=bulk
- `formula`: Reduced chemical formula
- `a, b, c`: Lattice parameters (Å)
- `alpha, beta, gamma`: Lattice angles (degrees)
- `volume`: Unit cell volume (Å³)
- `num_atoms`: Total number of atoms
- `space_group`: Space group symbol (optional)
- `space_group_number`: Space group number (optional)

### Enums

#### `StructureFormat`

Supported file formats:
- `StructureFormat.CIF` - Crystallographic Information File
- `StructureFormat.POSCAR` - VASP structure format
- `StructureFormat.XYZ` - XYZ Cartesian coordinates

### Functions

#### `parse_structure(text: str, format: StructureFormat) -> InternalStructureModel`

Parse structure from text in specified format.

**Parameters:**
- `text`: Raw file content as string
- `format`: File format (CIF, POSCAR, or XYZ)

**Returns:** InternalStructureModel

**Raises:** ParsingError with detailed error information

---

#### `to_cif(structure: InternalStructureModel) -> str`

Export structure to CIF format.

**Parameters:**
- `structure`: InternalStructureModel to export

**Returns:** CIF format string

**Raises:** ParsingError if export fails

---

#### `to_poscar(structure: InternalStructureModel) -> str`

Export structure to POSCAR format.

**Parameters:**
- `structure`: InternalStructureModel to export

**Returns:** POSCAR format string

**Raises:** ParsingError if export fails

---

#### `to_xyz(structure: InternalStructureModel) -> str`

Export structure to XYZ format.

**Note:** XYZ format does not preserve lattice information.

**Parameters:**
- `structure`: InternalStructureModel to export

**Returns:** XYZ format string

**Raises:** ParsingError if export fails

---

#### `convert_structure(input_text: str, input_format: StructureFormat, output_format: StructureFormat) -> str`

Convert structure from one format to another (convenience function).

**Parameters:**
- `input_text`: Input structure content
- `input_format`: Input format
- `output_format`: Desired output format

**Returns:** Converted structure as string

**Raises:** ParsingError if conversion fails

## Error Handling

All parsing and export functions raise `ParsingError` (from `src.api.exceptions`) with:

- **message**: Human-readable error description
- **details**: Dictionary with additional context
  - `format`: The file format being processed
  - `error`: Original error message
  - `hint`: Suggestions for fixing the issue (when available)
  - `traceback`: Full traceback (for debugging)

### Example

```python
from backend.common.structures import parse_structure, StructureFormat
from src.api.exceptions import ParsingError

try:
    structure = parse_structure(content, StructureFormat.CIF)
except ParsingError as e:
    print(f"Error: {e.message}")
    print(f"Format: {e.details['format']}")
    if 'hint' in e.details:
        print(f"Suggestion: {e.details['hint']}")
```

## Validation

The `InternalStructureModel` performs comprehensive validation:

1. **Lattice vectors**: Must be 3x3 matrix of numbers
2. **Atomic positions**: Must be Nx3 matrix (fractional coordinates)
3. **Consistency**: Species and positions must have matching lengths
4. **Dimensionality**: Must be 0, 1, 2, or 3
5. **Lattice parameters**: Must be positive
6. **Angles**: Must be between 0 and 180 degrees

## Testing

Run tests with:

```bash
# Install test dependencies
pip install pytest

# Run tests (requires pymatgen)
pytest tests/test_structures.py --run-pymatgen

# Run without pymatgen (tests will be skipped)
pytest tests/test_structures.py
```

## Examples

See [EXAMPLES.md](EXAMPLES.md) for comprehensive usage examples including:

- Basic parsing and export
- Format conversion
- Property access
- Error handling
- Integration with FastAPI
- Batch processing
- Custom analysis

## Supported Formats

### CIF (Crystallographic Information File)

- Industry standard for crystallographic data
- Preserves all structure information including symmetry
- Can contain multiple structures (parser uses the first one)

### POSCAR (VASP)

- VASP input format
- Supports both VASP 4.x and 5.x formats
- Preserves lattice and atomic positions
- May not contain space group information

### XYZ

- Simple Cartesian coordinate format
- **Warning**: Does not contain lattice information
- Parser creates a cubic box with padding around molecule
- Best for molecular structures (0D)

## Dimensionality Detection

The module automatically determines structure dimensionality:

- **3D (Bulk)**: All lattice parameters < 15 Å (no vacuum)
- **2D (Layer)**: One direction has > 15 Å (vacuum layer)
- **1D (Chain)**: Two directions have > 15 Å (vacuum)
- **0D (Molecule)**: All directions have > 15 Å (vacuum)

## Performance Considerations

- Parsing is fast for small-medium structures (< 1000 atoms)
- Large structures (> 10,000 atoms) may take seconds
- CIF files can be slower due to symmetry analysis
- Space group determination is optional and may be skipped for complex structures

## Dependencies

- **pymatgen** (>= 2023.5.0): Core structure manipulation
- **pydantic** (>= 2.0): Data validation and serialization
- **numpy**: Array operations (via pymatgen)

## License

Part of the ORION LLM Research Platform.

## Contributing

When adding new features:

1. Maintain type hints for all functions
2. Add comprehensive docstrings with examples
3. Include error handling with meaningful messages
4. Add tests to `tests/test_structures.py`
5. Update EXAMPLES.md with usage examples

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'pymatgen'`

**Solution**: Install pymatgen: `pip install pymatgen`

### Parsing Errors

**Problem**: Structure file fails to parse

**Solutions**:
1. Check file format matches the specified StructureFormat
2. Verify file is not corrupted (valid UTF-8 text)
3. Check error details for specific hints
4. Compare with example files in EXAMPLES.md

### Space Group Issues

**Problem**: `structure.space_group` is None

**Explanation**: Space group determination can fail for:
- Low-symmetry structures
- Molecular systems (0D)
- Structures parsed from POSCAR (no symmetry info)
- Perturbed/distorted structures

**Solution**: This is expected behavior. Space group is optional metadata.

### Dimensionality Detection

**Problem**: Dimensionality detected incorrectly

**Explanation**: Uses heuristic based on lattice parameters (> 15 Å = vacuum)

**Solution**: For edge cases, manually override if needed:

```python
structure = parse_structure(content, format)
structure.dimensionality = 2  # Force 2D classification
```

## Future Enhancements

Potential improvements for future versions:

- [ ] Support for additional formats (PDB, LAMMPS data)
- [ ] Automatic format detection
- [ ] Structure validation against physical constraints
- [ ] Supercell generation
- [ ] Structure comparison and matching
- [ ] Symmetry standardization
- [ ] Defect analysis tools

## Contact

For issues, questions, or contributions, please refer to the main ORION platform documentation.
