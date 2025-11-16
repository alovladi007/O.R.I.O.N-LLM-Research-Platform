#!/usr/bin/env python3
"""
Quick test script to verify engine implementation.

This script tests:
1. Engine registry functionality
2. Mock engine execution
3. QE engine setup and mock mode
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_registry():
    """Test engine registry."""
    print("\n" + "="*60)
    print("TEST 1: Engine Registry")
    print("="*60)

    from backend.common.engines import list_engines, is_engine_available, get_engine

    # List engines
    engines = list_engines()
    print(f"✓ Available engines: {list(engines.keys())}")

    # Check specific engines
    assert is_engine_available("MOCK"), "MOCK engine should be available"
    assert is_engine_available("QE"), "QE engine should be available"
    assert not is_engine_available("VASP"), "VASP engine should not be available yet"
    print("✓ Engine availability checks passed")

    # Get engine classes
    mock_class = get_engine("MOCK")
    qe_class = get_engine("QE")
    print(f"✓ Got MOCK engine: {mock_class.__name__}")
    print(f"✓ Got QE engine: {qe_class.__name__}")

    # Test case-insensitive lookup
    assert get_engine("mock") == mock_class
    assert get_engine("quantum_espresso") == qe_class
    print("✓ Case-insensitive lookups work")

    print("\n✅ Registry tests PASSED\n")


def test_mock_engine():
    """Test mock engine."""
    print("\n" + "="*60)
    print("TEST 2: Mock Engine")
    print("="*60)

    from backend.common.engines import get_engine

    # Create mock engine
    engine_class = get_engine("MOCK")
    engine = engine_class()
    print(f"✓ Created mock engine: {engine}")

    # Setup
    structure = {
        "id": "test-123",
        "atoms": ["Mo", "Mo", "S", "S", "S", "S"],
        "positions": [
            [0.333, 0.667, 0.5],
            [0.667, 0.333, 0.5],
            [0.333, 0.667, 0.621],
            [0.667, 0.333, 0.621],
            [0.333, 0.667, 0.379],
            [0.667, 0.333, 0.379],
        ],
        "formula": "MoS2",
        "n_atoms": 6,
        "dimensionality": 2,
    }
    parameters = {
        "functional": "PBE",
        "k_points": [4, 4, 1],
        "ecutwfc": 50.0,
    }

    engine.setup(structure, parameters)
    print("✓ Engine setup completed")

    # Run (this will use asyncio internally)
    results = engine.run()
    print("✓ Engine run completed")

    # Check results
    assert "summary" in results
    assert "convergence_reached" in results
    assert "metadata" in results
    print(f"✓ Results structure valid")
    print(f"  - Energy: {results['summary']['total_energy']:.6f} eV")
    print(f"  - Convergence: {results['convergence_reached']}")
    print(f"  - Quality: {results['quality_score']}")

    # Cleanup
    engine.cleanup()
    print("✓ Engine cleanup completed")

    print("\n✅ Mock engine tests PASSED\n")


def test_qe_engine():
    """Test QE engine (in mock mode)."""
    print("\n" + "="*60)
    print("TEST 3: Quantum ESPRESSO Engine (Mock Mode)")
    print("="*60)

    import os
    from backend.common.engines import get_engine

    # Force mock mode
    os.environ["QE_MOCK_MODE"] = "true"

    # Create QE engine
    engine_class = get_engine("QE")
    engine = engine_class()
    print(f"✓ Created QE engine: {engine}")
    print(f"  - Mock mode: {engine.mock_mode}")

    # Setup
    structure = {
        "id": "test-qe",
        "atoms": ["Si", "Si"],
        "positions": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        "cell": [[0.0, 2.7, 2.7], [2.7, 0.0, 2.7], [2.7, 2.7, 0.0]],
        "formula": "Si2",
        "n_atoms": 2,
    }
    parameters = {
        "calculation": "relax",
        "ecutwfc": 30.0,
        "k_points": [4, 4, 4],
    }

    engine.setup(structure, parameters)
    print("✓ Engine setup completed")
    print(f"  - Input file: {engine.input_file}")
    print(f"  - Work dir: {engine.work_dir}")

    # Check input file was created
    assert engine.input_file is not None
    assert Path(engine.input_file).exists()
    print("✓ Input file created successfully")

    # Read and verify input file
    with open(engine.input_file, 'r') as f:
        input_content = f.read()
    assert "calculation = 'relax'" in input_content
    assert "ecutwfc = 30.0" in input_content
    assert "Si" in input_content
    print("✓ Input file content verified")

    # Run (in mock mode)
    results = engine.run()
    print("✓ Engine run completed (mock mode)")

    # Check results
    assert "summary" in results
    assert results["summary"]["engine"] == "QE"
    assert "total_energy" in results["summary"]
    print(f"✓ Results structure valid")
    print(f"  - Engine: {results['metadata']['engine']}")
    print(f"  - Energy: {results['summary']['total_energy']:.6f} eV")
    print(f"  - Convergence: {results['convergence_reached']}")
    print(f"  - Mock mode: {results['metadata']['mock_mode']}")

    # Cleanup
    engine.cleanup()
    print("✓ Engine cleanup completed")

    print("\n✅ QE engine tests PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("NANO-OS Engine System Tests")
    print("="*60)

    try:
        test_registry()
        test_mock_engine()
        test_qe_engine()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")
        return 0

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TESTS FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
