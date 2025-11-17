"""
Example of lab integration using NANO-OS SDK.

Demonstrates instrument registration and experiment submission.

Session 28: Python SDK and Workflow DSL
"""

from nano_os import NanoOSClient
import time


def main():
    # Initialize client
    client = NanoOSClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    )

    print("NANO-OS Lab Integration Example")
    print("=" * 60)

    # Example 1: Register a mock CVD instrument
    print("\nExample 1: Registering lab instrument...")
    instrument = client.instruments.register(
        name="CVD Reactor 1",
        adapter_type="MOCK",
        connection_info={
            "instrument_type": "CVD",
            "simulation_delay": 2.0,
            "error_rate": 0.0
        },
        capabilities=["synthesis", "thin_film_deposition"],
        metadata={"location": "Lab A", "room": "101"}
    )
    print(f"✓ Registered instrument: {instrument.id}")
    print(f"  Name: {instrument.name}")
    print(f"  Type: {instrument.adapter_type}")
    print(f"  Capabilities: {instrument.capabilities}")

    # Example 2: Create a structure for synthesis
    print("\nExample 2: Creating target structure...")
    structure = client.structures.create(
        composition="MoS2",
        lattice_type="hexagonal",
        dimensionality=2,
        metadata={"synthesis_target": True}
    )
    print(f"✓ Created structure: {structure.id}")

    # Example 3: Submit synthesis experiment
    print("\nExample 3: Submitting synthesis experiment...")
    synthesis_exp = client.experiments.submit(
        instrument_id=instrument.id,
        type="synthesis",
        parameters={
            "temperature": 800,
            "duration": 3600,
            "precursors": ["Mo(CO)6", "S powder"],
            "carrier_gas": "Ar",
            "pressure": 1.0
        },
        linked_structure_id=structure.id,
        metadata={"researcher": "Dr. Smith"}
    )
    print(f"✓ Submitted synthesis experiment: {synthesis_exp.id}")
    print(f"  Type: {synthesis_exp.type}")
    print(f"  Status: {synthesis_exp.status}")
    print(f"  Linked structure: {synthesis_exp.linked_structure_id}")

    # Example 4: Monitor experiment status
    print("\nExample 4: Monitoring experiment status...")
    for i in range(3):
        time.sleep(1)
        status = client.experiments.get_status(synthesis_exp.id)
        print(f"  [{i+1}] Status: {status.get('status', 'UNKNOWN')}")

    # Example 5: Register a characterization instrument
    print("\nExample 5: Registering XRD instrument...")
    xrd_instrument = client.instruments.register(
        name="XRD Diffractometer",
        adapter_type="MOCK",
        connection_info={
            "instrument_type": "XRD",
            "simulation_delay": 1.5
        },
        capabilities=["characterization", "phase_analysis"],
        metadata={"location": "Lab B", "model": "Rigaku SmartLab"}
    )
    print(f"✓ Registered XRD instrument: {xrd_instrument.id}")

    # Example 6: Submit characterization experiment
    print("\nExample 6: Submitting XRD characterization...")
    xrd_exp = client.experiments.submit(
        instrument_id=xrd_instrument.id,
        type="characterization",
        parameters={
            "method": "XRD",
            "scan_range": [10, 80],
            "scan_speed": 2.0,
            "wavelength": 1.5406  # Cu K-alpha
        },
        linked_structure_id=structure.id,
        metadata={"sample_id": "MoS2-001"}
    )
    print(f"✓ Submitted XRD experiment: {xrd_exp.id}")

    # Example 7: List all experiments
    print("\nExample 7: Listing all experiments...")
    experiments = client.experiments.list(page=1, page_size=10)
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.type} on instrument {exp.instrument_id} ({exp.status})")

    # Example 8: List all instruments
    print("\nExample 8: Listing all instruments...")
    instruments = client.instruments.list()
    print(f"Found {len(instruments)} instruments:")
    for inst in instruments:
        print(f"  - {inst.name} ({inst.adapter_type}) - {inst.status}")

    print("\n" + "=" * 60)
    print("✓ Lab integration examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
