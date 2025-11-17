"""
Basic usage examples for NANO-OS Python SDK.

Session 28: Python SDK and Workflow DSL
"""

from nano_os import NanoOSClient


def main():
    # Initialize client
    client = NanoOSClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    )

    # Example 1: Create a structure
    print("Example 1: Creating a structure...")
    structure = client.structures.create(
        composition="MoS2",
        lattice_type="hexagonal",
        dimensionality=2,
        metadata={"source": "manual_entry", "tags": ["2D", "TMD"]}
    )
    print(f"Created structure: {structure.id}")
    print(f"  Composition: {structure.composition}")
    print(f"  Dimensionality: {structure.dimensionality}")

    # Example 2: Submit ML prediction job
    print("\nExample 2: Submitting ML prediction...")
    ml_job = client.jobs.submit_ml_prediction(
        structure_id=structure.id,
        model_name="cgcnn_bandgap_v1"
    )
    print(f"Submitted ML prediction job: {ml_job.id}")

    # Check job status
    status = client.jobs.get_status(ml_job.id)
    print(f"  Job status: {status.status}")

    # Example 3: Submit DFT job
    print("\nExample 3: Submitting DFT calculation...")
    dft_job = client.jobs.submit_dft(
        structure_id=structure.id,
        functional="PBE",
        kpoints_density=0.03,
        is_relaxation=True,
        priority=7
    )
    print(f"Submitted DFT job: {dft_job.id}")

    # Example 4: List structures
    print("\nExample 4: Listing structures...")
    structures = client.structures.list(page=1, page_size=5)
    print(f"Found {len(structures)} structures:")
    for s in structures[:3]:
        print(f"  - {s.composition} ({s.id})")

    # Example 5: Create a design campaign
    print("\nExample 5: Creating a design campaign...")
    campaign = client.campaigns.create(
        name="TMD Bandgap Optimization",
        description="Find 2D TMDs with 2eV bandgap",
        config={
            "target_properties": {
                "bandgap": {
                    "value": 2.0,
                    "tolerance": 0.2,
                    "weight": 1.0
                },
                "formation_energy": {
                    "max": -3.0,
                    "weight": 0.5
                }
            },
            "constraints": {
                "elements": ["Mo", "W", "S", "Se"],
                "dimensionality": 2,
                "max_atoms": 20
            },
            "max_iterations": 20,
            "candidates_per_iteration": 10,
            "generation_strategy": "bayesian"
        }
    )
    print(f"Created campaign: {campaign.id}")
    print(f"  Name: {campaign.name}")
    print(f"  Max iterations: {campaign.max_iterations}")

    # Run campaign iterations
    print("\nRunning 3 campaign iterations...")
    result = client.campaigns.run_iterations(
        campaign_id=campaign.id,
        num_iterations=3
    )
    print(f"Completed iterations. Current iteration: {result['campaign']['current_iteration']}")
    print(f"Best score: {result['campaign']['best_score']}")

    # Get campaign summary
    summary = client.campaigns.get_summary(campaign.id)
    print(f"\nCampaign summary:")
    print(f"  Total iterations: {summary.total_iterations}")
    print(f"  Structures created: {summary.total_structures_created}")
    print(f"  Best score: {summary.best_score_overall}")

    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    main()
