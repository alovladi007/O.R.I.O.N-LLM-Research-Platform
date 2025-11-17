"""
Example of running workflows using the NANO-OS SDK.

Session 28: Python SDK and Workflow DSL
"""

from nano_os import NanoOSClient
from nano_os.workflow import WorkflowRunner, WorkflowSpec
import sys
from pathlib import Path


def run_workflow_from_file(workflow_path: str, api_key: str):
    """Run a workflow from a YAML/JSON file."""
    # Initialize client
    client = NanoOSClient(
        base_url="http://localhost:8000",
        api_key=api_key
    )

    # Create workflow runner
    runner = WorkflowRunner(client)

    # Execute workflow
    print(f"Loading workflow from: {workflow_path}")
    result = runner.run_workflow_file(workflow_path)

    print("\n" + "="*60)
    print("Workflow Execution Results")
    print("="*60)
    print(f"Workflow: {result['workflow_name']}")
    print(f"Status: {result['status']}")
    print(f"\nStep Results:")
    for step_name, step_result in result['step_results'].items():
        print(f"  {step_name}:")
        if isinstance(step_result, list):
            print(f"    - Generated {len(step_result)} items")
        elif isinstance(step_result, dict):
            for key, value in list(step_result.items())[:3]:
                print(f"    - {key}: {value}")
        else:
            print(f"    - {step_result}")
    print("="*60)

    return result


def run_inline_workflow(api_key: str):
    """Run a workflow defined inline in Python."""
    # Initialize client
    client = NanoOSClient(
        base_url="http://localhost:8000",
        api_key=api_key
    )

    # Define workflow programmatically
    workflow_spec = WorkflowSpec(
        name="Quick TMD Screening",
        version="1.0",
        description="Quick screening workflow for TMD materials",
        steps=[
            {
                "name": "generate_structures",
                "type": "structure_generation",
                "params": {
                    "elements": ["Mo", "S"],
                    "num_structures": 5,
                    "dimensionality": 2
                },
                "outputs": ["structures"]
            },
            {
                "name": "predict_properties",
                "type": "ml_prediction",
                "inputs": ["structures"],
                "params": {
                    "model": "cgcnn_bandgap_v1",
                    "properties": ["bandgap"]
                },
                "outputs": ["predictions"]
            }
        ]
    )

    # Execute workflow
    runner = WorkflowRunner(client)
    print("Running inline workflow...")
    result = runner.run_workflow(workflow_spec)

    print("\n" + "="*60)
    print("Inline Workflow Results")
    print("="*60)
    print(f"Workflow: {result['workflow_name']}")
    print(f"Status: {result['status']}")
    print(f"Variables: {list(result['results'].keys())}")
    print("="*60)

    return result


def main():
    # Get API key from environment or argument
    import os
    api_key = os.environ.get("NANO_OS_API_KEY", "your-api-key")

    # Example 1: Run workflow from YAML file
    print("Example 1: Running workflow from YAML file")
    print("-" * 60)
    workflow_file = Path(__file__).parent / "tmd_screening_workflow.yaml"
    if workflow_file.exists():
        try:
            result1 = run_workflow_from_file(str(workflow_file), api_key)
        except Exception as e:
            print(f"Error running workflow: {e}")
    else:
        print(f"Workflow file not found: {workflow_file}")

    print("\n\n")

    # Example 2: Run inline workflow
    print("Example 2: Running inline workflow")
    print("-" * 60)
    try:
        result2 = run_inline_workflow(api_key)
    except Exception as e:
        print(f"Error running inline workflow: {e}")

    print("\n✓ Workflow examples completed!")


if __name__ == "__main__":
    main()
