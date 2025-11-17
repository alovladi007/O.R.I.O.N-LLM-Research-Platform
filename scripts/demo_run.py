#!/usr/bin/env python3
"""
NANO-OS Demonstration Script
=============================

This script demonstrates the full NANO-OS workflow programmatically:
1. Create a material and structure
2. Launch a mock simulation job
3. Get simulation results
4. Run ML property predictions
5. Start a design campaign and run one iteration

This shows how external AI agents or automation systems can interact with NANO-OS.

Usage:
    python scripts/demo_run.py
    python scripts/demo_run.py --skip-campaign  # Skip design campaign (faster)
    python scripts/demo_run.py --use-dft  # Use real DFT instead of mock (slower)

Requirements:
    - NANO-OS API server running on localhost:8000
    - Database initialized and seeded
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import httpx
import json
from datetime import datetime

# ANSI colors for output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


class NANOOSDemo:
    """Demonstration of NANO-OS programmatic API usage."""

    def __init__(self, base_url: str = "http://localhost:8000", use_dft: bool = False):
        self.base_url = base_url
        self.use_dft = use_dft
        self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
        self.demo_user_id = None

    def print_section(self, title: str):
        """Print a section header."""
        print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
        print(f"{BOLD}{BLUE}{title}{RESET}")
        print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")

    def print_success(self, message: str):
        """Print success message."""
        print(f"{GREEN}✓ {message}{RESET}")

    def print_info(self, message: str):
        """Print info message."""
        print(f"{BLUE}ℹ {message}{RESET}")

    def print_warning(self, message: str):
        """Print warning message."""
        print(f"{YELLOW}⚠ {message}{RESET}")

    def print_error(self, message: str):
        """Print error message."""
        print(f"{RED}✗ {message}{RESET}")

    async def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            health = response.json()
            self.print_success(f"API is healthy: {health.get('status')}")
            return True
        except Exception as e:
            self.print_error(f"API health check failed: {e}")
            return False

    async def create_material(self) -> Dict[str, Any]:
        """Create a new material."""
        self.print_section("Step 1: Creating a New Material")

        material_data = {
            "name": "Demo Material - Boron Nitride",
            "formula": "BN",
            "description": "Hexagonal boron nitride (h-BN) - wide bandgap 2D insulator",
            "tags": ["2D", "insulator", "high_thermal_conductivity"],
            "metadata": {
                "dimensionality": 2,
                "crystal_system": "hexagonal",
                "applications": ["thermal_management", "dielectric", "substrate"],
                "demo": True
            }
        }

        self.print_info(f"Creating material: {material_data['name']}")
        response = await self.client.post("/api/materials", json=material_data)
        response.raise_for_status()

        material = response.json()
        self.print_success(f"Material created with ID: {material['id']}")
        print(f"  Name: {material['name']}")
        print(f"  Formula: {material['formula']}")

        return material

    async def create_structure(self, material_id: str) -> Dict[str, Any]:
        """Create a structure for the material."""
        self.print_section("Step 2: Creating a Structure")

        # BN monolayer CIF
        cif_content = """data_bn_monolayer
_symmetry_space_group_name_H-M    'P -6 m 2'
_cell_length_a                    2.50
_cell_length_b                    2.50
_cell_length_c                    20.0
_cell_angle_alpha                 90.0
_cell_angle_beta                  90.0
_cell_angle_gamma                 120.0

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
B1 B 0.3333 0.6667 0.0000
N1 N 0.6667 0.3333 0.0000
"""

        structure_data = {
            "material_id": material_id,
            "name": "h-BN monolayer",
            "file_content": cif_content,
            "format": "CIF"
        }

        self.print_info("Uploading structure from CIF file...")
        response = await self.client.post("/api/structures", json=structure_data)
        response.raise_for_status()

        structure = response.json()
        self.print_success(f"Structure created with ID: {structure['id']}")
        print(f"  Name: {structure['name']}")
        print(f"  Formula: {structure.get('formula', 'N/A')}")
        print(f"  Atoms: {structure.get('num_atoms', 'N/A')}")
        print(f"  Dimensionality: {structure.get('dimensionality', 'N/A')}")

        return structure

    async def launch_simulation(self, structure_id: str) -> Dict[str, Any]:
        """Launch a simulation job."""
        self.print_section("Step 3: Launching a Simulation Job")

        # Choose workflow based on options
        if self.use_dft:
            workflow_name = "DFT_SCF_QE"
            self.print_warning("Using DFT workflow (this may take several minutes)")
        else:
            workflow_name = "MOCK_SIMULATION"
            self.print_info("Using mock workflow for quick demonstration")

        # Get workflow template
        response = await self.client.get("/api/workflows/templates")
        response.raise_for_status()
        templates = response.json()

        workflow_template = next(
            (t for t in templates if t["name"] == workflow_name),
            None
        )

        if not workflow_template:
            self.print_error(f"Workflow template '{workflow_name}' not found")
            return None

        self.print_info(f"Using workflow: {workflow_template['display_name']}")

        # Create job
        job_data = {
            "structure_id": structure_id,
            "workflow_template_id": workflow_template["id"],
            "name": "Demo simulation job",
            "parameters": workflow_template.get("default_parameters", {}),
            "priority": 5
        }

        response = await self.client.post("/api/jobs", json=job_data)
        response.raise_for_status()

        job = response.json()
        self.print_success(f"Job created with ID: {job['id']}")
        print(f"  Status: {job['status']}")

        return job

    async def wait_for_job(self, job_id: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for job to complete."""
        self.print_info("Waiting for job to complete...")

        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = await self.client.get(f"/api/jobs/{job_id}/status")
            response.raise_for_status()
            status_data = response.json()

            status = status_data.get("status")
            print(f"  Status: {status}", end="\r")

            if status == "COMPLETED":
                self.print_success("Job completed successfully!")
                break
            elif status == "FAILED":
                self.print_error("Job failed!")
                return None
            elif status in ["CANCELLED", "TIMEOUT"]:
                self.print_error(f"Job {status.lower()}!")
                return None

            await asyncio.sleep(2)  # Poll every 2 seconds

        else:
            self.print_error(f"Job did not complete within {timeout} seconds")
            return None

        # Get full job details
        response = await self.client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    async def get_simulation_results(self, job_id: str) -> Dict[str, Any]:
        """Get simulation results."""
        self.print_section("Step 4: Retrieving Simulation Results")

        response = await self.client.get(f"/api/jobs/{job_id}/result")

        if response.status_code == 404:
            self.print_warning("No results available yet")
            return None

        response.raise_for_status()
        result = response.json()

        self.print_success("Simulation results retrieved")
        print(f"  Success: {result.get('success')}")
        print(f"  Engine: {result.get('engine_name')}")

        summary = result.get("summary", {})
        if summary:
            print("\n  Results:")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")

        return result

    async def run_ml_prediction(self, structure_id: str) -> Dict[str, Any]:
        """Run ML property predictions."""
        self.print_section("Step 5: Running ML Property Predictions")

        prediction_data = {
            "structure_id": structure_id,
            "model_name": "STUB",
            "use_gnn": False
        }

        self.print_info("Requesting ML predictions...")
        response = await self.client.post("/api/ml/properties", json=prediction_data)
        response.raise_for_status()

        predictions = response.json()
        self.print_success("ML predictions completed")
        print(f"  Model: {predictions.get('model_name')} v{predictions.get('model_version')}")
        print(f"\n  Predicted Properties:")
        print(f"    Bandgap: {predictions.get('bandgap')} eV")
        print(f"    Formation Energy: {predictions.get('formation_energy')} eV/atom")
        print(f"    Stability Score: {predictions.get('stability_score')}")

        confidence = predictions.get("confidence", {})
        if confidence:
            print(f"\n  Confidence:")
            print(f"    Bandgap: {confidence.get('bandgap')}")
            print(f"    Formation Energy: {confidence.get('formation_energy')}")

        return predictions

    async def start_design_campaign(self) -> Dict[str, Any]:
        """Start a design campaign."""
        self.print_section("Step 6: Starting a Design Campaign")

        campaign_data = {
            "name": "Demo Campaign - Wide Bandgap 2D Materials",
            "description": "Search for 2D materials with bandgap > 3 eV",
            "search_space": {
                "allowed_elements": ["B", "N", "Al", "Ga"],
                "dimensionality": 2,
                "max_atoms_per_cell": 10
            },
            "objective": "maximize_bandgap",
            "config": {
                "algorithm": "random",
                "batch_size": 10,
                "evaluation_method": "ml_only",
                "ml_model": "STUB"
            },
            "max_iterations": 3
        }

        self.print_info("Creating design campaign...")
        response = await self.client.post("/api/campaigns", json=campaign_data)
        response.raise_for_status()

        campaign = response.json()
        self.print_success(f"Campaign created with ID: {campaign['id']}")
        print(f"  Name: {campaign['name']}")
        print(f"  Status: {campaign['status']}")
        print(f"  Max iterations: {campaign.get('max_iterations')}")

        return campaign

    async def check_campaign_progress(self, campaign_id: str) -> Dict[str, Any]:
        """Check campaign progress."""
        self.print_info("Checking campaign progress...")

        # Wait a bit for first iteration
        await asyncio.sleep(5)

        response = await self.client.get(f"/api/campaigns/{campaign_id}")
        response.raise_for_status()

        campaign = response.json()
        print(f"\n  Iterations completed: {campaign.get('iterations_completed', 0)}")
        print(f"  Status: {campaign['status']}")

        best_candidates = campaign.get("best_candidates", [])
        if best_candidates:
            print(f"\n  Best candidate so far:")
            best = best_candidates[0]
            print(f"    Formula: {best.get('formula', 'N/A')}")
            print(f"    Bandgap: {best.get('bandgap', 'N/A')} eV")
            print(f"    Stability: {best.get('stability_score', 'N/A')}")

        return campaign

    async def run_demo(self, skip_campaign: bool = False):
        """Run the full demonstration."""
        print(f"{BOLD}NANO-OS Demonstration{RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}\n")

        try:
            # Check API health
            if not await self.check_health():
                self.print_error("API is not available. Please start the server first.")
                return

            # Step 1: Create material
            material = await self.create_material()

            # Step 2: Create structure
            structure = await self.create_structure(material["id"])

            # Step 3: Launch simulation
            job = await self.launch_simulation(structure["id"])
            if not job:
                return

            # Step 4: Wait for job and get results
            completed_job = await self.wait_for_job(job["id"])
            if completed_job:
                await self.get_simulation_results(job["id"])

            # Step 5: Run ML predictions
            await self.run_ml_prediction(structure["id"])

            # Step 6: Design campaign (optional)
            if not skip_campaign:
                campaign = await self.start_design_campaign()
                await self.check_campaign_progress(campaign["id"])
            else:
                self.print_info("Skipping design campaign (use without --skip-campaign to see)")

            # Summary
            self.print_section("Demonstration Complete!")
            self.print_success("All steps completed successfully")
            print(f"\n{BOLD}What you can do next:{RESET}")
            print("  • Visit http://localhost:3000 to see the frontend")
            print("  • Check API docs at http://localhost:8000/docs")
            print("  • Explore the database with: make shell-db")
            print("  • View all documentation in /docs folder")

        except httpx.HTTPStatusError as e:
            self.print_error(f"HTTP error: {e.response.status_code}")
            print(f"  Response: {e.response.text}")
        except Exception as e:
            self.print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate NANO-OS programmatic workflow"
    )
    parser.add_argument(
        "--skip-campaign",
        action="store_true",
        help="Skip design campaign demonstration (faster)"
    )
    parser.add_argument(
        "--use-dft",
        action="store_true",
        help="Use real DFT instead of mock (slower, requires QE)"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="NANO-OS API base URL (default: http://localhost:8000)"
    )

    args = parser.parse_args()

    demo = NANOOSDemo(base_url=args.api_url, use_dft=args.use_dft)
    await demo.run_demo(skip_campaign=args.skip_campaign)


if __name__ == "__main__":
    asyncio.run(main())
