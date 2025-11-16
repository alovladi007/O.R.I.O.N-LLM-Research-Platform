#!/usr/bin/env python3
"""
Workflow Template Seeder
=========================

This script seeds the database with workflow templates for various simulation engines.

Session 5 templates:
- DFT relaxation with Quantum ESPRESSO
- DFT SCF with Quantum ESPRESSO
- DFT band structure with Quantum ESPRESSO

Usage:
    # Seed all templates
    python scripts/seed_workflows.py

    # Seed only QE templates
    python scripts/seed_workflows.py --engine QE

    # Dry run (don't commit to database)
    python scripts/seed_workflows.py --dry-run

    # Clear existing templates before seeding
    python scripts/seed_workflows.py --clear

Environment:
    DATABASE_URL: Database connection string (required)
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import async_session_factory, init_db
from src.api.models.workflow import WorkflowTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Template Definitions
# ============================================================================

QUANTUM_ESPRESSO_TEMPLATES = [
    {
        "name": "DFT_relaxation_QE",
        "display_name": "DFT Geometry Relaxation (Quantum ESPRESSO)",
        "description": (
            "Perform geometry relaxation using Quantum ESPRESSO. "
            "This workflow relaxes atomic positions to minimize forces while "
            "keeping the cell fixed. Suitable for optimizing molecular and "
            "crystal structures."
        ),
        "engine": "QE",
        "engine_version": "7.0+",
        "category": "DFT",
        "default_parameters": {
            "calculation": "relax",
            "ecutwfc": 50.0,  # Wavefunction cutoff (Ry)
            "ecutrho": 200.0,  # Density cutoff (Ry)
            "conv_thr": 1.0e-8,  # SCF convergence threshold
            "mixing_beta": 0.7,  # Mixing factor for SCF
            "k_points": [4, 4, 4],  # K-point grid
            "k_shift": [0, 0, 0],  # K-point shift
            "coord_type": "crystal",  # Coordinate type (crystal, angstrom, bohr)
            "ion_dynamics": "bfgs",  # Ionic relaxation algorithm
            "forc_conv_thr": 1.0e-4,  # Force convergence threshold (Ry/au)
            "spin_polarized": False,  # Enable spin polarization
            "tot_magnetization": 0.0,  # Total magnetization
        },
        "default_resources": {
            "cores": 4,
            "memory_gb": 8,
            "walltime_minutes": 60,
        },
        "is_active": True,
        "is_public": True,
        "documentation_url": "https://www.quantum-espresso.org/Doc/INPUT_PW.html",
    },
    {
        "name": "DFT_scf_QE",
        "display_name": "DFT Single-Point Energy (Quantum ESPRESSO)",
        "description": (
            "Perform single-point self-consistent field (SCF) calculation "
            "using Quantum ESPRESSO. This workflow calculates the total energy "
            "and electronic structure for a fixed geometry. Useful for "
            "calculating energies of known structures."
        ),
        "engine": "QE",
        "engine_version": "7.0+",
        "category": "DFT",
        "default_parameters": {
            "calculation": "scf",
            "ecutwfc": 50.0,  # Wavefunction cutoff (Ry)
            "ecutrho": 200.0,  # Density cutoff (Ry)
            "conv_thr": 1.0e-8,  # SCF convergence threshold
            "mixing_beta": 0.7,  # Mixing factor for SCF
            "k_points": [4, 4, 4],  # K-point grid
            "k_shift": [0, 0, 0],  # K-point shift
            "coord_type": "crystal",  # Coordinate type
            "spin_polarized": False,  # Enable spin polarization
            "tot_magnetization": 0.0,  # Total magnetization
        },
        "default_resources": {
            "cores": 4,
            "memory_gb": 4,
            "walltime_minutes": 30,
        },
        "is_active": True,
        "is_public": True,
        "documentation_url": "https://www.quantum-espresso.org/Doc/INPUT_PW.html",
    },
    {
        "name": "DFT_bands_QE",
        "display_name": "DFT Band Structure (Quantum ESPRESSO)",
        "description": (
            "Calculate electronic band structure using Quantum ESPRESSO. "
            "This is a two-step workflow: (1) SCF calculation on uniform grid, "
            "followed by (2) non-SCF calculation along high-symmetry k-points. "
            "Note: This template configures the initial SCF step. Band calculation "
            "requires a separate bands.x run (future implementation)."
        ),
        "engine": "QE",
        "engine_version": "7.0+",
        "category": "DFT",
        "default_parameters": {
            "calculation": "scf",  # First step: SCF
            "ecutwfc": 60.0,  # Higher cutoff for accuracy
            "ecutrho": 240.0,
            "conv_thr": 1.0e-10,  # Tighter convergence for bands
            "mixing_beta": 0.7,
            "k_points": [8, 8, 8],  # Denser k-point grid
            "k_shift": [0, 0, 0],
            "coord_type": "crystal",
            "spin_polarized": False,
            "tot_magnetization": 0.0,
            # Future: add band-specific parameters
            # "bands_k_path": "GXWKGLUWLK",  # High-symmetry path
            # "bands_points": 100,  # Number of points along path
        },
        "default_resources": {
            "cores": 8,
            "memory_gb": 16,
            "walltime_minutes": 120,
        },
        "is_active": True,
        "is_public": True,
        "documentation_url": "https://www.quantum-espresso.org/Doc/INPUT_PW.html",
    },
    {
        "name": "DFT_vc_relax_QE",
        "display_name": "DFT Variable-Cell Relaxation (Quantum ESPRESSO)",
        "description": (
            "Perform variable-cell relaxation using Quantum ESPRESSO. "
            "This workflow relaxes both atomic positions and cell parameters "
            "to minimize stress and forces. Suitable for finding equilibrium "
            "lattice constants and optimizing unit cells."
        ),
        "engine": "QE",
        "engine_version": "7.0+",
        "category": "DFT",
        "default_parameters": {
            "calculation": "vc-relax",
            "ecutwfc": 50.0,
            "ecutrho": 200.0,
            "conv_thr": 1.0e-8,
            "mixing_beta": 0.7,
            "k_points": [4, 4, 4],
            "k_shift": [0, 0, 0],
            "coord_type": "crystal",
            "ion_dynamics": "bfgs",  # Ionic relaxation algorithm
            "cell_dynamics": "bfgs",  # Cell relaxation algorithm
            "forc_conv_thr": 1.0e-4,  # Force convergence
            "press_conv_thr": 0.5,  # Pressure convergence (kbar)
            "spin_polarized": False,
            "tot_magnetization": 0.0,
        },
        "default_resources": {
            "cores": 4,
            "memory_gb": 8,
            "walltime_minutes": 120,
        },
        "is_active": True,
        "is_public": True,
        "documentation_url": "https://www.quantum-espresso.org/Doc/INPUT_PW.html",
    },
]

MOCK_TEMPLATES = [
    {
        "name": "MOCK_simulation",
        "display_name": "Mock Simulation (Testing)",
        "description": (
            "Mock simulation for testing and development. Returns realistic "
            "but fake results without running actual calculations. Useful for "
            "testing workflows, UI development, and CI/CD."
        ),
        "engine": "MOCK",
        "engine_version": "2.0.0",
        "category": "Testing",
        "default_parameters": {
            "functional": "PBE",
            "k_points": [4, 4, 4],
            "ecutwfc": 500,
            "smearing": 0.01,
            "spin_polarized": False,
        },
        "default_resources": {
            "cores": 1,
            "memory_gb": 1,
            "walltime_minutes": 5,
        },
        "is_active": True,
        "is_public": True,
        "documentation_url": None,
    },
]

# VASP templates (placeholders for future implementation)
VASP_TEMPLATES = [
    {
        "name": "DFT_relaxation_VASP",
        "display_name": "DFT Geometry Relaxation (VASP)",
        "description": (
            "Perform geometry relaxation using VASP. "
            "NOTE: VASP engine not yet implemented. This is a placeholder."
        ),
        "engine": "VASP",
        "engine_version": "6.0+",
        "category": "DFT",
        "default_parameters": {
            "IBRION": 2,  # Relaxation algorithm
            "ISIF": 2,  # Relax ions only
            "EDIFF": 1e-6,  # Electronic convergence
            "EDIFFG": -0.01,  # Ionic convergence
            "ENCUT": 520,  # Energy cutoff (eV)
            "ISMEAR": 0,  # Gaussian smearing
            "SIGMA": 0.05,  # Smearing width
        },
        "default_resources": {
            "cores": 4,
            "memory_gb": 8,
            "walltime_minutes": 60,
        },
        "is_active": False,  # Not active until VASP engine is implemented
        "is_public": True,
        "documentation_url": "https://www.vasp.at/wiki/index.php/The_VASP_Manual",
    },
]


# ============================================================================
# Seeder Functions
# ============================================================================

async def clear_templates(db: AsyncSession) -> int:
    """
    Clear all existing workflow templates.

    Args:
        db: Database session

    Returns:
        Number of templates deleted
    """
    logger.info("Clearing existing workflow templates...")
    result = await db.execute(delete(WorkflowTemplate))
    await db.commit()
    count = result.rowcount
    logger.info(f"Deleted {count} workflow templates")
    return count


async def seed_template(db: AsyncSession, template_data: dict, dry_run: bool = False) -> WorkflowTemplate:
    """
    Seed a single workflow template.

    Args:
        db: Database session
        template_data: Template data dictionary
        dry_run: If True, don't commit to database

    Returns:
        Created WorkflowTemplate instance
    """
    # Check if template already exists
    stmt = select(WorkflowTemplate).where(WorkflowTemplate.name == template_data["name"])
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        logger.info(f"Template '{template_data['name']}' already exists, updating...")
        # Update existing template
        for key, value in template_data.items():
            setattr(existing, key, value)
        template = existing
    else:
        logger.info(f"Creating template '{template_data['name']}'...")
        template = WorkflowTemplate(**template_data)
        db.add(template)

    if not dry_run:
        await db.commit()
        await db.refresh(template)
        logger.info(f"✓ Template '{template.name}' seeded successfully")
    else:
        logger.info(f"[DRY RUN] Would create/update template '{template_data['name']}'")

    return template


async def seed_templates(
    db: AsyncSession,
    templates: list[dict],
    dry_run: bool = False
) -> list[WorkflowTemplate]:
    """
    Seed multiple workflow templates.

    Args:
        db: Database session
        templates: List of template data dictionaries
        dry_run: If True, don't commit to database

    Returns:
        List of created WorkflowTemplate instances
    """
    created_templates = []

    for template_data in templates:
        template = await seed_template(db, template_data, dry_run)
        created_templates.append(template)

    return created_templates


async def main():
    """Main seeder function."""
    parser = argparse.ArgumentParser(description="Seed workflow templates")
    parser.add_argument(
        "--engine",
        choices=["QE", "MOCK", "VASP", "ALL"],
        default="ALL",
        help="Which engine templates to seed (default: ALL)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing templates before seeding"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run (don't commit to database)"
    )
    args = parser.parse_args()

    # Initialize database
    logger.info("Initializing database...")
    await init_db()

    # Select templates to seed
    templates_to_seed = []
    if args.engine in ["QE", "ALL"]:
        templates_to_seed.extend(QUANTUM_ESPRESSO_TEMPLATES)
    if args.engine in ["MOCK", "ALL"]:
        templates_to_seed.extend(MOCK_TEMPLATES)
    if args.engine in ["VASP", "ALL"]:
        templates_to_seed.extend(VASP_TEMPLATES)

    logger.info(f"Seeding {len(templates_to_seed)} templates...")

    # Seed templates
    async with async_session_factory() as db:
        # Clear existing templates if requested
        if args.clear:
            await clear_templates(db)

        # Seed templates
        created = await seed_templates(db, templates_to_seed, dry_run=args.dry_run)

        logger.info(f"\n{'='*60}")
        logger.info(f"Seeding complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Templates processed: {len(created)}")

        if not args.dry_run:
            # Show summary
            logger.info("\nSeeded templates:")
            for template in created:
                logger.info(
                    f"  - {template.name} ({template.engine}) - "
                    f"{'Active' if template.is_active else 'Inactive'}"
                )
        else:
            logger.info("\n[DRY RUN] No changes committed to database")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nSeeding cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error seeding templates: {e}", exc_info=True)
        sys.exit(1)
