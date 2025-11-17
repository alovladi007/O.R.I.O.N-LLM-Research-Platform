#!/usr/bin/env python3
"""
Database Seeder for NANO-OS
============================

Seeds the database with example data for development and demonstrations:
- Example materials (graphene, MoS2, Si, GaN)
- Example structures (CIF and POSCAR formats)
- Workflow templates for all engines

Usage:
    python scripts/seed_data.py
    python scripts/seed_data.py --clear  # Clear existing data first
    python scripts/seed_data.py --dry-run  # Preview without committing

Environment:
    DATABASE_URL: Database connection string (from .env)
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import async_session_factory, init_db
from src.api.models.material import Material
from src.api.models.structure import Structure, StructureFormat, StructureSource
from src.api.models.workflow import WorkflowTemplate
from src.api.models.user import User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example Materials
# ============================================================================

EXAMPLE_MATERIALS = [
    {
        "name": "Graphene",
        "formula": "C",
        "description": "Single-layer 2D carbon allotrope with exceptional electronic properties",
        "tags": ["2D", "carbon", "conductor", "high_mobility"],
        "metadata": {
            "dimensionality": 2,
            "crystal_system": "hexagonal",
            "applications": ["electronics", "sensors", "energy_storage"],
            "experimental_bandgap": 0.0,
            "reference": "Novoselov et al., Science 2004"
        }
    },
    {
        "name": "Molybdenum Disulfide",
        "formula": "MoS2",
        "description": "Transition metal dichalcogenide with tunable bandgap",
        "tags": ["2D", "TMD", "semiconductor", "direct_bandgap"],
        "metadata": {
            "dimensionality": 2,
            "crystal_system": "hexagonal",
            "applications": ["optoelectronics", "catalysis", "lubrication"],
            "experimental_bandgap": 1.8,
            "reference": "Mak et al., PRL 2010"
        }
    },
    {
        "name": "Silicon",
        "formula": "Si",
        "description": "Ubiquitous semiconductor for electronics",
        "tags": ["3D", "semiconductor", "indirect_bandgap", "group_IV"],
        "metadata": {
            "dimensionality": 3,
            "crystal_system": "cubic",
            "space_group": "Fd-3m",
            "applications": ["electronics", "solar_cells", "sensors"],
            "experimental_bandgap": 1.12,
            "lattice_constant": 5.43
        }
    },
    {
        "name": "Gallium Nitride",
        "formula": "GaN",
        "description": "Wide bandgap semiconductor for high-power and optoelectronic applications",
        "tags": ["3D", "semiconductor", "wide_bandgap", "III-V"],
        "metadata": {
            "dimensionality": 3,
            "crystal_system": "hexagonal",
            "space_group": "P63mc",
            "applications": ["LEDs", "power_electronics", "RF_devices"],
            "experimental_bandgap": 3.4,
            "lattice_constants": {"a": 3.19, "c": 5.19}
        }
    }
]


# ============================================================================
# Example Structures (CIF format)
# ============================================================================

GRAPHENE_CIF = """data_graphene
_symmetry_space_group_name_H-M    'P 6/m m m'
_cell_length_a                    2.46
_cell_length_b                    2.46
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
C1 C 0.3333 0.6667 0.0000
C2 C 0.6667 0.3333 0.0000
"""

MOS2_CIF = """data_mos2_monolayer
_symmetry_space_group_name_H-M    'P -6 m 2'
_cell_length_a                    3.16
_cell_length_b                    3.16
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
Mo1 Mo 0.3333 0.6667 0.0000
S1  S  0.6667 0.3333 0.0790
S2  S  0.6667 0.3333 -0.0790
"""

SILICON_CIF = """data_silicon
_symmetry_space_group_name_H-M    'F d -3 m'
_cell_length_a                    5.43
_cell_length_b                    5.43
_cell_length_c                    5.43
_cell_angle_alpha                 90.0
_cell_angle_beta                  90.0
_cell_angle_gamma                 90.0

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.0000 0.0000 0.0000
Si2 Si 0.2500 0.2500 0.2500
"""

GAN_CIF = """data_gan_wurtzite
_symmetry_space_group_name_H-M    'P 63 m c'
_cell_length_a                    3.19
_cell_length_b                    3.19
_cell_length_c                    5.19
_cell_angle_alpha                 90.0
_cell_angle_beta                  90.0
_cell_angle_gamma                 120.0

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ga1 Ga 0.3333 0.6667 0.0000
N1  N  0.3333 0.6667 0.3770
"""


# ============================================================================
# Workflow Templates
# ============================================================================

WORKFLOW_TEMPLATES = [
    # Mock Engine (for testing)
    {
        "name": "MOCK_SIMULATION",
        "display_name": "Mock Simulation (Testing)",
        "description": "Mock simulation for testing and development. Returns fake but realistic results.",
        "engine": "mock",
        "category": "Testing",
        "default_parameters": {
            "success_probability": 0.95,
            "execution_time_seconds": 2.0
        },
        "is_active": True,
        "is_public": True
    },

    # Quantum Espresso (DFT)
    {
        "name": "DFT_SCF_QE",
        "display_name": "DFT Single-Point (QE)",
        "description": "Single-point DFT energy calculation with Quantum Espresso",
        "engine": "qe",
        "category": "DFT",
        "default_parameters": {
            "ecutwfc": 50.0,
            "ecutrho": 400.0,
            "conv_thr": 1e-8,
            "kpoints": [6, 6, 1]
        },
        "is_active": True,
        "is_public": True
    },
    {
        "name": "DFT_RELAX_QE",
        "display_name": "DFT Geometry Optimization (QE)",
        "description": "Relax atomic positions with Quantum Espresso",
        "engine": "qe",
        "category": "DFT",
        "default_parameters": {
            "ecutwfc": 50.0,
            "ecutrho": 400.0,
            "conv_thr": 1e-8,
            "forc_conv_thr": 1e-4,
            "relax_type": "atoms",
            "kpoints": [6, 6, 1]
        },
        "is_active": True,
        "is_public": True
    },
    {
        "name": "DFT_BANDS_QE",
        "display_name": "DFT Band Structure (QE)",
        "description": "Calculate electronic band structure with Quantum Espresso",
        "engine": "qe",
        "category": "DFT",
        "default_parameters": {
            "ecutwfc": 50.0,
            "ecutrho": 400.0,
            "kpath": "automatic",
            "num_bands": 20
        },
        "is_active": True,
        "is_public": True
    },

    # LAMMPS (MD) - Session 17
    {
        "name": "MD_NVT_LAMMPS",
        "display_name": "MD Equilibration (NVT, LAMMPS)",
        "description": "Molecular dynamics in canonical ensemble (constant N, V, T)",
        "engine": "lammps",
        "category": "MD",
        "default_parameters": {
            "temperature": 300.0,
            "timestep": 1.0,
            "num_steps": 100000,
            "dump_interval": 1000,
            "potential": "tersoff"
        },
        "is_active": True,
        "is_public": True
    },
    {
        "name": "MD_NPT_LAMMPS",
        "display_name": "MD Equilibration (NPT, LAMMPS)",
        "description": "Molecular dynamics in isothermal-isobaric ensemble (constant N, P, T)",
        "engine": "lammps",
        "category": "MD",
        "default_parameters": {
            "temperature": 300.0,
            "pressure": 1.0,
            "timestep": 1.0,
            "num_steps": 100000,
            "potential": "tersoff"
        },
        "is_active": True,
        "is_public": True
    },
    {
        "name": "MD_ANNEAL_LAMMPS",
        "display_name": "MD Simulated Annealing (LAMMPS)",
        "description": "Simulated annealing for structure optimization",
        "engine": "lammps",
        "category": "MD",
        "default_parameters": {
            "temp_start": 1000.0,
            "temp_end": 0.0,
            "anneal_steps": 500000,
            "timestep": 1.0,
            "potential": "tersoff"
        },
        "is_active": True,
        "is_public": True
    }
]


# ============================================================================
# Seeding Functions
# ============================================================================

async def get_or_create_demo_user(db: AsyncSession) -> User:
    """Get or create the demo user."""
    result = await db.execute(
        select(User).where(User.username == "demo")
    )
    user = result.scalar_one_or_none()

    if not user:
        logger.info("Creating demo user...")
        user = User(
            username="demo",
            email="demo@nano-os.dev",
            full_name="Demo User",
            hashed_password="$2b$12$demo_hash",  # Not a real hash, for demo only
            is_active=True
        )
        db.add(user)
        await db.flush()
        logger.info(f"Demo user created: {user.id}")

    return user


async def seed_materials(db: AsyncSession, owner_id: uuid.UUID) -> Dict[str, Material]:
    """Seed example materials."""
    materials = {}

    for mat_data in EXAMPLE_MATERIALS:
        # Check if material already exists
        result = await db.execute(
            select(Material).where(
                Material.owner_id == owner_id,
                Material.formula == mat_data["formula"]
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.info(f"Material {mat_data['name']} already exists, skipping...")
            materials[mat_data["formula"]] = existing
            continue

        material = Material(
            owner_id=owner_id,
            name=mat_data["name"],
            formula=mat_data["formula"],
            description=mat_data["description"],
            tags=mat_data["tags"],
            metadata=mat_data["metadata"]
        )
        db.add(material)
        await db.flush()
        materials[mat_data["formula"]] = material
        logger.info(f"Created material: {material.name} ({material.formula})")

    return materials


async def seed_structures(db: AsyncSession, materials: Dict[str, Material], owner_id: uuid.UUID):
    """Seed example structures."""
    structures_data = [
        ("C", "Graphene monolayer", GRAPHENE_CIF, 2),
        ("MoS2", "MoS2 monolayer", MOS2_CIF, 2),
        ("Si", "Silicon diamond structure", SILICON_CIF, 3),
        ("GaN", "GaN wurtzite", GAN_CIF, 3)
    ]

    for formula, name, cif_content, dimensionality in structures_data:
        if formula not in materials:
            logger.warning(f"Material {formula} not found, skipping structure...")
            continue

        material = materials[formula]

        # Check if structure already exists
        result = await db.execute(
            select(Structure).where(
                Structure.owner_id == owner_id,
                Structure.material_id == material.id,
                Structure.name == name
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.info(f"Structure {name} already exists, skipping...")
            continue

        structure = Structure(
            owner_id=owner_id,
            material_id=material.id,
            name=name,
            format=StructureFormat.CIF,
            source=StructureSource.EXTERNAL_DB,
            raw_text=cif_content,
            formula=formula,
            dimensionality=dimensionality
        )
        db.add(structure)
        logger.info(f"Created structure: {name}")

    await db.flush()


async def seed_workflows(db: AsyncSession):
    """Seed workflow templates."""
    for template_data in WORKFLOW_TEMPLATES:
        # Check if template already exists
        result = await db.execute(
            select(WorkflowTemplate).where(
                WorkflowTemplate.name == template_data["name"]
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.info(f"Workflow template {template_data['name']} already exists, skipping...")
            continue

        template = WorkflowTemplate(
            name=template_data["name"],
            display_name=template_data["display_name"],
            description=template_data["description"],
            engine=template_data["engine"],
            category=template_data.get("category", "General"),
            default_parameters=template_data["default_parameters"],
            is_active=template_data.get("is_active", True),
            is_public=template_data.get("is_public", True)
        )
        db.add(template)
        logger.info(f"Created workflow template: {template.display_name}")

    await db.flush()


async def clear_existing_data(db: AsyncSession):
    """Clear existing seed data (materials, structures, workflows)."""
    logger.warning("Clearing existing seed data...")

    # Delete in reverse order of dependencies
    await db.execute(delete(Structure))
    await db.execute(delete(Material))
    await db.execute(delete(WorkflowTemplate))
    await db.commit()

    logger.info("Existing data cleared")


async def seed_database(clear: bool = False, dry_run: bool = False):
    """Main seeding function."""
    logger.info("=" * 60)
    logger.info("NANO-OS Database Seeder")
    logger.info("=" * 60)

    # Initialize database
    await init_db()

    async with async_session_factory() as db:
        try:
            # Clear existing data if requested
            if clear and not dry_run:
                await clear_existing_data(db)

            # Get or create demo user
            demo_user = await get_or_create_demo_user(db)

            # Seed materials
            logger.info("\nSeeding materials...")
            materials = await seed_materials(db, demo_user.id)
            logger.info(f"Seeded {len(materials)} materials")

            # Seed structures
            logger.info("\nSeeding structures...")
            await seed_structures(db, materials, demo_user.id)
            logger.info("Structures seeded")

            # Seed workflow templates
            logger.info("\nSeeding workflow templates...")
            await seed_workflows(db)
            logger.info("Workflow templates seeded")

            # Commit or rollback
            if dry_run:
                logger.info("\nDry run mode - rolling back changes")
                await db.rollback()
            else:
                logger.info("\nCommitting changes...")
                await db.commit()
                logger.info("✓ Database seeded successfully!")

        except Exception as e:
            logger.error(f"Error during seeding: {e}", exc_info=True)
            await db.rollback()
            raise

    logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Seed NANO-OS database with example data")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before seeding")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without committing")

    args = parser.parse_args()

    asyncio.run(seed_database(clear=args.clear, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
