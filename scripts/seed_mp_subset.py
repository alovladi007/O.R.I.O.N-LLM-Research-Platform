#!/usr/bin/env python3
"""
Materials Project seed loader (Phase 1 / Session 1.5).

Loads a curated subset into the ORION database:

- With ``MP_API_KEY`` set → pulls a specified subset from Materials
  Project via ``mp-api`` / pymatgen and caches the CIFs locally.
- Without ``MP_API_KEY`` → falls back to the bundled offline fixture
  under ``tests/fixtures/mp_offline/`` (20 representative structures).

Seed lists (MP query, when key is set):

- ~200 oxides and ABO3 perovskites with PBE bandgap 0–6 eV
- ~50 elemental metals
- ~50 2D materials

Idempotent: re-running skips rows whose ``structure_hash`` already
exists. Also seeds three users (admin / scientist / viewer) with
passwords sourced from ``ORION_SEED_PASSWORD`` or defaults.

Usage
-----

::

    # offline mode (no MP key required) — good for tests & CI
    python scripts/seed_mp_subset.py --offline

    # MP-live mode
    export MP_API_KEY=...
    python scripts/seed_mp_subset.py --formula-family oxides

    # Backfill structure_hash on pre-1.2 rows
    python scripts/seed_mp_subset.py --backfill-hashes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make `backend`, `src`, etc. importable when this script is run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from backend.common.structures.hashing import structure_hash  # noqa: E402

logger = logging.getLogger("orion.seed_mp_subset")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
)


DEFAULT_SEED_USERS = [
    {
        "email": "admin@orion.dev",
        "username": "admin",
        "full_name": "ORION Admin",
        "role": "admin",
    },
    {
        "email": "scientist@orion.dev",
        "username": "scientist",
        "full_name": "Research Scientist",
        "role": "scientist",
    },
    {
        "email": "viewer@orion.dev",
        "username": "viewer",
        "full_name": "Viewer",
        "role": "viewer",
    },
]


OFFLINE_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "mp_offline"
MP_CACHE_DIR = ROOT / "data" / "mp_cache"


# ---------------------------------------------------------------------------
# Offline fixture loading
# ---------------------------------------------------------------------------


def load_offline_fixtures() -> List[Dict[str, Any]]:
    """
    Read every JSON file in ``tests/fixtures/mp_offline/`` and return a
    list of structure-seed dicts. Missing dir → empty list.

    Each file holds a single JSON object with keys:
        mp_id, formula, cif, bandgap, formation_energy_per_atom,
        density, source.
    """
    if not OFFLINE_FIXTURE_DIR.exists():
        logger.warning("offline fixture dir not present at %s", OFFLINE_FIXTURE_DIR)
        return []

    items: List[Dict[str, Any]] = []
    for path in sorted(OFFLINE_FIXTURE_DIR.glob("*.json")):
        try:
            with path.open() as f:
                items.append(json.load(f))
        except Exception as exc:  # noqa: BLE001
            logger.error("bad fixture %s: %s", path.name, exc)
    logger.info("loaded %d offline fixture structures", len(items))
    return items


# ---------------------------------------------------------------------------
# Live MP fetch (gated on MP_API_KEY)
# ---------------------------------------------------------------------------


def fetch_mp_subset(
    family: str = "oxides", limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Query Materials Project for a named subset.

    ``family`` values:
      - ``oxides``    — band_gap 0..6, contains O, ABO3-like stoichiometries.
      - ``metals``    — elemental metals with bandgap == 0.
      - ``2d``        — dimensionality 2 (MP's ``is_metal`` + size heuristic).

    Returns the same dict shape as ``load_offline_fixtures``. Requires
    ``MP_API_KEY``. Caches each mp-id's CIF under ``data/mp_cache/<mp-id>.cif``.
    """
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MP_API_KEY not set. Either export it or run with --offline."
        )

    try:
        from mp_api.client import MPRester  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "mp-api package not installed. `pip install mp-api` or run with --offline."
        ) from exc

    MP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    filters: Dict[str, Any] = {}
    if family == "oxides":
        filters = dict(elements=["O"], band_gap=(0.0, 6.0), num_sites=(1, 30))
    elif family == "metals":
        filters = dict(band_gap=(0.0, 0.01), num_elements=1, num_sites=(1, 20))
    elif family == "2d":
        filters = dict(band_gap=(0.0, 6.0), num_sites=(1, 20))
    else:
        raise ValueError(f"unknown family: {family!r}")

    items: List[Dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(**filters, chunk_size=min(limit, 1000))
        for d in docs[:limit]:
            mp_id = d.material_id
            cif_path = MP_CACHE_DIR / f"{mp_id}.cif"
            if cif_path.exists():
                cif_text = cif_path.read_text()
            else:
                try:
                    cif_text = d.structure.to(fmt="cif")
                    cif_path.write_text(cif_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to cache %s: %s", mp_id, exc)
                    continue
            items.append(
                {
                    "mp_id": mp_id,
                    "formula": d.formula_pretty,
                    "cif": cif_text,
                    "bandgap": d.band_gap,
                    "formation_energy_per_atom": d.formation_energy_per_atom,
                    "density": d.density,
                    "source": "materials_project",
                }
            )
    logger.info("fetched %d MP structures (family=%s)", len(items), family)
    return items


# ---------------------------------------------------------------------------
# DB persistence — idempotent
# ---------------------------------------------------------------------------


async def _seed_users(session: AsyncSession) -> None:
    """Insert the three default users if they don't already exist."""
    from src.api.models import User  # local import; avoids heavy boot
    from passlib.context import CryptContext

    pw = CryptContext(schemes=["bcrypt"], deprecated="auto").hash(
        os.getenv("ORION_SEED_PASSWORD", "CHANGE_ME_local_only"),
    )

    for spec in DEFAULT_SEED_USERS:
        existing = await session.execute(
            select(User).where(User.email == spec["email"])
        )
        if existing.scalar_one_or_none() is not None:
            logger.info("user %s already present — skip", spec["email"])
            continue
        user = User(
            email=spec["email"],
            username=spec["username"],
            full_name=spec["full_name"],
            hashed_password=pw,
            role=spec["role"],
            is_active=True,
        )
        session.add(user)
    await session.commit()


async def _seed_structures(
    session: AsyncSession, items: List[Dict[str, Any]], owner_email: str,
) -> Dict[str, int]:
    """
    Insert each structure (and its Material) if its ``structure_hash``
    isn't already present. Returns a counts dict.
    """
    from src.api.models import Material, Structure, User
    from src.api.models.structure import StructureFormat, StructureSource

    # Import here to avoid circular loading issues when this module is
    # imported in tests.
    from backend.common.structures import (
        StructureFormat as ParserFmt,
        parse_structure,
    )

    owner = await session.execute(select(User).where(User.email == owner_email))
    owner = owner.scalar_one_or_none()
    if owner is None:
        raise RuntimeError(
            f"owner user {owner_email!r} not found — run _seed_users first"
        )

    counts = {"inserted": 0, "skipped": 0, "failed": 0}
    for item in items:
        try:
            parsed = parse_structure(item["cif"], ParserFmt.CIF)
            shash = structure_hash(
                lattice=parsed.lattice_vectors,
                atoms=[
                    {"species": sp, "position": pos}
                    for sp, pos in zip(parsed.atomic_species, parsed.atomic_positions)
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("parse failed for %s: %s", item.get("mp_id"), exc)
            counts["failed"] += 1
            continue

        existing = await session.execute(
            select(Structure.id).where(Structure.structure_hash == shash)
        )
        if existing.scalar_one_or_none() is not None:
            counts["skipped"] += 1
            continue

        # Material row (idempotent by formula + external_id).
        mat_stmt = select(Material).where(
            Material.external_id == item.get("mp_id"),
        )
        mat = (await session.execute(mat_stmt)).scalar_one_or_none()
        if mat is None:
            mat = Material(
                owner_id=owner.id,
                name=item["formula"],
                formula=item["formula"],
                source=item.get("source", "materials_project"),
                external_id=item.get("mp_id"),
                composition={},
            )
            session.add(mat)
            await session.flush()

        structure = Structure(
            owner_id=owner.id,
            material_id=mat.id,
            name=f"{item['formula']} ({item.get('mp_id', 'fixture')})",
            format=StructureFormat.CIF,
            source=StructureSource.EXTERNAL_DB,
            raw_text=item["cif"],
            formula=parsed.formula,
            num_atoms=parsed.num_atoms,
            dimensionality=parsed.dimensionality,
            a=parsed.a, b=parsed.b, c=parsed.c,
            alpha=parsed.alpha, beta=parsed.beta, gamma=parsed.gamma,
            volume=parsed.volume,
            space_group=parsed.space_group,
            space_group_number=parsed.space_group_number,
            structure_hash=shash,
            extra_metadata={
                "mp_id": item.get("mp_id"),
                "bandgap_ev": item.get("bandgap"),
                "formation_energy_per_atom_ev": item.get("formation_energy_per_atom"),
                "density_g_cm3": item.get("density"),
            },
        )
        session.add(structure)
        counts["inserted"] += 1

    await session.commit()
    return counts


async def _backfill_hashes(session: AsyncSession) -> int:
    """
    Compute ``structure_hash`` for rows missing one. Used to populate
    legacy rows inserted before Session 1.2 added the column.
    """
    from src.api.models import Structure

    result = await session.execute(
        select(Structure).where(Structure.structure_hash.is_(None))
    )
    updated = 0
    for s in result.scalars():
        if not s.atoms or not s.lattice:
            continue
        lattice = (
            s.lattice.get("vectors") if isinstance(s.lattice, dict) else s.lattice
        )
        try:
            h = structure_hash(
                lattice=lattice,
                atoms=s.atoms,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill skipped for %s: %s", s.id, exc)
            continue
        s.structure_hash = h
        updated += 1
    await session.commit()
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true",
        help="Use the bundled offline fixture instead of hitting MP.",
    )
    parser.add_argument(
        "--family", default="oxides", choices=("oxides", "metals", "2d"),
        help="Which MP subset to pull (ignored in --offline mode).",
    )
    parser.add_argument(
        "--limit", type=int, default=50,
        help="Max rows to request from MP.",
    )
    parser.add_argument(
        "--backfill-hashes", action="store_true",
        help="Only compute structure_hash for rows missing one and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load + validate but don't commit to the DB.",
    )
    args = parser.parse_args()

    from src.api.config import settings

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        if args.backfill_hashes:
            n = await _backfill_hashes(session)
            logger.info("backfilled %d structure_hash rows", n)
            return 0

        # Always make sure the seed users exist first.
        await _seed_users(session)

        if args.offline or not os.getenv("MP_API_KEY"):
            if not args.offline:
                logger.warning(
                    "MP_API_KEY missing; falling back to offline fixtures"
                )
            items = load_offline_fixtures()
        else:
            items = fetch_mp_subset(family=args.family, limit=args.limit)

        if args.dry_run:
            logger.info("dry-run: %d candidate structures; not persisting", len(items))
            return 0

        counts = await _seed_structures(
            session, items, owner_email="admin@orion.dev",
        )
        logger.info("done: %s", counts)
    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
