"""Rename metadata columns, add structure_hash + embedding, add experiments tables

Phase 1 / Session 1.2b wraps up the data-model foundation:

1. **Rename `metadata` → `extra_metadata`** on every table that had it.
   SQLAlchemy reserves ``Base.metadata``; Session 0.1 renamed the Python
   attribute as a stopgap while keeping the DB column named
   ``metadata`` via ``mapped_column("metadata", ...)``. This migration
   performs the real rename so the column and attribute match. Affects:

       materials, structures, simulation_jobs, simulation_results,
       predicted_properties, design_campaigns, design_iterations,
       vector_embeddings

2. **Add `structures.structure_hash`** (VARCHAR(64), unique, indexed,
   nullable for now). Deterministic SHA-256 fingerprint populated by
   ``backend.common.structures.hashing.structure_hash``. Nullable because
   pre-existing rows won't have a hash until the backfill in Session 1.5.

3. **Add `structures.embedding`** (Text, nullable). Learned 256-d
   representation; Text serialization for now. Phase 6 upgrades this to
   a typed pgvector ``Vector(256)`` column + ANN index.

4. **Add `predicted_properties.value_unit` and
   `predicted_properties.uncertainty`**. Stepping stones for the
   Session 1.3 bulk property importer and Session 6 uncertainty-aware
   ML predictions.

5. **Create `instruments` and `experiment_runs` tables.** Their Python
   models landed in Session 1.2a (``src/api/models/experiment.py``) but
   no matching migration existed — this adds one.

Revision ID: 009_rename_metadata_and_add_structure_hash
Revises: 008_add_orchestrator
Create Date: 2026-04-17
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "009_rename_metadata_and_add_structure_hash"
down_revision: Union[str, None] = "008_add_orchestrator"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Tables that held a ``metadata`` JSON/JSONB column in prior migrations.
_METADATA_TABLES = [
    "materials",
    "structures",
    "simulation_jobs",
    "simulation_results",
    "predicted_properties",
    "design_campaigns",
    "design_iterations",
    "vector_embeddings",
]


def upgrade() -> None:
    # ---- 1. Rename metadata → extra_metadata -----------------------------
    for table in _METADATA_TABLES:
        op.alter_column(
            table,
            "metadata",
            new_column_name="extra_metadata",
        )

    # ---- 2. structures.structure_hash ------------------------------------
    op.add_column(
        "structures",
        sa.Column(
            "structure_hash",
            sa.String(length=64),
            nullable=True,
            comment="SHA-256 fingerprint of symmetrized structure (64 hex chars).",
        ),
    )
    op.create_index(
        "ix_structures_structure_hash",
        "structures",
        ["structure_hash"],
        unique=True,
    )

    # ---- 3. structures.embedding -----------------------------------------
    op.add_column(
        "structures",
        sa.Column(
            "embedding",
            sa.Text(),
            nullable=True,
            comment="Learned 256-d representation (Text; pgvector in Phase 6).",
        ),
    )

    # ---- 4. predicted_properties value_unit + uncertainty ---------------
    op.add_column(
        "predicted_properties",
        sa.Column(
            "value_unit",
            sa.String(length=32),
            nullable=True,
            comment="Unit the raw value was reported in (e.g. 'eV'). SI stored elsewhere.",
        ),
    )
    op.add_column(
        "predicted_properties",
        sa.Column(
            "uncertainty",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Uncertainty descriptor; shape varies by source.",
        ),
    )

    # ---- 5. Instruments + ExperimentRuns --------------------------------
    op.create_table(
        "instruments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(length=255), nullable=False, index=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "adapter_type",
            sa.String(length=50),
            nullable=False,
            server_default="MOCK",
            comment="Adapter kind: MOCK | REST | OPCUA | SSH | CUSTOM.",
        ),
        sa.Column(
            "connection_info",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "capabilities",
            postgresql.ARRAY(sa.String(length=100)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default="IDLE",
            index=True,
        ),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column(
            "extra_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "experiment_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "linked_structure_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("structures.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("name", sa.String(length=255), nullable=False, index=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "type",
            sa.String(length=100),
            nullable=False,
            comment="synthesis | measurement | characterization | testing",
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default="PENDING",
            index=True,
        ),
        sa.Column(
            "parameters",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "results",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("external_job_id", sa.String(length=255), nullable=True),
        sa.Column(
            "extra_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            index=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    # Drop experiments tables first (newest).
    op.drop_table("experiment_runs")
    op.drop_table("instruments")

    # Drop predicted_properties new columns.
    op.drop_column("predicted_properties", "uncertainty")
    op.drop_column("predicted_properties", "value_unit")

    # Drop structures new columns.
    op.drop_column("structures", "embedding")
    op.drop_index("ix_structures_structure_hash", table_name="structures")
    op.drop_column("structures", "structure_hash")

    # Rename extra_metadata → metadata on every affected table.
    for table in _METADATA_TABLES:
        op.alter_column(
            table,
            "extra_metadata",
            new_column_name="metadata",
        )
