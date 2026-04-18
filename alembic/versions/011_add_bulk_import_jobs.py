"""Add bulk_import_jobs table

Session 1.3 introduces the BulkImportJob model for tracking CSV/Parquet
property imports with per-row validation, progress counters, and a
capped error log. Session 2.1 extends usage (MinIO-backed payload,
Celery execution); the schema already supports that via source_kind.

Revision ID: 011_add_bulk_import_jobs
Revises: 010_add_structure_spacegroup_columns
Create Date: 2026-04-18
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "011_add_bulk_import_jobs"
down_revision: Union[str, None] = "010_add_structure_spacegroup_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bulk_import_jobs",
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False,
        ),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "kind",
            sa.String(length=50),
            nullable=False,
            server_default="PROPERTY",
            index=True,
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default="PENDING",
            index=True,
        ),
        sa.Column(
            "mapping",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "source_kind",
            sa.String(length=16),
            nullable=False,
            server_default="inline",
        ),
        sa.Column("source_payload", sa.Text(), nullable=True),
        sa.Column(
            "rows_total", sa.Integer(), nullable=False, server_default="0",
        ),
        sa.Column(
            "rows_ok", sa.Integer(), nullable=False, server_default="0",
        ),
        sa.Column(
            "rows_failed", sa.Integer(), nullable=False, server_default="0",
        ),
        sa.Column(
            "errors",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            index=True,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("bulk_import_jobs")
