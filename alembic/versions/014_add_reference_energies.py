"""Add reference_energies table (Session 3.4).

Revision ID: 014_add_reference_energies
Revises: 013_add_workflow_runs
Create Date: 2026-04-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision: str = "014_add_reference_energies"
down_revision: Union[str, None] = "013_add_workflow_runs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "reference_energies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("element", sa.String(length=8), nullable=False),
        sa.Column("functional", sa.String(length=32), nullable=False),
        sa.Column("pseudo_family", sa.String(length=64), nullable=False),
        sa.Column("energy_per_atom_ev", sa.Float, nullable=False),
        sa.Column("n_atoms_in_reference_cell", sa.Integer, nullable=False),
        sa.Column(
            "source_job_id",
            UUID(as_uuid=True),
            sa.ForeignKey("simulation_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "reference_prototype",
            sa.String(length=32),
            nullable=False,
            server_default="auto",
        ),
        sa.Column("relaxed_a_ang", sa.Float, nullable=True),
        sa.Column("extra_metadata", JSONB, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "element", "functional", "pseudo_family",
            name="uq_reference_energy_triple",
        ),
    )
    op.create_index(
        "ix_reference_energies_element", "reference_energies", ["element"],
    )
    op.create_index(
        "ix_reference_energies_functional", "reference_energies", ["functional"],
    )
    op.create_index(
        "ix_reference_energies_pseudo_family",
        "reference_energies",
        ["pseudo_family"],
    )
    op.create_index(
        "ix_reference_energies_source_job_id",
        "reference_energies",
        ["source_job_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_reference_energies_source_job_id", "reference_energies")
    op.drop_index("ix_reference_energies_pseudo_family", "reference_energies")
    op.drop_index("ix_reference_energies_functional", "reference_energies")
    op.drop_index("ix_reference_energies_element", "reference_energies")
    op.drop_table("reference_energies")
