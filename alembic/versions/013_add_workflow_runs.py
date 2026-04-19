"""Add workflow_runs + workflow_run_steps tables (Session 2.4).

Revision ID: 013_add_workflow_runs
Revises: 012_add_simulation_job_kind
Create Date: 2026-04-18
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision: str = "013_add_workflow_runs"
down_revision: Union[str, None] = "012_add_simulation_job_kind"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "workflow_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "owner_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("spec", JSONB, nullable=False),
        sa.Column("manifest", JSONB, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_workflow_runs_owner_id", "workflow_runs", ["owner_id"])
    op.create_index("ix_workflow_runs_status", "workflow_runs", ["status"])

    op.create_table(
        "workflow_run_steps",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workflow_run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("step_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=50), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("topo_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("spec", JSONB, nullable=False),
        sa.Column("outputs", JSONB, nullable=True),
        sa.Column(
            "simulation_job_id",
            UUID(as_uuid=True),
            sa.ForeignKey("simulation_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("workflow_run_id", "step_id", name="uq_workflow_step"),
    )
    op.create_index(
        "ix_workflow_run_steps_workflow_run_id",
        "workflow_run_steps",
        ["workflow_run_id"],
    )
    op.create_index(
        "ix_workflow_run_steps_status", "workflow_run_steps", ["status"]
    )
    op.create_index(
        "ix_workflow_run_steps_simulation_job_id",
        "workflow_run_steps",
        ["simulation_job_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_workflow_run_steps_simulation_job_id", "workflow_run_steps")
    op.drop_index("ix_workflow_run_steps_status", "workflow_run_steps")
    op.drop_index("ix_workflow_run_steps_workflow_run_id", "workflow_run_steps")
    op.drop_table("workflow_run_steps")
    op.drop_index("ix_workflow_runs_status", "workflow_runs")
    op.drop_index("ix_workflow_runs_owner_id", "workflow_runs")
    op.drop_table("workflow_runs")
