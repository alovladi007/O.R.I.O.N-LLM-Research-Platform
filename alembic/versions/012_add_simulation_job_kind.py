"""Add simulation_jobs.kind column

Phase 1 / Session 1.4 introduces a semantic ``kind`` column (String(50))
on ``simulation_jobs``. The existing ``engine`` column continues to
say *which* physics package; ``kind`` says *what* the job is doing
(dft_relax, md_nvt, ml_train, agent_step, ...).

Nullable for backwards compatibility with rows created before 1.4.

Revision ID: 012_add_simulation_job_kind
Revises: 011_add_bulk_import_jobs
Create Date: 2026-04-18
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012_add_simulation_job_kind"
down_revision: Union[str, None] = "011_add_bulk_import_jobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "simulation_jobs",
        sa.Column(
            "kind",
            sa.String(length=50),
            nullable=True,
            comment="Semantic job kind: dft_relax, md_nvt, ml_train, ... (see JobKind).",
        ),
    )
    op.create_index(
        "ix_simulation_jobs_kind", "simulation_jobs", ["kind"], unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_simulation_jobs_kind", table_name="simulation_jobs")
    op.drop_column("simulation_jobs", "kind")
