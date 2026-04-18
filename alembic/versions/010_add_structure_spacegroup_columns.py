"""Add structures.space_group, space_group_number, crystal_system, density

Phase 1 / Session 1.1 populates these columns from pymatgen's
SpacegroupAnalyzer at parse time. Keeping the columns nullable because:

- Molecular / XYZ inputs have no spacegroup.
- SpacegroupAnalyzer can raise at pathological inputs; we'd rather fall
  back to ``NULL`` than refuse the row.

Revision ID: 010_add_structure_spacegroup_columns
Revises: 009_rename_metadata_and_add_structure_hash
Create Date: 2026-04-17
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010_add_structure_spacegroup_columns"
down_revision: Union[str, None] = "009_rename_metadata_and_add_structure_hash"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "structures",
        sa.Column(
            "space_group",
            sa.String(length=32),
            nullable=True,
            comment="International spacegroup symbol (e.g. 'Fd-3m').",
        ),
    )
    op.add_column(
        "structures",
        sa.Column(
            "space_group_number",
            sa.Integer(),
            nullable=True,
            comment="International spacegroup number 1–230.",
        ),
    )
    op.add_column(
        "structures",
        sa.Column(
            "crystal_system",
            sa.String(length=32),
            nullable=True,
            comment="cubic | tetragonal | orthorhombic | hexagonal | trigonal | "
            "monoclinic | triclinic",
        ),
    )
    op.add_column(
        "structures",
        sa.Column(
            "density",
            sa.Float(),
            nullable=True,
            comment="Mass density (g/cm³), computed from cell mass / volume.",
        ),
    )
    op.create_index(
        "ix_structures_space_group", "structures", ["space_group"], unique=False
    )
    op.create_index(
        "ix_structures_space_group_number",
        "structures",
        ["space_group_number"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_structures_space_group_number", table_name="structures")
    op.drop_index("ix_structures_space_group", table_name="structures")
    op.drop_column("structures", "density")
    op.drop_column("structures", "crystal_system")
    op.drop_column("structures", "space_group_number")
    op.drop_column("structures", "space_group")
