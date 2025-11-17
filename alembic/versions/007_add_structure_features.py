"""Add StructureFeatures model for ML feature caching (Session 14)

This migration creates a table for caching pre-computed ML features for structures,
avoiding recomputation for GNN inference and training.

Revision ID: 007_add_structure_features
Revises: 006_add_design_campaigns
Create Date: 2025-11-17 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '007_add_structure_features'
down_revision: Union[str, None] = '006_add_design_campaigns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create structure_features table for caching ML features.

    This table stores:
    1. Graph representations (neighbor lists, bond distances) for GNN models
    2. Scalar features (composition, avg properties) for traditional ML
    3. Feature extraction parameters for reproducibility
    4. Version tracking in case feature extraction changes
    """

    # Create structure_features table
    op.create_table(
        'structure_features',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'structure_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('structures.id', ondelete='CASCADE'),
            nullable=False,
            unique=True,  # One features entry per structure
            index=True,
            comment="Structure this feature set belongs to"
        ),
        sa.Column(
            'graph_repr',
            postgresql.JSONB(),
            nullable=False,
            comment="Graph representation for GNN input"
        ),
        sa.Column(
            'scalar_features',
            postgresql.JSONB(),
            nullable=False,
            comment="Scalar features for traditional ML"
        ),
        sa.Column(
            'extraction_params',
            postgresql.JSONB(),
            nullable=True,
            comment="Parameters used during extraction (cutoff_radius, etc.)"
        ),
        sa.Column(
            'feature_version',
            sa.String(length=50),
            nullable=False,
            server_default='1.0.0',
            comment="Version of feature extraction code"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(),
            nullable=False,
            server_default=sa.text('now()'),
            comment="When features were computed"
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            nullable=False,
            server_default=sa.text('now()'),
            onupdate=sa.text('now()'),
            comment="When features were last updated"
        ),
    )

    # Add index on feature_version for querying by version
    op.create_index(
        'ix_structure_features_version',
        'structure_features',
        ['feature_version']
    )

    # Add lattice_vectors column to structures table (if not exists)
    # This is used by the feature extractor
    op.add_column(
        'structures',
        sa.Column(
            'lattice_vectors',
            postgresql.JSONB(),
            nullable=True,
            comment="Lattice vectors as 3x3 matrix"
        )
    )


def downgrade() -> None:
    """
    Remove structure_features table and related changes.
    """

    # Drop lattice_vectors column from structures
    op.drop_column('structures', 'lattice_vectors')

    # Drop index
    op.drop_index('ix_structure_features_version', table_name='structure_features')

    # Drop structure_features table
    op.drop_table('structure_features')
