"""Add predicted_properties table for ML predictions

This migration adds support for ML-based property prediction by creating
the predicted_properties table. This table stores:
- Predicted material properties (bandgap, formation energy, stability, etc.)
- ML model information (name, version)
- Confidence scores for predictions
- Metadata about the prediction process

The table supports:
- Multiple predictions per structure (different models, versions)
- Efficient querying by structure and model
- Tracking of prediction history

Revision ID: 002_add_predicted_properties
Revises: 001_initial_schema
Create Date: 2025-11-16 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_add_predicted_properties'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add predicted_properties table.

    Creates a table to store ML-predicted material properties with:
    - Foreign key to structures table
    - Model name and version tracking
    - JSON columns for properties and confidence scores
    - Composite indexes for efficient queries
    """
    # Create predicted_properties table
    op.create_table(
        'predicted_properties',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('structure_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column(
            'properties',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Predicted material properties"
        ),
        sa.Column(
            'confidence_scores',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Confidence scores for each predicted property"
        ),
        sa.Column(
            'metadata',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Additional metadata about the prediction"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
            comment="When the prediction was made"
        ),
        # Foreign key constraint
        sa.ForeignKeyConstraint(
            ['structure_id'],
            ['structures.id'],
            name='fk_predicted_properties_structure_id',
            ondelete='CASCADE'
        ),
    )

    # Create indexes
    # Index on structure_id for finding all predictions for a structure
    op.create_index(
        'ix_predicted_properties_structure_id',
        'predicted_properties',
        ['structure_id']
    )

    # Index on model_name for finding all predictions by a specific model
    op.create_index(
        'ix_predicted_properties_model_name',
        'predicted_properties',
        ['model_name']
    )

    # Index on created_at for time-based queries
    op.create_index(
        'ix_predicted_properties_created_at',
        'predicted_properties',
        ['created_at']
    )

    # Composite index for finding latest prediction for structure + model
    op.create_index(
        'ix_predicted_properties_structure_model',
        'predicted_properties',
        ['structure_id', 'model_name', 'created_at'],
        unique=False
    )

    # Composite index for queries by model and version
    op.create_index(
        'ix_predicted_properties_model',
        'predicted_properties',
        ['model_name', 'model_version', 'created_at'],
        unique=False
    )


def downgrade() -> None:
    """
    Remove predicted_properties table.

    Drops the table and all associated indexes.
    """
    # Drop indexes first
    op.drop_index('ix_predicted_properties_model', table_name='predicted_properties')
    op.drop_index('ix_predicted_properties_structure_model', table_name='predicted_properties')
    op.drop_index('ix_predicted_properties_created_at', table_name='predicted_properties')
    op.drop_index('ix_predicted_properties_model_name', table_name='predicted_properties')
    op.drop_index('ix_predicted_properties_structure_id', table_name='predicted_properties')

    # Drop table
    op.drop_table('predicted_properties')
