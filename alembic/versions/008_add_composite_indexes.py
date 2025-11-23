"""add composite indexes for query optimization

Revision ID: 008
Revises: 007
Create Date: 2025-01-23

Adds composite indexes for common query patterns to improve performance.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    """
    Add composite indexes for query optimization.

    These indexes significantly improve performance for common query patterns:
    - Material searches by formula/owner with deleted_at filter
    - Structure lookups by material_id
    - Simulation job queries by owner/status
    - ML training job queries by owner/status
    - Predicted properties lookups by structure_id
    """

    # Material indexes
    op.create_index(
        'ix_materials_formula_deleted',
        'materials',
        ['formula', 'deleted_at'],
        unique=False
    )
    op.create_index(
        'ix_materials_owner_deleted',
        'materials',
        ['owner_id', 'deleted_at'],
        unique=False
    )
    op.create_index(
        'ix_materials_tags_gin',
        'materials',
        ['tags'],
        unique=False,
        postgresql_using='gin'
    )
    op.create_index(
        'ix_materials_created_deleted',
        'materials',
        ['created_at', 'deleted_at'],
        unique=False
    )

    # Structure indexes
    op.create_index(
        'ix_structures_material_created',
        'structures',
        ['material_id', 'created_at'],
        unique=False
    )
    op.create_index(
        'ix_structures_formula_deleted',
        'structures',
        ['formula', 'deleted_at'],
        unique=False
    )
    op.create_index(
        'ix_structures_dim_formula',
        'structures',
        ['dimensionality', 'formula'],
        unique=False
    )

    # Simulation job indexes
    op.create_index(
        'ix_jobs_owner_status',
        'simulation_jobs',
        ['owner_id', 'status'],
        unique=False
    )
    op.create_index(
        'ix_jobs_status_priority',
        'simulation_jobs',
        ['status', 'priority'],
        unique=False
    )
    op.create_index(
        'ix_jobs_structure_status',
        'simulation_jobs',
        ['structure_id', 'status'],
        unique=False
    )
    op.create_index(
        'ix_jobs_created_status',
        'simulation_jobs',
        ['created_at', 'status'],
        unique=False
    )

    # ML training job indexes
    op.create_index(
        'ix_ml_training_owner_status',
        'ml_training_jobs',
        ['owner_id', 'status'],
        unique=False
    )
    op.create_index(
        'ix_ml_training_target_status',
        'ml_training_jobs',
        ['target_property', 'status'],
        unique=False
    )
    op.create_index(
        'ix_ml_training_created_status',
        'ml_training_jobs',
        ['created_at', 'status'],
        unique=False
    )

    # Predicted properties indexes
    op.create_index(
        'ix_predicted_structure_created',
        'predicted_properties',
        ['structure_id', 'created_at'],
        unique=False
    )
    op.create_index(
        'ix_predicted_model_target',
        'predicted_properties',
        ['model_type', 'target'],
        unique=False
    )

    # Campaign indexes
    op.create_index(
        'ix_campaigns_owner_status',
        'design_campaigns',
        ['owner_id', 'status'],
        unique=False
    )
    op.create_index(
        'ix_campaigns_created_status',
        'design_campaigns',
        ['created_at', 'status'],
        unique=False
    )


def downgrade():
    """Remove composite indexes"""

    # Campaign indexes
    op.drop_index('ix_campaigns_created_status', table_name='design_campaigns')
    op.drop_index('ix_campaigns_owner_status', table_name='design_campaigns')

    # Predicted properties indexes
    op.drop_index('ix_predicted_model_target', table_name='predicted_properties')
    op.drop_index('ix_predicted_structure_created', table_name='predicted_properties')

    # ML training job indexes
    op.drop_index('ix_ml_training_created_status', table_name='ml_training_jobs')
    op.drop_index('ix_ml_training_target_status', table_name='ml_training_jobs')
    op.drop_index('ix_ml_training_owner_status', table_name='ml_training_jobs')

    # Simulation job indexes
    op.drop_index('ix_jobs_created_status', table_name='simulation_jobs')
    op.drop_index('ix_jobs_structure_status', table_name='simulation_jobs')
    op.drop_index('ix_jobs_status_priority', table_name='simulation_jobs')
    op.drop_index('ix_jobs_owner_status', table_name='simulation_jobs')

    # Structure indexes
    op.drop_index('ix_structures_dim_formula', table_name='structures')
    op.drop_index('ix_structures_formula_deleted', table_name='structures')
    op.drop_index('ix_structures_material_created', table_name='structures')

    # Material indexes
    op.drop_index('ix_materials_created_deleted', table_name='materials')
    op.drop_index('ix_materials_tags_gin', table_name='materials')
    op.drop_index('ix_materials_owner_deleted', table_name='materials')
    op.drop_index('ix_materials_formula_deleted', table_name='materials')
