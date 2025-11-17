"""Add multi-tenancy with owner_id to all models

This migration adds owner_id foreign keys to enable multi-tenant data isolation.
Users can only see and manage their own resources unless they are admins.

Revision ID: 004_add_multi_tenancy
Revises: 003_add_provenance_tracking
Create Date: 2025-11-17 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004_add_multi_tenancy'
down_revision: Union[str, None] = '003_add_provenance_tracking'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add owner_id columns to all main data tables.

    Adds foreign keys to users table for:
    - materials
    - structures
    - simulation_jobs
    - simulation_results
    - predicted_properties
    """

    # Add owner_id to materials
    op.add_column(
        'materials',
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,  # Temporarily nullable for migration
            comment="User who owns this material"
        )
    )
    op.create_index(
        'ix_materials_owner_id',
        'materials',
        ['owner_id']
    )
    op.create_foreign_key(
        'fk_materials_owner_id_users',
        'materials',
        'users',
        ['owner_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Add owner_id to structures
    op.add_column(
        'structures',
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="User who owns this structure"
        )
    )
    op.create_index(
        'ix_structures_owner_id',
        'structures',
        ['owner_id']
    )
    op.create_foreign_key(
        'fk_structures_owner_id_users',
        'structures',
        'users',
        ['owner_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Add owner_id to simulation_jobs
    op.add_column(
        'simulation_jobs',
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="User who owns this job"
        )
    )
    op.create_index(
        'ix_simulation_jobs_owner_id',
        'simulation_jobs',
        ['owner_id']
    )
    op.create_foreign_key(
        'fk_simulation_jobs_owner_id_users',
        'simulation_jobs',
        'users',
        ['owner_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Add owner_id to simulation_results
    op.add_column(
        'simulation_results',
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="User who owns this result"
        )
    )
    op.create_index(
        'ix_simulation_results_owner_id',
        'simulation_results',
        ['owner_id']
    )
    op.create_foreign_key(
        'fk_simulation_results_owner_id_users',
        'simulation_results',
        'users',
        ['owner_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Add owner_id to predicted_properties
    op.add_column(
        'predicted_properties',
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="User who owns this prediction"
        )
    )
    op.create_index(
        'ix_predicted_properties_owner_id',
        'predicted_properties',
        ['owner_id']
    )
    op.create_foreign_key(
        'fk_predicted_properties_owner_id_users',
        'predicted_properties',
        'users',
        ['owner_id'],
        ['id'],
        ondelete='CASCADE'
    )

    # Note: In production, you would run UPDATE statements here to set owner_id
    # for existing records, perhaps to a system/admin user. For new installations,
    # all records will be created with an owner from the start.

    # After data migration, you could make owner_id NOT NULL:
    # op.alter_column('materials', 'owner_id', nullable=False)
    # etc.


def downgrade() -> None:
    """
    Remove owner_id columns and foreign keys.
    """

    # Drop foreign keys and columns in reverse order
    op.drop_constraint('fk_predicted_properties_owner_id_users', 'predicted_properties', type_='foreignkey')
    op.drop_index('ix_predicted_properties_owner_id', table_name='predicted_properties')
    op.drop_column('predicted_properties', 'owner_id')

    op.drop_constraint('fk_simulation_results_owner_id_users', 'simulation_results', type_='foreignkey')
    op.drop_index('ix_simulation_results_owner_id', table_name='simulation_results')
    op.drop_column('simulation_results', 'owner_id')

    op.drop_constraint('fk_simulation_jobs_owner_id_users', 'simulation_jobs', type_='foreignkey')
    op.drop_index('ix_simulation_jobs_owner_id', table_name='simulation_jobs')
    op.drop_column('simulation_jobs', 'owner_id')

    op.drop_constraint('fk_structures_owner_id_users', 'structures', type_='foreignkey')
    op.drop_index('ix_structures_owner_id', table_name='structures')
    op.drop_column('structures', 'owner_id')

    op.drop_constraint('fk_materials_owner_id_users', 'materials', type_='foreignkey')
    op.drop_index('ix_materials_owner_id', table_name='materials')
    op.drop_column('materials', 'owner_id')
