"""Add provenance_records table for complete audit trails

This migration adds comprehensive provenance tracking by creating the
provenance_records table. This table stores:
- Complete event history for all entities (jobs, predictions, structures, materials)
- Detailed context for each event (parameters, system info, errors, etc.)
- Timing information for performance analysis
- User actions and system operations

The table supports:
- Full audit trails for reproducibility
- Lineage tracking for derived data
- Performance monitoring and debugging
- Compliance and accountability

Revision ID: 003_add_provenance_tracking
Revises: 002_add_predicted_properties
Create Date: 2025-11-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003_add_provenance_tracking'
down_revision: Union[str, None] = '002_add_predicted_properties'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add provenance_records table.

    Creates a table to track all significant events and state changes with:
    - Entity type and ID for the tracked entity
    - Event type (CREATED, STARTED, COMPLETED, FAILED, etc.)
    - Timestamp for when the event occurred
    - Details JSON for flexible context storage
    - Composite indexes for efficient querying
    """
    # Create provenance_records table
    op.create_table(
        'provenance_records',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'entity_type',
            sa.String(length=50),
            nullable=False,
            comment="Type of entity being tracked (JOB, PREDICTION, STRUCTURE, etc.)"
        ),
        sa.Column(
            'entity_id',
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="ID of the entity being tracked"
        ),
        sa.Column(
            'event_type',
            sa.String(length=50),
            nullable=False,
            comment="Type of event (CREATED, STARTED, COMPLETED, FAILED, etc.)"
        ),
        sa.Column(
            'timestamp',
            sa.DateTime(timezone=True),
            nullable=False,
            comment="When the event occurred (UTC)"
        ),
        sa.Column(
            'details',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Detailed event information and context"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
            comment="When this record was created (UTC)"
        ),
    )

    # Create indexes for efficient queries

    # Index on entity_type for filtering by type
    op.create_index(
        'ix_provenance_records_entity_type',
        'provenance_records',
        ['entity_type']
    )

    # Index on entity_id for finding all records for an entity
    op.create_index(
        'ix_provenance_records_entity_id',
        'provenance_records',
        ['entity_id']
    )

    # Index on timestamp for time-based queries
    op.create_index(
        'ix_provenance_records_timestamp',
        'provenance_records',
        ['timestamp']
    )

    # Index on event_type for filtering by event type
    op.create_index(
        'ix_provenance_records_event_type',
        'provenance_records',
        ['event_type']
    )

    # Composite index for fetching all events for a specific entity (most common query)
    # This supports queries like: "Get all provenance records for job X"
    op.create_index(
        'ix_provenance_entity_lookup',
        'provenance_records',
        ['entity_type', 'entity_id', 'timestamp'],
        unique=False
    )

    # Composite index for querying events by type and time
    # This supports queries like: "Get all FAILED events in the last hour"
    op.create_index(
        'ix_provenance_event_timeline',
        'provenance_records',
        ['entity_type', 'event_type', 'timestamp'],
        unique=False
    )


def downgrade() -> None:
    """
    Remove provenance_records table.

    Drops the table and all associated indexes.
    """
    # Drop indexes first
    op.drop_index('ix_provenance_event_timeline', table_name='provenance_records')
    op.drop_index('ix_provenance_entity_lookup', table_name='provenance_records')
    op.drop_index('ix_provenance_records_event_type', table_name='provenance_records')
    op.drop_index('ix_provenance_records_timestamp', table_name='provenance_records')
    op.drop_index('ix_provenance_records_entity_id', table_name='provenance_records')
    op.drop_index('ix_provenance_records_entity_type', table_name='provenance_records')

    # Drop table
    op.drop_table('provenance_records')
