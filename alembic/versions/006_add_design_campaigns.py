"""Add design campaign models for AI-driven materials discovery

This migration creates tables for managing design campaigns and iterations,
enabling autonomous AI-driven materials discovery loops.

Revision ID: 006_add_design_campaigns
Revises: 005_add_multiscale_models
Create Date: 2025-11-17 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '006_add_design_campaigns'
down_revision: Union[str, None] = '005_add_multiscale_models'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create design campaign tables.

    Design campaigns enable AI agents to autonomously:
    1. Propose candidate structures
    2. Predict properties
    3. Run simulations
    4. Update database with results
    5. Iterate toward optimization goals
    """

    # Create design_campaigns table
    op.create_table(
        'design_campaigns',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('users.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
            comment="User who owns this campaign"
        ),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column(
            'status',
            sa.String(length=50),
            nullable=False,
            default='CREATED',
            comment="Status: CREATED, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED"
        ),
        sa.Column(
            'config',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Campaign configuration: targets, constraints, strategy"
        ),
        sa.Column(
            'max_iterations',
            sa.Integer(),
            nullable=False,
            default=10,
            comment="Maximum number of iterations to run"
        ),
        sa.Column(
            'current_iteration',
            sa.Integer(),
            nullable=False,
            default=0,
            comment="Current iteration number (0-based)"
        ),
        sa.Column(
            'best_score',
            sa.Float(),
            nullable=True,
            comment="Best score achieved so far (0-1)"
        ),
        sa.Column(
            'best_structure_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('structures.id', ondelete='SET NULL'),
            nullable=True,
            comment="Structure ID with best score"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
            index=True
        ),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Create indexes for design_campaigns
    op.create_index(
        'ix_design_campaigns_status',
        'design_campaigns',
        ['status']
    )
    op.create_index(
        'ix_design_campaigns_owner_status',
        'design_campaigns',
        ['owner_id', 'status']
    )

    # Create design_iterations table
    op.create_table(
        'design_iterations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'campaign_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('design_campaigns.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
            comment="Associated campaign"
        ),
        sa.Column(
            'iteration_index',
            sa.Integer(),
            nullable=False,
            comment="Iteration number (0-based)"
        ),
        sa.Column(
            'created_structures',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='[]',
            comment="Array of structure UUIDs created this iteration"
        ),
        sa.Column(
            'evaluated_structures',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='[]',
            comment="Array of structure UUIDs evaluated this iteration"
        ),
        sa.Column(
            'best_score_this_iter',
            sa.Float(),
            nullable=True,
            comment="Best score in this iteration"
        ),
        sa.Column(
            'best_structure_id_this_iter',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('structures.id', ondelete='SET NULL'),
            nullable=True,
            comment="Best structure from this iteration"
        ),
        sa.Column(
            'metrics',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Iteration metrics: scores, improvements, diversity"
        ),
        sa.Column(
            'strategy_used',
            sa.String(length=100),
            nullable=True,
            comment="Strategy used: random, bayesian, genetic, etc."
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()')
        ),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Create composite index for campaign iterations lookup
    op.create_index(
        'ix_design_iterations_campaign_index',
        'design_iterations',
        ['campaign_id', 'iteration_index'],
        unique=True
    )


def downgrade() -> None:
    """
    Drop design campaign tables.
    """
    # Drop indexes first
    op.drop_index('ix_design_iterations_campaign_index', table_name='design_iterations')
    op.drop_table('design_iterations')

    op.drop_index('ix_design_campaigns_owner_status', table_name='design_campaigns')
    op.drop_index('ix_design_campaigns_status', table_name='design_campaigns')
    op.drop_table('design_campaigns')
