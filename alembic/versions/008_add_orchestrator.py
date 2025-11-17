"""Add orchestrator models for AGI control plane

This migration creates tables for the orchestrator service that enables
autonomous control of NANO-OS by external LLM/AI agents.

Revision ID: 008_add_orchestrator
Revises: 007_add_structure_features
Create Date: 2025-01-17 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '008_add_orchestrator'
down_revision: Union[str, None] = '007_add_structure_features'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create orchestrator tables for AGI control plane.

    The orchestrator enables:
    1. Autonomous decision-making for campaigns
    2. Intelligent model retraining scheduling
    3. Simulation and experiment orchestration
    4. LLM/agent control interface
    """

    # Create orchestrator_state table
    op.create_table(
        'orchestrator_state',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('name', sa.Text(), nullable=False, unique=True, index=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column(
            'mode',
            sa.Text(),
            nullable=False,
            server_default='MANUAL',
            comment="Operation mode: MANUAL, SCHEDULED, CONTINUOUS, PAUSED"
        ),
        sa.Column(
            'config',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Business rules: budgets, thresholds, priorities"
        ),
        sa.Column(
            'last_run_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of last orchestrator run"
        ),
        sa.Column(
            'last_training_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of last model training trigger"
        ),
        sa.Column(
            'last_experiment_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp of last experiment scheduling"
        ),
        sa.Column(
            'run_count',
            sa.Integer(),
            nullable=False,
            server_default='0',
            comment="Total number of orchestrator runs"
        ),
        sa.Column(
            'total_simulations_launched',
            sa.Integer(),
            nullable=False,
            server_default='0',
            comment="Total simulations launched"
        ),
        sa.Column(
            'total_experiments_launched',
            sa.Integer(),
            nullable=False,
            server_default='0',
            comment="Total experiments launched"
        ),
        sa.Column(
            'total_trainings_launched',
            sa.Integer(),
            nullable=False,
            server_default='0',
            comment="Total model trainings launched"
        ),
        sa.Column(
            'is_active',
            sa.Boolean(),
            nullable=False,
            server_default='true',
            comment="Whether orchestrator is active"
        ),
        sa.Column(
            'error_message',
            sa.Text(),
            nullable=True,
            comment="Last error message if any"
        ),
        sa.Column(
            'stats',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Current system statistics"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
        ),
        sa.Column(
            'metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional metadata"
        ),
    )

    # Create orchestrator_runs table
    op.create_table(
        'orchestrator_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'orchestrator_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('orchestrator_state.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
        ),
        sa.Column(
            'started_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
            index=True,
            comment="When the orchestrator run started"
        ),
        sa.Column(
            'completed_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When the orchestrator run completed"
        ),
        sa.Column(
            'duration_seconds',
            sa.Float(),
            nullable=True,
            comment="Duration of the run in seconds"
        ),
        sa.Column(
            'actions',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Actions taken: campaigns advanced, simulations/experiments launched"
        ),
        sa.Column(
            'success',
            sa.Boolean(),
            nullable=False,
            server_default='true',
            index=True,
            comment="Whether the run succeeded"
        ),
        sa.Column(
            'error_message',
            sa.Text(),
            nullable=True,
            comment="Error message if run failed"
        ),
        sa.Column(
            'stats_before',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="System statistics before run"
        ),
        sa.Column(
            'stats_after',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="System statistics after run"
        ),
        sa.Column(
            'triggered_by',
            sa.Text(),
            nullable=True,
            index=True,
            comment="What triggered this run: schedule, manual, api, agent"
        ),
        sa.Column(
            'trigger_context',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional context about trigger"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
        ),
    )

    # Create agent_commands table
    op.create_table(
        'agent_commands',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'agent_id',
            sa.Text(),
            nullable=True,
            index=True,
            comment="Identifier for the agent/LLM"
        ),
        sa.Column(
            'command_type',
            sa.Text(),
            nullable=False,
            index=True,
            comment="Type of command: create_campaign, advance_campaign, etc."
        ),
        sa.Column(
            'payload',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            comment="Command payload/request data"
        ),
        sa.Column(
            'executed_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When the command was executed"
        ),
        sa.Column(
            'completed_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When the command completed"
        ),
        sa.Column(
            'duration_seconds',
            sa.Float(),
            nullable=True,
            comment="Duration of command execution"
        ),
        sa.Column(
            'success',
            sa.Boolean(),
            nullable=False,
            server_default='true',
            index=True,
            comment="Whether the command succeeded"
        ),
        sa.Column(
            'result',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Command result data"
        ),
        sa.Column(
            'error_message',
            sa.Text(),
            nullable=True,
            comment="Error message if command failed"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
            index=True,
        ),
        sa.Column(
            'metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Additional metadata"
        ),
    )

    # Create indexes for performance
    op.create_index(
        'ix_orchestrator_runs_started_at_desc',
        'orchestrator_runs',
        [sa.text('started_at DESC')],
    )
    op.create_index(
        'ix_agent_commands_created_at_desc',
        'agent_commands',
        [sa.text('created_at DESC')],
    )
    op.create_index(
        'ix_agent_commands_agent_id_created_at',
        'agent_commands',
        ['agent_id', sa.text('created_at DESC')],
    )


def downgrade() -> None:
    """Remove orchestrator tables."""

    # Drop indexes
    op.drop_index('ix_agent_commands_agent_id_created_at', table_name='agent_commands')
    op.drop_index('ix_agent_commands_created_at_desc', table_name='agent_commands')
    op.drop_index('ix_orchestrator_runs_started_at_desc', table_name='orchestrator_runs')

    # Drop tables in reverse order
    op.drop_table('agent_commands')
    op.drop_table('orchestrator_runs')
    op.drop_table('orchestrator_state')
