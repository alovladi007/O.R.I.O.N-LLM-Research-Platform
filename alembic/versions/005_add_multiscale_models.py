"""Add multi-scale simulation models (mesoscale and continuum)

This migration creates tables for multi-scale simulations that bridge
atomic → mesoscale → continuum scales.

Revision ID: 005_add_multiscale_models
Revises: 004_add_multi_tenancy
Create Date: 2025-11-17 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005_add_multiscale_models'
down_revision: Union[str, None] = '004_add_multi_tenancy'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create mesoscale and continuum simulation tables.

    Mesoscale simulations (Phase Field, Monte Carlo, KMC) take results from
    atomic simulations and produce microstructure data.

    Continuum simulations (FEM, FVM, BEM) take mesoscale results and compute
    effective properties at macroscopic scales.
    """

    # Create mesoscale_simulation_jobs table
    op.create_table(
        'mesoscale_simulation_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('users.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
            comment="User who owns this job"
        ),
        sa.Column(
            'parent_atomic_job_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('simulation_jobs.id', ondelete='SET NULL'),
            nullable=True,
            index=True,
            comment="Parent atomic-scale simulation job"
        ),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column(
            'engine_type',
            sa.String(length=50),
            nullable=False,
            comment="Engine type: PHASE_FIELD, MONTE_CARLO, KMC"
        ),
        sa.Column(
            'status',
            sa.String(length=50),
            nullable=False,
            default='PENDING',
            comment="Job status: PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED"
        ),
        sa.Column(
            'parameters',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Simulation parameters"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
            index=True
        ),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True, index=True),
    )

    # Create indexes for mesoscale jobs
    op.create_index(
        'ix_mesoscale_simulation_jobs_status',
        'mesoscale_simulation_jobs',
        ['status']
    )
    op.create_index(
        'ix_mesoscale_simulation_jobs_engine_type',
        'mesoscale_simulation_jobs',
        ['engine_type']
    )

    # Create mesoscale_simulation_results table
    op.create_table(
        'mesoscale_simulation_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'job_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('mesoscale_simulation_jobs.id', ondelete='CASCADE'),
            nullable=False,
            unique=True,
            index=True,
            comment="Associated mesoscale job"
        ),
        sa.Column(
            'microstructure_metrics',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Grain size, phase fractions, etc."
        ),
        sa.Column(
            'domain_size',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Domain dimensions [nx, ny, nz]"
        ),
        sa.Column('timesteps', sa.Integer(), nullable=True),
        sa.Column(
            'output_files',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            server_default='[]',
            comment="List of output file paths"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()')
        ),
    )

    # Create continuum_simulation_jobs table
    op.create_table(
        'continuum_simulation_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'owner_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('users.id', ondelete='CASCADE'),
            nullable=False,
            index=True,
            comment="User who owns this job"
        ),
        sa.Column(
            'parent_mesoscale_job_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('mesoscale_simulation_jobs.id', ondelete='SET NULL'),
            nullable=True,
            index=True,
            comment="Parent mesoscale simulation job"
        ),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column(
            'engine_type',
            sa.String(length=50),
            nullable=False,
            comment="Engine type: FEM, FVM, BEM"
        ),
        sa.Column(
            'status',
            sa.String(length=50),
            nullable=False,
            default='PENDING',
            comment="Job status: PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED"
        ),
        sa.Column(
            'parameters',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=False,
            server_default='{}',
            comment="Simulation parameters"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()'),
            index=True
        ),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True, index=True),
    )

    # Create indexes for continuum jobs
    op.create_index(
        'ix_continuum_simulation_jobs_status',
        'continuum_simulation_jobs',
        ['status']
    )
    op.create_index(
        'ix_continuum_simulation_jobs_engine_type',
        'continuum_simulation_jobs',
        ['engine_type']
    )

    # Create continuum_simulation_results table
    op.create_table(
        'continuum_simulation_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            'job_id',
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey('continuum_simulation_jobs.id', ondelete='CASCADE'),
            nullable=False,
            unique=True,
            index=True,
            comment="Associated continuum job"
        ),
        sa.Column(
            'effective_properties',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Elastic moduli, thermal conductivity, etc."
        ),
        sa.Column(
            'mesh_info',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Mesh nodes, elements, etc."
        ),
        sa.Column(
            'solution_fields',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            comment="Displacement, stress, temperature fields, etc."
        ),
        sa.Column(
            'output_files',
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
            server_default='[]',
            comment="List of output file paths"
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('now()')
        ),
    )


def downgrade() -> None:
    """
    Drop multi-scale simulation tables.
    """
    # Drop tables in reverse order (results before jobs)
    op.drop_table('continuum_simulation_results')
    op.drop_index('ix_continuum_simulation_jobs_engine_type', table_name='continuum_simulation_jobs')
    op.drop_index('ix_continuum_simulation_jobs_status', table_name='continuum_simulation_jobs')
    op.drop_table('continuum_simulation_jobs')

    op.drop_table('mesoscale_simulation_results')
    op.drop_index('ix_mesoscale_simulation_jobs_engine_type', table_name='mesoscale_simulation_jobs')
    op.drop_index('ix_mesoscale_simulation_jobs_status', table_name='mesoscale_simulation_jobs')
    op.drop_table('mesoscale_simulation_jobs')
