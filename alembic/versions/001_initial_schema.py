"""Initial database schema for NANO-OS

This migration creates the complete initial database schema for the NANO-OS platform,
including:
- User management tables (users)
- Materials and structures (materials, structures)
- Workflow and simulation tables (workflow_templates, simulation_jobs, simulation_results)
- Vector embeddings for ML/search (vector_embeddings, structure_similarities)

The schema supports:
- PostgreSQL with pgvector extension
- Async SQLAlchemy with asyncpg driver
- UUID primary keys
- JSON/JSONB columns for flexible metadata
- Proper foreign keys with cascade behavior
- Indexes for query optimization

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-11-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Apply the initial schema to the database.

    Creates all tables, indexes, and constraints for the NANO-OS platform.
    Enables pgvector extension for vector similarity search.
    """
    # Enable pgvector extension (required for vector_embeddings table)
    # This is also done in env.py, but we include it here for completeness
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ============================================================================
    # Users Table
    # ============================================================================
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('oauth_provider', sa.String(length=50), nullable=True),
        sa.Column('oauth_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Users indexes
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    # ============================================================================
    # Materials Table
    # ============================================================================
    op.create_table(
        'materials',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('formula', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String(length=50)), nullable=True),
        sa.Column('composition', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('source', sa.String(length=255), nullable=True),
        sa.Column('external_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Materials indexes
    op.create_index('ix_materials_name', 'materials', ['name'])
    op.create_index('ix_materials_formula', 'materials', ['formula'])
    op.create_index('ix_materials_external_id', 'materials', ['external_id'])
    op.create_index('ix_materials_created_at', 'materials', ['created_at'])

    # ============================================================================
    # Structures Table
    # ============================================================================
    op.create_table(
        'structures',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('material_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('format', sa.String(length=50), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=True),
        sa.Column('lattice', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('atoms', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('dimensionality', sa.Integer(), nullable=True),
        sa.Column('num_atoms', sa.Integer(), nullable=True),
        sa.Column('formula', sa.String(length=100), nullable=True),
        sa.Column('a', sa.Float(), nullable=True),
        sa.Column('b', sa.Float(), nullable=True),
        sa.Column('c', sa.Float(), nullable=True),
        sa.Column('alpha', sa.Float(), nullable=True),
        sa.Column('beta', sa.Float(), nullable=True),
        sa.Column('gamma', sa.Float(), nullable=True),
        sa.Column('volume', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['material_id'], ['materials.id'], ondelete='CASCADE'),
    )

    # Structures indexes
    op.create_index('ix_structures_material_id', 'structures', ['material_id'])
    op.create_index('ix_structures_formula', 'structures', ['formula'])
    op.create_index('ix_structures_created_at', 'structures', ['created_at'])

    # ============================================================================
    # Workflow Templates Table
    # ============================================================================
    op.create_table(
        'workflow_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('engine', sa.String(length=50), nullable=False),
        sa.Column('engine_version', sa.String(length=50), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('default_parameters', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('default_resources', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('parameter_schema', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('documentation_url', sa.String(length=500), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    )

    # Workflow templates indexes
    op.create_index('ix_workflow_templates_name', 'workflow_templates', ['name'], unique=True)
    op.create_index('ix_workflow_templates_engine', 'workflow_templates', ['engine'])
    op.create_index('ix_workflow_templates_category', 'workflow_templates', ['category'])

    # ============================================================================
    # Simulation Jobs Table
    # ============================================================================
    op.create_table(
        'simulation_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('structure_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workflow_template_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('engine', sa.String(length=50), nullable=False),
        sa.Column('parameters', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('resources', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('current_step', sa.String(length=255), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('worker_id', sa.String(length=255), nullable=True),
        sa.Column('worker_hostname', sa.String(length=255), nullable=True),
        sa.Column('celery_task_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['structure_id'], ['structures.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['workflow_template_id'], ['workflow_templates.id']),
    )

    # Simulation jobs indexes
    op.create_index('ix_simulation_jobs_structure_id', 'simulation_jobs', ['structure_id'])
    op.create_index('ix_simulation_jobs_workflow_template_id', 'simulation_jobs', ['workflow_template_id'])
    op.create_index('ix_simulation_jobs_status', 'simulation_jobs', ['status'])
    op.create_index('ix_simulation_jobs_priority', 'simulation_jobs', ['priority'])
    op.create_index('ix_simulation_jobs_submitted_at', 'simulation_jobs', ['submitted_at'])
    op.create_index('ix_simulation_jobs_celery_task_id', 'simulation_jobs', ['celery_task_id'], unique=True)

    # ============================================================================
    # Simulation Results Table
    # ============================================================================
    op.create_table(
        'simulation_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('simulation_job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('summary', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('artifacts_path', sa.String(length=500), nullable=True),
        sa.Column('artifacts', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('detailed_results', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('convergence_reached', sa.Boolean(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['simulation_job_id'], ['simulation_jobs.id'], ondelete='CASCADE'),
    )

    # Simulation results indexes
    op.create_index('ix_simulation_results_simulation_job_id', 'simulation_results', ['simulation_job_id'], unique=True)

    # ============================================================================
    # Vector Embeddings Table
    # ============================================================================
    # Note: This table uses the pgvector extension for storing vector embeddings
    op.create_table(
        'vector_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=False),
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), nullable=False),
        # Vector column - using raw SQL type since pgvector might not be in sqlalchemy.dialects
        sa.Column('embedding', sa.Text(), nullable=False),  # Will be cast to vector(512) in raw SQL
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('embedding_dimension', sa.Integer(), nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
    )

    # Alter the embedding column to use the vector type
    # This is done separately because Alembic might not recognize the vector type directly
    op.execute("ALTER TABLE vector_embeddings ALTER COLUMN embedding TYPE vector(512) USING embedding::vector(512)")

    # Vector embeddings indexes
    op.create_index('ix_vector_embeddings_entity_type', 'vector_embeddings', ['entity_type'])
    op.create_index('ix_vector_embeddings_entity_id', 'vector_embeddings', ['entity_id'])
    op.create_index('ix_vector_embeddings_model_name', 'vector_embeddings', ['model_name'])
    op.create_index('ix_vector_embeddings_created_at', 'vector_embeddings', ['created_at'])
    op.create_index('ix_vector_embeddings_entity', 'vector_embeddings', ['entity_type', 'entity_id'])

    # Create IVFFLAT index for fast similarity search (optional, can be created later)
    # This requires data in the table, so we comment it out for initial migration
    # op.execute(
    #     "CREATE INDEX ix_vector_embeddings_embedding_ivfflat "
    #     "ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) "
    #     "WITH (lists = 100)"
    # )

    # ============================================================================
    # Structure Similarities Table
    # ============================================================================
    op.create_table(
        'structure_similarities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('structure_a_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('structure_b_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('metric', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
    )

    # Structure similarities indexes
    op.create_index('ix_structure_similarities_structure_a_id', 'structure_similarities', ['structure_a_id'])
    op.create_index('ix_structure_similarities_structure_b_id', 'structure_similarities', ['structure_b_id'])
    op.create_index(
        'ix_structure_similarities_pair',
        'structure_similarities',
        ['structure_a_id', 'structure_b_id'],
        unique=True
    )


def downgrade() -> None:
    """
    Revert the initial schema from the database.

    Drops all tables in reverse order of creation to respect foreign key constraints.
    """
    # Drop tables in reverse order
    op.drop_table('structure_similarities')
    op.drop_table('vector_embeddings')
    op.drop_table('simulation_results')
    op.drop_table('simulation_jobs')
    op.drop_table('workflow_templates')
    op.drop_table('structures')
    op.drop_table('materials')
    op.drop_table('users')

    # Note: We don't drop the pgvector extension as it might be used by other databases
    # If you want to drop it, uncomment the following line:
    # op.execute("DROP EXTENSION IF EXISTS vector")
