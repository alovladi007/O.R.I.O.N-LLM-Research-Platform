"""
Alembic environment configuration for NANO-OS.

This module configures Alembic to work with async SQLAlchemy and the
NANO-OS database models. It handles:
- Async database connections
- Automatic model metadata detection
- Migration context configuration
- Online and offline migration modes

For more information:
https://alembic.sqlalchemy.org/en/latest/tutorial.html#the-migration-environment
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Add the project root to the path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Base and all models to register them with metadata
from src.api.database import Base
from src.api.models import (
    User,
    Material,
    Structure,
    WorkflowTemplate,
    SimulationJob,
    SimulationResult,
    VectorEmbedding,
)
from src.api.models.embedding import StructureSimilarity

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Override sqlalchemy.url with environment variable if set
# This allows us to use the same DATABASE_URL as the application
import os
database_url = os.getenv("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Important: render_as_batch for SQLite compatibility (if needed)
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Run migrations with the provided connection.

    This is called by both sync and async migration runners.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Compare types to detect column type changes
        compare_type=True,
        # Compare server defaults
        compare_server_default=True,
        # Render batch operations for better compatibility
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async support.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    This is an async version that works with asyncpg.
    """
    # Create async engine from configuration
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # Use NullPool for migrations
    )

    async with connectable.connect() as connection:
        # Enable pgvector extension before running migrations
        await connection.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await connection.commit()

        # Run migrations in a sync context
        await connection.run_sync(do_run_migrations)

    # Dispose of the connection pool
    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Wrapper that calls the async migration function.
    """
    asyncio.run(run_async_migrations())


# Determine which mode to run in
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
