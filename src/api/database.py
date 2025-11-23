"""
Database connection and session management for NANO-OS.

This module provides:
- Async SQLAlchemy engine and session factory
- Connection pooling with proper configuration
- Database lifecycle management (init_db, close_db)
- Dependency injection for FastAPI routes

Future sessions will extend this with:
- Connection health checks
- Read replicas support
- Database sharding for multi-tenant deployments
"""

from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event, text, create_engine, Engine
import logging
from contextlib import contextmanager

from .config import settings

logger = logging.getLogger(__name__)

# Base class for all ORM models
Base = declarative_base()

# Global engine and session factory
engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None

# Synchronous engine and session factory for Celery tasks
sync_engine: Engine | None = None
sync_session_factory: sessionmaker[Session] | None = None


def create_engine_with_pool() -> AsyncEngine:
    """
    Create async SQLAlchemy engine with appropriate connection pooling.

    For production: Uses QueuePool with conservative settings
    For testing: Uses NullPool to avoid connection leaks
    """
    pool_class = NullPool if settings.TESTING else QueuePool

    engine_args = {
        "echo": settings.DB_ECHO,
        "pool_pre_ping": True,  # Verify connections before using
        "pool_recycle": 3600,   # Recycle connections after 1 hour
    }

    if pool_class == QueuePool:
        engine_args.update({
            "poolclass": QueuePool,
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": 30,
        })
    else:
        engine_args["poolclass"] = NullPool

    return create_async_engine(
        settings.DATABASE_URL,
        **engine_args
    )


def create_sync_engine_with_pool() -> Engine:
    """
    Create synchronous SQLAlchemy engine for Celery tasks.

    Uses similar pooling configuration as async engine.
    """
    pool_class = NullPool if settings.TESTING else QueuePool

    # Convert async database URL to sync (postgresql+asyncpg -> postgresql+psycopg2)
    sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

    engine_args = {
        "echo": settings.DB_ECHO,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }

    if pool_class == QueuePool:
        engine_args.update({
            "poolclass": QueuePool,
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": 30,
        })
    else:
        engine_args["poolclass"] = NullPool

    return create_engine(
        sync_url,
        **engine_args
    )


async def init_db() -> None:
    """
    Initialize database connection pool.

    Called during FastAPI startup (lifespan context).
    Creates engine, session factory, and optionally creates tables.
    """
    global engine, async_session_factory, sync_engine, sync_session_factory

    logger.info("Initializing database connection pool...")

    # Initialize async engine and session factory
    engine = create_engine_with_pool()
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Initialize sync engine and session factory for Celery tasks
    sync_engine = create_sync_engine_with_pool()
    sync_session_factory = sessionmaker(
        sync_engine,
        class_=Session,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Register event listeners
    @event.listens_for(engine.sync_engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        """Set connection parameters on connect."""
        logger.debug(f"New database connection established: {id(dbapi_conn)}")

    # Enable pgvector extension if needed
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("Enabled pgvector extension")

    # In development, create tables (in production, use migrations)
    if settings.DB_AUTO_CREATE_TABLES:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified")

    logger.info("Database initialization complete (async + sync)")


async def close_db() -> None:
    """
    Close database connection pool.

    Called during FastAPI shutdown (lifespan context).
    Properly disposes engine and closes all connections.
    """
    global engine, async_session_factory, sync_engine, sync_session_factory

    if engine:
        logger.info("Closing database connection pool...")
        await engine.dispose()
        engine = None
        async_session_factory = None
        logger.info("Async database connection pool closed")

    if sync_engine:
        logger.info("Closing sync database connection pool...")
        sync_engine.dispose()
        sync_engine = None
        sync_session_factory = None
        logger.info("Sync database connection pool closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage in routes:
        @router.get("/materials")
        async def list_materials(db: AsyncSession = Depends(get_db)):
            ...

    Ensures:
    - Proper session lifecycle (commit on success, rollback on error)
    - Automatic cleanup
    - Connection returns to pool
    """
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """
    Context manager for database sessions (for use outside FastAPI routes).

    Usage:
        async with get_db_context() as db:
            result = await db.execute(select(Material))
    """
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@contextmanager
def get_sync_db():
    """
    Context manager for synchronous database sessions (for Celery tasks).

    Usage:
        with get_sync_db() as db:
            result = db.execute(select(Material))
    """
    if not sync_session_factory:
        raise RuntimeError("Sync database not initialized. Call init_db() first.")

    session = sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def check_db_health() -> dict:
    """
    Check database health for monitoring/health endpoints.

    Returns:
        dict with status, latency, and connection pool stats
    """
    if not engine:
        return {
            "status": "unhealthy",
            "error": "Database not initialized"
        }

    try:
        import time
        start = time.time()

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        latency_ms = (time.time() - start) * 1000

        pool = engine.pool
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "pool": {
                "size": pool.size() if hasattr(pool, 'size') else None,
                "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else None,
                "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else None,
                "overflow": pool.overflow() if hasattr(pool, 'overflow') else None,
            }
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
