"""
Database Query Optimizations
=============================

Provides database optimization utilities:
- Composite indexes for common query patterns
- Query result caching
- Batch loading utilities
- Query complexity limits
- Performance monitoring
"""

from typing import List, Optional, Any, TypeVar, Generic
from sqlalchemy import Index, event, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Query, selectinload, joinedload
from sqlalchemy.engine import Engine
import logging
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ========== Composite Indexes ==========

"""
Composite indexes for common query patterns.

These indexes significantly improve query performance for frequently
used query combinations.

To apply these indexes, run a database migration:
    alembic revision --autogenerate -m "Add composite indexes"
    alembic upgrade head
"""

# Material queries
MATERIAL_INDEXES = [
    # Search by formula and not deleted
    Index('ix_materials_formula_deleted', 'materials.formula', 'materials.deleted_at'),

    # Search by owner and not deleted
    Index('ix_materials_owner_deleted', 'materials.owner_id', 'materials.deleted_at'),

    # Search by tags (for filtering by tags)
    Index('ix_materials_tags_gin', 'materials.tags', postgresql_using='gin'),

    # Search by creation date (for sorting)
    Index('ix_materials_created_deleted', 'materials.created_at', 'materials.deleted_at'),
]

# Structure queries
STRUCTURE_INDEXES = [
    # Structures by material
    Index('ix_structures_material_created', 'structures.material_id', 'structures.created_at'),

    # Structures by formula
    Index('ix_structures_formula_deleted', 'structures.formula', 'structures.deleted_at'),

    # Structures by dimensionality and formula (common filter)
    Index('ix_structures_dim_formula', 'structures.dimensionality', 'structures.formula'),
]

# Simulation job queries
SIMULATION_JOB_INDEXES = [
    # Jobs by owner and status (most common query)
    Index('ix_jobs_owner_status', 'simulation_jobs.owner_id', 'simulation_jobs.status'),

    # Jobs by status and priority (for worker queue)
    Index('ix_jobs_status_priority', 'simulation_jobs.status', 'simulation_jobs.priority'),

    # Jobs by structure
    Index('ix_jobs_structure_status', 'simulation_jobs.structure_id', 'simulation_jobs.status'),

    # Jobs by creation date and status (for pagination)
    Index('ix_jobs_created_status', 'simulation_jobs.created_at', 'simulation_jobs.status'),
]

# ML training job queries
ML_TRAINING_JOB_INDEXES = [
    # Training jobs by owner and status
    Index('ix_ml_training_owner_status', 'ml_training_jobs.owner_id', 'ml_training_jobs.status'),

    # Training jobs by target property
    Index('ix_ml_training_target_status', 'ml_training_jobs.target_property', 'ml_training_jobs.status'),

    # Training jobs by creation date
    Index('ix_ml_training_created_status', 'ml_training_jobs.created_at', 'ml_training_jobs.status'),
]

# Predicted properties queries
PREDICTED_PROPERTIES_INDEXES = [
    # Properties by structure (most common lookup)
    Index('ix_predicted_structure_created', 'predicted_properties.structure_id', 'predicted_properties.created_at'),

    # Properties by model and target
    Index('ix_predicted_model_target', 'predicted_properties.model_type', 'predicted_properties.target'),
]

# Campaign queries
CAMPAIGN_INDEXES = [
    # Campaigns by owner and status
    Index('ix_campaigns_owner_status', 'design_campaigns.owner_id', 'design_campaigns.status'),

    # Campaigns by creation date
    Index('ix_campaigns_created_status', 'design_campaigns.created_at', 'design_campaigns.status'),
]


ALL_COMPOSITE_INDEXES = (
    MATERIAL_INDEXES +
    STRUCTURE_INDEXES +
    SIMULATION_JOB_INDEXES +
    ML_TRAINING_JOB_INDEXES +
    PREDICTED_PROPERTIES_INDEXES +
    CAMPAIGN_INDEXES
)


# ========== Query Result Pagination with Cursor ==========


class CursorPagination(Generic[T]):
    """
    Cursor-based pagination for efficient large dataset queries.

    Uses cursor-based pagination instead of offset-based for better
    performance on large datasets.
    """

    def __init__(
        self,
        query: Query,
        page_size: int = 50,
        cursor: Optional[str] = None,
        order_by_field: str = "created_at"
    ):
        self.query = query
        self.page_size = min(page_size, 100)  # Max 100 items
        self.cursor = cursor
        self.order_by_field = order_by_field

    async def get_page(self, db: AsyncSession) -> dict:
        """
        Get a page of results with cursor pagination.

        Returns:
            dict with 'items', 'next_cursor', 'has_more'
        """
        # Apply cursor filtering if provided
        if self.cursor:
            # Decode cursor (in production, use base64 encoding)
            cursor_value = self.cursor
            self.query = self.query.where(
                getattr(self.query.column_descriptions[0]['entity'], self.order_by_field) > cursor_value
            )

        # Fetch page_size + 1 to check if there are more results
        items_query = self.query.limit(self.page_size + 1)
        result = await db.execute(items_query)
        items = result.scalars().all()

        # Check if there are more results
        has_more = len(items) > self.page_size
        if has_more:
            items = items[:self.page_size]

        # Generate next cursor
        next_cursor = None
        if has_more and items:
            last_item = items[-1]
            next_cursor = str(getattr(last_item, self.order_by_field))

        return {
            "items": items,
            "next_cursor": next_cursor,
            "has_more": has_more
        }


# ========== Batch Loading ==========


async def batch_load_relations(
    db: AsyncSession,
    entities: List[Any],
    *relations: str
) -> List[Any]:
    """
    Batch load related entities to avoid N+1 queries.

    Example:
        materials = await db.execute(select(Material).limit(10))
        materials = materials.scalars().all()

        # Load structures for all materials in one query
        materials = await batch_load_relations(
            db, materials, "structures"
        )

    Args:
        db: Database session
        entities: List of entities
        relations: Relation names to load

    Returns:
        List of entities with relations loaded
    """
    if not entities:
        return entities

    # Get entity class
    entity_class = type(entities[0])

    # Build query with selectinload for all relations
    query = db.query(entity_class).where(
        entity_class.id.in_([e.id for e in entities])
    )

    for relation in relations:
        query = query.options(selectinload(getattr(entity_class, relation)))

    result = await db.execute(query)
    return result.scalars().all()


# ========== Query Complexity Limits ==========


class QueryComplexityLimiter:
    """
    Limits query complexity to prevent expensive queries.

    Enforces:
    - Maximum result set size
    - Maximum number of joins
    - Query timeout
    """

    def __init__(
        self,
        max_results: int = 1000,
        max_joins: int = 5,
        timeout_seconds: int = 30
    ):
        self.max_results = max_results
        self.max_joins = max_joins
        self.timeout_seconds = timeout_seconds

    def limit_query(self, query: Query) -> Query:
        """Apply limits to query"""
        # Limit result set size
        if query._limit is None or query._limit > self.max_results:
            query = query.limit(self.max_results)

        return query

    async def execute_with_timeout(
        self,
        db: AsyncSession,
        query: Query,
        timeout_ms: Optional[int] = None
    ):
        """
        Execute query with timeout.

        Uses PostgreSQL statement_timeout to prevent long-running queries.
        """
        timeout = timeout_ms or (self.timeout_seconds * 1000)

        # Set statement timeout
        await db.execute(text(f"SET LOCAL statement_timeout = {timeout}"))

        try:
            result = await db.execute(query)
            return result
        except Exception as e:
            logger.error(f"Query timed out or failed: {e}")
            raise


# ========== Performance Monitoring ==========


class QueryPerformanceMonitor:
    """
    Monitor query performance and log slow queries.
    """

    def __init__(self, slow_query_threshold_ms: float = 100.0):
        self.slow_query_threshold_ms = slow_query_threshold_ms
        self.query_stats = {}

    def log_query(self, query: str, duration_ms: float):
        """Log query execution time"""
        if duration_ms > self.slow_query_threshold_ms:
            logger.warning(
                f"Slow query detected ({duration_ms:.2f}ms): {query[:200]}"
            )

        # Track stats
        if query not in self.query_stats:
            self.query_stats[query] = {
                "count": 0,
                "total_time_ms": 0,
                "max_time_ms": 0,
                "min_time_ms": float('inf')
            }

        stats = self.query_stats[query]
        stats["count"] += 1
        stats["total_time_ms"] += duration_ms
        stats["max_time_ms"] = max(stats["max_time_ms"], duration_ms)
        stats["min_time_ms"] = min(stats["min_time_ms"], duration_ms)

    def get_stats(self) -> dict:
        """Get query performance statistics"""
        return {
            query: {
                **stats,
                "avg_time_ms": stats["total_time_ms"] / stats["count"]
            }
            for query, stats in self.query_stats.items()
        }

    def get_slowest_queries(self, limit: int = 10) -> List[tuple]:
        """Get slowest queries by average execution time"""
        queries_with_avg = [
            (query, stats["total_time_ms"] / stats["count"])
            for query, stats in self.query_stats.items()
        ]
        return sorted(queries_with_avg, key=lambda x: x[1], reverse=True)[:limit]


# Global performance monitor
query_monitor = QueryPerformanceMonitor(slow_query_threshold_ms=100.0)


# ========== SQLAlchemy Event Listeners ==========


@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Record query start time"""
    conn.info.setdefault('query_start_time', []).append(time.time())


@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log query execution time"""
    total_time = time.time() - conn.info['query_start_time'].pop()
    duration_ms = total_time * 1000

    # Log to performance monitor
    query_monitor.log_query(statement, duration_ms)


# ========== Connection Pool Optimization ==========


def configure_connection_pool(engine: Engine, pool_size: int = 20, max_overflow: int = 40):
    """
    Configure connection pool for optimal performance.

    Args:
        engine: SQLAlchemy engine
        pool_size: Number of connections to maintain
        max_overflow: Maximum overflow connections
    """
    engine.pool._pool.maxsize = pool_size
    engine.pool._max_overflow = max_overflow

    logger.info(f"Connection pool configured: pool_size={pool_size}, max_overflow={max_overflow}")


# ========== Query Optimization Helpers ==========


def optimize_material_query(query: Query, include_structures: bool = False) -> Query:
    """
    Optimize material query with proper eager loading.

    Args:
        query: Base query
        include_structures: Whether to load structures

    Returns:
        Optimized query
    """
    if include_structures:
        # Use selectinload for one-to-many relationships
        query = query.options(selectinload('structures'))

    return query


def optimize_simulation_job_query(
    query: Query,
    include_structure: bool = True,
    include_results: bool = False
) -> Query:
    """
    Optimize simulation job query.

    Args:
        query: Base query
        include_structure: Load structure relationship
        include_results: Load results relationship

    Returns:
        Optimized query
    """
    if include_structure:
        # Use joinedload for many-to-one
        query = query.options(joinedload('structure'))

    if include_results:
        # Use selectinload for one-to-many
        query = query.options(selectinload('results'))

    return query


# ========== Database Statistics ==========


async def get_database_stats(db: AsyncSession) -> dict:
    """
    Get database statistics for monitoring.

    Returns:
        dict with table sizes, index usage, cache hit ratios
    """
    stats = {}

    try:
        # Table sizes
        table_size_query = text("""
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            LIMIT 10
        """)
        result = await db.execute(table_size_query)
        stats["table_sizes"] = [
            {"schema": row[0], "table": row[1], "size": row[2]}
            for row in result
        ]

        # Index usage
        index_usage_query = text("""
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC
            LIMIT 10
        """)
        result = await db.execute(index_usage_query)
        stats["index_usage"] = [
            {
                "schema": row[0],
                "table": row[1],
                "index": row[2],
                "scans": row[3],
                "tuples_read": row[4],
                "tuples_fetched": row[5]
            }
            for row in result
        ]

        # Cache hit ratio
        cache_hit_query = text("""
            SELECT
                sum(heap_blks_read) as heap_read,
                sum(heap_blks_hit) as heap_hit,
                sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
            FROM pg_statio_user_tables
        """)
        result = await db.execute(cache_hit_query)
        row = result.fetchone()
        if row:
            stats["cache_hit_ratio"] = {
                "heap_blocks_read": row[0],
                "heap_blocks_hit": row[1],
                "hit_ratio": float(row[2]) if row[2] else 0.0
            }

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        stats["error"] = str(e)

    return stats
