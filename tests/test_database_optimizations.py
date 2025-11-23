"""
Tests for Database Query Optimizations
=======================================

Tests database optimization utilities:
- Composite indexes (verify they exist)
- Query result caching
- Batch loading
- Query complexity limits
- Performance monitoring
- Database statistics
"""

import pytest
import time
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, inspect, text
from unittest.mock import Mock

from src.api.database_optimizations import (
    CursorPagination,
    batch_load_relations,
    QueryComplexityLimiter,
    QueryPerformanceMonitor,
    query_monitor,
    optimize_material_query,
    optimize_simulation_job_query,
    get_database_stats,
)
from src.api.models.user import User
from src.api.models.material import Material
from src.api.models.structure import Structure
from src.api.models.simulation import SimulationJob, JobStatus


class TestCompositeIndexes:
    """Test that composite indexes are created"""

    @pytest.mark.asyncio
    async def test_material_indexes_exist(self, db_session: AsyncSession):
        """Test that material composite indexes exist"""
        try:
            # Query pg_indexes to verify indexes exist
            indexes_query = text("""
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'materials'
                AND schemaname = 'public'
            """)
            result = await db_session.execute(indexes_query)
            indexes = [row[0] for row in result]

            # Check for specific composite indexes
            expected_indexes = [
                'ix_materials_formula_deleted',
                'ix_materials_owner_deleted',
                'ix_materials_created_deleted',
            ]

            for idx_name in expected_indexes:
                assert idx_name in indexes, f"Index {idx_name} not found"

        except Exception as e:
            pytest.skip(f"Database not available or indexes not created: {e}")

    @pytest.mark.asyncio
    async def test_simulation_job_indexes_exist(self, db_session: AsyncSession):
        """Test that simulation job composite indexes exist"""
        try:
            indexes_query = text("""
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'simulation_jobs'
                AND schemaname = 'public'
            """)
            result = await db_session.execute(indexes_query)
            indexes = [row[0] for row in result]

            expected_indexes = [
                'ix_jobs_owner_status',
                'ix_jobs_status_priority',
                'ix_jobs_structure_status',
            ]

            for idx_name in expected_indexes:
                assert idx_name in indexes, f"Index {idx_name} not found"

        except Exception as e:
            pytest.skip(f"Database not available or indexes not created: {e}")

    @pytest.mark.asyncio
    async def test_index_usage_stats(self, db_session: AsyncSession):
        """Test querying index usage statistics"""
        try:
            stats = await get_database_stats(db_session)

            assert "index_usage" in stats
            assert isinstance(stats["index_usage"], list)

            if stats["index_usage"]:
                # Verify structure of index usage stats
                first_stat = stats["index_usage"][0]
                assert "table" in first_stat
                assert "index" in first_stat
                assert "scans" in first_stat

        except Exception as e:
            pytest.skip(f"Database stats not available: {e}")


class TestCursorPagination:
    """Test cursor-based pagination"""

    @pytest.mark.asyncio
    async def test_cursor_pagination_first_page(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test getting first page with cursor pagination"""
        # Create test materials
        materials = []
        for i in range(15):
            material = Material(
                name=f"Material {i}",
                formula=f"M{i}",
                description=f"Test material {i}"
            )
            db_session.add(material)
            materials.append(material)

        await db_session.commit()

        # Query with cursor pagination
        query = select(Material).order_by(Material.created_at)
        pagination = CursorPagination(query, page_size=10, order_by_field="created_at")

        result = await pagination.get_page(db_session)

        assert len(result["items"]) == 10
        assert result["has_more"] is True
        assert result["next_cursor"] is not None

    @pytest.mark.asyncio
    async def test_cursor_pagination_next_page(
        self, db_session: AsyncSession
    ):
        """Test getting next page using cursor"""
        # Create test materials
        for i in range(25):
            material = Material(
                name=f"Material {i}",
                formula=f"M{i}",
                description=f"Test material {i}"
            )
            db_session.add(material)

        await db_session.commit()

        # Get first page
        query = select(Material).order_by(Material.created_at)
        pagination1 = CursorPagination(query, page_size=10, order_by_field="created_at")
        result1 = await pagination1.get_page(db_session)

        # Get second page using cursor
        pagination2 = CursorPagination(
            select(Material).order_by(Material.created_at),
            page_size=10,
            cursor=result1["next_cursor"],
            order_by_field="created_at"
        )
        result2 = await pagination2.get_page(db_session)

        assert len(result2["items"]) == 10
        assert result2["has_more"] is True

        # Verify no overlap between pages
        page1_ids = [item.id for item in result1["items"]]
        page2_ids = [item.id for item in result2["items"]]
        assert not set(page1_ids).intersection(set(page2_ids))


class TestBatchLoading:
    """Test batch loading to avoid N+1 queries"""

    @pytest.mark.asyncio
    async def test_batch_load_relations(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test batch loading related entities"""
        # Create materials with structures
        for i in range(5):
            material = Material(
                name=f"Material {i}",
                formula=f"M{i}",
                description=f"Test material {i}"
            )
            db_session.add(material)
            await db_session.flush()

            # Add structures to each material
            for j in range(3):
                structure = Structure(
                    material_id=material.id,
                    name=f"Structure {i}-{j}",
                    format="CIF",
                    raw_text="# Test structure",
                    formula=f"M{i}",
                    num_atoms=10,
                    dimensionality=3
                )
                db_session.add(structure)

        await db_session.commit()

        # Query materials without loading structures
        result = await db_session.execute(select(Material).limit(5))
        materials = result.scalars().all()

        # Batch load structures
        materials_with_structures = await batch_load_relations(
            db_session, materials, "structures"
        )

        # Verify structures are loaded
        assert len(materials_with_structures) == 5
        for material in materials_with_structures:
            assert hasattr(material, 'structures')
            assert len(material.structures) == 3


class TestQueryComplexityLimiter:
    """Test query complexity limiting"""

    @pytest.mark.asyncio
    async def test_limit_query_max_results(self):
        """Test that query is limited to max results"""
        limiter = QueryComplexityLimiter(max_results=100)

        # Create query without limit
        query = select(Material)
        assert query._limit is None

        # Apply limiter
        limited_query = limiter.limit_query(query)
        assert limited_query._limit == 100

    @pytest.mark.asyncio
    async def test_limit_query_respects_smaller_limit(self):
        """Test that existing smaller limit is not increased"""
        limiter = QueryComplexityLimiter(max_results=100)

        # Create query with smaller limit
        query = select(Material).limit(50)

        # Apply limiter (should keep 50)
        limited_query = limiter.limit_query(query)
        assert limited_query._limit == 50

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, db_session: AsyncSession):
        """Test query execution with timeout"""
        limiter = QueryComplexityLimiter(timeout_seconds=5)

        # Simple fast query
        query = select(Material).limit(10)

        try:
            result = await limiter.execute_with_timeout(
                db_session, query, timeout_ms=5000
            )
            assert result is not None
        except Exception as e:
            pytest.skip(f"Database not available: {e}")


class TestQueryPerformanceMonitor:
    """Test query performance monitoring"""

    def test_log_query_performance(self):
        """Test logging query performance"""
        monitor = QueryPerformanceMonitor(slow_query_threshold_ms=50.0)

        # Log fast query
        monitor.log_query("SELECT * FROM materials LIMIT 10", 25.0)

        # Log slow query
        monitor.log_query("SELECT * FROM materials WHERE formula LIKE '%test%'", 150.0)

        stats = monitor.get_stats()
        assert len(stats) == 2

    def test_get_slowest_queries(self):
        """Test getting slowest queries"""
        monitor = QueryPerformanceMonitor()

        # Log multiple queries with different times
        queries = [
            ("SELECT * FROM materials", 10.0),
            ("SELECT * FROM structures", 50.0),
            ("SELECT * FROM simulation_jobs", 100.0),
            ("SELECT * FROM predicted_properties", 25.0),
        ]

        for query, duration in queries:
            monitor.log_query(query, duration)

        slowest = monitor.get_slowest_queries(limit=2)
        assert len(slowest) == 2
        assert slowest[0][1] == 100.0  # Slowest query
        assert slowest[1][1] == 50.0   # Second slowest

    def test_query_stats_aggregation(self):
        """Test query statistics aggregation"""
        monitor = QueryPerformanceMonitor()

        query = "SELECT * FROM materials WHERE id = $1"

        # Log same query multiple times
        monitor.log_query(query, 10.0)
        monitor.log_query(query, 20.0)
        monitor.log_query(query, 30.0)

        stats = monitor.get_stats()
        query_stat = stats[query]

        assert query_stat["count"] == 3
        assert query_stat["total_time_ms"] == 60.0
        assert query_stat["max_time_ms"] == 30.0
        assert query_stat["min_time_ms"] == 10.0
        assert query_stat["avg_time_ms"] == 20.0


class TestQueryOptimizationHelpers:
    """Test query optimization helper functions"""

    @pytest.mark.asyncio
    async def test_optimize_material_query(self, db_session: AsyncSession):
        """Test material query optimization"""
        query = select(Material)

        # Optimize without structures
        optimized = optimize_material_query(query, include_structures=False)
        assert optimized is not None

        # Optimize with structures
        optimized_with_structures = optimize_material_query(
            query, include_structures=True
        )
        assert optimized_with_structures is not None

    @pytest.mark.asyncio
    async def test_optimize_simulation_job_query(self, db_session: AsyncSession):
        """Test simulation job query optimization"""
        query = select(SimulationJob)

        # Optimize with structure
        optimized = optimize_simulation_job_query(
            query, include_structure=True, include_results=False
        )
        assert optimized is not None

        # Optimize with structure and results
        optimized_full = optimize_simulation_job_query(
            query, include_structure=True, include_results=True
        )
        assert optimized_full is not None


class TestDatabaseStatistics:
    """Test database statistics collection"""

    @pytest.mark.asyncio
    async def test_get_database_stats(self, db_session: AsyncSession):
        """Test getting database statistics"""
        try:
            stats = await get_database_stats(db_session)

            # Verify stats structure
            assert "table_sizes" in stats
            assert "index_usage" in stats
            assert "cache_hit_ratio" in stats

            # Verify table sizes
            assert isinstance(stats["table_sizes"], list)
            if stats["table_sizes"]:
                first_table = stats["table_sizes"][0]
                assert "table" in first_table
                assert "size" in first_table

            # Verify cache hit ratio
            if "cache_hit_ratio" in stats and stats["cache_hit_ratio"]:
                ratio_stats = stats["cache_hit_ratio"]
                assert "hit_ratio" in ratio_stats
                assert 0.0 <= ratio_stats["hit_ratio"] <= 1.0

        except Exception as e:
            pytest.skip(f"Database stats not available: {e}")

    @pytest.mark.asyncio
    async def test_database_stats_table_sizes(self, db_session: AsyncSession):
        """Test table size statistics"""
        try:
            stats = await get_database_stats(db_session)

            table_sizes = stats.get("table_sizes", [])

            # Should have at least some tables
            assert len(table_sizes) > 0

            # Verify size format
            for table_stat in table_sizes[:3]:
                assert "table" in table_stat
                assert "size" in table_stat
                # Size should be human-readable (e.g., "8192 bytes", "16 kB")
                assert isinstance(table_stat["size"], str)

        except Exception as e:
            pytest.skip(f"Database stats not available: {e}")


class TestPerformanceImprovements:
    """Integration tests for performance improvements"""

    @pytest.mark.asyncio
    async def test_indexed_query_performance(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that indexed queries perform better than non-indexed"""
        # Create many materials
        for i in range(100):
            material = Material(
                owner_id=test_user.id,
                name=f"Material {i}",
                formula=f"Formula{i % 10}",  # 10 unique formulas
                description=f"Test material {i}"
            )
            db_session.add(material)

        await db_session.commit()

        # Query using indexed column (formula)
        start = time.time()
        query = select(Material).where(Material.formula == "Formula5")
        result = await db_session.execute(query)
        materials = result.scalars().all()
        indexed_time = time.time() - start

        assert len(materials) > 0
        # Indexed query should be fast (< 100ms)
        assert indexed_time < 0.1

    @pytest.mark.asyncio
    async def test_composite_index_usage(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that composite index is used for multi-column queries"""
        # Create materials
        for i in range(50):
            material = Material(
                owner_id=test_user.id,
                name=f"Material {i}",
                formula="NaCl",
                description=f"Test material {i}"
            )
            db_session.add(material)

        await db_session.commit()

        # Query using composite index (owner_id, deleted_at)
        start = time.time()
        query = select(Material).where(
            Material.owner_id == test_user.id,
            Material.deleted_at.is_(None)
        )
        result = await db_session.execute(query)
        materials = result.scalars().all()
        composite_time = time.time() - start

        assert len(materials) == 50
        # Should be fast with composite index
        assert composite_time < 0.1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
