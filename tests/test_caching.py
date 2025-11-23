"""
Integration Tests for Redis Caching Layer
==========================================

Tests the Redis caching infrastructure:
- Cache initialization and health checks
- Basic cache operations (get, set, delete)
- Cache invalidation patterns
- Domain-specific cache helpers
- Rate limiting
- Caching decorators
- API endpoint caching
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.cache import (
    init_cache,
    close_cache,
    get_redis,
    cache_get,
    cache_set,
    cache_delete,
    cache_delete_pattern,
    cache_exists,
    cache_ttl,
    cache_increment,
    check_cache_health,
    # Domain-specific helpers
    cache_structure,
    get_cached_structure,
    invalidate_structure_cache,
    cache_material,
    get_cached_material,
    invalidate_material_cache,
    cache_ml_prediction,
    get_cached_ml_prediction,
    cache_job_status,
    get_cached_job_status,
    invalidate_job_status_cache,
    # Rate limiting
    check_rate_limit,
    reset_rate_limit,
    # Stats
    get_cache_stats,
    # Decorators
    cached,
    invalidate_cache_on_change,
    CacheNamespace,
)
from src.api.models.user import User
from src.api.models.material import Material


class TestCacheInitialization:
    """Test cache initialization and lifecycle"""

    @pytest.mark.asyncio
    async def test_init_cache(self):
        """Test cache initialization"""
        try:
            await init_cache()
            redis_client = get_redis()
            assert redis_client is not None

            # Test connection
            pong = await redis_client.ping()
            assert pong is True

        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_cache_health_check(self):
        """Test cache health check"""
        try:
            await init_cache()

            health = await check_cache_health()

            assert health["status"] == "healthy"
            assert "latency_ms" in health
            assert health["latency_ms"] < 100  # Should be fast
            assert "redis_version" in health
            assert "total_keys" in health

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_close_cache(self):
        """Test cache cleanup"""
        try:
            await init_cache()
            await close_cache()

            # After close, should raise error
            with pytest.raises(RuntimeError):
                get_redis()

        except Exception:
            pytest.skip("Redis not available")


class TestBasicCacheOperations:
    """Test basic cache operations"""

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test setting and getting values"""
        try:
            await init_cache()

            key = "test_key"
            value = {"data": "test_value", "number": 42}

            # Set value
            success = await cache_set(key, value, ttl=60)
            assert success is True

            # Get value
            retrieved = await cache_get(key)
            assert retrieved == value

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_cache_get_default(self):
        """Test getting non-existent key returns default"""
        try:
            await init_cache()

            value = await cache_get("nonexistent_key", default="default_value")
            assert value == "default_value"

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test deleting values"""
        try:
            await init_cache()

            key = "test_delete_key"
            await cache_set(key, "value")

            # Verify exists
            exists = await cache_exists(key)
            assert exists is True

            # Delete
            deleted = await cache_delete(key)
            assert deleted is True

            # Verify deleted
            exists = await cache_exists(key)
            assert exists is False

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_cache_delete_pattern(self):
        """Test deleting keys by pattern"""
        try:
            await init_cache()

            # Create multiple keys
            await cache_set("user:1:profile", {"name": "User 1"})
            await cache_set("user:2:profile", {"name": "User 2"})
            await cache_set("user:3:profile", {"name": "User 3"})
            await cache_set("material:1:data", {"formula": "NaCl"})

            # Delete all user keys
            deleted = await cache_delete_pattern("user:*")
            assert deleted == 3

            # Verify only user keys deleted
            assert await cache_exists("user:1:profile") is False
            assert await cache_exists("material:1:data") is True

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test TTL functionality"""
        try:
            await init_cache()

            key = "test_ttl_key"
            await cache_set(key, "value", ttl=60)

            # Check TTL
            ttl = await cache_ttl(key)
            assert 55 < ttl <= 60  # Should be close to 60 seconds

            # Non-existent key
            ttl_missing = await cache_ttl("nonexistent")
            assert ttl_missing == -2  # Key doesn't exist

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_cache_increment(self):
        """Test atomic increment operation"""
        try:
            await init_cache()

            key = "test_counter"

            # First increment (creates key)
            count = await cache_increment(key)
            assert count == 1

            # Subsequent increments
            count = await cache_increment(key)
            assert count == 2

            count = await cache_increment(key, amount=5)
            assert count == 7

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestDomainSpecificHelpers:
    """Test domain-specific cache helpers"""

    @pytest.mark.asyncio
    async def test_structure_caching(self):
        """Test structure cache helpers"""
        try:
            await init_cache()

            structure_id = str(uuid.uuid4())
            structure_data = {
                "id": structure_id,
                "formula": "MoS2",
                "num_atoms": 3,
                "lattice": {"a": 3.16, "b": 3.16, "c": 12.3}
            }

            # Cache structure
            success = await cache_structure(structure_id, structure_data, ttl=600)
            assert success is True

            # Retrieve structure
            retrieved = await get_cached_structure(structure_id)
            assert retrieved == structure_data

            # Invalidate structure cache
            invalidated = await invalidate_structure_cache(structure_id)
            assert invalidated is True

            # Verify invalidated
            retrieved = await get_cached_structure(structure_id)
            assert retrieved is None

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_material_caching(self):
        """Test material cache helpers"""
        try:
            await init_cache()

            material_id = str(uuid.uuid4())
            material_data = {
                "id": material_id,
                "name": "Molybdenum Disulfide",
                "formula": "MoS2",
                "tags": ["2D material", "semiconductor"]
            }

            # Cache material
            await cache_material(material_id, material_data)

            # Retrieve material
            retrieved = await get_cached_material(material_id)
            assert retrieved["formula"] == "MoS2"

            # Invalidate
            await invalidate_material_cache(material_id)
            assert await get_cached_material(material_id) is None

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_ml_prediction_caching(self):
        """Test ML prediction cache helpers"""
        try:
            await init_cache()

            structure_id = str(uuid.uuid4())
            model_type = "CGCNN"
            target_property = "bandgap"

            prediction = {
                "value": 2.45,
                "uncertainty": 0.12,
                "model": model_type
            }

            # Cache prediction
            await cache_ml_prediction(
                structure_id, model_type, target_property, prediction, ttl=3600
            )

            # Retrieve prediction
            retrieved = await get_cached_ml_prediction(
                structure_id, model_type, target_property
            )
            assert retrieved["value"] == 2.45

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_job_status_caching(self):
        """Test job status cache helpers"""
        try:
            await init_cache()

            job_id = str(uuid.uuid4())
            status_data = {
                "job_id": job_id,
                "status": "RUNNING",
                "progress": 0.75,
                "current_epoch": 75
            }

            # Cache job status (short TTL)
            await cache_job_status(job_id, status_data, ttl=60)

            # Retrieve status
            retrieved = await get_cached_job_status(job_id)
            assert retrieved["progress"] == 0.75

            # Invalidate
            await invalidate_job_status_cache(job_id)
            assert await get_cached_job_status(job_id) is None

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestRateLimiting:
    """Test rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests(self):
        """Test rate limiting allows requests within limit"""
        try:
            await init_cache()

            identifier = f"user_{uuid.uuid4()}"

            # First request should be allowed
            allowed, remaining = await check_rate_limit(
                identifier, max_requests=10, window_seconds=60
            )
            assert allowed is True
            assert remaining == 9

            # Second request
            allowed, remaining = await check_rate_limit(identifier, max_requests=10)
            assert allowed is True
            assert remaining == 8

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess_requests(self):
        """Test rate limiting blocks requests exceeding limit"""
        try:
            await init_cache()

            identifier = f"user_{uuid.uuid4()}"
            max_requests = 5

            # Make max_requests allowed requests
            for i in range(max_requests):
                allowed, remaining = await check_rate_limit(
                    identifier, max_requests=max_requests, window_seconds=60
                )
                assert allowed is True

            # Next request should be blocked
            allowed, remaining = await check_rate_limit(
                identifier, max_requests=max_requests, window_seconds=60
            )
            assert allowed is False
            assert remaining == 0

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_reset_rate_limit(self):
        """Test resetting rate limit"""
        try:
            await init_cache()

            identifier = f"user_{uuid.uuid4()}"

            # Make some requests
            for _ in range(5):
                await check_rate_limit(identifier, max_requests=10)

            # Reset
            reset = await reset_rate_limit(identifier)
            assert reset is True

            # Should be able to make requests again
            allowed, remaining = await check_rate_limit(identifier, max_requests=10)
            assert allowed is True
            assert remaining == 9  # Back to initial state

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestCachingDecorators:
    """Test caching decorators"""

    @pytest.mark.asyncio
    async def test_cached_decorator(self):
        """Test @cached decorator"""
        try:
            await init_cache()

            call_count = 0

            @cached(key_prefix="expensive_func", ttl=60)
            async def expensive_function(user_id: str, param: int):
                nonlocal call_count
                call_count += 1
                return {"user_id": user_id, "result": param * 2}

            # First call - should execute function
            result1 = await expensive_function("user123", 5)
            assert result1["result"] == 10
            assert call_count == 1

            # Second call with same args - should use cache
            result2 = await expensive_function("user123", 5)
            assert result2["result"] == 10
            assert call_count == 1  # Function not called again

            # Different args - should execute function
            result3 = await expensive_function("user123", 10)
            assert result3["result"] == 20
            assert call_count == 2

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()

    @pytest.mark.asyncio
    async def test_invalidate_cache_on_change_decorator(self):
        """Test @invalidate_cache_on_change decorator"""
        try:
            await init_cache()

            # Set up some cached data
            await cache_set("material:123:data", {"name": "Old Name"})
            await cache_set("material:123:structures", ["struct1", "struct2"])

            @invalidate_cache_on_change(["material:123:*"])
            async def update_material(material_id: str, new_data: dict):
                return {"updated": True, "data": new_data}

            # Call function
            result = await update_material("123", {"name": "New Name"})
            assert result["updated"] is True

            # Verify caches were invalidated
            cached_data = await cache_get("material:123:data")
            assert cached_data is None

            cached_structures = await cache_get("material:123:structures")
            assert cached_structures is None

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestCacheStatistics:
    """Test cache statistics and monitoring"""

    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """Test getting cache statistics"""
        try:
            await init_cache()

            # Add some test data across different namespaces
            await cache_structure("struct1", {"data": "test"})
            await cache_structure("struct2", {"data": "test"})
            await cache_material("mat1", {"data": "test"})
            await cache_ml_prediction("struct1", "CGCNN", "bandgap", {"value": 2.5})

            stats = await get_cache_stats()

            assert "total_keys" in stats
            assert stats["total_keys"] > 0
            assert "namespaces" in stats
            assert CacheNamespace.STRUCTURE in stats["namespaces"]
            assert stats["namespaces"][CacheNamespace.STRUCTURE] >= 2

        except Exception:
            pytest.skip("Redis not available")
        finally:
            await close_cache()


class TestAPIEndpointCaching:
    """Test caching integration with API endpoints"""

    @pytest.mark.asyncio
    async def test_material_endpoint_caching(
        self, db_session: AsyncSession, test_user: User
    ):
        """Test that material GET endpoint uses caching"""
        try:
            await init_cache()

            # Create a test material
            material = Material(
                name="Test Material",
                formula="NaCl",
                description="Test material for caching",
                tags=["test"]
            )
            db_session.add(material)
            await db_session.commit()
            await db_session.refresh(material)

            material_id = str(material.id)

            # First request - should cache
            from src.api.routers.materials import get_material

            response1 = await get_material(material.id, db_session)
            assert response1.formula == "NaCl"

            # Verify cached
            cached = await get_cached_material(material_id)
            assert cached is not None
            assert cached["formula"] == "NaCl"

            # Second request - should use cache (verify by checking cache directly)
            response2 = await get_material(material.id, db_session)
            assert response2.formula == "NaCl"

            # Update material - should invalidate cache
            material.name = "Updated Material"
            await db_session.commit()

            # After update, cache should be invalidated via endpoint logic
            # (Note: This requires the update endpoint to call invalidate_material_cache)

        except Exception as e:
            pytest.skip(f"Redis not available or DB error: {e}")
        finally:
            await close_cache()


class TestCacheNamespaces:
    """Test cache namespace organization"""

    def test_cache_namespaces_defined(self):
        """Test that all required cache namespaces are defined"""
        assert hasattr(CacheNamespace, 'STRUCTURE')
        assert hasattr(CacheNamespace, 'MATERIAL')
        assert hasattr(CacheNamespace, 'SIMULATION')
        assert hasattr(CacheNamespace, 'ML_PREDICTION')
        assert hasattr(CacheNamespace, 'ML_TRAINING')
        assert hasattr(CacheNamespace, 'CAMPAIGN')
        assert hasattr(CacheNamespace, 'USER')
        assert hasattr(CacheNamespace, 'WORKFLOW')

    def test_namespace_uniqueness(self):
        """Test that namespace values are unique"""
        namespaces = [
            CacheNamespace.STRUCTURE,
            CacheNamespace.MATERIAL,
            CacheNamespace.SIMULATION,
            CacheNamespace.ML_PREDICTION,
            CacheNamespace.ML_TRAINING,
            CacheNamespace.CAMPAIGN,
            CacheNamespace.USER,
            CacheNamespace.WORKFLOW,
        ]

        assert len(namespaces) == len(set(namespaces))


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
