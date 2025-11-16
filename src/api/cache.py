"""
Redis cache management for NANO-OS.

Provides:
- Redis connection pooling
- Cache lifecycle management (init_cache, close_cache)
- Helper functions for common caching patterns
- Session storage for user authentication

Future sessions will add:
- Distributed caching strategies
- Cache invalidation patterns
- Result caching for expensive computations
"""

import json
import logging
from typing import Any, Optional
from datetime import timedelta

import redis.asyncio as aioredis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError

from .config import settings

logger = logging.getLogger(__name__)

# Global Redis client
redis_client: Optional[Redis] = None
connection_pool: Optional[ConnectionPool] = None


async def init_cache() -> None:
    """
    Initialize Redis connection pool.

    Called during FastAPI startup (lifespan context).
    Creates connection pool and verifies connectivity.
    """
    global redis_client, connection_pool

    logger.info("Initializing Redis cache connection...")

    try:
        # Create connection pool
        connection_pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=False,  # We'll handle encoding ourselves for flexibility
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True,
        )

        # Create Redis client
        redis_client = Redis(connection_pool=connection_pool)

        # Test connection
        await redis_client.ping()

        logger.info("Redis cache initialization complete")

    except RedisError as e:
        logger.error(f"Failed to initialize Redis cache: {e}")
        # Don't fail app startup if cache is unavailable
        redis_client = None
        connection_pool = None


async def close_cache() -> None:
    """
    Close Redis connection pool.

    Called during FastAPI shutdown (lifespan context).
    Properly closes all connections.
    """
    global redis_client, connection_pool

    if redis_client:
        logger.info("Closing Redis cache connection...")
        await redis_client.close()
        redis_client = None

    if connection_pool:
        await connection_pool.disconnect()
        connection_pool = None

    logger.info("Redis cache connection closed")


def get_redis() -> Redis:
    """
    Get Redis client instance.

    Raises:
        RuntimeError if cache is not initialized
    """
    if not redis_client:
        raise RuntimeError("Cache not initialized. Call init_cache() first.")
    return redis_client


# ========== Cache Helper Functions ==========


async def cache_get(key: str, default: Any = None) -> Any:
    """
    Get value from cache.

    Args:
        key: Cache key
        default: Default value if key not found

    Returns:
        Cached value (deserialized from JSON) or default
    """
    if not redis_client:
        return default

    try:
        value = await redis_client.get(key)
        if value is None:
            return default

        # Try to deserialize as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Return raw bytes if not JSON
            return value.decode('utf-8') if isinstance(value, bytes) else value

    except RedisError as e:
        logger.warning(f"Cache get error for key '{key}': {e}")
        return default


async def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None
) -> bool:
    """
    Set value in cache.

    Args:
        key: Cache key
        value: Value to cache (will be JSON serialized)
        ttl: Time-to-live in seconds (None = no expiration)

    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False

    try:
        # Serialize to JSON for complex types
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value)
        elif not isinstance(value, (str, bytes, int, float)):
            value = json.dumps(value)

        if ttl:
            await redis_client.setex(key, ttl, value)
        else:
            await redis_client.set(key, value)

        return True

    except (RedisError, TypeError, json.JSONEncodeError) as e:
        logger.warning(f"Cache set error for key '{key}': {e}")
        return False


async def cache_delete(key: str) -> bool:
    """
    Delete key from cache.

    Args:
        key: Cache key

    Returns:
        True if key was deleted, False otherwise
    """
    if not redis_client:
        return False

    try:
        deleted = await redis_client.delete(key)
        return deleted > 0
    except RedisError as e:
        logger.warning(f"Cache delete error for key '{key}': {e}")
        return False


async def cache_delete_pattern(pattern: str) -> int:
    """
    Delete all keys matching pattern.

    Args:
        pattern: Redis key pattern (e.g., "user:*", "material:123:*")

    Returns:
        Number of keys deleted
    """
    if not redis_client:
        return 0

    try:
        deleted = 0
        async for key in redis_client.scan_iter(match=pattern):
            await redis_client.delete(key)
            deleted += 1
        return deleted
    except RedisError as e:
        logger.warning(f"Cache delete pattern error for '{pattern}': {e}")
        return 0


async def cache_exists(key: str) -> bool:
    """Check if key exists in cache."""
    if not redis_client:
        return False

    try:
        return await redis_client.exists(key) > 0
    except RedisError:
        return False


async def cache_ttl(key: str) -> int:
    """
    Get remaining TTL for key.

    Returns:
        Seconds remaining (-1 = no expiration, -2 = key doesn't exist)
    """
    if not redis_client:
        return -2

    try:
        return await redis_client.ttl(key)
    except RedisError:
        return -2


async def cache_increment(key: str, amount: int = 1) -> Optional[int]:
    """
    Increment counter in cache.

    Args:
        key: Cache key
        amount: Amount to increment (default 1)

    Returns:
        New value after increment, or None on error
    """
    if not redis_client:
        return None

    try:
        return await redis_client.incrby(key, amount)
    except RedisError as e:
        logger.warning(f"Cache increment error for key '{key}': {e}")
        return None


async def check_cache_health() -> dict:
    """
    Check Redis cache health for monitoring.

    Returns:
        dict with status, latency, and stats
    """
    if not redis_client:
        return {
            "status": "unhealthy",
            "error": "Cache not initialized"
        }

    try:
        import time
        start = time.time()

        # Test connectivity
        await redis_client.ping()

        latency_ms = (time.time() - start) * 1000

        # Get Redis info
        info = await redis_client.info()

        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
            "redis_version": info.get("redis_version"),
            "connected_clients": info.get("connected_clients"),
            "used_memory_human": info.get("used_memory_human"),
            "total_keys": await redis_client.dbsize(),
        }

    except RedisError as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ========== Session Management ==========


async def store_session(session_id: str, user_data: dict, ttl: int = 3600) -> bool:
    """
    Store user session data.

    Args:
        session_id: Unique session identifier
        user_data: User data to store
        ttl: Session TTL in seconds (default 1 hour)
    """
    key = f"session:{session_id}"
    return await cache_set(key, user_data, ttl=ttl)


async def get_session(session_id: str) -> Optional[dict]:
    """Retrieve user session data."""
    key = f"session:{session_id}"
    return await cache_get(key)


async def delete_session(session_id: str) -> bool:
    """Delete user session."""
    key = f"session:{session_id}"
    return await cache_delete(key)


async def extend_session(session_id: str, ttl: int = 3600) -> bool:
    """Extend session TTL."""
    if not redis_client:
        return False

    try:
        key = f"session:{session_id}"
        return await redis_client.expire(key, ttl)
    except RedisError:
        return False
