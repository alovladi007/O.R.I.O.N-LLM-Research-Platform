"""
Health check and monitoring router for NANO-OS API.

Provides:
- System health checks (database, cache, overall status)
- API version information
- Service status monitoring
"""

from fastapi import APIRouter, status
from pydantic import BaseModel, Field
from typing import Dict, Any
from datetime import datetime
import logging

from ..database import check_db_health
from ..cache import check_cache_health
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        500: {"description": "Service unhealthy"}
    }
)


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall health status (healthy/degraded/unhealthy)")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, Any] = Field(..., description="Individual service health status")


class VersionResponse(BaseModel):
    """API version response schema."""
    name: str = Field(..., description="Application name")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment (development/staging/production)")
    api_prefix: str = Field(..., description="API prefix path")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check",
    description="""
    Comprehensive health check for all NANO-OS services.

    Checks:
    - Database connectivity and latency
    - Redis cache connectivity and latency
    - Overall system status

    Returns:
    - **healthy**: All services operational
    - **degraded**: Some services have issues but API is functional
    - **unhealthy**: Critical services are down
    """,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "version": "2.0.0",
                        "services": {
                            "database": {
                                "status": "healthy",
                                "latency_ms": 2.5
                            },
                            "cache": {
                                "status": "healthy",
                                "latency_ms": 0.8
                            }
                        }
                    }
                }
            }
        },
        503: {
            "description": "System is unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "version": "2.0.0",
                        "services": {
                            "database": {
                                "status": "unhealthy",
                                "error": "Connection refused"
                            },
                            "cache": {
                                "status": "healthy",
                                "latency_ms": 0.8
                            }
                        }
                    }
                }
            }
        }
    }
)
async def health_check() -> HealthResponse:
    """
    Check health of all system components.

    Returns HTTP 200 if all services are healthy or degraded.
    Returns HTTP 503 if critical services are unhealthy.
    """
    logger.debug("Performing health check")

    # Check database
    db_health = await check_db_health()

    # Check cache
    cache_health = await check_cache_health()

    # Aggregate service status
    services = {
        "database": db_health,
        "cache": cache_health,
    }

    # Determine overall status
    # Critical: database must be healthy
    # Degraded: cache can be down (API still works, just slower)
    if db_health.get("status") == "unhealthy":
        overall_status = "unhealthy"
        response_status = status.HTTP_503_SERVICE_UNAVAILABLE
    elif cache_health.get("status") == "unhealthy":
        overall_status = "degraded"
        response_status = status.HTTP_200_OK
        logger.warning("Cache is unhealthy, system running in degraded mode")
    else:
        overall_status = "healthy"
        response_status = status.HTTP_200_OK

    response = HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        services=services
    )

    # Log unhealthy status
    if overall_status == "unhealthy":
        logger.error(f"Health check failed: {services}")

    return response


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Get API version",
    description="""
    Get NANO-OS API version and environment information.

    Useful for:
    - Verifying API deployment version
    - Checking environment (dev/staging/prod)
    - API documentation and debugging
    """,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "API version information",
            "content": {
                "application/json": {
                    "example": {
                        "name": "ORION Platform API",
                        "version": "2.0.0",
                        "environment": "production",
                        "api_prefix": "/api/v1"
                    }
                }
            }
        }
    }
)
async def get_version() -> VersionResponse:
    """
    Get API version and environment information.
    """
    return VersionResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        api_prefix=settings.api_prefix
    )


@router.get(
    "/db",
    summary="Database health check",
    description="Check database connectivity and performance metrics",
    status_code=status.HTTP_200_OK
)
async def db_health() -> Dict[str, Any]:
    """
    Detailed database health check.

    Returns connection pool stats, latency, and database status.
    """
    return await check_db_health()


@router.get(
    "/cache",
    summary="Cache health check",
    description="Check Redis cache connectivity and performance metrics",
    status_code=status.HTTP_200_OK
)
async def cache_health() -> Dict[str, Any]:
    """
    Detailed cache health check.

    Returns Redis connection status, latency, and memory usage.
    """
    return await check_cache_health()
