"""API routers for NANO-OS endpoints."""

from .health import router as health_router
from .auth import router as auth_router
from .materials import router as materials_router
from .structures import router as structures_router
from .workflows import router as workflows_router
from .jobs import router as jobs_router
from .ml import router as ml_router

__all__ = [
    "health_router",
    "auth_router",
    "materials_router",
    "structures_router",
    "workflows_router",
    "jobs_router",
    "ml_router",
]
