"""API routers for NANO-OS endpoints."""

from .health import router as health_router
from .auth import router as auth_router
from .materials import router as materials_router
from .structures import router as structures_router
from .workflows import router as workflows_router
from .jobs import router as jobs_router
from .ml import router as ml_router
from .design import router as design_router
from .provenance import router as provenance_router
from .campaigns import router as campaigns_router
from .mesoscale import router as mesoscale_router
from .continuum import router as continuum_router
from .orchestrator import router as orchestrator_router
from .agent import router as agent_router
from .experiments import router as experiments_router
from .photonics import router as photonics_router
from .batteries import router as batteries_router
from .quantum import router as quantum_router
from .metamaterials import router as metamaterials_router
from .pcm import router as pcm_router
from .execution import router as execution_router

__all__ = [
    "health_router",
    "auth_router",
    "materials_router",
    "structures_router",
    "workflows_router",
    "jobs_router",
    "ml_router",
    "design_router",
    "provenance_router",
    "campaigns_router",
    "mesoscale_router",
    "continuum_router",
    "orchestrator_router",
    "agent_router",
    "experiments_router",
    "photonics_router",
    "batteries_router",
    "quantum_router",
    "metamaterials_router",
    "pcm_router",
    "execution_router",
]
