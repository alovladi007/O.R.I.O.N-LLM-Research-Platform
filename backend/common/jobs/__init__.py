"""Job-side schemas and helpers shared by the API router and Celery tasks.

Phase 2 / Session 2.2 introduces this package so that the mock-engine
end-to-end path can validate its outputs against a single authoritative
shape (``MockStaticOutput``). Later phases add one sibling module per
engine kind (DFT relax/static/bands/DOS, MD NVE/NVT/NPT, continuum,
mesoscale, ML).

Nothing in here imports FastAPI, Celery, or SQLAlchemy — the output
schemas are pure pydantic so they're safe to import from worker code
that does not have a DB session.
"""

from .artifacts import (
    DEFAULT_ARTIFACTS_BUCKET,
    DEFAULT_PRESIGN_EXPIRY,
    build_minio_client,
    ensure_bucket,
    presign_artifact,
)
from .mock_static import (
    FORCE_NOISE_SIGMA_EV_PER_ANG,
    TRAJECTORY_N_FRAMES,
    run_mock_static,
    write_trajectory_xyz,
)
from .schemas import (
    MOCK_STATIC_OUTPUT_SCHEMA_VERSION,
    MockForceEntry,
    MockStaticOutput,
    MockTrajectoryFrame,
)

__all__ = [
    "DEFAULT_ARTIFACTS_BUCKET",
    "DEFAULT_PRESIGN_EXPIRY",
    "FORCE_NOISE_SIGMA_EV_PER_ANG",
    "MOCK_STATIC_OUTPUT_SCHEMA_VERSION",
    "MockForceEntry",
    "MockStaticOutput",
    "MockTrajectoryFrame",
    "TRAJECTORY_N_FRAMES",
    "build_minio_client",
    "ensure_bucket",
    "presign_artifact",
    "run_mock_static",
    "write_trajectory_xyz",
]
