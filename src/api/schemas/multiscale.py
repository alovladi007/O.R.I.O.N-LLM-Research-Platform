"""
Pydantic schemas for multi-scale simulation API.

These schemas define the request and response formats for mesoscale and
continuum simulation endpoints, including:
- Job creation requests
- Job responses
- Result responses
- Status updates
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ============================================================================
# Mesoscale Simulation Schemas
# ============================================================================

class MesoscaleJobCreate(BaseModel):
    """
    Request schema for creating a mesoscale simulation job.

    Used to create jobs for phase field, Monte Carlo, or kinetic Monte Carlo
    simulations at the mesoscale.
    """
    name: str = Field(
        ...,
        description="Descriptive name for the mesoscale simulation",
        min_length=1,
        max_length=255
    )
    engine_type: str = Field(
        ...,
        description="Type of mesoscale simulation engine (PHASE_FIELD, MONTE_CARLO, KMC)"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Simulation parameters (domain size, timesteps, material properties, etc.)"
    )
    parent_atomic_job_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional parent atomistic simulation providing input parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "MoS2 grain growth simulation",
                "engine_type": "MONTE_CARLO",
                "parameters": {
                    "domain_size": [100, 100, 100],
                    "timesteps": 10000,
                    "temperature": 800.0,
                    "grain_boundary_energy": 1.5
                },
                "parent_atomic_job_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }


class MesoscaleJobResponse(BaseModel):
    """
    Response schema for mesoscale simulation job.

    Returns complete job information including status, timing, and relationships.
    """
    id: uuid.UUID = Field(..., description="Job ID")
    owner_id: uuid.UUID = Field(..., description="ID of user who owns this job")
    parent_atomic_job_id: Optional[uuid.UUID] = Field(
        None,
        description="Parent atomistic simulation ID"
    )
    name: str = Field(..., description="Job name")
    engine_type: str = Field(..., description="Mesoscale engine type")
    status: str = Field(..., description="Current job status")
    parameters: Dict[str, Any] = Field(..., description="Simulation parameters")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    celery_task_id: Optional[str] = Field(None, description="Celery task ID")
    duration_seconds: Optional[float] = Field(
        None,
        description="Job duration in seconds (if completed)"
    )

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "423e4567-e89b-12d3-a456-426614174000",
                "owner_id": "523e4567-e89b-12d3-a456-426614174000",
                "parent_atomic_job_id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "MoS2 grain growth simulation",
                "engine_type": "MONTE_CARLO",
                "status": "COMPLETED",
                "parameters": {
                    "domain_size": [100, 100, 100],
                    "timesteps": 10000
                },
                "created_at": "2025-11-17T10:00:00Z",
                "started_at": "2025-11-17T10:01:00Z",
                "completed_at": "2025-11-17T10:30:00Z",
                "celery_task_id": "task-abc-123",
                "duration_seconds": 1740.0
            }
        }


class MesoscaleResultResponse(BaseModel):
    """
    Response schema for mesoscale simulation results.

    Returns microstructure metrics, domain information, and output files.
    """
    id: uuid.UUID = Field(..., description="Result ID")
    job_id: uuid.UUID = Field(..., description="Associated job ID")
    microstructure_metrics: Dict[str, Any] = Field(
        ...,
        description="Microstructure analysis (grain size, phase fractions, etc.)"
    )
    domain_size: Dict[str, Any] = Field(
        ...,
        description="Simulation domain dimensions"
    )
    timesteps: int = Field(..., description="Number of timesteps simulated")
    output_files: List[str] = Field(
        ...,
        description="List of output file paths (VTK, HDF5, images, etc.)"
    )
    created_at: datetime = Field(..., description="Result creation timestamp")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "623e4567-e89b-12d3-a456-426614174000",
                "job_id": "423e4567-e89b-12d3-a456-426614174000",
                "microstructure_metrics": {
                    "average_grain_size": 15.3,
                    "grain_size_distribution": [10.2, 12.5, 15.3, 18.7, 21.2],
                    "phase_fractions": {"alpha": 0.6, "beta": 0.4},
                    "grain_boundary_density": 0.023
                },
                "domain_size": {
                    "nx": 100,
                    "ny": 100,
                    "nz": 100
                },
                "timesteps": 10000,
                "output_files": [
                    "/data/mesoscale/job123/final_microstructure.vtk",
                    "/data/mesoscale/job123/evolution.h5"
                ],
                "created_at": "2025-11-17T10:30:00Z"
            }
        }


# ============================================================================
# Continuum Simulation Schemas
# ============================================================================

class ContinuumJobCreate(BaseModel):
    """
    Request schema for creating a continuum simulation job.

    Used to create jobs for FEM, FVM, or BEM simulations at the continuum scale.
    """
    name: str = Field(
        ...,
        description="Descriptive name for the continuum simulation",
        min_length=1,
        max_length=255
    )
    engine_type: str = Field(
        ...,
        description="Type of continuum simulation engine (FEM, FVM, BEM)"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Simulation parameters (mesh, boundary conditions, material properties, etc.)"
    )
    parent_mesoscale_job_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional parent mesoscale simulation providing material properties"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Tensile test FEM simulation",
                "engine_type": "FEM",
                "parameters": {
                    "mesh_size": 0.01,
                    "boundary_conditions": {
                        "fixed": "bottom_face",
                        "load": {"face": "top_face", "force": 1000.0}
                    },
                    "material_properties": {
                        "elastic_modulus": 200e9,
                        "poisson_ratio": 0.3
                    }
                },
                "parent_mesoscale_job_id": "423e4567-e89b-12d3-a456-426614174000"
            }
        }


class ContinuumJobResponse(BaseModel):
    """
    Response schema for continuum simulation job.

    Returns complete job information including status, timing, and relationships.
    """
    id: uuid.UUID = Field(..., description="Job ID")
    owner_id: uuid.UUID = Field(..., description="ID of user who owns this job")
    parent_mesoscale_job_id: Optional[uuid.UUID] = Field(
        None,
        description="Parent mesoscale simulation ID"
    )
    name: str = Field(..., description="Job name")
    engine_type: str = Field(..., description="Continuum engine type")
    status: str = Field(..., description="Current job status")
    parameters: Dict[str, Any] = Field(..., description="Simulation parameters")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    celery_task_id: Optional[str] = Field(None, description="Celery task ID")
    duration_seconds: Optional[float] = Field(
        None,
        description="Job duration in seconds (if completed)"
    )

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "723e4567-e89b-12d3-a456-426614174000",
                "owner_id": "523e4567-e89b-12d3-a456-426614174000",
                "parent_mesoscale_job_id": "423e4567-e89b-12d3-a456-426614174000",
                "name": "Tensile test FEM simulation",
                "engine_type": "FEM",
                "status": "COMPLETED",
                "parameters": {
                    "mesh_size": 0.01,
                    "material_properties": {
                        "elastic_modulus": 200e9
                    }
                },
                "created_at": "2025-11-17T11:00:00Z",
                "started_at": "2025-11-17T11:01:00Z",
                "completed_at": "2025-11-17T11:45:00Z",
                "celery_task_id": "task-def-456",
                "duration_seconds": 2640.0
            }
        }


class ContinuumResultResponse(BaseModel):
    """
    Response schema for continuum simulation results.

    Returns effective properties, mesh information, solution fields, and output files.
    """
    id: uuid.UUID = Field(..., description="Result ID")
    job_id: uuid.UUID = Field(..., description="Associated job ID")
    effective_properties: Dict[str, Any] = Field(
        ...,
        description="Homogenized material properties (elastic moduli, thermal conductivity, etc.)"
    )
    mesh_info: Dict[str, Any] = Field(
        ...,
        description="Mesh metadata (number of nodes, elements, element types, etc.)"
    )
    solution_fields: Dict[str, Any] = Field(
        ...,
        description="Summary of solution fields (displacement, stress, temperature, etc.)"
    )
    output_files: List[str] = Field(
        ...,
        description="List of output file paths (VTK, ParaView, result databases, etc.)"
    )
    created_at: datetime = Field(..., description="Result creation timestamp")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "823e4567-e89b-12d3-a456-426614174000",
                "job_id": "723e4567-e89b-12d3-a456-426614174000",
                "effective_properties": {
                    "effective_elastic_modulus": 195e9,
                    "effective_poisson_ratio": 0.31,
                    "effective_thermal_conductivity": 45.2
                },
                "mesh_info": {
                    "num_nodes": 125000,
                    "num_elements": 100000,
                    "element_type": "C3D8",
                    "mesh_quality": 0.92
                },
                "solution_fields": {
                    "displacement_max": 0.005,
                    "stress_max": 250e6,
                    "strain_max": 0.00125
                },
                "output_files": [
                    "/data/continuum/job456/displacement.vtk",
                    "/data/continuum/job456/stress.vtk",
                    "/data/continuum/job456/results.odb"
                ],
                "created_at": "2025-11-17T11:45:00Z"
            }
        }
