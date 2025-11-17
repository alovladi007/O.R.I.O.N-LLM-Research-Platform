"""
Minimal Development FastAPI Application
========================================

Simplified FastAPI app for frontend testing without all production dependencies.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI instance
app = FastAPI(
    title="ORION Platform API",
    description="AI-Driven Materials Science Research Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS - allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "ok",
        "message": "ORION API is running (development mode)",
        "version": "1.0.0"
    }

# API v1 endpoints with mock data
@app.get("/api/v1/structures")
async def list_structures(skip: int = 0, limit: int = 20):
    """List structures with mock data"""
    mock_structures = [
        {
            "id": f"struct_{i}",
            "formula": ["Si", "GaN", "Fe2O3", "TiO2", "Al2O3"][i % 5],
            "space_group": ["Fd-3m", "P63mc", "R-3c", "P42/mnm", "R-3c"][i % 5],
            "crystal_system": "cubic",
            "lattice_params": {
                "a": 5.43 + i * 0.1,
                "b": 5.43 + i * 0.1,
                "c": 5.43 + i * 0.1,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0
            },
            "num_atoms": 8 + i,
            "dimensionality": "3D",
            "tags": ["semiconductor", "stable"],
            "created_at": "2025-11-16T00:00:00Z"
        }
        for i in range(skip, min(skip + limit, 50))
    ]

    return {
        "items": mock_structures,
        "total": 50,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/v1/structures/{structure_id}")
async def get_structure(structure_id: str):
    """Get structure details with mock data"""
    return {
        "id": structure_id,
        "formula": "Si",
        "space_group": "Fd-3m",
        "crystal_system": "cubic",
        "lattice_params": {
            "a": 5.43,
            "b": 5.43,
            "c": 5.43,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0
        },
        "atoms": [
            {"element": "Si", "position": [0.0, 0.0, 0.0]},
            {"element": "Si", "position": [0.25, 0.25, 0.25]},
            {"element": "Si", "position": [0.5, 0.5, 0.0]},
            {"element": "Si", "position": [0.75, 0.75, 0.25]},
            {"element": "Si", "position": [0.5, 0.0, 0.5]},
            {"element": "Si", "position": [0.75, 0.25, 0.75]},
            {"element": "Si", "position": [0.0, 0.5, 0.5]},
            {"element": "Si", "position": [0.25, 0.75, 0.75]}
        ],
        "properties": {
            "band_gap": 1.12,
            "formation_energy": -5.42,
            "magnetic_moment": 0.0,
            "density": 2.33
        },
        "created_at": "2025-11-16T00:00:00Z"
    }

@app.get("/api/v1/design/stats")
async def get_design_stats():
    """Get design statistics with mock data"""
    return {
        "total_designs": 1250,
        "successful_designs": 892,
        "avg_fitness": 0.78,
        "best_fitness": 0.95,
        "total_evaluations": 45000,
        "avg_generation_time": 12.5
    }

@app.post("/api/v1/design/search")
async def search_designs(params: dict):
    """Mock design search endpoint"""
    return {
        "best_candidates": [
            {
                "id": f"design_{i}",
                "formula": f"Material_{i}",
                "fitness": 0.95 - i * 0.05,
                "properties": {
                    "band_gap": 2.0 + i * 0.1,
                    "formation_energy": -4.5 - i * 0.2
                },
                "generation": i + 1
            }
            for i in range(5)
        ],
        "generations": 10,
        "total_evaluations": 500,
        "best_fitness": 0.95,
        "convergence_history": [0.5, 0.65, 0.75, 0.82, 0.88, 0.91, 0.93, 0.94, 0.95, 0.95]
    }

@app.post("/api/v1/jobs")
async def submit_job(job_data: dict):
    """Mock job submission endpoint"""
    return {
        "id": "job_12345",
        "structure_id": job_data.get("structure_id"),
        "simulation_type": job_data.get("simulation_type", "dft"),
        "status": "pending",
        "created_at": "2025-11-16T00:00:00Z"
    }

@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    """Mock job status endpoint"""
    return {
        "id": job_id,
        "status": "completed",
        "progress": 100,
        "results": {
            "energy": -123.45,
            "forces": [0.01, 0.02, 0.01],
            "stress": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
        },
        "created_at": "2025-11-16T00:00:00Z",
        "completed_at": "2025-11-16T00:10:00Z"
    }

@app.post("/api/v1/ml/predict")
async def predict_properties(prediction_data: dict):
    """Mock ML prediction endpoint"""
    return {
        "structure_id": prediction_data.get("structure_id"),
        "properties": [
            {
                "property_name": prop,
                "predicted_value": 1.5,
                "uncertainty": 0.1,
                "unit": "eV"
            }
            for prop in prediction_data.get("properties", ["band_gap"])
        ],
        "model_version": "v1.0",
        "timestamp": "2025-11-16T00:00:00Z"
    }

@app.get("/api/v1/provenance/{entity_type}/{entity_id}")
async def get_provenance(entity_type: str, entity_id: str):
    """Mock provenance endpoint"""
    return [
        {
            "id": f"prov_{i}",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": ["created", "modified", "analyzed"][i % 3],
            "actor": "user@example.com",
            "timestamp": f"2025-11-16T00:{i:02d}:00Z",
            "metadata": {}
        }
        for i in range(5)
    ]

logger.info("Development API server initialized")
logger.info("CORS enabled for: http://localhost:3000, http://localhost:3001, http://localhost:8000")
logger.info("All endpoints return mock data - no database required")
