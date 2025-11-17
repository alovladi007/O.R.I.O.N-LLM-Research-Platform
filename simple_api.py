"""
Simple standalone ORION API Server
Provides basic endpoints without complex dependencies
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

# Create app
app = FastAPI(
    title="ORION Platform API",
    description="AI-driven materials science platform",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ORION Platform API",
        "version": "2.0.0",
        "status": "operational"
    }

@app.get("/api/v1/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/v1/structures")
async def list_structures(skip: int = 0, limit: int = 12):
    """List crystal structures"""
    # Mock data for now
    structures = [
        {
            "id": f"struct-{i}",
            "formula": f"TiO{i%3+1}",
            "name": f"Titanium Oxide {i}",
            "crystal_system": "tetragonal",
            "space_group": "P4/mmm",
            "num_atoms": 12 + i,
            "properties": {
                "band_gap": 3.2 + (i * 0.1),
                "formation_energy": -2.5 - (i * 0.05)
            }
        }
        for i in range(skip, min(skip + limit, 100))
    ]

    return {
        "structures": structures,
        "total": 100,
        "skip": skip,
        "limit": limit
    }

@app.get("/api/v1/structures/{structure_id}")
async def get_structure(structure_id: str):
    """Get a specific structure"""
    return {
        "id": structure_id,
        "formula": "TiO2",
        "name": "Titanium Dioxide",
        "crystal_system": "tetragonal",
        "space_group": "P42/mnm",
        "lattice_parameters": {
            "a": 4.593,
            "b": 4.593,
            "c": 2.959,
            "alpha": 90,
            "beta": 90,
            "gamma": 90
        },
        "num_atoms": 6,
        "properties": {
            "band_gap": 3.2,
            "formation_energy": -2.5,
            "density": 4.23
        }
    }

@app.get("/api/v1/workflows")
async def list_workflows():
    """List workflows"""
    return {
        "workflows": [],
        "total": 0
    }

@app.get("/api/v1/jobs")
async def list_jobs():
    """List computational jobs"""
    return {
        "jobs": [],
        "total": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
