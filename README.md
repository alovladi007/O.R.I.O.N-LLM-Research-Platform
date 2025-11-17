# NANO-OS: Nanomaterials Operating System

**A comprehensive platform for computational materials research with multi-scale simulations, machine learning, and automated design workflows.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Next.js-14-black.svg" alt="Next.js 14">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

---

## 🚀 Overview

NANO-OS is a modular platform that integrates:

- **Multi-Scale Simulations**: DFT (Quantum Espresso), MD (LAMMPS), Continuum, Mesoscale
- **Machine Learning**: GNN-based property predictions, model training & deployment
- **Automated Design**: AI-driven materials discovery campaigns with Bayesian optimization
- **Structure Management**: Upload, parse, and visualize crystal structures (CIF, POSCAR)
- **Worker System**: Background job processing with Celery
- **RESTful API**: Comprehensive API for programmatic access
- **Modern Frontend**: Next.js-based UI with 3D structure visualization

## ✨ Key Features

### 🔬 Simulation Engines
- **Mock Engine**: Fast testing with realistic fake data
- **Quantum Espresso (QE)**: DFT calculations (SCF, relaxation, band structure)
- **LAMMPS**: Classical MD simulations (NVT, NPT, annealing)
- **Continuum & Mesoscale**: Stubs for FEM and KMC (future integration)

### 🤖 Machine Learning
- **Stub ML Models**: Deterministic predictions for testing
- **GNN Infrastructure** (Sessions 14-16):
  - Feature extraction pipeline for crystal structures
  - CGCNN-style graph neural networks
  - Model training and registry system
  - Inference API with uncertainty quantification

### 🎯 Materials Design
- **Design Campaigns**: Multi-iteration optimization loops
- **Search Algorithms**: Random, genetic, Bayesian optimization, active learning
- **AGI Integration**: Ready for external AI agent control
- **Provenance Tracking**: Full reproducibility of design decisions

### 📊 Data Management
- **PostgreSQL**: Relational data (materials, structures, jobs, results)
- **Redis**: Caching and Celery broker
- **Alembic**: Database migrations
- **Multi-tenancy**: User-based data isolation

## 🏃 Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **OR** Local installation:
  - Python 3.10+
  - Node.js 18+
  - PostgreSQL 14+
  - Redis 7+

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform.git
cd O.R.I.O.N-LLM-Research-Platform

# Create environment file
make env  # or: cp .env.example .env

# Start all services
make up

# Run database migrations
make migrate

# Seed database with example data
make seed

# Check service health
make health
```

**Access the platform:**
- **Frontend**: http://localhost:3000 (or 3001)
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Option 2: Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials

# Run database migrations
alembic upgrade head

# Seed database
python scripts/seed_data.py

# Start backend (terminal 1)
uvicorn src.api.app:app --reload --port 8000

# Start worker (terminal 2)
celery -A src.worker.celery_app worker --loglevel=info

# Start frontend (terminal 3)
cd frontend && npm run dev
```

### Running the Demo

See the platform in action:

```bash
# Make sure API server is running first
make dev  # or: uvicorn src.api.app:app --reload

# In another terminal, run the demo
python scripts/demo_run.py

# Skip design campaign for faster demo
python scripts/demo_run.py --skip-campaign
```

The demo script will:
1. Create a new material (h-BN)
2. Upload a crystal structure (CIF format)
3. Launch a simulation job
4. Run ML property predictions
5. Start a design campaign

## 📖 Documentation

Comprehensive documentation is available in the `/docs` folder:

- **[Architecture Overview](docs/architecture.md)**
  - System architecture and data flow
  - Service descriptions
  - Multi-scale integration
  - AGI loop design

- **[API Reference](docs/api-overview.md)**
  - Complete endpoint documentation
  - Request/response examples
  - Authentication & authorization
  - Rate limiting & pagination

- **[Simulation Engines](docs/engines.md)**
  - Engine abstraction layer
  - Available engines (Mock, QE, LAMMPS)
  - Adding new engines
  - Multi-engine workflows

- **[Design Loops & Campaigns](docs/design-loops.md)**
  - Campaign lifecycle
  - Search algorithms
  - Objective functions & constraints
  - AGI agent integration guide

## 🛠️ Development

### Project Structure

```
NANO-OS/
├── backend/
│   └── common/
│       ├── engines/      # Simulation engine implementations
│       ├── ml/           # ML models & property predictions
│       ├── design/       # Design search algorithms
│       ├── campaigns/    # Multi-iteration campaign loops
│       ├── structures/   # Structure parsers
│       └── provenance/   # Reproducibility tracking
│
├── src/
│   ├── api/
│   │   ├── app.py       # FastAPI application
│   │   ├── routes/      # API endpoints
│   │   └── models/      # SQLAlchemy ORM models
│   └── worker/          # Celery worker tasks
│
├── frontend/            # Next.js frontend
│   ├── src/
│   │   ├── app/        # App router pages
│   │   ├── components/ # React components
│   │   └── lib/        # Utilities & API client
│   └── package.json
│
├── docs/               # Documentation (Sessions 13+)
├── scripts/            # Utility scripts
│   ├── seed_data.py   # Database seeder
│   └── demo_run.py    # Platform demonstration
│
├── tests/              # Test suite
├── alembic/            # Database migrations
├── docker-compose.yml  # Docker services
├── Makefile            # Development commands
└── pyproject.toml      # Python dependencies
```

### Common Make Commands

```bash
make help              # Show all available commands
make up                # Start all services
make down              # Stop all services
make logs              # View logs from all services
make shell-db          # Open PostgreSQL shell
make migrate           # Run database migrations
make seed              # Seed database with example data
make test              # Run test suite
make lint              # Run code linters
make format            # Format code (black, isort)
make status            # Check service status
```

### Adding a New Simulation Engine

1. Create engine file: `backend/common/engines/my_engine.py`
2. Inherit from `SimulationEngine` base class
3. Implement: `prepare_input()`, `execute()`, `parse_output()`
4. Register in `backend/common/engines/registry.py`
5. Add workflow templates to `scripts/seed_data.py`

See [docs/engines.md](docs/engines.md) for detailed instructions.

### Adding a New ML Model

1. Implement model: `backend/common/ml/models/my_model.py`
2. Register in model registry
3. Add inference route or integrate with existing `/ml/properties`
4. (Optional) Add training pipeline

See Sessions 14-16 implementation for examples.

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/test_api.py -v
pytest tests/test_structures.py -v
pytest tests/test_engines.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Frontend tests
cd frontend && npm test
```

## 📡 API Examples

### Create a Material

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/materials",
        json={
            "name": "Graphene",
            "formula": "C",
            "description": "2D carbon allotrope",
            "tags": ["2D", "conductor"]
        }
    )
    material = response.json()
    print(f"Created material: {material['id']}")
```

### Upload a Structure

```python
cif_content = """
data_graphene
_cell_length_a 2.46
_cell_length_b 2.46
_cell_length_c 20.0
...
"""

response = await client.post(
    "http://localhost:8000/api/structures",
    json={
        "material_id": material["id"],
        "name": "Graphene monolayer",
        "file_content": cif_content,
        "format": "CIF"
    }
)
```

### Launch a Simulation

```python
response = await client.post(
    "http://localhost:8000/api/jobs",
    json={
        "structure_id": structure["id"],
        "workflow_template_id": "uuid-of-template",
        "name": "DFT relaxation",
        "parameters": {
            "ecutwfc": 50.0,
            "kpoints": [6, 6, 1]
        }
    }
)
job = response.json()
```

### ML Property Prediction

```python
response = await client.post(
    "http://localhost:8000/api/ml/properties",
    json={
        "structure_id": structure["id"],
        "model_name": "STUB"  # or "cgcnn_bandgap_v1"
    }
)
predictions = response.json()
print(f"Predicted bandgap: {predictions['bandgap']} eV")
```

See [docs/api-overview.md](docs/api-overview.md) for complete API documentation.

## 🎯 Sessions & Milestones

NANO-OS is developed through incremental sessions:

- **Sessions 1-3**: Core infrastructure, database models, structure parsing
- **Sessions 4-6**: Job orchestration, QE engine, ML predictions (stub)
- **Sessions 7-9**: Frontend structure viewer, job dashboard, UI polish
- **Sessions 10-12**: Authentication, multi-scale engines, design campaigns
- **Session 13** ✅: Documentation, seeding, developer convenience
- **Session 14** 🚧: ML infrastructure (feature extraction, dataset builder)
- **Session 15** 🚧: GNN model integration (CGCNN-style)
- **Session 16** 🚧: Model training pipeline & registry
- **Session 17** 🚧: LAMMPS integration for MD simulations

See `SESSIONS_*.md` files for detailed implementation notes.

## 🤖 AGI Integration

NANO-OS is designed to be controlled by AI agents. External agents can:

1. **Read** all data via REST API
2. **Submit** structures, jobs, and campaigns programmatically
3. **Monitor** progress through status endpoints (future: WebSockets)
4. **Analyze** results and propose next candidates
5. **Orchestrate** multi-scale workflows

Example: An AGI agent running a materials discovery campaign can propose candidates based on past iterations, submit them for evaluation (ML or DFT), analyze results, and iteratively refine the search strategy.

See [docs/design-loops.md](docs/design-loops.md) for AGI integration patterns.

## 🔒 Security & Multi-Tenancy

- **Authentication**: JWT-based (Sessions 10-12)
- **Authorization**: Row-level security via `owner_id`
- **Input Validation**: Pydantic schemas on all API inputs
- **SQL Injection**: Prevented by SQLAlchemy ORM
- **Rate Limiting**: Configurable per endpoint

## 🌟 Future Enhancements

- [ ] Real-time updates via WebSocket
- [ ] HPC cluster integration (SLURM submission)
- [ ] Advanced GNN architectures (ALIGNN, M3GNET)
- [ ] Experiment-in-the-loop with lab automation
- [ ] Multi-objective Pareto optimization
- [ ] Federated learning for collaborative model training
- [ ] Knowledge graph integration (Neo4j)

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Quantum Espresso**: DFT simulations
- **LAMMPS**: Molecular dynamics
- **PyMatGen**: Materials analysis
- **ASE**: Atomic simulation environment
- **3Dmol.js**: Structure visualization
- Materials Project, OQMD, and the open materials science community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform/issues)
- **Documentation**: [/docs](/docs) folder
- **Email**: Contact repository maintainers

---

**Built for materials scientists, by materials scientists (with AI assistance).**
