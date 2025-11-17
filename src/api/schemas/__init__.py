"""
Pydantic schemas for request/response validation.

These schemas provide:
- Input validation for API requests
- Type-safe response serialization
- API documentation via OpenAPI/Swagger
- Clear separation between DB models and API contracts
"""

from .auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
    TokenRefresh,
)
from .material import (
    MaterialCreate,
    MaterialUpdate,
    MaterialResponse,
    MaterialListResponse,
)
from .structure import (
    StructureCreate,
    StructureUpdate,
    StructureResponse,
    StructureParseRequest,
    StructureParseResponse,
)
from .workflow import (
    WorkflowTemplateCreate,
    WorkflowTemplateUpdate,
    WorkflowTemplateResponse,
)
from .simulation import (
    SimulationJobCreate,
    SimulationJobUpdate,
    SimulationJobResponse,
    SimulationResultResponse,
)
from .ml import (
    PropertyPredictionRequest,
    PropertyPredictionResponse,
    ModelInfoResponse,
    ModelsListResponse,
    PropertyComparisonRequest,
    PropertyComparisonResponse,
    PredictionHistoryResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from .design import (
    PropertyConstraint,
    DesignSearchRequest,
    CandidateStructure,
    DesignSearchResponse,
)
from .common import (
    PaginationParams,
    PaginatedResponse,
    HealthResponse,
    VersionResponse,
)
from .provenance import (
    ProvenanceRecordResponse,
    ProvenanceChainResponse,
    ProvenanceTimelineResponse,
    ProvenanceSummaryResponse,
    TimelineEvent,
)

__all__ = [
    # Auth
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "Token",
    "TokenRefresh",
    # Material
    "MaterialCreate",
    "MaterialUpdate",
    "MaterialResponse",
    "MaterialListResponse",
    # Structure
    "StructureCreate",
    "StructureUpdate",
    "StructureResponse",
    "StructureParseRequest",
    "StructureParseResponse",
    # Workflow
    "WorkflowTemplateCreate",
    "WorkflowTemplateUpdate",
    "WorkflowTemplateResponse",
    # Simulation
    "SimulationJobCreate",
    "SimulationJobUpdate",
    "SimulationJobResponse",
    "SimulationResultResponse",
    # ML
    "PropertyPredictionRequest",
    "PropertyPredictionResponse",
    "ModelInfoResponse",
    "ModelsListResponse",
    "PropertyComparisonRequest",
    "PropertyComparisonResponse",
    "PredictionHistoryResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    # Design
    "PropertyConstraint",
    "DesignSearchRequest",
    "CandidateStructure",
    "DesignSearchResponse",
    # Provenance
    "ProvenanceRecordResponse",
    "ProvenanceChainResponse",
    "ProvenanceTimelineResponse",
    "ProvenanceSummaryResponse",
    "TimelineEvent",
    # Common
    "PaginationParams",
    "PaginatedResponse",
    "HealthResponse",
    "VersionResponse",
]
