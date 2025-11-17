"""
NANO-OS API client.

Session 28: Python SDK and Workflow DSL
"""

import logging
from typing import Optional, Dict, Any, List
from uuid import UUID
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import (
    Structure, StructureCreate,
    Job, DFTJobCreate, MLPredictionCreate,
    Campaign, CampaignCreate, CampaignStepRequest, CampaignSummary,
    Instrument, InstrumentCreate,
    Experiment, ExperimentCreate,
    MaterialProperties, JobStatusResponse, PaginatedResponse
)

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors."""
    pass


class ResourceNotFoundError(APIError):
    """Resource not found (404)."""
    pass


class AuthenticationError(APIError):
    """Authentication error (401)."""
    pass


class ValidationError(APIError):
    """Validation error (422)."""
    pass


class NanoOSClient:
    """
    Client for NANO-OS API.

    Provides programmatic access to all NANO-OS endpoints with automatic
    retry logic, authentication, and type-safe request/response handling.

    Example:
        >>> client = NanoOSClient(
        ...     base_url="http://localhost:8000",
        ...     api_key="your-api-key"
        ... )
        >>> structure = client.structures.create(
        ...     composition="MoS2",
        ...     lattice_type="hexagonal"
        ... )
        >>> job = client.jobs.submit_dft(structure.id, functional="PBE")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize NANO-OS client.

        Args:
            base_url: Base URL of NANO-OS API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers["Content-Type"] = "application/json"

        # Initialize resource managers
        self.structures = StructureManager(self)
        self.jobs = JobManager(self)
        self.campaigns = CampaignManager(self)
        self.instruments = InstrumentManager(self)
        self.experiments = ExperimentManager(self)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (relative to base_url)
            json: JSON request body
            params: Query parameters

        Returns:
            Response object

        Raises:
            APIError: On API errors
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json,
                params=params,
                timeout=self.timeout
            )

            # Handle HTTP errors
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed. Check your API key.")
            elif response.status_code == 404:
                raise ResourceNotFoundError(f"Resource not found: {endpoint}")
            elif response.status_code == 422:
                raise ValidationError(f"Validation error: {response.json()}")
            elif response.status_code >= 400:
                raise APIError(
                    f"API error {response.status_code}: {response.text}"
                )

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise APIError(f"Request failed: {e}")

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request."""
        response = self._request("GET", endpoint, params=params)
        return response.json()

    def post(self, endpoint: str, json: Dict[str, Any]) -> Dict[str, Any]:
        """POST request."""
        response = self._request("POST", endpoint, json=json)
        return response.json()

    def put(self, endpoint: str, json: Dict[str, Any]) -> Dict[str, Any]:
        """PUT request."""
        response = self._request("PUT", endpoint, json=json)
        return response.json()

    def delete(self, endpoint: str) -> None:
        """DELETE request."""
        self._request("DELETE", endpoint)


class StructureManager:
    """Manager for structure-related endpoints."""

    def __init__(self, client: NanoOSClient):
        self.client = client

    def create(
        self,
        composition: str,
        lattice_type: Optional[str] = None,
        num_atoms: Optional[int] = None,
        space_group: Optional[int] = None,
        dimensionality: Optional[int] = None,
        structure_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Structure:
        """
        Create a new structure.

        Args:
            composition: Chemical composition (e.g., "MoS2")
            lattice_type: Lattice type (e.g., "hexagonal")
            num_atoms: Number of atoms
            space_group: Space group number
            dimensionality: Dimensionality (0=molecule, 1=1D, 2=2D, 3=bulk)
            structure_data: Detailed structure data (CIF, POSCAR, etc.)
            metadata: Additional metadata

        Returns:
            Created structure
        """
        data = StructureCreate(
            composition=composition,
            lattice_type=lattice_type,
            num_atoms=num_atoms,
            space_group=space_group,
            dimensionality=dimensionality,
            structure_data=structure_data,
            metadata=metadata
        ).model_dump(exclude_none=True)

        response = self.client.post("/api/structures", json=data)
        return Structure(**response)

    def get(self, structure_id: UUID) -> Structure:
        """Get structure by ID."""
        response = self.client.get(f"/api/structures/{structure_id}")
        return Structure(**response)

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        composition: Optional[str] = None,
        dimensionality: Optional[int] = None
    ) -> List[Structure]:
        """
        List structures.

        Args:
            page: Page number
            page_size: Items per page
            composition: Filter by composition
            dimensionality: Filter by dimensionality

        Returns:
            List of structures
        """
        params = {
            "page": page,
            "page_size": page_size,
            "composition": composition,
            "dimensionality": dimensionality
        }
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get("/api/structures", params=params)
        return [Structure(**item) for item in response["structures"]]

    def update(
        self,
        structure_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
        structure_data: Optional[Dict[str, Any]] = None
    ) -> Structure:
        """Update structure."""
        data = {}
        if metadata is not None:
            data["metadata"] = metadata
        if structure_data is not None:
            data["structure_data"] = structure_data

        response = self.client.put(f"/api/structures/{structure_id}", json=data)
        return Structure(**response)

    def delete(self, structure_id: UUID) -> None:
        """Delete structure."""
        self.client.delete(f"/api/structures/{structure_id}")


class JobManager:
    """Manager for job-related endpoints."""

    def __init__(self, client: NanoOSClient):
        self.client = client

    def submit_dft(
        self,
        structure_id: UUID,
        functional: str = "PBE",
        kpoints_density: float = 0.03,
        energy_cutoff: Optional[float] = None,
        is_relaxation: bool = True,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Job:
        """
        Submit DFT job.

        Args:
            structure_id: Structure to compute
            functional: Exchange-correlation functional
            kpoints_density: K-points density
            energy_cutoff: Plane wave energy cutoff
            is_relaxation: Whether to relax structure
            priority: Job priority (1-10)
            metadata: Additional metadata

        Returns:
            Created job
        """
        data = DFTJobCreate(
            structure_id=structure_id,
            functional=functional,
            kpoints_density=kpoints_density,
            energy_cutoff=energy_cutoff,
            is_relaxation=is_relaxation,
            priority=priority,
            metadata=metadata
        ).model_dump(exclude_none=True)

        response = self.client.post("/api/jobs/dft", json=data)
        return Job(**response)

    def submit_ml_prediction(
        self,
        structure_id: UUID,
        model_name: str = "cgcnn_bandgap_v1",
        properties: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Job:
        """
        Submit ML prediction job.

        Args:
            structure_id: Structure to predict
            model_name: ML model to use
            properties: Properties to predict
            metadata: Additional metadata

        Returns:
            Created job
        """
        data = MLPredictionCreate(
            structure_id=structure_id,
            model_name=model_name,
            properties=properties,
            metadata=metadata
        ).model_dump(exclude_none=True)

        response = self.client.post("/api/jobs/ml-prediction", json=data)
        return Job(**response)

    def get(self, job_id: UUID) -> Job:
        """Get job by ID."""
        response = self.client.get(f"/api/jobs/{job_id}")
        return Job(**response)

    def get_status(self, job_id: UUID) -> JobStatusResponse:
        """Get job status."""
        response = self.client.get(f"/api/jobs/{job_id}/status")
        return JobStatusResponse(**response)

    def get_results(self, job_id: UUID) -> Dict[str, Any]:
        """Get job results."""
        response = self.client.get(f"/api/jobs/{job_id}/results")
        return response

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        job_type: Optional[str] = None
    ) -> List[Job]:
        """List jobs."""
        params = {
            "page": page,
            "page_size": page_size,
            "status": status,
            "job_type": job_type
        }
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get("/api/jobs", params=params)
        return [Job(**item) for item in response["jobs"]]


class CampaignManager:
    """Manager for campaign-related endpoints."""

    def __init__(self, client: NanoOSClient):
        self.client = client

    def create(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Campaign:
        """
        Create design campaign.

        Args:
            name: Campaign name
            config: Campaign configuration
            description: Campaign description
            metadata: Additional metadata

        Returns:
            Created campaign
        """
        data = CampaignCreate(
            name=name,
            description=description,
            config=config,
            metadata=metadata
        ).model_dump(exclude_none=True)

        response = self.client.post("/api/campaigns", json=data)
        return Campaign(**response)

    def get(self, campaign_id: UUID) -> Campaign:
        """Get campaign by ID."""
        response = self.client.get(f"/api/campaigns/{campaign_id}")
        return Campaign(**response)

    def run_iterations(
        self,
        campaign_id: UUID,
        num_iterations: int = 1,
        override_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run campaign iterations.

        Args:
            campaign_id: Campaign ID
            num_iterations: Number of iterations to run
            override_config: Temporary config overrides

        Returns:
            Step response with updated campaign and iterations
        """
        data = CampaignStepRequest(
            num_iterations=num_iterations,
            override_config=override_config
        ).model_dump(exclude_none=True)

        response = self.client.post(f"/api/campaigns/{campaign_id}/step", json=data)
        return response

    def get_summary(self, campaign_id: UUID) -> CampaignSummary:
        """Get campaign summary statistics."""
        response = self.client.get(f"/api/campaigns/{campaign_id}/summary")
        return CampaignSummary(**response)

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None
    ) -> List[Campaign]:
        """List campaigns."""
        params = {
            "page": page,
            "page_size": page_size,
            "status": status
        }
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get("/api/campaigns", params=params)
        return [Campaign(**item) for item in response["campaigns"]]


class InstrumentManager:
    """Manager for instrument-related endpoints."""

    def __init__(self, client: NanoOSClient):
        self.client = client

    def register(
        self,
        name: str,
        adapter_type: str,
        connection_info: Dict[str, Any],
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Instrument:
        """
        Register lab instrument.

        Args:
            name: Instrument name
            adapter_type: Adapter type (MOCK, REST, OPCUA, SSH)
            connection_info: Connection configuration
            capabilities: Instrument capabilities
            metadata: Additional metadata

        Returns:
            Registered instrument
        """
        data = InstrumentCreate(
            name=name,
            adapter_type=adapter_type,
            connection_info=connection_info,
            capabilities=capabilities,
            metadata=metadata
        ).model_dump(exclude_none=True)

        response = self.client.post("/api/instruments", json=data)
        return Instrument(**response)

    def get(self, instrument_id: UUID) -> Instrument:
        """Get instrument by ID."""
        response = self.client.get(f"/api/instruments/{instrument_id}")
        return Instrument(**response)

    def list(self, page: int = 1, page_size: int = 20) -> List[Instrument]:
        """List instruments."""
        params = {"page": page, "page_size": page_size}
        response = self.client.get("/api/instruments", params=params)
        return [Instrument(**item) for item in response["instruments"]]


class ExperimentManager:
    """Manager for experiment-related endpoints."""

    def __init__(self, client: NanoOSClient):
        self.client = client

    def submit(
        self,
        instrument_id: UUID,
        type: str,
        parameters: Dict[str, Any],
        linked_structure_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Submit experiment to instrument.

        Args:
            instrument_id: Instrument ID
            type: Experiment type (synthesis, measurement, etc.)
            parameters: Experiment parameters
            linked_structure_id: Optional linked structure
            metadata: Additional metadata

        Returns:
            Created experiment
        """
        data = ExperimentCreate(
            instrument_id=instrument_id,
            type=type,
            parameters=parameters,
            linked_structure_id=linked_structure_id,
            metadata=metadata
        ).model_dump(exclude_none=True)

        response = self.client.post("/api/experiments", json=data)
        return Experiment(**response)

    def get(self, experiment_id: UUID) -> Experiment:
        """Get experiment by ID."""
        response = self.client.get(f"/api/experiments/{experiment_id}")
        return Experiment(**response)

    def get_status(self, experiment_id: UUID) -> Dict[str, Any]:
        """Get experiment status."""
        response = self.client.get(f"/api/experiments/{experiment_id}/status")
        return response

    def list(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None
    ) -> List[Experiment]:
        """List experiments."""
        params = {
            "page": page,
            "page_size": page_size,
            "status": status
        }
        params = {k: v for k, v in params.items() if v is not None}

        response = self.client.get("/api/experiments", params=params)
        return [Experiment(**item) for item in response["experiments"]]
