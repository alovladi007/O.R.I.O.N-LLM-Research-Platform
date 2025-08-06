"""
ORION Custom Exceptions
======================

Custom exception hierarchy for the ORION platform.
"""

from typing import Optional, Dict, Any


class ORIONException(Exception):
    """Base exception for all ORION-specific errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(ORIONException):
    """Raised when there's an error in configuration"""
    pass


class ValidationError(ORIONException):
    """Raised when validation fails"""
    pass


class MaterialValidationError(ValidationError):
    """Raised when material validation fails"""
    pass


class SimulationError(ORIONException):
    """Base class for simulation-related errors"""
    pass


class SimulationSetupError(SimulationError):
    """Raised when simulation setup fails"""
    pass


class SimulationExecutionError(SimulationError):
    """Raised when simulation execution fails"""
    pass


class SimulationTimeoutError(SimulationError):
    """Raised when simulation times out"""
    pass


class KnowledgeGraphError(ORIONException):
    """Base class for knowledge graph errors"""
    pass


class GraphConnectionError(KnowledgeGraphError):
    """Raised when connection to graph database fails"""
    pass


class GraphQueryError(KnowledgeGraphError):
    """Raised when graph query fails"""
    pass


class RAGError(ORIONException):
    """Base class for RAG-related errors"""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails"""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails"""
    pass


class LLMError(ORIONException):
    """Base class for LLM-related errors"""
    pass


class LLMConnectionError(LLMError):
    """Raised when LLM connection fails"""
    pass


class LLMGenerationError(LLMError):
    """Raised when LLM generation fails"""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class DatabaseError(ORIONException):
    """Base class for database errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails"""
    pass


class AuthenticationError(ORIONException):
    """Raised when authentication fails"""
    pass


class AuthorizationError(ORIONException):
    """Raised when authorization fails"""
    pass


class ResourceNotFoundError(ORIONException):
    """Raised when a requested resource is not found"""
    pass


class ResourceExistsError(ORIONException):
    """Raised when trying to create a resource that already exists"""
    pass


class ProcessingError(ORIONException):
    """Raised when processing fails"""
    pass


class DataIngestionError(ProcessingError):
    """Raised when data ingestion fails"""
    pass


class ProtocolGenerationError(ProcessingError):
    """Raised when protocol generation fails"""
    pass


class ExperimentalDesignError(ProcessingError):
    """Raised when experimental design fails"""
    pass


class CacheError(ORIONException):
    """Raised when cache operations fail"""
    pass


class StorageError(ORIONException):
    """Raised when storage operations fail"""
    pass


class NetworkError(ORIONException):
    """Raised when network operations fail"""
    pass


class APIError(ORIONException):
    """Base class for API-related errors"""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.status_code = status_code


class BadRequestError(APIError):
    """Raised for bad API requests"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 400, details)


class NotFoundError(APIError):
    """Raised when API resource is not found"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 404, details)


class ConflictError(APIError):
    """Raised when there's a conflict in API request"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 409, details)


class InternalServerError(APIError):
    """Raised for internal server errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 500, details)


class ServiceUnavailableError(APIError):
    """Raised when service is unavailable"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, 503, details)