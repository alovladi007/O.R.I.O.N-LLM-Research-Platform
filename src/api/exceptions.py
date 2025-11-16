"""
Custom exceptions and error handlers for NANO-OS API.

Provides:
- Domain-specific exception classes
- Consistent error response format
- Exception handlers for FastAPI

Future sessions will add:
- Detailed error codes for different failure modes
- Error recovery strategies
- User-friendly error messages for frontend
"""

from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)


# ========== Custom Exception Classes ==========


class NANOOSException(Exception):
    """Base exception for all NANO-OS errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class DatabaseError(NANOOSException):
    """Database operation failed."""

    def __init__(self, message: str = "Database error occurred", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class NotFoundError(NANOOSException):
    """Resource not found."""

    def __init__(self, resource: str, identifier: Any):
        super().__init__(
            message=f"{resource} with id '{identifier}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource": resource, "id": str(identifier)}
        )


class ValidationError(NANOOSException):
    """Data validation failed."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class AuthenticationError(NANOOSException):
    """Authentication failed."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={}
        )


class AuthorizationError(NANOOSException):
    """User not authorized for this operation."""

    def __init__(self, message: str = "Not authorized"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details={}
        )


class ConflictError(NANOOSException):
    """Resource conflict (e.g., duplicate entry)."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )


class RateLimitError(NANOOSException):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


class ExternalServiceError(NANOOSException):
    """External service (simulation engine, etc.) failed."""

    def __init__(self, service: str, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"{service} error: {message}",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={"service": service, **(details or {})}
        )


class ParsingError(NANOOSException):
    """Structure file parsing failed."""

    def __init__(self, file_format: str, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Failed to parse {file_format} file: {message}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"format": file_format, **(details or {})}
        )


class SimulationError(NANOOSException):
    """Simulation job execution failed."""

    def __init__(self, job_id: str, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=f"Simulation job {job_id} failed: {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"job_id": job_id, **(details or {})}
        )


# ========== Exception Handlers ==========


async def nanoos_exception_handler(request: Request, exc: NANOOSException) -> JSONResponse:
    """
    Handle custom NANO-OS exceptions.

    Returns consistent error response format:
    {
        "error": {
            "message": "Error message",
            "type": "ExceptionClassName",
            "details": {...}
        }
    }
    """
    logger.error(
        f"{exc.__class__.__name__}: {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "details": exc.details
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.__class__.__name__,
                "details": exc.details
            }
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle standard HTTP exceptions.
    """
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "HTTPException",
                "details": {}
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    Reformats validation errors into user-friendly format.
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(
        f"Validation error: {len(errors)} field(s) invalid",
        extra={
            "path": request.url.path,
            "method": request.method,
            "errors": errors
        }
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation failed",
                "type": "ValidationError",
                "details": {"fields": errors}
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions.

    Logs full traceback and returns generic error to client.
    """
    logger.exception(
        f"Unhandled exception: {exc}",
        extra={
            "path": request.url.path,
            "method": request.method
        }
    )

    # Don't expose internal error details in production
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "An internal error occurred",
                "type": "InternalServerError",
                "details": {}
            }
        }
    )


# ========== Helper Functions ==========


def register_exception_handlers(app):
    """
    Register all exception handlers with FastAPI app.

    Usage in app.py:
        from .exceptions import register_exception_handlers
        register_exception_handlers(app)
    """
    app.add_exception_handler(NANOOSException, nanoos_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
