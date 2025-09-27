"""
API Middleware
==============

Custom middleware for logging, metrics, security, and request tracking.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
import logging
from typing import Callable
import json

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Get request ID
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": f"{duration:.3f}s",
            }
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect metrics for monitoring"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Update metrics
            self.request_count += 1
            self.total_duration += time.time() - start_time
            
            if response.status_code >= 400:
                self.error_count += 1
            
            # Add metrics headers
            response.headers["X-Request-Count"] = str(self.request_count)
            response.headers["X-Error-Rate"] = f"{(self.error_count / self.request_count * 100):.2f}%"
            
            return response
            
        except Exception as e:
            self.error_count += 1
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://api.orion-platform.ai"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """Custom compression middleware with more control"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if compression is needed
        accept_encoding = request.headers.get("accept-encoding", "")
        
        if "gzip" in accept_encoding and len(response.body) > 1000:
            # Response will be compressed by GZipMiddleware
            response.headers["Vary"] = "Accept-Encoding"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Custom rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        if client_ip in self.request_times:
            self.request_times[client_ip] = [
                t for t in self.request_times[client_ip] if t > minute_ago
            ]
        else:
            self.request_times[client_ip] = []
        
        # Check if limit exceeded
        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429,
                headers={"Retry-After": "60"},
            )
        
        # Record request time
        self.request_times[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.request_times[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(int(minute_ago + 60))
        
        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Add cache control headers based on endpoint"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Determine cache policy based on path
        path = request.url.path
        
        if path.startswith("/api/v1/materials/search"):
            # Cache search results for 5 minutes
            response.headers["Cache-Control"] = "public, max-age=300"
        elif path.startswith("/api/v1/knowledge"):
            # Cache knowledge graph queries for 10 minutes
            response.headers["Cache-Control"] = "public, max-age=600"
        elif path.startswith("/api/v1/auth") or path.startswith("/api/v1/users"):
            # Never cache auth/user endpoints
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        elif request.method == "GET":
            # Default cache for GET requests
            response.headers["Cache-Control"] = "public, max-age=60"
        else:
            # No cache for non-GET requests
            response.headers["Cache-Control"] = "no-cache"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            
            logger.error(
                f"Unhandled exception",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                },
                exc_info=True
            )
            
            # Return generic error response
            return Response(
                content=json.dumps({
                    "detail": "Internal server error",
                    "request_id": request_id,
                }),
                status_code=500,
                media_type="application/json",
            )