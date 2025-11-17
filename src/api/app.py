"""
Main FastAPI Application
========================

Production-ready FastAPI application with middleware, security, and monitoring.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
import logging
import time
from typing import Optional

from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import settings
from .middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    SecurityHeadersMiddleware,
    RequestIDMiddleware,
)
from .database import init_db, close_db
from .cache import init_cache, close_cache
from .routers import (
    auth_router,
    health_router,
    materials_router,
    structures_router,
    workflows_router,
    jobs_router,
    ml_router,
    design_router,
    provenance_router,
    campaigns_router,
    mesoscale_router,
    continuum_router,
    orchestrator_router,
    agent_router,
    experiments_router,
    photonics_router,
    batteries_router,
    quantum_router,
    metamaterials_router,
    pcm_router,
    execution_router,
)
from .exceptions import (
    ORIONAPIException,
    handle_orion_exception,
    handle_validation_exception,
    handle_http_exception,
    handle_generic_exception,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if settings.log_format == "text" 
           else '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize cache
        await init_cache()
        logger.info("Cache initialized")
        
        # Initialize tracing if enabled
        if settings.enable_tracing:
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()
            
            jaeger_exporter = JaegerExporter(
                agent_host_name=settings.jaeger_host,
                agent_port=settings.jaeger_port,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
            logger.info("Tracing initialized")
        
        # Initialize other services
        # await init_knowledge_graph()
        # await init_simulation_engine()
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    try:
        # Close database connections
        await close_db()
        logger.info("Database connections closed")
        
        # Close cache connections
        await close_cache()
        logger.info("Cache connections closed")
        
        # Cleanup other resources
        # await close_knowledge_graph()
        # await close_simulation_engine()
        
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    
    # Create FastAPI instance
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url=settings.docs_url if not settings.is_production else None,
        redoc_url=settings.redoc_url if not settings.is_production else None,
        openapi_url=settings.openapi_url if not settings.is_production else None,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add trusted host middleware for production
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.orion-platform.ai", "orion-platform.ai"]
        )
    
    # Add session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key.get_secret_value()
    )
    
    # Add custom middleware
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Add rate limiting
    if settings.rate_limit_enabled:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add exception handlers
    app.add_exception_handler(ORIONAPIException, handle_orion_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_exception)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_generic_exception)
    
    # Include routers
    app.include_router(health_router, tags=["health"])
    app.include_router(
        auth_router,
        prefix=f"{settings.api_prefix}/auth",
        tags=["authentication"]
    )
    app.include_router(
        materials_router,
        prefix=f"{settings.api_prefix}/materials",
        tags=["materials"]
    )
    app.include_router(
        structures_router,
        prefix=f"{settings.api_prefix}/structures",
        tags=["structures"]
    )
    app.include_router(
        workflows_router,
        prefix=f"{settings.api_prefix}/workflows",
        tags=["workflows"]
    )
    app.include_router(
        jobs_router,
        prefix=f"{settings.api_prefix}/jobs",
        tags=["jobs"]
    )
    app.include_router(
        ml_router,
        prefix=f"{settings.api_prefix}",
        tags=["machine-learning"]
    )
    app.include_router(
        design_router,
        prefix=f"{settings.api_prefix}",
        tags=["design"]
    )
    app.include_router(
        provenance_router,
        prefix=f"{settings.api_prefix}",
        tags=["provenance"]
    )
    app.include_router(
        campaigns_router,
        prefix=f"{settings.api_prefix}/campaigns",
        tags=["campaigns"]
    )
    app.include_router(
        mesoscale_router,
        prefix=f"{settings.api_prefix}",
        tags=["mesoscale-simulations"]
    )
    app.include_router(
        continuum_router,
        prefix=f"{settings.api_prefix}",
        tags=["continuum-simulations"]
    )
    app.include_router(
        orchestrator_router,
        prefix=f"{settings.api_prefix}",
        tags=["orchestrator"]
    )
    app.include_router(
        agent_router,
        prefix=f"{settings.api_prefix}",
        tags=["agent"]
    )
    app.include_router(
        experiments_router,
        prefix=f"{settings.api_prefix}",
        tags=["experiments"]
    )
    app.include_router(
        photonics_router,
        prefix=f"{settings.api_prefix}",
        tags=["photonics"]
    )
    app.include_router(
        batteries_router,
        prefix=f"{settings.api_prefix}",
        tags=["batteries"]
    )
    app.include_router(
        quantum_router,
        prefix=f"{settings.api_prefix}",
        tags=["quantum"]
    )
    app.include_router(
        metamaterials_router,
        prefix=f"{settings.api_prefix}",
        tags=["metamaterials"]
    )
    app.include_router(
        pcm_router,
        prefix=f"{settings.api_prefix}",
        tags=["pcm"]
    )
    app.include_router(
        execution_router,
        prefix=f"{settings.api_prefix}",
        tags=["execution"]
    )

    # Add Prometheus metrics if enabled
    if settings.enable_metrics:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint=settings.metrics_path)
    
    # Instrument with OpenTelemetry if tracing is enabled
    if settings.enable_tracing:
        FastAPIInstrumentor.instrument_app(app)
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": settings.app_description,
            "environment": settings.environment,
            "docs": f"{settings.docs_url}" if settings.docs_url else None,
            "health": "/health",
            "metrics": settings.metrics_path if settings.enable_metrics else None,
        }
    
    # API info endpoint
    @app.get(f"{settings.api_prefix}", tags=["api"])
    async def api_info():
        """API information endpoint"""
        return {
            "version": "v1",
            "endpoints": {
                "auth": f"{settings.api_prefix}/auth",
                "materials": f"{settings.api_prefix}/materials",
                "structures": f"{settings.api_prefix}/structures",
                "workflows": f"{settings.api_prefix}/workflows",
                "jobs": f"{settings.api_prefix}/jobs",
                "ml": f"{settings.api_prefix}/ml",
                "design": f"{settings.api_prefix}/design",
                "provenance": f"{settings.api_prefix}/provenance",
                "campaigns": f"{settings.api_prefix}/campaigns",
                "mesoscale": f"{settings.api_prefix}/mesoscale",
                "continuum": f"{settings.api_prefix}/continuum",
            },
        }
    
    return app


# Create application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        workers=settings.worker_count if not settings.is_development else 1,
        log_level=settings.log_level.lower(),
        access_log=True,
    )