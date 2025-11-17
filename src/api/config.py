"""
API Configuration
=================

Production-ready configuration management using Pydantic Settings.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator, SecretStr
from functools import lru_cache
import secrets
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "ORION Platform API"
    app_version: str = "2.0.0"
    app_description: str = "AI-driven materials science platform"
    environment: str = Field("production", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # API Settings
    api_prefix: str = "/api/v1"
    docs_url: Optional[str] = "/docs"
    redoc_url: Optional[str] = "/redoc"
    openapi_url: Optional[str] = "/openapi.json"
    
    # Security
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        env="SECRET_KEY"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # CORS
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Database
    database_url: str = Field(
        "postgresql+asyncpg://orion:orion_secure_pwd@localhost:5432/orion_db",
        env="DATABASE_URL"
    )
    database_pool_size: int = 20
    database_max_overflow: int = 40
    database_pool_timeout: int = 30
    
    # Redis
    redis_url: str = Field(
        "redis://:orion_redis_pwd@localhost:6379/0",
        env="REDIS_URL"
    )
    redis_pool_size: int = 10
    
    # Neo4j
    neo4j_uri: str = Field(
        "bolt://localhost:7687",
        env="NEO4J_URI"
    )
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: SecretStr = Field(
        SecretStr("orion_secure_pwd"),
        env="NEO4J_PASSWORD"
    )
    
    # Elasticsearch
    elasticsearch_hosts: List[str] = Field(
        ["http://localhost:9200"],
        env="ELASTICSEARCH_HOSTS"
    )
    elasticsearch_index_prefix: str = "orion"
    
    # MinIO
    minio_endpoint: str = Field("localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field("orion_admin", env="MINIO_ACCESS_KEY")
    minio_secret_key: SecretStr = Field(
        SecretStr("orion_minio_pwd"),
        env="MINIO_SECRET_KEY"
    )
    minio_secure: bool = False
    minio_bucket: str = "orion-data"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # OpenAI
    openai_api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    
    # Anthropic
    anthropic_api_key: Optional[SecretStr] = Field(None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-3-opus-20240229"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_path: str = "/metrics"
    enable_tracing: bool = True
    jaeger_host: str = Field("localhost", env="JAEGER_HOST")
    jaeger_port: int = Field(6831, env="JAEGER_PORT")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = "json"
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Email (for notifications)
    smtp_host: Optional[str] = Field(None, env="SMTP_HOST")
    smtp_port: int = Field(587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(None, env="SMTP_USER")
    smtp_password: Optional[SecretStr] = Field(None, env="SMTP_PASSWORD")
    smtp_from: str = Field("noreply@orion-platform.ai", env="SMTP_FROM")
    
    # OAuth2 Providers
    google_client_id: Optional[str] = Field(None, env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[SecretStr] = Field(None, env="GOOGLE_CLIENT_SECRET")
    github_client_id: Optional[str] = Field(None, env="GITHUB_CLIENT_ID")
    github_client_secret: Optional[SecretStr] = Field(None, env="GITHUB_CLIENT_SECRET")
    
    # Feature Flags
    enable_websocket: bool = True
    enable_graphql: bool = True
    enable_admin_panel: bool = True
    
    # Performance
    worker_count: int = Field(4, env="WORKER_COUNT")
    worker_class: str = "uvicorn.workers.UvicornWorker"
    keepalive: int = 5
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("elasticsearch_hosts", pre=True)
    def parse_elasticsearch_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL"""
        return self.database_url.replace("+asyncpg", "")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Hide sensitive fields in string representation
        fields = {
            "secret_key": {"exclude": True},
            "neo4j_password": {"exclude": True},
            "minio_secret_key": {"exclude": True},
            "openai_api_key": {"exclude": True},
            "anthropic_api_key": {"exclude": True},
            "smtp_password": {"exclude": True},
            "google_client_secret": {"exclude": True},
            "github_client_secret": {"exclude": True},
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export settings instance
settings = get_settings()