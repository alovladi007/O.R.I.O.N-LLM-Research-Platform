"""
API Configuration
=================

Production-ready configuration management using Pydantic Settings (v2).

Migrated from Pydantic v1 style in Phase 0 / Session 0.1 to unblock import of
the canonical backend entry point. Full security hardening (CORS validation,
JWT length enforcement, env-based secret backends) lands in Session 0.4.
"""

from typing import List, Optional
from functools import lru_cache
import secrets

from pydantic import Field, SecretStr, field_validator, computed_field
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


# Fields we expect as CSV strings in the env / .env file. Without this, pydantic-
# settings v2 tries JSON-decoding their values and rejects plain CSV.
_CSV_LIST_FIELDS = {"cors_origins", "elasticsearch_hosts"}


class _CsvEnvSource(EnvSettingsSource):
    """EnvSettingsSource that refuses to JSON-decode our CSV list fields."""

    def prepare_field_value(self, field_name, field, value, value_is_complex):
        if field_name in _CSV_LIST_FIELDS and isinstance(value, str):
            return value  # validator will split
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class _CsvDotEnvSource(DotEnvSettingsSource):
    """DotEnvSettingsSource with the same CSV-list carveout."""

    def prepare_field_value(self, field_name, field, value, value_is_complex):
        if field_name in _CSV_LIST_FIELDS and isinstance(value, str):
            return value
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # accept unknown env vars without blowing up
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        # Replace the default env / dotenv sources with our CSV-aware versions
        # so fields like CORS_ORIGINS can be plain comma-separated strings.
        return (
            init_settings,
            _CsvEnvSource(settings_cls),
            _CsvDotEnvSource(settings_cls),
            file_secret_settings,
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "ORION Platform API"
    app_version: str = "2.0.0"
    app_description: str = "AI-driven materials science platform"
    environment: str = Field("production", alias="ENVIRONMENT")
    debug: bool = Field(False, alias="DEBUG")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    api_prefix: str = "/api/v1"
    docs_url: Optional[str] = "/docs"
    redoc_url: Optional[str] = "/redoc"
    openapi_url: Optional[str] = "/openapi.json"

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        alias="SECRET_KEY",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    # Typed as str at the env-parsing boundary; the validator splits into a
    # list. pydantic-settings tries to JSON-decode List[str] fields from env,
    # which breaks on plain CSV values.
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:3002",
            "http://localhost:8000",
            "http://localhost:8002",
        ],
        alias="CORS_ORIGINS",
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]  # tightened in Session 0.4
    cors_allow_headers: List[str] = ["*"]  # tightened in Session 0.4

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(
        "postgresql+asyncpg://orion:orion_secure_pwd@localhost:5432/orion_db",
        alias="DATABASE_URL",
    )
    database_pool_size: int = 20
    database_max_overflow: int = 40
    database_pool_timeout: int = 30

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    redis_url: str = Field(
        "redis://:orion_redis_pwd@localhost:6379/0",
        alias="REDIS_URL",
    )
    redis_pool_size: int = 10

    # ------------------------------------------------------------------
    # Neo4j  (scheduled for removal in Session 0.2 if unused)
    # ------------------------------------------------------------------
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: SecretStr = Field(
        default=SecretStr("orion_secure_pwd"), alias="NEO4J_PASSWORD"
    )

    # ------------------------------------------------------------------
    # Elasticsearch
    # ------------------------------------------------------------------
    elasticsearch_hosts: List[str] = Field(
        default=["http://localhost:9200"], alias="ELASTICSEARCH_HOSTS"
    )
    elasticsearch_index_prefix: str = "orion"

    # ------------------------------------------------------------------
    # MinIO
    # ------------------------------------------------------------------
    minio_endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field("orion_admin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: SecretStr = Field(
        default=SecretStr("orion_minio_pwd"), alias="MINIO_SECRET_KEY"
    )
    minio_secure: bool = False
    minio_bucket: str = "orion-data"

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds

    # ------------------------------------------------------------------
    # LLM providers
    # ------------------------------------------------------------------
    openai_api_key: Optional[SecretStr] = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000

    anthropic_api_key: Optional[SecretStr] = Field(None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-opus-4-7"

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_path: str = "/metrics"
    enable_tracing: bool = False  # off by default; opt in via ENABLE_TRACING=true
    jaeger_host: str = Field("localhost", alias="JAEGER_HOST")
    jaeger_port: int = Field(6831, alias="JAEGER_PORT")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_format: str = "json"
    log_file: Optional[str] = Field(None, alias="LOG_FILE")

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------
    smtp_host: Optional[str] = Field(None, alias="SMTP_HOST")
    smtp_port: int = Field(587, alias="SMTP_PORT")
    smtp_user: Optional[str] = Field(None, alias="SMTP_USER")
    smtp_password: Optional[SecretStr] = Field(None, alias="SMTP_PASSWORD")
    smtp_from: str = Field("noreply@orion-platform.ai", alias="SMTP_FROM")

    # ------------------------------------------------------------------
    # OAuth
    # ------------------------------------------------------------------
    google_client_id: Optional[str] = Field(None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[SecretStr] = Field(None, alias="GOOGLE_CLIENT_SECRET")
    github_client_id: Optional[str] = Field(None, alias="GITHUB_CLIENT_ID")
    github_client_secret: Optional[SecretStr] = Field(None, alias="GITHUB_CLIENT_SECRET")

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    enable_websocket: bool = True
    enable_graphql: bool = True
    enable_admin_panel: bool = True

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    worker_count: int = Field(4, alias="WORKER_COUNT")
    worker_class: str = "uvicorn.workers.UvicornWorker"
    keepalive: int = 5

    # ==================================================================
    # Validators
    # ==================================================================
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production", "testing"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {sorted(allowed)}")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("elasticsearch_hosts", mode="before")
    @classmethod
    def parse_elasticsearch_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",") if host.strip()]
        return v

    # ==================================================================
    # Computed / helpers
    # ==================================================================
    @computed_field  # type: ignore[misc]
    @property
    def database_url_sync(self) -> str:
        """Synchronous SQLAlchemy URL (strips +asyncpg driver suffix)."""
        return self.database_url.replace("+asyncpg", "")

    @computed_field  # type: ignore[misc]
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @computed_field  # type: ignore[misc]
    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()


# Module-level settings instance used by the rest of the app.
settings = get_settings()
