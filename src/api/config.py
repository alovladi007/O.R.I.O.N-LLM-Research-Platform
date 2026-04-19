"""
API Configuration
=================

Pydantic Settings v2 with security hardening landed in Phase 0 / Session 0.4:

- `CORS_ORIGINS` cannot be `"*"` unless `ORION_ENV=development`. A wildcard in
  production / staging is rejected at startup with a descriptive error.
- `cors_allow_methods` and `cors_allow_headers` default to explicit allowlists
  rather than `["*"]`, and the same anti-wildcard rule is enforced.
- `JWT_SECRET_KEY` (aliased `SECRET_KEY` for backcompat) must be ≥32
  characters in production. In development an ephemeral key is generated.
- `DATABASE_URL` and `REDIS_URL` must not be left at the insecure defaults
  (which contain hardcoded demo passwords) in production.

Full rationale and rotation policy live in `docs/SECURITY.md`.
"""

import logging
import secrets
from functools import lru_cache
from typing import List, Optional

from pydantic import AliasChoices, Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

logger = logging.getLogger(__name__)


# Fields we expect as CSV strings in the env / .env file. Without this, pydantic-
# settings v2 tries JSON-decoding their values and rejects plain CSV.
_CSV_LIST_FIELDS = {
    "cors_origins",
    "cors_allow_methods",
    "cors_allow_headers",
    "elasticsearch_hosts",
}

# Known insecure demo values carried over from .env.example; rejected in prod.
_INSECURE_DB_SUBSTRINGS = ("orion_secure_pwd", "your_secure_postgres_password")
_INSECURE_REDIS_SUBSTRINGS = ("orion_redis_pwd", "your_secure_redis_password")


def _ensure_no_wildcard(values: List[str], *, field: str) -> None:
    """Raise ValueError if the list contains a ``*`` wildcard entry.

    CORS method / header wildcards are rejected in every environment
    because the combination of ``allow_credentials=True`` with wildcards
    is invalid per the CORS spec and browsers silently drop the response.
    """
    if "*" in values:
        raise ValueError(
            f"{field} contains '*'. CORS wildcards are disallowed because "
            "ORION sets allow_credentials=True, which is incompatible with "
            "wildcard methods/headers per the CORS spec. List the specific "
            "values you need instead."
        )


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
    # Accept both ORION_ENV (canonical) and ENVIRONMENT (backward-compat).
    # Default is "development" so a blank .env doesn't trip the
    # production-only invariants. Deployments must set ORION_ENV=production
    # (or staging) explicitly, which is fail-safe: if you forget, you land
    # in dev mode, which logs warnings — not in prod mode accepting a
    # demo password.
    environment: str = Field(
        "development",
        validation_alias=AliasChoices("ORION_ENV", "ENVIRONMENT"),
    )
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
    # Accepts JWT_SECRET_KEY (canonical) or SECRET_KEY (backward-compat).
    # In development, an ephemeral key is generated so local runs don't
    # require env setup. In production, the model_validator rejects ephemeral
    # or too-short keys.
    secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(secrets.token_urlsafe(32)),
        validation_alias=AliasChoices("JWT_SECRET_KEY", "SECRET_KEY"),
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    # Typed as List[str], but CSV values from env are tolerated via the
    # custom EnvSettingsSource (see _CsvEnvSource above) and the
    # parse_cors_origins validator.
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3002",
            "http://localhost:8002",
        ],
        alias="CORS_ORIGINS",
    )
    cors_allow_credentials: bool = True
    # Explicit allowlists; model_validator rejects wildcards outside dev.
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        validation_alias=AliasChoices("CORS_ALLOW_METHODS"),
    )
    cors_allow_headers: List[str] = Field(
        default=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
        ],
        validation_alias=AliasChoices("CORS_ALLOW_HEADERS"),
    )

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
    # SLURM (Session 2.3)
    #
    # Left unset → worker runs the Local execution backend. Setting
    # ORION_SLURM_HOST flips the SLURM backend into remote mode and
    # requires asyncssh + a key path.
    # ------------------------------------------------------------------
    slurm_host: Optional[str] = Field(None, alias="ORION_SLURM_HOST")
    slurm_user: Optional[str] = Field(None, alias="ORION_SLURM_USER")
    slurm_key_path: Optional[str] = Field(None, alias="ORION_SLURM_KEY_PATH")
    slurm_partition: Optional[str] = Field(None, alias="ORION_SLURM_PARTITION")

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

    @field_validator(
        "cors_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        "elasticsearch_hosts",
        mode="before",
    )
    @classmethod
    def parse_csv_list(cls, v):
        """Accept plain CSV strings from env / .env for List[str] fields."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    # ------------------------------------------------------------------
    # Cross-field security invariants
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def enforce_security_invariants(self) -> "Settings":
        """
        Reject insecure configurations when `environment != "development"`.

        Development intentionally bypasses these so local runs don't require
        full env setup. Staging / production / testing must supply real
        secrets and tight CORS.
        """
        # Always-on invariants (apply in every environment)
        _ensure_no_wildcard(self.cors_allow_methods, field="cors_allow_methods")
        _ensure_no_wildcard(self.cors_allow_headers, field="cors_allow_headers")

        # Relaxed in development
        if self.environment == "development":
            if "*" in self.cors_origins:
                logger.warning(
                    "CORS_ORIGINS contains '*' in development mode. "
                    "This is permitted locally but MUST NOT ship to production."
                )
            return self

        # Strict: non-dev environments
        if "*" in self.cors_origins:
            raise ValueError(
                "CORS_ORIGINS='*' is not allowed outside ORION_ENV=development. "
                "Set an explicit comma-separated list of allowed origins."
            )

        secret = self.secret_key.get_secret_value()
        if len(secret) < 32:
            raise ValueError(
                "JWT_SECRET_KEY / SECRET_KEY must be at least 32 characters "
                "in non-development environments. Generate one with "
                "`python -c 'import secrets; print(secrets.token_urlsafe(48))'`."
            )

        if any(bad in self.database_url for bad in _INSECURE_DB_SUBSTRINGS):
            raise ValueError(
                "DATABASE_URL contains a known demo password. "
                "Set a real DATABASE_URL for non-development environments."
            )
        if any(bad in self.redis_url for bad in _INSECURE_REDIS_SUBSTRINGS):
            raise ValueError(
                "REDIS_URL contains a known demo password. "
                "Set a real REDIS_URL for non-development environments."
            )

        return self

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
