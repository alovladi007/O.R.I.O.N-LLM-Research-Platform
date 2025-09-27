"""
Configuration Manager for ORION Platform
=======================================

Handles loading, validation, and management of configuration settings.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    
    class PostgresConfig(BaseModel):
        host: str = "localhost"
        port: int = 5432
        database: str = "orion_db"
        user: str = "orion"
        password: str = "orion_secure_pwd"
        pool_size: int = 20
        max_overflow: int = 40
        
        @property
        def connection_string(self) -> str:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Neo4jConfig(BaseModel):
        uri: str = "bolt://localhost:7687"
        user: str = "neo4j"
        password: str = "neo4j_secure_pwd"
        encrypted: bool = False
        trust: str = "TRUST_ALL_CERTIFICATES"
    
    class RedisConfig(BaseModel):
        host: str = "localhost"
        port: int = 6379
        db: int = 0
        password: Optional[str] = None
        decode_responses: bool = True
        socket_timeout: int = 5
        connection_pool_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"max_connections": 50})
    
    class MongoDBConfig(BaseModel):
        uri: str = "mongodb://localhost:27017"
        database: str = "orion_documents"
        collection: str = "materials"
    
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)


class KnowledgeGraphConfig(BaseModel):
    """Knowledge Graph configuration"""
    
    class OntologyConfig(BaseModel):
        namespaces: Dict[str, str] = Field(default_factory=lambda: {
            "mat": "http://orion.ai/ontology/material#",
            "proc": "http://orion.ai/ontology/process#",
            "prop": "http://orion.ai/ontology/property#",
            "meth": "http://orion.ai/ontology/method#",
            "unit": "http://orion.ai/ontology/unit#"
        })
    
    class SchemaConfig(BaseModel):
        version: str = "1.0.0"
        auto_index: bool = True
        constraints: List[str] = Field(default_factory=list)
    
    class ETLConfig(BaseModel):
        batch_size: int = 1000
        parallel_workers: int = 4
        checkpoint_interval: int = 5000
    
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    schema: SchemaConfig = Field(default_factory=SchemaConfig)
    etl: ETLConfig = Field(default_factory=ETLConfig)


class RAGConfig(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration"""
    
    class EmbeddingConfig(BaseModel):
        model: str = "sentence-transformers/all-mpnet-base-v2"
        dimension: int = 768
        batch_size: int = 32
        normalize: bool = True
    
    class RetrievalConfig(BaseModel):
        sparse_weight: float = 0.4
        dense_weight: float = 0.6
        top_k: int = 8
        rerank_top_n: int = 12
        score_threshold: float = 0.2
        chunk_size: int = 512
        chunk_overlap: int = 20
        
        @validator('sparse_weight')
        def validate_weights(cls, v, values):
            if 'dense_weight' in values:
                assert abs(v + values['dense_weight'] - 1.0) < 0.001, "Weights must sum to 1.0"
            return v
    
    class FAISSConfig(BaseModel):
        index_type: str = "HNSWFlat"
        hnsw_m: int = 32
        ef_construction: int = 200
        ef_search: int = 64
    
    class ElasticsearchConfig(BaseModel):
        index_name: str = "orion_materials"
        shards: int = 3
        replicas: int = 1
    
    class CacheConfig(BaseModel):
        ttl: int = 21600  # 6 hours
        max_size: int = 10000
    
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    fallback_models: List[str] = Field(default_factory=lambda: ["gpt-3.5-turbo-16k", "claude-2.1"])
    
    class LocalConfig(BaseModel):
        model_path: str = "/models/llama-2-70b"
        device: str = "cuda"
        load_in_8bit: bool = True
    
    local: LocalConfig = Field(default_factory=LocalConfig)


class ConfigManager:
    """
    Manages configuration loading and access for the ORION platform.
    
    Features:
    - Environment variable substitution
    - Configuration validation
    - Hot reloading support
    - Multi-source configuration (file, env, CLI)
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self._config: Optional[Dict[str, Any]] = None
        self._last_modified: Optional[float] = None
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.reload()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration path"""
        # Check environment variable first
        if env_path := os.getenv("ORION_CONFIG"):
            return Path(env_path)
        
        # Check common locations
        for path in [
            Path("config/config.yaml"),
            Path("/workspace/orion-platform/config/config.yaml"),
            Path.home() / ".orion" / "config.yaml",
        ]:
            if path.exists():
                return path
        
        raise FileNotFoundError("No configuration file found. Please set ORION_CONFIG environment variable.")
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively substitute environment variables in configuration.
        
        Supports ${VAR_NAME:default_value} syntax.
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Check for environment variable pattern
            import re
            pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) or ""
                return os.getenv(var_name, default_value)
            
            return re.sub(pattern, replacer, config)
        else:
            return config
    
    def reload(self) -> None:
        """Reload configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Substitute environment variables
            self._config = self._substitute_env_vars(raw_config)
            
            # Update last modified time
            self._last_modified = os.path.getmtime(self.config_path)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key.
        
        Example:
            config.get("database.postgres.host")
        """
        if self._config is None:
            self.reload()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {})
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration"""
        return DatabaseConfig(**self.get_section("database"))
    
    @property
    def knowledge_graph(self) -> KnowledgeGraphConfig:
        """Get knowledge graph configuration"""
        return KnowledgeGraphConfig(**self.get_section("knowledge_graph"))
    
    @property
    def rag(self) -> RAGConfig:
        """Get RAG configuration"""
        return RAGConfig(**self.get_section("rag"))
    
    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration"""
        return LLMConfig(**self.get_section("llm"))
    
    def check_reload(self) -> bool:
        """Check if configuration file has been modified and reload if necessary"""
        if self.config_path.exists():
            current_mtime = os.path.getmtime(self.config_path)
            if current_mtime != self._last_modified:
                self.reload()
                return True
        return False
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Validate each section
            _ = self.database
            _ = self.knowledge_graph
            _ = self.rag
            _ = self.llm
            
            logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def export(self, path: Union[str, Path], format: str = "yaml") -> None:
        """Export current configuration to file"""
        path = Path(path)
        
        if format == "yaml":
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration exported to {path}")


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager