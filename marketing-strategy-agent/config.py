"""
Configuration settings for the Marketing Strategy Agent.

This module handles all configuration settings using Pydantic Settings
for type safety and validation.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Application
    app_name: str = Field(default="Marketing Strategy Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL"
    )
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    
    # TiDB Configuration
    tidb_host: str = Field(..., env="TIDB_HOST")
    tidb_port: int = Field(default=4000, env="TIDB_PORT")
    tidb_user: str = Field(..., env="TIDB_USER")
    tidb_password: str = Field(..., env="TIDB_PASSWORD")
    tidb_database: str = Field(..., env="TIDB_DATABASE")
    tidb_ssl_ca: Optional[str] = Field(default=None, env="TIDB_SSL_CA")
    tidb_ssl_verify: bool = Field(default=True, env="TIDB_SSL_VERIFY")
    
    # Vector Store Configuration
    vector_dimension: int = Field(default=1536, env="VECTOR_DIMENSION")
    vector_distance_metric: str = Field(default="cosine", env="VECTOR_DISTANCE_METRIC")
    max_vector_results: int = Field(default=10, env="MAX_VECTOR_RESULTS")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Cache Configuration
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    # External APIs (Optional)
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    serpapi_key: Optional[str] = Field(default=None, env="SERPAPI_KEY")
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    facebook_access_token: Optional[str] = Field(default=None, env="FACEBOOK_ACCESS_TOKEN")
    linkedin_access_token: Optional[str] = Field(default=None, env="LINKEDIN_ACCESS_TOKEN")
    
    # Logging Configuration
    log_file_path: str = Field(default="./logs/marketing_agent.log", env="LOG_FILE_PATH")
    structured_logging: bool = Field(default=True, env="STRUCTURED_LOGGING")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Security
    secret_key: str = Field(default="development-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # Knowledge Base
    knowledge_base_path: str = Field(default="data/knowledge_base", env="KNOWLEDGE_BASE_PATH")
    
    # Vector Store (using compatible names with .env)
    vector_table_name: str = Field(default="marketing_knowledge", env="VECTOR_TABLE_NAME")
    vector_distance_strategy: str = Field(default="cosine", env="VECTOR_DISTANCE_STRATEGY")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("vector_distance_metric")
    def validate_distance_metric(cls, v):
        """Validate vector distance metric."""
        valid_metrics = ["cosine", "euclidean", "dot_product"]
        if v.lower() not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v.lower()
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    def get_database_url(self) -> str:
        """Get the complete database URL for TiDB."""
        ssl_params = ""
        if self.tidb_ssl_verify and self.tidb_ssl_ca:
            ssl_params = f"?ssl_ca={self.tidb_ssl_ca}&ssl_verify_cert=true"
        elif self.tidb_ssl_verify:
            ssl_params = "?ssl_verify_cert=true"
        
        return (
            f"mysql+pymysql://{self.tidb_user}:{self.tidb_password}"
            f"@{self.tidb_host}:{self.tidb_port}/{self.tidb_database}{ssl_params}"
        )
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_level = getattr(logging, self.log_level)
        
        if self.structured_logging and self.log_format == "json":
            import structlog
            
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
        else:
            # Standard logging configuration
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                handlers=[
                    logging.FileHandler(self.log_file_path),
                    logging.StreamHandler()
                ]
            )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra environment variables


# Global settings instance
settings = Settings()

# Setup logging on import
settings.setup_logging()


def get_settings() -> Settings:
    """Get application settings."""
    return settings