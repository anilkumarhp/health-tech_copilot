"""
Global configuration and logging setup for Healthcare Copilot.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = {"env_file": ".env"}
    
    # API Configuration
    api_host: str = "localhost"
    api_port: int = 8000
    api_reload: bool = True
    
    # Vector Store Configuration
    chroma_persist_directory: str = "./data/chroma"
    chroma_collection_name: str = "healthcare_policies"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "./logs/healthcare_copilot.log"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # JWT Configuration
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    jwt_refresh_days: int = 7
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    
    # Monitoring
    enable_metrics: bool = True
    metrics_endpoint: str = "/metrics"
    
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost/healthcare_copilot"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_temperature: float = 0.1
    ollama_timeout: int = 60
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # Application Settings
    max_file_size_mb: int = 50
    supported_file_types: str = "pdf,txt,docx"
    



def setup_logging(settings: Settings) -> None:
    """
    Configure global logging with file and console output.
    
    Args:
        settings: Application settings containing log configuration
    """
    # Remove default logger
    logger.remove()
    
    # Create logs directory if it doesn't exist
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Console logging
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    logger.info("Logging configured successfully")


def create_directories(settings: Settings) -> None:
    """
    Create necessary directories for the application.
    
    Args:
        settings: Application settings containing directory paths
    """
    directories = [
        Path(settings.chroma_persist_directory),
        Path(settings.log_file).parent,
        Path("./data/uploads"),
        Path("./data/processed")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


# Global settings instance
settings = Settings()

# Setup logging and directories
setup_logging(settings)
create_directories(settings)

logger.info("Healthcare Copilot configuration loaded successfully")