"""
Configuration settings for the Marketing Strategy Agent
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    # Load environment variables from .env file
    load_dotenv()
    
    config = {
        # OpenAI Configuration
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "OPENAI_TEMPERATURE": float(os.getenv("OPENAI_TEMPERATURE", "0.4")),
        "OPENAI_MAX_TOKENS": int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
        
        # Gemini Configuration
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest"),
        
        # Tavily Configuration
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "TAVILY_MAX_RESULTS": int(os.getenv("TAVILY_MAX_RESULTS", "10")),
        
        # TiDB Vector Database Configuration
        "TIDB_HOST": os.getenv("TIDB_HOST"),
        "TIDB_PORT": int(os.getenv("TIDB_PORT", "4000")),
        "TIDB_USER": os.getenv("TIDB_USER"),
        "TIDB_PASSWORD": os.getenv("TIDB_PASSWORD"),
        "TIDB_DATABASE": os.getenv("TIDB_DATABASE"),
        
        # Application Configuration
        "APP_NAME": os.getenv("APP_NAME", "Marketing Strategy Agent"),
        "APP_VERSION": os.getenv("APP_VERSION", "1.0.0"),
        "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        
        # API Configuration
        "HOST": os.getenv("HOST", "0.0.0.0"),
        "PORT": int(os.getenv("PORT", "8000")),
        "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "*").split(","),
        
        # Workflow Configuration
        "DEFAULT_EXECUTION_MODE": os.getenv("DEFAULT_EXECUTION_MODE", "hybrid"),
        "MAX_CONCURRENT_WORKFLOWS": int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "5")),
        "WORKFLOW_TIMEOUT_SECONDS": int(os.getenv("WORKFLOW_TIMEOUT_SECONDS", "1800")),
        
        # File Storage Configuration
        "STORAGE_PATH": os.getenv("STORAGE_PATH", "./data/storage"),
        "REPORTS_PATH": os.getenv("REPORTS_PATH", "./data/reports"),
        "LOGS_PATH": os.getenv("LOGS_PATH", "./data/logs"),
    }
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate that required configuration values are present"""
    required_keys = [
        "OPENAI_API_KEY"
    ]
    
    missing_keys = []
    for key in required_keys:
        if not config.get(key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True


def get_database_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get database configuration"""
    return {
        "host": config.get("TIDB_HOST"),
        "port": config.get("TIDB_PORT"),
        "user": config.get("TIDB_USER"),
        "password": config.get("TIDB_PASSWORD"),
        "database": config.get("TIDB_DATABASE")
    }


def get_api_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get API configuration"""
    return {
        "openai": {
            "api_key": config.get("OPENAI_API_KEY"),
            "model": config.get("OPENAI_MODEL"),
            "temperature": config.get("OPENAI_TEMPERATURE"),
            "max_tokens": config.get("OPENAI_MAX_TOKENS")
        },
        "gemini": {
            "api_key": config.get("GEMINI_API_KEY"),
            "model": config.get("GEMINI_MODEL")
        },
        "tavily": {
            "api_key": config.get("TAVILY_API_KEY"),
            "max_results": config.get("TAVILY_MAX_RESULTS")
        }
    }