#!/usr/bin/env python
# api/core/config.py
# Configuration settings for the API

import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """API configuration settings."""
    
    # API settings
    API_VERSION: str = "v1"
    API_PREFIX: str = f"/api/{API_VERSION}"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]  # Allow all origins for development
    
    # Model settings
    MODEL_DIR: str = os.getenv("MODEL_DIR", "../models/checkpoints")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "mediapipe_ml")
    
    # Storage settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
