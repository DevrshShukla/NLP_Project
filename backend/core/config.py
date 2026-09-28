import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Study Topic Extractor"
    PIPELINE_VERSION: str = "1.0.0"
    
    # Input Limits
    MAX_TEXT_CHARS: int = 150000
    MIN_TEXT_CHARS: int = 200
    MAX_PDF_MB: int = 10
    MAX_PDF_PAGES: int = 200
    MAX_TOPICS: int = 25
    MIN_TOPICS: int = 5
    MAX_TOPICS_LIMIT: int = 40
    
    # NLP Weights (default sum = 1.0)
    WEIGHT_TFIDF: float = 0.40
    WEIGHT_FREQUENCY: float = 0.25
    WEIGHT_POSITION: float = 0.20
    WEIGHT_SEMANTIC: float = 0.15
    
    # Priority Thresholds
    PRIORITY_HIGH_MIN: float = 80.0
    PRIORITY_MEDIUM_MIN: float = 50.0
    
    # Feature Flags
    ENABLE_SEMANTIC: bool = False
    ENABLE_OCR: bool = False
    ENABLE_LLM: bool = False
    
    # Security and CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "*"]
    RATE_LIMIT_ANALYZE: str = "10/minute"
    RATE_LIMIT_UPLOAD: str = "5/minute"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
