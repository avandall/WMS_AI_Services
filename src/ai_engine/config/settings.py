"""
Configuration settings for WMS AI Engine
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Central configuration class for WMS AI Engine"""
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    EVALUATOR_MODEL: str = os.getenv("EVALUATOR_MODEL", "llama-3.3-70b-versatile")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Embedding Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # Database Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    DB_CONNECTION_STRING: str = os.getenv("DB_CONNECTION_STRING", "")
    
    # Retrieval Configuration
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "3"))
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.5"))
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))
    
    # Evaluation Configuration (scale 0-10 to match evaluator)
    QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", "5.0"))
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Optional fine-tuned local model for data/query extraction
    FINE_TUNED_MODEL_PATH: str = os.getenv("FINE_TUNED_MODEL_PATH", "").strip()
    FINE_TUNED_MODEL_DEVICE: str = os.getenv("FINE_TUNED_MODEL_DEVICE", "cpu").strip()
    FINE_TUNED_MAX_NEW_TOKENS: int = int(os.getenv("FINE_TUNED_MAX_NEW_TOKENS", "256"))
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration dictionary"""
        return {
            "model": cls.LLM_MODEL,
            "temperature": cls.TEMPERATURE,
            "api_key": cls.GROQ_API_KEY
        }
    
    @classmethod
    def get_evaluator_config(cls) -> Dict[str, Any]:
        """Get evaluator LLM configuration"""
        return {
            "model": cls.EVALUATOR_MODEL,
            "temperature": 0.0,
            "api_key": cls.GROQ_API_KEY
        }

settings = Settings()
