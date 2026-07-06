"""
WMS AI Engine - Modular & Agentic RAG Architecture

A comprehensive AI engine for Warehouse Management Systems featuring:
- Hybrid retrieval (vector + keyword search)
- Advanced RAG workflows with quality control
- WMS-specific agents with database tools
- Clean, modular architecture
"""

from .core import WMSEngine, ProcessingMode
from .config import settings
from .models import Document, GraphState, AgentState
from .retrieval import HybridRetriever, DocumentProcessor
from .generation import LLMGenerator, QualityEvaluator
from .agents import WMSAgent
from .workflows import AdvancedRAGWorkflow
from .utils import logger

__version__ = "1.0.0"

__all__ = [
    # Core
    "WMSEngine",
    "ProcessingMode",
    
    # Configuration
    "settings",
    
    # Models
    "Document",
    "GraphState", 
    "AgentState",
    
    # Components
    "HybridRetriever",
    "DocumentProcessor",
    "LLMGenerator",
    "QualityEvaluator",
    "WMSAgent",
    "AdvancedRAGWorkflow",
    
    # Utilities
    "logger"
]