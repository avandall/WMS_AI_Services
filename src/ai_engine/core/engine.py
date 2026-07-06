"""
Main WMS AI Engine - orchestrates all components
"""
from typing import Dict, Any, Optional
from enum import Enum
import hashlib

def cached(prefix="", ttl=0):
    """No-op cache decorator for standalone ai-engine runtime."""
    def decorator(fn):
        return fn
    return decorator
from ..workflows import AdvancedRAGWorkflow
from ..agents import WMSAgent
from ..retrieval import HybridRetriever, DocumentProcessor
from ..generation import LLMGenerator
from ..utils import logger, validate_api_keys, create_wms_sample_data
from ..config import settings


class ProcessingMode(Enum):
    """Processing modes for the AI engine"""
    RAG = "rag"
    AGENT = "agent"
    HYBRID = "hybrid"


class WMSEngine:
    """Main WMS AI Engine that orchestrates all components"""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.RAG):
        self.mode = mode
        self._validate_configuration()
        self._initialize_components()
        
        logger.info(f"WMS AI Engine initialized in {mode.value} mode")
    
    def _validate_configuration(self):
        """Validate API keys and configuration"""
        validation_results = validate_api_keys()
        
        if not all(validation_results.values()):
            missing_keys = [k for k, v in validation_results.items() if not v]
            raise ValueError(f"Missing API keys: {missing_keys}")
        
        logger.info("Configuration validation passed")
    
    def _initialize_components(self):
        """Initialize all engine components"""
        # Core components
        self.retriever = HybridRetriever()
        self.document_processor = DocumentProcessor()
        self.generator = LLMGenerator()
        
        # Workflow components
        self.rag_workflow = AdvancedRAGWorkflow()
        self.wms_agent = None
        
        logger.info("All components initialized successfully")
    
    @cached(prefix="ai_query", ttl=86400)  # 24 hour cache for AI responses
    async def process_query(self, question: str, mode: Optional[ProcessingMode] = None) -> Dict[str, Any]:
        """
        Process a query using the specified or default mode
        
        Args:
            question: The user's question
            mode: Processing mode (overrides default if specified)
            
        Returns:
            Dictionary containing response and metadata
        """
        processing_mode = mode or self.mode
        
        logger.info(f"Processing query in {processing_mode.value} mode: {question[:100]}...")
        
        try:
            if processing_mode == ProcessingMode.RAG:
                response = self.rag_workflow.process(question)
                return {
                    "response": response,
                    "mode": "rag",
                    "success": True
                }
            
            elif processing_mode == ProcessingMode.AGENT:
                response = self._get_wms_agent().process(question)
                return {
                    "response": response,
                    "mode": "agent",
                    "success": True
                }
            
            elif processing_mode == ProcessingMode.HYBRID:
                # Try RAG first, fallback to agent if needed
                rag_response = self.rag_workflow.process(question)
                
                # Simple heuristic: if RAG response seems insufficient, try agent
                if len(rag_response) < 50 or "couldn't find" in rag_response.lower():
                    logger.info("RAG response insufficient, trying agent mode")
                    agent_response = self._get_wms_agent().process(question)
                    return {
                        "response": agent_response,
                        "mode": "agent_fallback",
                        "success": True,
                        "rag_response": rag_response
                    }
                
                return {
                    "response": rag_response,
                    "mode": "rag",
                    "success": True
                }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"Error processing your request: {str(e)}",
                "mode": processing_mode.value,
                "success": False,
                "error": str(e)
            }

    def _get_wms_agent(self):
        if self.wms_agent is None:
            self.wms_agent = WMSAgent()
        return self.wms_agent
    
    def add_documents(self, documents: list, metadatas: list = None):
        """Add documents to the knowledge base"""
        try:
            processed_docs = self.document_processor.load_from_text(documents, metadatas)
            self.retriever.add_documents(processed_docs)
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def add_web_documents(self, urls: list):
        """Add documents from web URLs"""
        try:
            processed_docs = self.document_processor.load_from_web(urls)
            self.retriever.add_documents(processed_docs)
            logger.info(f"Added documents from {len(urls)} URLs")
            return True
        except Exception as e:
            logger.error(f"Error adding web documents: {str(e)}")
            return False
    
    def initialize_sample_data(self):
        """Initialize with sample WMS data for testing"""
        sample_data = create_wms_sample_data()
        documents = [item["text"] for item in sample_data]
        metadatas = [item["metadata"] for item in sample_data]
        
        return self.add_documents(documents, metadatas)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine configuration"""
        return {
            "mode": self.mode.value,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "vector_db_path": settings.VECTOR_DB_PATH,
            "retrieval_k": settings.RETRIEVAL_K,
            "quality_threshold": settings.QUALITY_THRESHOLD
        }
