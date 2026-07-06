"""
Base models and interfaces for WMS AI Engine
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, TypedDict, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation for RAG"""
    page_content: str
    metadata: Dict[str, Any]


class GraphState(TypedDict):
    """Base state for graph workflows"""
    question: str
    documents: List[str]
    generation: str
    is_relevant: Optional[str]
    eval_score: Optional[float]
    eval_critique: Optional[str]


class AgentState(TypedDict):
    """State for agent workflows"""
    messages: List[Any]
    question: str
    generation: Optional[str]


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve documents based on query"""
        pass


class BaseGenerator(ABC):
    """Abstract base class for generators"""
    
    @abstractmethod
    def generate(self, question: str, context: List[str]) -> str:
        """Generate response based on question and context"""
        pass


class BaseEvaluator(ABC):
    """Abstract base class for evaluators"""
    
    @abstractmethod
    def evaluate(self, question: str, generation: str, context: List[str]) -> Dict[str, Any]:
        """Evaluate generation quality"""
        pass


class BaseAgent(ABC):
    """Abstract base class for agents"""
    
    @abstractmethod
    def process(self, question: str) -> str:
        """Process question using agent capabilities"""
        pass
