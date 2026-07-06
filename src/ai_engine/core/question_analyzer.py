"""
Question Analyzer AI Layer - Analyzes questions to determine optimal processing mode
"""
from typing import Dict, Any, Optional
from enum import Enum
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from ..config import settings
from ..utils import logger


class QuestionType(Enum):
    """Question types for mode selection"""
    KNOWLEDGE = "knowledge"  # RAG - documentation, processes, policies
    DATA_QUERY = "data_query"  # AGENT - specific database queries
    COMPLEX = "complex"  # HYBRID - needs both knowledge and data


class QuestionAnalyzer:
    """AI-powered question analyzer for intelligent mode selection"""
    
    def __init__(self):
        """Initialize the question analyzer with LLM"""
        self.llm_config = settings.get_llm_config()
        self.llm = ChatGroq(**self.llm_config)
        logger.info("Question Analyzer initialized")
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze question and determine optimal processing mode
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with analysis results and recommended mode
        """
        try:
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(question)
            
            # Get LLM analysis
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis_result = self._parse_analysis_response(response.content)
            
            # Map question type to processing mode
            recommended_mode = self._map_to_processing_mode(analysis_result["question_type"])
            
            logger.info(f"Question analyzed: {analysis_result['question_type']} -> {recommended_mode}")
            
            return {
                "question": question,
                "question_type": analysis_result["question_type"],
                "confidence": analysis_result["confidence"],
                "reasoning": analysis_result["reasoning"],
                "recommended_mode": recommended_mode,
                "entities": analysis_result.get("entities", []),
                "keywords": analysis_result.get("keywords", [])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            # Fallback to RAG mode on analysis failure
            return {
                "question": question,
                "question_type": "knowledge",
                "confidence": 0.5,
                "reasoning": f"Analysis failed, defaulting to RAG mode: {str(e)}",
                "recommended_mode": "rag",
                "entities": [],
                "keywords": []
            }
    
    def _create_analysis_prompt(self, question: str) -> str:
        """Create the analysis prompt for the LLM"""
        return f"""You are a question classifier for a Warehouse Management System (WMS) AI assistant.

Analyze the following question and classify it into one of these categories:

1. **KNOWLEDGE** - Questions about:
   - WMS processes, procedures, workflows
   - Documentation, policies, best practices
   - General information about warehousing
   - "How to", "What is", "Explain" type questions
   - Questions that don't require specific database lookups

2. **DATA_QUERY** - Questions that require specific database queries:
   - Inventory levels for specific SKUs/products
   - Order status and tracking
   - Location information and warehouse data
   - ABC analysis and slotting recommendations
   - Questions containing specific identifiers (SKU codes, order IDs, location codes)
   - "How many", "Where is", "What is the status of" type questions with specific entities
   - "ABC analysis", "slotting", "storage recommendation", "optimal location" queries

3. **COMPLEX** - Questions that likely need both:
   - Multi-step processes requiring both knowledge and data
   - Questions combining general procedures with specific data
   - Complex scenarios requiring analysis of multiple data points

Question to analyze: "{question}"

Provide your response in this exact format:
QUESTION_TYPE: [knowledge|data_query|complex]
CONFIDENCE: [0.1-1.0]
REASONING: [Brief explanation of your choice]
ENTITIES: [List any specific entities found (SKUs, order IDs, locations)]
KEYWORDS: [List key terms that influenced your decision]

Example:
QUESTION_TYPE: data_query
CONFIDENCE: 0.9
REASONING: User is asking for specific inventory information for SKU code ABC-123
ENTITIES: ["ABC-123"]
KEYWORDS: ["inventory", "quantity", "ABC-123"]
"""
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM analysis response"""
        result = {
            "question_type": "knowledge",
            "confidence": 0.5,
            "reasoning": "Default analysis",
            "entities": [],
            "keywords": []
        }
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('QUESTION_TYPE:'):
                    result["question_type"] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('CONFIDENCE:'):
                    result["confidence"] = float(line.split(':', 1)[1].strip())
                elif line.startswith('REASONING:'):
                    result["reasoning"] = line.split(':', 1)[1].strip()
                elif line.startswith('ENTITIES:'):
                    entities_str = line.split(':', 1)[1].strip()
                    if entities_str and entities_str != "[]":
                        # Simple parsing - could be improved
                        result["entities"] = [e.strip() for e in entities_str.strip('[]').split(',')]
                elif line.startswith('KEYWORDS:'):
                    keywords_str = line.split(':', 1)[1].strip()
                    if keywords_str and keywords_str != "[]":
                        result["keywords"] = [k.strip() for k in keywords_str.strip('[]').split(',')]
        except Exception as e:
            logger.warning(f"Error parsing analysis response: {str(e)}")
        
        return result
    
    def _map_to_processing_mode(self, question_type: str) -> str:
        """Map question type to processing mode"""
        mapping = {
            "knowledge": "rag",
            "data_query": "agent", 
            "complex": "hybrid"
        }
        return mapping.get(question_type, "rag")
    
    def get_analysis_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the analysis"""
        return f"""Question Analysis Summary:
- Type: {analysis_result['question_type'].upper()}
- Recommended Mode: {analysis_result['recommended_mode'].upper()}
- Confidence: {analysis_result['confidence']:.1f}
- Reasoning: {analysis_result['reasoning']}
- Entities Found: {len(analysis_result['entities'])}
- Keywords: {', '.join(analysis_result['keywords'])}
"""
