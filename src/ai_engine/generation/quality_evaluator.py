"""
Quality evaluation for generated responses
"""
import json
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from ..models.base import BaseEvaluator
from ..config import settings


class QualityEvaluator(BaseEvaluator):
    """Evaluates the quality of generated responses"""
    
    def __init__(self):
        self.llm_config = settings.get_evaluator_config()
        
        # Initialize Groq evaluator LLM
        self.evaluator_llm = ChatGroq(**self.llm_config)
    
    def evaluate(self, question: str, generation: str, context: List[str]) -> Dict[str, Any]:
        """Evaluate generation quality and return score with critique"""
        context_text = "\n\n".join(context)
        
        eval_prompt = f"""You are an expert evaluator for WMS (Warehouse Management System) AI responses.
        
        Evaluate the following response based on:
        1. Accuracy: Does the response correctly use information from the context?
        2. Relevance: Does the response directly answer the question?
        3. Completeness: Does the response provide sufficient information?
        4. Hallucination: Does the response contain information not present in the context?
        
        Context: {context_text}
        
        Question: {question}
        
        Response: {generation}
        
        Return a JSON object with:
        {{
            "score": <0-10>,
            "critique": "<detailed evaluation explaining the score>"
        }}
        
        Be strict - penalize any hallucinated information heavily.
        """
        
        try:
            response = self.evaluator_llm.invoke([HumanMessage(content=eval_prompt)])
            result = json.loads(response.content)
            
            # Ensure required fields exist
            if "score" not in result:
                result["score"] = 0
            if "critique" not in result:
                result["critique"] = "Invalid evaluation format"
                
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            return {
                "score": 0,
                "critique": f"Evaluation error: {str(e)}"
            }
