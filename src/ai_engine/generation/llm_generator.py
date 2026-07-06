"""
LLM-based generation for WMS RAG system
"""
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from ..models.base import BaseGenerator
from ..config import settings


class LLMGenerator(BaseGenerator):
    """LLM-based response generator"""
    
    def __init__(self, use_evaluator_model: bool = False):
        self.llm_config = settings.get_evaluator_config() if use_evaluator_model else settings.get_llm_config()
        
        # Initialize Groq LLM
        self.llm = ChatGroq(**self.llm_config)
        
        # Default prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        You are a WMS (Warehouse Management System) AI assistant. 
        Use the provided context to answer the question accurately and concisely.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """)
    
    def generate(self, question: str, context: List[str]) -> str:
        """Generate response based on question and context"""
        context_text = "\n\n".join(context)
        
        # Format prompt
        formatted_prompt = self.prompt.format(
            context=context_text,
            question=question
        )
        
        # Generate response
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return response.content
    
    def set_prompt_template(self, template: str):
        """Set custom prompt template"""
        self.prompt = ChatPromptTemplate.from_template(template)
