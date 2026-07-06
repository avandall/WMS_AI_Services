"""
Advanced RAG workflow with quality control and fallback mechanisms
"""
from typing import List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from ..models.base import GraphState
from ..retrieval import HybridRetriever
from ..generation import LLMGenerator, QualityEvaluator
from ..config import settings


class AdvancedRAGWorkflow:
    """Advanced RAG workflow with retrieval, generation, and quality control"""
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = LLMGenerator()
        self.evaluator = QualityEvaluator()
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the RAG workflow graph"""
        
        def retrieve_node(state: GraphState):
            """Retrieve relevant documents"""
            print("--- [RETRIEVE] Fetching relevant documents ---")
            docs = self.retriever.retrieve(state["question"])
            return {"documents": [doc.page_content for doc in docs]}
        
        def grade_node(state: GraphState):
            """Grade document relevance"""
            print("--- [GRADE] Evaluating document relevance ---")
            question = state["question"]
            docs = state["documents"]
            
            if not docs:
                return {"is_relevant": "no"}
            
            # Simple relevance check using LLM
            grader_prompt = f"""You are a relevance evaluator. 
            Determine if the following documents contain information to answer the question.
            Respond with only 'yes' or 'no'.
            
            Question: {question}
            Document: {docs[0][:500]}...
            """
            
            try:
                response = self.generator.llm.invoke([HumanMessage(content=grader_prompt)])
                score = response.content.lower().strip()
                return {"is_relevant": "yes" if "yes" in score else "no"}
            except:
                return {"is_relevant": "no"}
        
        def transform_query_node(state: GraphState):
            """Transform query for better retrieval"""
            print("--- [TRANSFORM] Reformulating query for better results ---")
            original_question = state["question"]
            transformed_question = f"Provide detailed information about: {original_question}"
            return {"question": transformed_question}
        
        def generate_node(state: GraphState):
            """Generate response"""
            print("--- [GENERATE] Creating response ---")
            question = state["question"]
            documents = state["documents"]
            
            response = self.generator.generate(question, documents)
            return {"generation": response}
        
        def evaluate_node(state: GraphState):
            """Evaluate response quality"""
            print("--- [EVALUATE] Assessing response quality ---")
            question = state["question"]
            generation = state["generation"]
            documents = state["documents"]
            
            evaluation = self.evaluator.evaluate(question, generation, documents)
            return {
                "eval_score": evaluation["score"],
                "eval_critique": evaluation["critique"]
            }
        
        def fallback_node(state: GraphState):
            """Fallback response for failed cases"""
            print("--- [FALLBACK] Providing fallback response ---")
            return {
                "generation": "I apologize, but I couldn't find reliable information to answer your question. Please try rephrasing or contact support for assistance."
            }
        
        # Decision functions
        def decide_to_generate(state: GraphState) -> Literal["generate", "transform_query"]:
            """Decide whether to generate or transform query"""
            if state.get("is_relevant") == "yes":
                return "generate"
            return "transform_query"
        
        def check_quality(state: GraphState) -> Literal["__end__", "fallback"]:
            """Check if response quality meets threshold"""
            score = state.get("eval_score", 0)
            if score >= settings.QUALITY_THRESHOLD:
                print(f"--- [QUALITY] Response accepted with score {score}/10 ---")
                return "__end__"
            print(f"--- [QUALITY] Response rejected with score {score}/10 ---")
            return "fallback"
        
        # Build workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("grade", grade_node)
        workflow.add_node("transform_query", transform_query_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("evaluate", evaluate_node)
        workflow.add_node("fallback", fallback_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges(
            "grade",
            decide_to_generate,
            {
                "generate": "generate",
                "transform_query": "transform_query"
            }
        )
        workflow.add_edge("transform_query", "retrieve")  # Loop back with transformed query
        workflow.add_edge("generate", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            check_quality,
            {
                "__end__": "__end__",
                "fallback": "fallback"
            }
        )
        workflow.add_edge("fallback", "__end__")
        
        return workflow
    
    def process(self, question: str) -> str:
        """Process question through the RAG workflow"""
        try:
            result = self.app.invoke({"question": question})
            return result.get("generation", "Error: No generation produced")
        except Exception as e:
            return f"Error processing question: {str(e)}"
