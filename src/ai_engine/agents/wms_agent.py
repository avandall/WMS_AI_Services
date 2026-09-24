"""
WMS-specific agent with database tools
"""
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from ..models.base import BaseAgent, AgentState
from ..config import settings
from .tools import EnhancedWMSTools
from .policy_engine import DeterministicPolicyEngine


class WMSAgent(BaseAgent):
    """WMS agent with database query capabilities"""
    
    def __init__(self):
        # Initialize Groq LLM
        self.llm_config = settings.get_llm_config()
        self.llm = ChatGroq(**self.llm_config)
        
        # Initialize enhanced tools
        self.enhanced_tools = EnhancedWMSTools()
        
        # Setup tools - use all operational tools
        self.tools = self.enhanced_tools.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
        
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph"""
        
        def call_model(state: AgentState):
            """Agent reasoning node with SOX/GAAP guardrails system prompt"""
            system_prompt = (
                "You are an AI assistant for WMS (Warehouse Management System). "
                "MAKER-CHECKER SECURITY GUARDRAIL ENFORCED: You CANNOT modify inventory quantity or move stock directly in the database. "
                "Any inventory update or stock movement request will generate a DraftAdjustmentProposal or DraftMovementProposal "
                "which must be reviewed and confirmed by a human Warehouse Manager before actual stock levels change."
            )
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: AgentState):
            """Determine if tools are needed"""
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return "__end__"
        
        # Create workflow
        workflow = StateGraph(AgentState)
        
        tool_node = ToolNode(self.tools)
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        return workflow
    
    def process(self, question: str, user_id: str = "authenticated_user") -> str:
        """Process question using agent capabilities with deterministic guardrails"""
        # 1. Deterministic Hard Policy Engine Gate
        policy_eval = DeterministicPolicyEngine.evaluate(question)
        if policy_eval.get("is_violation"):
            return policy_eval.get("message", "Request rejected by policy engine.")

        messages = [HumanMessage(content=question)]
        
        try:
            result = self.app.invoke({"messages": messages})
            final_message = result["messages"][-1]
            return final_message.content
        except Exception as e:
            return f"Error processing request: {str(e)}"
