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


class WMSAgent(BaseAgent):
    """WMS agent with database query capabilities"""
    
    def __init__(self):
        # Initialize Groq LLM
        self.llm_config = settings.get_llm_config()
        self.llm = ChatGroq(**self.llm_config)
        
        # Initialize enhanced tools
        self.enhanced_tools = EnhancedWMSTools()
        
        # Setup tools - use all operational tools
        self.tools = [
            # Analytical Tools
            self.enhanced_tools.enhanced_inventory_query,
            self.enhanced_tools.abc_analysis_report,
            self.enhanced_tools.smart_slotting_optimizer,
            
            # Inventory Transaction Tools
            self.enhanced_tools.update_inventory_quantity,
            self.enhanced_tools.adjust_inventory_for_reason,
            
            # Inbound/Outbound Tools
            self.enhanced_tools.move_stock_between_locations,
            
            # Logs & System Tools
            self.enhanced_tools.get_transaction_history,
            self.enhanced_tools.get_stock_movement_history,
            self.enhanced_tools.get_user_activity_summary,
            self.enhanced_tools.create_system_alert,
            
            # Enhanced Operational Tools
            self.enhanced_tools.get_low_stock_report,
            self.enhanced_tools.verify_location_empty
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
        
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph"""
        
        def call_model(state: AgentState):
            """Agent reasoning node"""
            messages = state["messages"]
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
    
    def process(self, question: str) -> str:
        """Process question using agent capabilities"""
        messages = [HumanMessage(content=question)]
        
        try:
            result = self.app.invoke({"messages": messages})
            final_message = result["messages"][-1]
            return final_message.content
        except Exception as e:
            return f"Error processing request: {str(e)}"
