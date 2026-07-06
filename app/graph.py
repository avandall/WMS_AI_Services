"""
LangGraph Agent state graph implementation utilizing WMS MCP tools
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ai_engine.config import settings
from ai_engine.models.base import AgentState
from ai_engine.utils.logger import logger
from app.mcp_client import WMSMCPClient


class WMSLangGraphAgent:
    """LangGraph orchestrator agent dynamically loading and executing tools from WMS MCP Server"""

    def __init__(self):
        self.llm_config = settings.get_llm_config()
        self.llm = ChatGroq(**self.llm_config)
        self.mcp_client = WMSMCPClient()
        self.app = None
        self.tools = []

    async def initialize(self) -> None:
        """Connects to the MCP client, loads all tools, and compiles the workflow graph"""
        if self.app is not None:
            return

        # Fetch and wrap tools from the MCP server
        self.tools = await self.mcp_client.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Define the state machine nodes
        def call_model(state: AgentState) -> Dict[str, Any]:
            """Agent node that decides what to do next"""
            logger.info("LangGraph Agent executing reasoning node...")
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """Conditional edge to check if tool calls are requested"""
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                logger.info(f"Agent requested tool calls: {last_message.tool_calls}")
                return "tools"
            logger.info("Agent decided to terminate reasoning loop.")
            return END

        # Define the workflow graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", call_model)
        
        # ToolNode takes the tool list and acts as a router executing requested tools
        tool_node = ToolNode(self.tools)
        workflow.add_node("tools", tool_node)

        # Add edges and flows
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {
            "tools": "tools",
            END: END
        })
        workflow.add_edge("tools", "agent")

        # Compile graph
        self.app = workflow.compile()
        logger.info("LangGraph agent workflow successfully compiled.")

    async def process(self, question: str) -> str:
        """Runs the LangGraph agent to answer a question"""
        await self.initialize()
        assert self.app is not None

        state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "generation": None
        }

        try:
            result = await self.app.ainvoke(state)
            final_message = result["messages"][-1]
            return str(final_message.content)
        except Exception as e:
            logger.error(f"Error executing LangGraph agent flow: {e}")
            return f"Agent failed to process request: {e}"
        finally:
            # We can choose to keep the MCP session alive or shut it down.
            # To avoid subprocess leaks, let's close the client connection on termination.
            await self.mcp_client.close()
