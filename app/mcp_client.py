"""
MCP Client for connecting to WMS MCP Server
"""
from __future__ import annotations

import os
import sys
from typing import Any, List, Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import tool

from ai_engine.utils.logger import logger


class WMSMCPClient:
    """Client to connect to and invoke tools on WMS MCP Server"""

    def __init__(self):
        # Retrieve server launch command from environment
        command_str = os.getenv("MCP_SERVER_COMMAND", "python -m main")
        self.command = command_str.split()
        
        # Determine the working directory for WMS_MCP_Server
        # It should be a sibling directory to WMS_AI_Services
        self.cwd = os.getenv(
            "MCP_SERVER_CWD",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "WMS_MCP_Server"))
        )
        
        self.session: ClientSession | None = None
        self.client_context = None

    async def connect(self) -> None:
        """Establishes connection to the MCP server via stdio transport"""
        if self.session is not None:
            return

        logger.info(f"Connecting to MCP server via: {self.command} in cwd: {self.cwd}")
        
        if not os.path.exists(self.cwd):
            logger.warning(f"MCP CWD does not exist: {self.cwd}. Using default execution path.")
            self.cwd = None

        server_params = StdioServerParameters(
            command=self.command[0],
            args=self.command[1:],
            env=os.environ.copy(),
        )
        
        # Override working directory if running in stdio
        self.client_context = stdio_client(server_params)
        read_stream, write_stream = await self.client_context.__aenter__()
        
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        logger.info("Successfully connected and initialized WMS MCP Server session.")

    async def get_tools(self) -> List[Any]:
        """Lists tools on the MCP server and wraps them for LangChain compatibility"""
        await self.connect()
        assert self.session is not None
        
        resp = await self.session.list_tools()
        langchain_tools = []
        
        for mcp_tool in resp.tools:
            langchain_tools.append(self._wrap_tool(mcp_tool))
            
        logger.info(f"Dynamically registered {len(langchain_tools)} tools from MCP server.")
        return langchain_tools

    def _wrap_tool(self, mcp_tool) -> Any:
        """Wraps an MCP tool into a LangChain StructuredTool"""
        # Define a dynamic function wrapper
        async def call_mcp_tool(**kwargs) -> str:
            assert self.session is not None
            logger.info(f"Calling MCP tool '{mcp_tool.name}' with args: {kwargs}")
            try:
                res = await self.session.call_tool(mcp_tool.name, arguments=kwargs)
                # Combine contents into a single string
                text_contents = [c.text for c in res.content if hasattr(c, 'text')]
                return "\n".join(text_contents)
            except Exception as e:
                logger.error(f"Error calling MCP tool '{mcp_tool.name}': {e}")
                return f"Error executing tool: {e}"

        # Assign correct name and docstring for the wrapped tool
        call_mcp_tool.__name__ = mcp_tool.name
        call_mcp_tool.__doc__ = mcp_tool.description or f"WMS MCP tool: {mcp_tool.name}"

        # Build schema using the parameters schema provided by the MCP server
        # LangChain's StructuredTool helper will build validation from it
        from langchain_core.tools import StructuredTool
        
        return StructuredTool.from_function(
            coroutine=call_mcp_tool,
            name=mcp_tool.name,
            description=mcp_tool.description,
        )

    async def close(self) -> None:
        """Clean up connections and streams"""
        if self.session is not None:
            await self.session.__aexit__(None, None, None)
            self.session = None
        if self.client_context is not None:
            await self.client_context.__aexit__(None, None, None)
            self.client_context = None
        logger.info("Closed WMS MCP Server session.")
