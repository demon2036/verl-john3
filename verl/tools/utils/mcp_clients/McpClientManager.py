# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from verl.tools.utils.mcp_clients.utils import TokenBucket, mcp2openai

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP tool with session mapping"""
    name: str
    session_index: int
    schema: dict


class ToolRegistry:
    """Registry for MCP tools"""

    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register_batch(self, tools: List[dict], session_index: int) -> None:
        """Register multiple tools from a session"""
        for tool_data in tools:
            tool_name = tool_data['name']
            self._tools[tool_name] = MCPTool(
                name=tool_name,
                session_index=session_index,
                schema=mcp2openai(tool_data)
            )

    def get(self, tool_name: str) -> Optional[MCPTool]:
        """Get tool by name"""
        return self._tools.get(tool_name)

    def list_schemas(self) -> List[dict]:
        """Get all tool schemas"""
        return [tool.schema for tool in self._tools.values()]


class MCPClientManager:
    """MCP Client Manager with streamable HTTP support"""
    initialized = False
    sessions: List[ClientSession] = []
    registry = ToolRegistry()
    rate_limiter: Optional[TokenBucket] = None
    stack: Optional[AsyncExitStack] = None
    _root_server_name = "mcpServers"



    async def initialize(self, config_path: str, rate_limit: float = 10.0) -> None:
        """Initialize the MCP Client Manager and start all clients"""
        if self.initialized:
            return

        # Load configuration
        servers = self._load_servers_config(config_path)

        # Initialize rate limiter
        self.rate_limiter = TokenBucket(rate_limit)

        # Setup connections
        self.stack = AsyncExitStack()
        await self._setup_connections(servers)

        self.initialized = True

    async def _setup_connections(self, servers: Dict[str, Any]) -> None:
        """Setup connections for all servers"""
        session_index = 0

        for server_name, config in servers.items():
            try:
                # Determine client type
                if "url" in config and "auth_token" in config:
                    # SSE client for auth-enabled servers
                    url = config["url"]
                    headers = {"Authorization": f"Bearer {config['auth_token']}"}

                    *connection_args, = await self.stack.enter_async_context(
                        sse_client(url=url, headers=headers)
                    )
                elif "url" in config:
                    # Standard streamablehttp client
                    url = config["url"]

                    *connection_args, = await self.stack.enter_async_context(
                        streamablehttp_client(url=url)
                    )
                else:
                    logger.warning(f"Skipping server {server_name}: missing URL")
                    continue

                # Create session
                read, write = connection_args[:2]
                session = await self.stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                # Store session
                self.sessions.append(session)
                session_index += 1

            except Exception as e:
                logger.error(f"Failed to connect to {server_name}: {e}")

    async def fetch_tool_schemas(self, tool_selected_list: Optional[List[str]] = None) -> List[dict]:
        """Fetch and register tool schemas from all sessions"""
        tool_schemas = []

        for i, session in enumerate(self.sessions):
            try:
                # List tools from session
                tools_response = await session.list_tools()

                # Convert to dict format
                tools_data = []
                for tool in tools_response.tools:
                    tool_dict = tool.model_dump() if hasattr(tool, 'model_dump') else {
                        'name': tool.name,
                        'description': tool.description,
                        'inputSchema': tool.inputSchema
                    }

                    # Filter by selection
                    if not tool_selected_list or tool_dict['name'] in tool_selected_list:
                        tools_data.append(tool_dict)

                # Register tools for this session
                self.registry.register_batch(tools_data, i)

                # Collect schemas
                for tool_dict in tools_data:
                    tool_schemas.append(mcp2openai(tool_dict))

            except Exception as e:
                logger.error(f"Failed to fetch tools from session {i}: {e}")

        return tool_schemas

    async def call_tool(self, tool_name: str, parameters: dict, timeout: float = None) -> Any:
        """Call a tool by name with rate limiting"""
        # Apply rate limiting
        while not self.rate_limiter.acquire():
            await asyncio.sleep(0.1)

        # Get tool info
        tool = self.registry.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Get session
        if tool.session_index >= len(self.sessions):
            raise ValueError(f"Invalid session index for tool {tool_name}")

        session = self.sessions[tool.session_index]

        # Execute tool
        return await session.call_tool(tool_name, parameters)

    def get_client_with_tool_name(self, tool_name: str) -> ClientSession:
        """Get session associated with a tool (for backward compatibility)"""
        tool = self.registry.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        if tool.session_index >= len(self.sessions):
            raise ValueError(f"Invalid session index for tool {tool_name}")

        return self.sessions[tool.session_index]

    def _load_servers_config(self, config_path: str) -> Dict[str, Any]:
        """Load server configuration from file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get(self._root_server_name, {})
        except FileNotFoundError:
            logger.warning(f'Config file "{config_path}" not found')
            return {}
        except Exception as e:
            logger.error(f'Error reading config file "{config_path}": {e}')
            return {}

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup connections on exit"""
        if self.stack:
            await self.stack.__aexit__(exc_type, exc_val, exc_tb)


# Singleton instance for backward compatibility
ClientManager = MCPClientManager()