"""mcp_client.py — MCP toolset integration with native fallbacks for Code Helper agents."""

import os
import logging
from pathlib import Path
from typing import Any

try:
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioServerParameters
    _MCP_AVAILABLE = True
except ImportError:
    McpToolset = None
    StdioServerParameters = None
    _MCP_AVAILABLE = False

from .common import PROJECT_DIR
from .code_tools import (
    read_file as native_read_file,
    import_graph_mermaid as native_import_graph_mermaid,
)

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages MCP server connections with automatic fallback to native tools."""

    def __init__(self):
        self._filesystem_toolset: McpToolset | None = None
        self._mermaid_toolset: McpToolset | None = None
        self._filesystem_available = False
        self._mermaid_available = False

    def get_filesystem_toolset(self):
        """Get or create the filesystem MCP toolset with fallback."""
        if not _MCP_AVAILABLE:
            self._filesystem_available = False
            return None

        if self._filesystem_toolset is not None:
            return self._filesystem_toolset

        try:
            project_dir_str = str(PROJECT_DIR.resolve())
            self._filesystem_toolset = McpToolset(
                connection_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", project_dir_str],
                )
            )
            self._filesystem_available = True
            logger.info("Filesystem MCP toolset initialized successfully")
            return self._filesystem_toolset
        except Exception as e:
            logger.warning(f"Filesystem MCP unavailable, using native fallback: {e}")
            self._filesystem_available = False
            return None

    def get_mermaid_toolset(self):
        """Get or create the mermaid MCP toolset with fallback."""
        if not _MCP_AVAILABLE:
            self._mermaid_available = False
            return None

        if self._mermaid_toolset is not None:
            return self._mermaid_toolset

        try:
            self._mermaid_toolset = McpToolset(
                connection_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-mermaid"],
                )
            )
            self._mermaid_available = True
            logger.info("Mermaid MCP toolset initialized successfully")
            return self._mermaid_toolset
        except Exception as e:
            logger.warning(f"Mermaid MCP unavailable, using native fallback: {e}")
            self._mermaid_available = False
            return None

    def is_filesystem_available(self) -> bool:
        """Check if filesystem MCP is available."""
        if not self._filesystem_available:
            self.get_filesystem_toolset()
        return self._filesystem_available

    def is_mermaid_available(self) -> bool:
        """Check if mermaid MCP is available."""
        if not self._mermaid_available:
            self.get_mermaid_toolset()
        return self._mermaid_available


_mcp_client = MCPClient()


async def mcp_read_file(file_name: str) -> str:
    """Read a file using filesystem MCP with native fallback."""
    if not _MCP_AVAILABLE:
        return native_read_file(file_name)

    toolset = _mcp_client.get_filesystem_toolset()
    if toolset:
        try:
            tools = await toolset.get_tools()
            read_tool = next((t for t in tools if t.name == "read_file"), None)
            if read_tool:
                result = await read_tool.run_async({"path": file_name})
                return result
        except Exception as e:
            logger.warning(f"MCP read_file failed, falling back: {e}")

    return native_read_file(file_name)


async def mcp_generate_mermaid(diagram_code: str) -> str:
    """Generate mermaid diagram using MCP with native fallback."""
    if not _MCP_AVAILABLE:
        return native_import_graph_mermaid()

    toolset = _mcp_client.get_mermaid_toolset()
    if toolset:
        try:
            tools = await toolset.get_tools()
            mermaid_tool = next((t for t in tools if t.name == "generate_diagram"), None)
            if mermaid_tool:
                result = await mermaid_tool.run_async({"code": diagram_code})
                return result
        except Exception as e:
            logger.warning(f"MCP mermaid generation failed, falling back: {e}")

    return native_import_graph_mermaid()


def get_mcp_client() -> MCPClient:
    """Get the singleton MCP client instance."""
    return _mcp_client


async def mcp_read_file_or_fallback(file_name: str) -> str:
    """Read a file using filesystem MCP with native fallback.
    Args:
        file_name: The name or relative path of the file to read (e.g. 'vector_store.py').
    """
    return await mcp_read_file(file_name)


async def mcp_mermaid_or_fallback(diagram_code: str = "") -> str:
    """Generate mermaid diagram using MCP with native fallback.
    Args:
        diagram_code: Optional mermaid diagram source code string.
    """
    return await mcp_generate_mermaid(diagram_code)