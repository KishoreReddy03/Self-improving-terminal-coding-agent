"""Terminal Coding Agent package."""

from coding_agent.client import LLMClient
from coding_agent.config import Config, load_config_from_env
from coding_agent.models import AgentConfig, ModelResponse, ToolCall
from coding_agent.parser import ResponseParser, StreamToolCallAccumulator
from coding_agent.registry import ToolRegistry
from coding_agent.tools import EditFileTool, ReadFileTool, ShellCommandTool, Tool, WriteFileTool

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "Config",
    "EditFileTool",
    "LLMClient",
    "ModelResponse",
    "ReadFileTool",
    "ResponseParser",
    "ShellCommandTool",
    "StreamToolCallAccumulator",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "WriteFileTool",
    "load_config_from_env",
    "__version__",
]
