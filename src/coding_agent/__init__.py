"""Terminal Coding Agent package."""

from coding_agent.agent import Agent
from coding_agent.client import LLMClient
from coding_agent.config import Config, load_config_from_env
from coding_agent.evaluator import EvaluationResult, TrajectoryEvaluator
from coding_agent.memory import Experience, ExperienceMemory
from coding_agent.models import AgentConfig, AgentResult, ModelResponse, ToolCall
from coding_agent.parser import ResponseParser, StreamToolCallAccumulator
from coding_agent.pipeline import ExperiencePipeline, PipelineResult
from coding_agent.reflection import ReflectionGenerator, ReflectionResult
from coding_agent.registry import ToolRegistry
from coding_agent.run_record import RunEvent, RunEventType, RunOutcome, RunRecord
from coding_agent.tools import EditFileTool, ReadFileTool, ShellCommandTool, Tool, WriteFileTool
from coding_agent.trajectory_store import TrajectoryStore
from coding_agent.weakness_analyzer import (
    ImprovementProposal,
    Severity,
    TrajectoryEntry,
    WeaknessAnalyzer,
    WeaknessEvidence,
    WeaknessReport,
    WeaknessType,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Config",
    "EditFileTool",
    "EvaluationResult",
    "Experience",
    "ExperienceMemory",
    "ExperiencePipeline",
    "LLMClient",
    "ModelResponse",
    "PipelineResult",
    "ReadFileTool",
    "ResponseParser",
    "ReflectionGenerator",
    "ReflectionResult",
    "RunEvent",
    "RunEventType",
    "RunOutcome",
    "RunRecord",
    "ShellCommandTool",
    "StreamToolCallAccumulator",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "TrajectoryEntry",
    "TrajectoryEvaluator",
    "TrajectoryStore",
    "ImprovementProposal",
    "Severity",
    "WeaknessAnalyzer",
    "WeaknessEvidence",
    "WeaknessReport",
    "WeaknessType",
    "WriteFileTool",
    "load_config_from_env",
    "__version__",
]
