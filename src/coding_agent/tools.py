"""Tool abstraction and implementations for the terminal coding agent."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    is_read_only: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_read_only: Optional[bool] = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        if is_read_only is not None:
            self.is_read_only = is_read_only

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the provided arguments."""
        pass

    def to_function_schema(self) -> Dict[str, Any]:
        """Convert the tool into the function schema expected by model APIs."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ReadFileTool(Tool):
    """Tool for reading text file contents from disk."""

    name = "read_file"
    description = "Read and return the text contents of a specified file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute file path to read.",
            },
        },
        "required": ["path"],
    }
    is_read_only = True

    def execute(self, path: str = "", **kwargs: Any) -> str:
        """Read and return contents of a file cleanly handling error conditions."""
        if not path:
            return "Error: 'path' parameter is required."

        target_path = Path(path)

        if not target_path.exists():
            return f"Error: File '{path}' does not exist."

        if target_path.is_dir():
            return f"Error: Path '{path}' is a directory, not a file."

        try:
            return target_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: Unable to decode file '{path}' as UTF-8 text."
        except PermissionError:
            return f"Error: Permission denied when accessing '{path}'."
        except OSError as e:
            return f"Error reading file '{path}': {e}"


class WriteFileTool(Tool):
    """Tool for creating or replacing a text file on disk with supplied content."""

    name = "write_file"
    description = "Create or overwrite a file with the supplied text content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute file path to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }
    is_read_only = False

    def execute(self, path: str = "", content: str = "", **kwargs: Any) -> str:
        """Create or overwrite file cleanly handling error conditions."""
        if not path:
            return "Error: 'path' parameter is required."

        target_path = Path(path)

        if target_path.exists() and target_path.is_dir():
            return f"Error: Path '{path}' is a directory, not a file."

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote to '{path}'."
        except PermissionError:
            return f"Error: Permission denied when writing to '{path}'."
        except OSError as e:
            return f"Error writing to file '{path}': {e}"
