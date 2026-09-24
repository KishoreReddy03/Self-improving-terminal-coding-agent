"""Tool abstraction and implementations for the terminal coding agent."""

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
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


class EditFileTool(Tool):
    """Tool for performing exact string replacement in a text file."""

    name = "edit_file"
    description = "Replace an exact string in a file with new content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute file path to edit.",
            },
            "old_str": {
                "type": "string",
                "description": "Exact text snippet to search for and replace.",
            },
            "new_str": {
                "type": "string",
                "description": "New text snippet to insert in place of old_str.",
            },
        },
        "required": ["path", "old_str", "new_str"],
    }
    is_read_only = False

    def execute(self, path: str = "", old_str: str = "", new_str: str = "", **kwargs: Any) -> str:
        """Perform exact string replacement in the specified file."""
        if not path:
            return "Error: 'path' parameter is required."
        if not old_str:
            return "Error: 'old_str' parameter is required and cannot be empty."

        target_path = Path(path)

        if not target_path.exists():
            return f"Error: File '{path}' does not exist."

        if target_path.is_dir():
            return f"Error: Path '{path}' is a directory, not a file."

        try:
            file_content = target_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: Unable to decode file '{path}' as UTF-8 text."
        except PermissionError:
            return f"Error: Permission denied when reading '{path}'."
        except OSError as e:
            return f"Error reading file '{path}': {e}"

        count = file_content.count(old_str)
        if count == 0:
            return f"Error: Target text to replace was not found in '{path}'."

        new_content = file_content.replace(old_str, new_str)

        try:
            target_path.write_text(new_content, encoding="utf-8")
            return f"Successfully edited '{path}' (replaced {count} occurrence(s))."
        except PermissionError:
            return f"Error: Permission denied when writing to '{path}'."
        except OSError as e:
            return f"Error writing to file '{path}': {e}"


class ShellCommandTool(Tool):
    """Tool for executing shell commands and capturing output and status."""

    name = "shell_command"
    description = (
        "Execute a shell command, capture stdout and stderr, and return exit status and output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Optional maximum execution time in seconds (default: 30.0).",
            },
        },
        "required": ["command"],
    }
    is_read_only = False

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_read_only: Optional[bool] = None,
        default_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            is_read_only=is_read_only,
        )
        self.default_timeout = default_timeout

    def execute(
        self, command: str = "", timeout: Optional[float] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute a shell command safely capturing output and handling timeouts."""
        if not command or not command.strip():
            return {
                "exit_code": -1,
                "exit_status": -1,
                "stdout": "",
                "stderr": "Error: 'command' parameter is required and cannot be empty.",
            }

        exec_timeout = float(timeout) if timeout is not None else self.default_timeout

        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=exec_timeout,
            )
            return {
                "exit_code": completed.returncode,
                "exit_status": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired as e:
            stdout_str = (
                e.stdout
                if isinstance(e.stdout, str)
                else (e.stdout.decode("utf-8", errors="replace") if e.stdout else "")
            )
            stderr_str = (
                e.stderr
                if isinstance(e.stderr, str)
                else (e.stderr.decode("utf-8", errors="replace") if e.stderr else "")
            )
            error_msg = f"Error: Command timed out after {exec_timeout} seconds."
            full_stderr = f"{stderr_str}\n{error_msg}".strip() if stderr_str else error_msg
            return {
                "exit_code": -1,
                "exit_status": -1,
                "stdout": stdout_str,
                "stderr": full_stderr,
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "exit_status": -1,
                "stdout": "",
                "stderr": f"Error executing command: {e}",
            }

