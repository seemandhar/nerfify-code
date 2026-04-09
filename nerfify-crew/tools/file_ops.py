"""
File operation tools for CrewAI agents.
"""
from __future__ import annotations

import glob as glob_module
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ── Explicit arg schemas so CrewAI/Pydantic gives clear errors ──

class FileReadSchema(BaseModel):
    """Arguments for file_read."""
    file_path: str = Field(..., description="Absolute or relative path to the file to read")


class FileWriteSchema(BaseModel):
    """Arguments for file_write."""
    file_path: str = Field(..., description="Absolute or relative path to the file to write")
    content: str = Field(default="", description="The text content to write to the file. REQUIRED — you must provide the content.")


class FileGlobSchema(BaseModel):
    """Arguments for file_glob."""
    pattern: str = Field(..., description="Glob pattern (e.g. '**/*.py')")
    directory: str = Field(default=".", description="Base directory to search in")


class FileReadTool(BaseTool):
    """Read a file from the filesystem."""

    name: str = "file_read"
    description: str = (
        "Read the contents of a file given its absolute or relative path. "
        "Use this to read research papers, template files, generated code, "
        "and configuration files."
    )
    args_schema: type[BaseModel] = FileReadSchema

    max_chars: int = 12000  # Cap file reads to limit token growth in conversation history

    def _run(self, file_path: str) -> str:
        try:
            p = Path(file_path).expanduser()
            if not p.exists():
                return f"File not found: {file_path}"
            if p.stat().st_size > 500_000:
                return f"File too large ({p.stat().st_size} bytes): {file_path}"
            content = p.read_text(encoding="utf-8")
            if len(content) > self.max_chars:
                content = content[:self.max_chars] + f"\n\n... [TRUNCATED — {len(content)} total chars, showing first {self.max_chars}]"
            return content
        except Exception as e:
            return f"Error reading {file_path}: {e}"


class FileWriteTool(BaseTool):
    """Write content to a file."""

    name: str = "file_write"
    description: str = (
        "Write content to a file. You MUST provide both 'file_path' and 'content' arguments. "
        "Creates parent directories if they don't exist. Overwrites existing files."
    )
    args_schema: type[BaseModel] = FileWriteSchema

    def _run(self, file_path: str, content: str = "") -> str:
        if not content or not content.strip():
            return (
                "Error: 'content' argument is empty. This usually means the generated code was too long "
                "and got truncated. Try writing the file in smaller pieces: write one class or function at a time, "
                "then use file_read to verify, and append the rest."
            )
        try:
            p = Path(file_path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            return f"Error writing {file_path}: {e}"


class FileGlobTool(BaseTool):
    """Find files matching a glob pattern."""

    name: str = "file_glob"
    description: str = (
        "Find files matching a glob pattern (e.g. '**/*.py', 'method_template/*.py'). "
        "Returns a list of matching file paths."
    )
    args_schema: type[BaseModel] = FileGlobSchema

    def _run(self, pattern: str, directory: str = ".") -> str:
        try:
            base = Path(directory).expanduser()
            matches = sorted(base.glob(pattern))
            if not matches:
                return f"No files matching '{pattern}' in {directory}"
            return "\n".join(str(m) for m in matches[:100])
        except Exception as e:
            return f"Glob error: {e}"
