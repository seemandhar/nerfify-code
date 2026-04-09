"""
Shell command execution tool for CrewAI agents.
"""
from __future__ import annotations

import subprocess

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ShellToolSchema(BaseModel):
    """Input schema for shell_command tool."""
    command: str = Field(default="", description="The shell command to execute")


class ShellTool(BaseTool):
    """Execute shell commands."""

    name: str = "shell_command"
    description: str = (
        "Execute a shell command and return the output. Use this for running "
        "mineru, pip install, ns-train, conda activate, and other system commands. "
        "Provide the full command as a string. Commands timeout after 10 minutes."
    )
    args_schema: type = ShellToolSchema
    timeout: int = Field(default=600, description="Command timeout in seconds")
    max_output_chars: int = Field(default=8000, description="Max chars to return per stream")

    def _run(self, command: str = "") -> str:
        if not command or not command.strip():
            return "Error: No command provided. Please provide a command string."
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                executable="/bin/bash",
            )
            output_parts = []
            if result.stdout:
                stdout = result.stdout
                if len(stdout) > self.max_output_chars:
                    # Keep first and last portions for context
                    half = self.max_output_chars // 2
                    stdout = stdout[:half] + f"\n... [{len(result.stdout) - self.max_output_chars} chars truncated] ...\n" + stdout[-half:]
                output_parts.append(f"STDOUT:\n{stdout}")
            if result.stderr:
                stderr = result.stderr
                if len(stderr) > self.max_output_chars:
                    # For errors, keep the end (most useful)
                    stderr = f"... [{len(result.stderr) - self.max_output_chars} chars truncated] ...\n" + stderr[-self.max_output_chars:]
                output_parts.append(f"STDERR:\n{stderr}")
            if result.returncode != 0:
                output_parts.append(f"EXIT CODE: {result.returncode}")
            return "\n".join(output_parts) if output_parts else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout}s: {command}"
        except Exception as e:
            return f"Error running command: {e}"
