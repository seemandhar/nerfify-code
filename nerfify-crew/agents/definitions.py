"""
CrewAI agent definitions for the Nerfify pipeline.

Maps the 9 specialized agents from the Claude SDK version to CrewAI Agent objects
with appropriate tools, LLM models, and system prompts.
"""
from __future__ import annotations

from pathlib import Path

from crewai import Agent

from agents.prompts import AGENT_PROMPTS
from config import PipelineConfig, get_model_for_agent
from tools.web_search import WebSearchTool, WebFetchTool
from tools.file_ops import FileReadTool, FileWriteTool, FileGlobTool
from tools.shell import ShellTool
from tools.clean_paper import CleanPaperTool


def _list_template_files(template_root: Path) -> str:
    """Return a short listing of template files and their sizes (no content)."""
    files = [
        "method_template/__init__.py",
        "method_template/template_config.py",
        "method_template/template_datamanager.py",
        "method_template/template_field.py",
        "method_template/template_model.py",
        "method_template/template_pipeline.py",
        "pyproject.toml",
        "README.md",
    ]
    lines = []
    for rel in files:
        fpath = template_root / rel
        if fpath.exists():
            size = len(fpath.read_text(encoding="utf-8"))
            lines.append(f"  - {template_root}/{rel} ({size} chars)")
    return "\n".join(lines)


def _list_example_files(papers_and_code: Path) -> str:
    """Return a short listing of example files (no content)."""
    priority = ["VanillaNerfOriginal.py", "code1.py", "code2.py"]
    lines = []
    for fname in priority:
        fpath = papers_and_code / fname
        if fpath.exists():
            size = len(fpath.read_text(encoding="utf-8"))
            lines.append(f"  - {papers_and_code}/{fname} ({size} chars)")
    return "\n".join(lines)


# Shared tool instances
_web_search = WebSearchTool()
_web_fetch = WebFetchTool()
_file_read = FileReadTool()
_file_write = FileWriteTool()
_file_glob = FileGlobTool()
_shell = ShellTool()
_clean_paper = CleanPaperTool()

# Tool sets matching the original agent tool assignments
TOOL_SETS: dict[str, list] = {
    "parser":             [_file_read, _file_write, _shell, _file_glob, _web_search, _web_fetch, _clean_paper],
    "citation_recovery":  [_file_read, _file_write, _web_search, _web_fetch],
    "planner":            [_file_read, _file_write, _web_search, _web_fetch],
    "coder":              [_file_read, _file_write, _file_glob, _shell, _web_search, _web_fetch],
    "reviewer":           [_file_read, _file_write, _file_glob, _web_search, _web_fetch],
    "tester":             [_file_read, _shell],
    "debugger":           [_file_read, _file_write, _shell, _file_glob, _web_search, _web_fetch],
}


def build_agents(config: PipelineConfig) -> dict[str, Agent]:
    """Build all CrewAI agents with config-aware prompts and tools."""

    # Configure the clean_paper tool to use the pipeline's default model
    _clean_paper.model = config.default_model

    template_listing = _list_template_files(config.template_root)
    example_listing = _list_example_files(config.papers_and_code) if config.include_examples else ""

    agents = {}

    for agent_name, prompt_data in AGENT_PROMPTS.items():
        # Determine the LLM model for this agent
        if agent_name == "coder":
            llm_model = config.coder_model
        elif agent_name in ("parser", "tester", "integrator"):
            llm_model = config.cheap_model
        elif agent_name in ("debugger",):
            llm_model = config.default_model
        else:
            llm_model = config.default_model

        if config.tiered_models:
            llm_model = get_model_for_agent(agent_name, config.default_model, tiered=True)

        # Build backstory with embedded context for coder
        backstory = prompt_data["backstory"]
        if agent_name == "coder":
            backstory += (
                f"\n\n## Template Files (READ THESE with file_read before writing code)\n"
                f"Use file_read to read each template file for API reference:\n{template_listing}\n\n"
                f"IMPORTANT: Read at least template_config.py, template_model.py, and template_field.py "
                f"before generating code. These define the NeRFStudio API you must follow."
            )
            if example_listing:
                backstory += (
                    f"\n\n## In-Context Examples (READ with file_read for reference)\n"
                    f"Example implementations to learn from:\n{example_listing}\n"
                    f"Read VanillaNerfOriginal.py first — it shows the base NeRF pattern."
                )
        elif agent_name == "planner":
            backstory += f"\n\nTemplate files location: {config.template_root}"
        elif agent_name == "tester":
            backstory = backstory.replace("{conda_env}", config.conda_env)
        elif agent_name == "debugger":
            backstory += f"\n\nTemplate files at: {config.template_root}"
            backstory += f"\nConda environment: {config.conda_env}"

        tools = TOOL_SETS.get(agent_name, [_file_read, _file_write])

        # Tight iteration limits to control token usage
        # Each iteration = full conversation history resent to LLM
        iter_limits = {
            "parser": 10,
            "citation_recovery": 8,
            "planner": 8,
            "coder": 15,      # needs more for reading templates + writing 8 files
            "reviewer": 10,
            "tester": 8,
            "debugger": 12,
        }

        agents[agent_name] = Agent(
            role=prompt_data["role"],
            goal=prompt_data["goal"],
            backstory=backstory,
            tools=tools,
            llm=llm_model,
            verbose=True,
            allow_delegation=False,
            max_iter=iter_limits.get(agent_name, 10),
            max_retry_limit=2,
        )

    return agents
