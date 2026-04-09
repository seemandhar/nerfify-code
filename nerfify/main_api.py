#!/usr/bin/env python3
"""
Nerfify — LiteLLM Multi-Model Pipeline

Uses LiteLLM to support Claude, GPT, Gemini, and open-source models with a unified
tool-use interface. Each agent is a separate conversation with specialized system prompt.

Supported models (via --model flag):
    claude-sonnet-4-20250514     (Anthropic)
    claude-opus-4-6              (Anthropic)
    gpt-4o                       (OpenAI)
    gpt-4.1                      (OpenAI)
    gemini/gemini-2.5-pro        (Google)
    ollama/llama3.3              (Local)
    ... any model LiteLLM supports

Usage:
    python main_api.py --arxiv 2308.12345
    python main_api.py --arxiv 2308.12345 --model gpt-4o
    python main_api.py --pdf /path/to/paper.pdf --model gemini/gemini-2.5-pro
    python main_api.py --arxiv 2308.12345 --model ollama/llama3.3 --no-review
    python main_api.py --arxiv 2308.12345 --fast --tiered
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

from config import PipelineConfig, get_model_for_agent

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


# ═══════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS (OpenAI function-calling format — LiteLLM standard)
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk. Returns the file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory, optionally with a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return stdout/stderr. Use for pip install, ns-train, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 300)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Useful for finding papers, APIs, debugging errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════
# TOOL EXECUTION
# ═══════════════════════════════════════════════════════════════════

def execute_tool(name: str, input_data: dict) -> str:
    """Execute a tool and return the result as a string."""
    if name == "read_file":
        path = Path(input_data["path"])
        if not path.exists():
            return f"Error: File not found: {path}"
        try:
            content = path.read_text(encoding="utf-8")
            if len(content) > 100000:
                content = content[:50000] + "\n...[TRUNCATED]...\n" + content[-25000:]
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"

    elif name == "write_file":
        path = Path(input_data["path"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(input_data["content"], encoding="utf-8")
            return f"Written: {path} ({len(input_data['content'])} chars)"
        except Exception as e:
            return f"Error writing {path}: {e}"

    elif name == "list_files":
        directory = Path(input_data["directory"])
        pattern = input_data.get("pattern", "*")
        if not directory.exists():
            return f"Error: Directory not found: {directory}"
        try:
            files = sorted(str(f.relative_to(directory)) for f in directory.rglob(pattern) if f.is_file())
            return "\n".join(files) if files else "(empty)"
        except Exception as e:
            return f"Error listing {directory}: {e}"

    elif name == "run_command":
        command = input_data["command"]
        cwd = input_data.get("cwd")
        timeout = input_data.get("timeout", 300)
        try:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            result = subprocess.run(
                ["bash", "-lc", command],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[EXIT CODE: {result.returncode}]"
            if len(output) > 20000:
                output = output[:10000] + "\n...[TRUNCATED]...\n" + output[-5000:]
            return output
        except subprocess.TimeoutExpired:
            return f"[TIMEOUT after {timeout}s]"
        except Exception as e:
            return f"Error running command: {e}"

    elif name == "web_search":
        query = input_data["query"]
        try:
            from duckduckgo_search import DDGS
            results = list(DDGS().text(query, max_results=5))
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.get('title', '')}")
                lines.append(f"   {r.get('href', r.get('link', ''))}")
                lines.append(f"   {r.get('body', r.get('snippet', ''))}")
                lines.append("")
            return "\n".join(lines) if lines else "No results found."
        except ImportError:
            return "Web search unavailable. Install with: pip install duckduckgo-search"
        except Exception as e:
            return f"Search error: {e}"

    return f"Unknown tool: {name}"


# ═══════════════════════════════════════════════════════════════════
# AGENT RUNNER (LiteLLM — works with any model)
# ═══════════════════════════════════════════════════════════════════

class Agent:
    """A single LLM agent with a system prompt and tool access.

    Uses LiteLLM's completion() which supports 100+ providers:
    Claude, GPT, Gemini, Mistral, Ollama, vLLM, Together, etc.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "claude-sonnet-4-20250514",
        tools: list | None = None,
        max_tokens: int = 16384,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools or TOOLS
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def run(self, user_prompt: str, *, max_iterations: int = 50) -> str:
        """Run the agent with tool use loop until it produces a final response."""
        print(f"\n{'─' * 40}")
        print(f"[{self.name}] Starting... (model: {self.model})")
        print(f"{'─' * 40}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for iteration in range(max_iterations):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    max_tokens=self.max_tokens,
                    stream=True,
                )

                # Collect streamed response
                collected_content = ""
                tool_calls_data: dict[int, dict] = {}
                finish_reason = None

                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # Stream text
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        collected_content += delta.content

                    # Collect tool calls from deltas
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {
                                    "id": tc.id or "",
                                    "function_name": "",
                                    "arguments": "",
                                }
                            if tc.id:
                                tool_calls_data[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls_data[idx]["function_name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc.function.arguments

                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                # Track token usage from the last chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    self.total_input_tokens += getattr(chunk.usage, "prompt_tokens", 0) or 0
                    self.total_output_tokens += getattr(chunk.usage, "completion_tokens", 0) or 0

            except Exception as e:
                print(f"\n[{self.name}] API error: {e}")
                return f"Error: {e}"

            # If no tool calls, we're done
            if not tool_calls_data:
                print(f"\n[{self.name}] Done.")
                return collected_content

            if finish_reason == "stop" and not tool_calls_data:
                print(f"\n[{self.name}] Done.")
                return collected_content

            # Build assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if collected_content:
                assistant_msg["content"] = collected_content
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function_name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls_data.values()
            ]
            messages.append(assistant_msg)

            # Execute tools and append results
            for tc in tool_calls_data.values():
                func_name = tc["function_name"]
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {}

                print(f"  [{self.name}] Tool: {func_name}({_summarize_input(args)})")
                result = execute_tool(func_name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        print(f"\n[{self.name}] Max iterations reached.")
        return "(Agent reached max iterations)"


def _summarize_input(input_data: dict) -> str:
    """Create a short summary of tool input for logging."""
    if "path" in input_data:
        return input_data["path"]
    if "command" in input_data:
        cmd = input_data["command"]
        return cmd[:80] + "..." if len(cmd) > 80 else cmd
    if "query" in input_data:
        return input_data["query"]
    if "directory" in input_data:
        return input_data["directory"]
    return str(input_data)[:60]


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════

def _parser_system(config: PipelineConfig) -> str:
    return f"""You are the Paper Parser agent for the Nerfify pipeline.
Your job: download PDF (if arXiv or URL), run mineru for markdown extraction, then clean the markdown.
If given a direct PDF URL (not arXiv), download it with wget first.

CLEANING RULES:
1) De-hyphenate words, fix OCR, normalize quotes/dashes
2) NEVER delete equations. Preserve all $...$ and $$...$$ math exactly.
3) Remove narrative (Intro, Related Work, qualitative Results). Keep ONLY: architecture, losses, algorithms, hyperparams, training details.
4) Delete benchmark tables. Keep implementation tables as bullet lists.
5) Remove figures/images but keep nearby implementation text.
6) Strip HTML artifacts.
7) Preserve citation markers so the citation recovery agent can resolve them later.
8) Compress verbose paragraphs to bullets.

Template files are at: {config.template_root}
"""


def _citation_recovery_system() -> str:
    return """You are the Citation Recovery agent for the Nerfify pipeline.

Your job: read the cleaned paper and identify IMPLEMENTATION GAPS — places where the
paper references a cited work for details it does NOT fully describe itself.

Only search for cited papers when the target paper's own text is insufficient to
implement that component. If the paper already provides all details, report that
no recovery is needed.

PROCESS:
1. Read the cleaned paper and identify gaps: "we follow [X]", "as in [Y]", loss/arch
   referenced by citation but not defined inline, etc.
2. For each gap (typically 1-5): search the web, find the paper, extract the SPECIFIC
   missing details (exact equations, layer sizes, hyperparameters, algorithms).
3. Save citation_details.json with the gaps and recovered details.
4. Save citation_recovery.md as a readable summary for the coder agent.

SKIP citations where the target paper already gives full details inline, or citations
that are just for background/comparison/datasets."""


def _planner_system(config: PipelineConfig) -> str:
    return f"""You are the Architecture Planner agent for Nerfify.
Read the cleaned paper, citation details, and template files, then produce a JSON DAG plan.

Template files at: {config.template_root}
Also read citation_details.json if it exists for additional implementation context.

Return JSON with: method_name, nodes, edges, files (with depends_on), base_architecture (nerfacto|vanilla_nerf), summary.
File paths must be from the template tree only."""


def _coder_system(config: PipelineConfig) -> str:
    return f"""You are the Code Generator agent for Nerfify. You are a senior NeRFStudio engineer.

RESOURCES TO READ:
- Template files: {config.template_root} (read ALL method_template/*.py, README.md, pyproject.toml)
- Examples: {config.papers_and_code} (read paper1-5.md, code1-5.py, VanillaNerfOriginal.py)
- Citation recovery: read citation_recovery.md if it exists for implementation details from cited papers

REQUIREMENTS:
- Generate exactly 8 files: method_template/__init__.py, template_config.py, template_datamanager.py, template_field.py, template_model.py, template_pipeline.py, README.md, pyproject.toml
- Match Nerfstudio APIs from templates exactly
- Use type hints, relative imports within method_template
- Encode paper's losses exactly
- Inherit from Nerfacto (if hash grids) or VanillaNeRF (if classic)
- pip install -e . must work
- No placeholders or TODOs"""


def _reviewer_system(config: PipelineConfig) -> str:
    return f"""You are the Code Reviewer agent for Nerfify.
Review generated NeRFStudio code for: import errors, API mismatches, type errors,
METHOD_NAME consistency, missing implementations, dependency issues, logical errors, training issues.

Template reference: {config.template_root}

Output JSON: {{"approved": bool, "issues": [...], "summary": "..."}}"""


def _validator_system() -> str:
    return """You are the Novelty Validator for Nerfify.
Compare paper novelties/equations against generated code.
Output JSON gap analysis with: novelties, equations, summary (with coverage_pct), approved."""


def _integrator_system() -> str:
    return """You are the Integration Checker for Nerfify.
Check: METHOD_NAME consistency, import chains, missing files, config wiring, pyproject.toml deps.
Output JSON: {passed, issues, method_name_consistent, all_files_present, imports_valid}"""


def _tester_system(config: PipelineConfig) -> str:
    return f"""You are the Smoke Tester for Nerfify.
Run tests in the '{config.conda_env}' conda environment.
All commands: eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; <command>

Steps: pip install -e . → ns-install-cli → import check → ns-train METHOD_NAME --max-num-iterations 10
Auto-detect dataparser: if transforms_train.json exists, use blender-data.
Output JSON: {{passed, steps_completed, failed_step, error_log}}"""


def _debugger_system(config: PipelineConfig) -> str:
    return f"""You are the Debug agent for Nerfify.
Diagnose and fix errors. Search the web for solutions if needed.
Template reference: {config.template_root}
Test in conda env: {config.conda_env}
Write complete fixed files. Output JSON diagnosis: {{error_category, root_cause, files_modified, fix_description}}"""


# ═══════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def _paper_is_self_contained(workspace: Path) -> bool:
    """Heuristic: check if the cleaned paper has few citation-dependent implementation gaps.

    Returns True if the paper likely doesn't need citation recovery (few "we follow [X]"
    or "as in [Y]" patterns for implementation details).
    """
    cleaned = workspace / "cleaned_paper.md"
    if not cleaned.exists():
        return False
    text = cleaned.read_text(encoding="utf-8")
    # Count gap-indicating patterns
    gap_patterns = [
        r"we follow \[",
        r"as in \[",
        r"we adopt .* from \[",
        r"we use the .* of \[",
        r"following \[.*?\]",
        r"borrowed from \[",
    ]
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in gap_patterns)
    return count <= 1  # 0-1 matches → likely self-contained


def run_pipeline(
    *,
    arxiv: str | None = None,
    pdf_url: str | None = None,
    pdf_path: str | None = None,
    method_name: str | None = None,
    data_path: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    config: PipelineConfig | None = None,
) -> dict:
    """Run the full pipeline using LiteLLM (any model provider).

    Optimizations (controlled by config):
    - fast_mode: skip citation_recovery if paper is self-contained
    - tiered_models: use cheap/mid/expensive models per agent role
    - Parallel execution: citation_recovery + planner run concurrently
    """

    if config is None:
        config = PipelineConfig()
    config.ensure_dirs()

    tiered = config.tiered_models

    # Create workspace
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    workspace = config.workspace_dir / run_id
    output_dir = config.generated_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID:    {run_id}")
    print(f"Model:     {model}")
    print(f"Tiered:    {tiered}")
    print(f"Fast mode: {config.fast_mode}")
    print(f"Workspace: {workspace}")
    print(f"Output:    {output_dir}")
    print("=" * 60)

    total_input_tokens = 0
    total_output_tokens = 0

    def _track(agent: Agent):
        nonlocal total_input_tokens, total_output_tokens
        total_input_tokens += agent.total_input_tokens
        total_output_tokens += agent.total_output_tokens

    def _model(agent_name: str) -> str:
        return get_model_for_agent(agent_name, model, tiered=tiered)

    # ── Step 1: Parse ──
    parser = Agent("parser", _parser_system(config), model=_model("parser"))
    if arxiv:
        input_desc = f"arXiv: {arxiv}"
    elif pdf_url:
        input_desc = f"PDF URL (download with wget first): {pdf_url}"
    else:
        input_desc = f"PDF: {pdf_path}"
    parser.run(
        f"Parse this paper: {input_desc}\n"
        f"Workspace: {workspace}\n"
        f"Save cleaned markdown to: {workspace}/cleaned_paper.md\n"
        f"Save raw markdown to: {workspace}/raw_paper.md"
    )
    _track(parser)

    # ── Step 2+3: Citation Recovery + Planning (PARALLEL) ──
    # These can run concurrently since planner reads citation_details.json
    # only if it exists (optional dependency).
    skip_citations = config.fast_mode and _paper_is_self_contained(workspace)
    if skip_citations:
        print("\n[orchestrator] Fast mode: skipping citation recovery (paper is self-contained)")

    def _run_citation():
        if skip_citations:
            # Write empty results so downstream agents don't error
            (workspace / "citation_details.json").write_text(
                json.dumps({"target_paper_title": "", "gaps_found": 0,
                             "no_gaps_needed": ["Skipped — fast mode"], "critical_citations": [],
                             "implementation_context": "No citation recovery needed."}, indent=2)
            )
            (workspace / "citation_recovery.md").write_text(
                "# Citation Recovery\nSkipped — paper appears self-contained (fast mode).\n"
            )
            return None
        agent = Agent("citation_recovery", _citation_recovery_system(), model=_model("citation_recovery"))
        agent.run(
            f"Read the cleaned paper at: {workspace}/cleaned_paper.md\n"
            f"Identify implementation gaps where cited papers need to be consulted.\n"
            f"Save results to: {workspace}/citation_details.json\n"
            f"Save readable summary to: {workspace}/citation_recovery.md"
        )
        return agent

    def _run_planner():
        agent = Agent("planner", _planner_system(config), model=_model("planner"))
        agent.run(
            f"Read the cleaned paper at: {workspace}/cleaned_paper.md\n"
            f"Read citation details at: {workspace}/citation_details.json (if exists)\n"
            f"Read template files from: {config.template_root}\n"
            f"Write the DAG plan to: {workspace}/dag_plan.json"
        )
        return agent

    # Run citation recovery and planner in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_citation = executor.submit(_run_citation)
        future_planner = executor.submit(_run_planner)

        citation_agent = future_citation.result()
        planner_agent = future_planner.result()

    if citation_agent:
        _track(citation_agent)
    _track(planner_agent)

    # Read plan to get method name
    plan_path = workspace / "dag_plan.json"
    resolved_method_name = method_name
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text())
            if not resolved_method_name:
                resolved_method_name = plan.get("method_name", "custom_nerf")
        except Exception:
            pass
    if not resolved_method_name:
        resolved_method_name = "custom_nerf"

    print(f"\nMethod name: {resolved_method_name}")

    # ── Step 4: Generate Code ──
    coder = Agent("coder", _coder_system(config), model=_model("coder"))
    coder.run(
        f"Generate a complete NeRFStudio implementation.\n"
        f"Method name: {resolved_method_name}\n"
        f"Read cleaned paper: {workspace}/cleaned_paper.md\n"
        f"Read citation recovery: {workspace}/citation_recovery.md (if exists)\n"
        f"Read DAG plan: {workspace}/dag_plan.json\n"
        f"Read template files from: {config.template_root}\n"
        f"Read examples from: {config.papers_and_code}\n"
        f"Write all generated files to: {output_dir}\n"
        f"Generate these files:\n" +
        "\n".join(f"  - {f}" for f in config.REQUIRED_FILES)
    )
    _track(coder)

    # ── Step 5: Review (optional loop) ──
    review_approved = True
    if config.enable_review:
        for review_iter in range(config.max_review_iterations):
            reviewer = Agent("reviewer", _reviewer_system(config), model=_model("reviewer"))
            reviewer.run(
                f"Review the generated code at: {output_dir}\n"
                f"Read all files in method_template/ plus README.md and pyproject.toml\n"
                f"Compare against templates at: {config.template_root}\n"
                f"Method name: {resolved_method_name}\n"
                f"Write review to: {workspace}/review_result.json"
            )
            _track(reviewer)

            review_path = workspace / "review_result.json"
            review_approved = True
            if review_path.exists():
                try:
                    review = json.loads(review_path.read_text())
                    review_approved = review.get("approved", True)
                except Exception:
                    pass

            if review_approved:
                print(f"\n[orchestrator] Review approved (iteration {review_iter + 1})")
                break
            else:
                print(f"\n[orchestrator] Review rejected (iteration {review_iter + 1}), fixing...")
                fix_coder = Agent("coder-fix", _coder_system(config), model=_model("coder"))
                fix_coder.run(
                    f"Fix the code based on review feedback.\n"
                    f"Review result: {workspace}/review_result.json\n"
                    f"Current code at: {output_dir}\n"
                    f"Method name: {resolved_method_name}\n"
                    f"Rewrite affected files in: {output_dir}"
                )
                _track(fix_coder)

    # ── Step 6: Validate (optional) ──
    validation_approved = True
    if config.enable_validation:
        validator = Agent("validator", _validator_system(), model=_model("validator"))
        validator.run(
            f"Validate that the paper's novelties are implemented.\n"
            f"Read cleaned paper: {workspace}/cleaned_paper.md\n"
            f"Read generated code at: {output_dir}\n"
            f"Write gap analysis to: {workspace}/validation_result.json"
        )
        _track(validator)

        val_path = workspace / "validation_result.json"
        coverage_pct = 100
        if val_path.exists():
            try:
                val = json.loads(val_path.read_text())
                validation_approved = val.get("approved", True)
                coverage_pct = val.get("summary", {}).get("coverage_pct", 100)
            except Exception:
                pass

        # Conditional skip: if validation coverage is high enough, skip review re-run
        if not validation_approved and coverage_pct >= config.validation_skip_threshold:
            print(f"\n[orchestrator] Validation coverage {coverage_pct}% >= {config.validation_skip_threshold}% threshold, skipping gap-fill")
            validation_approved = True

        if not validation_approved:
            print("\n[orchestrator] Validation found gaps, fixing...")
            gap_coder = Agent("coder-gaps", _coder_system(config), model=_model("coder"))
            gap_coder.run(
                f"Fill novelty gaps identified by the validator.\n"
                f"Gap analysis: {workspace}/validation_result.json\n"
                f"Cleaned paper: {workspace}/cleaned_paper.md\n"
                f"Current code at: {output_dir}\n"
                f"Method name: {resolved_method_name}\n"
                f"Rewrite affected files in: {output_dir}"
            )
            _track(gap_coder)

    # ── Step 7: Integration Check ──
    integrator = Agent("integrator", _integrator_system(), model=_model("integrator"))
    integrator.run(
        f"Check cross-file consistency of the code at: {output_dir}\n"
        f"Method name: {resolved_method_name}\n"
        f"Write results to: {workspace}/integration_result.json"
    )
    _track(integrator)

    # ── Step 8: Smoke Test (optional debug loop) ──
    test_passed = True
    if config.enable_smoke_test:
        data = data_path or config.default_dataset
        for debug_iter in range(config.max_debug_iterations + 1):
            tester = Agent("tester", _tester_system(config), model=_model("tester"))
            tester.run(
                f"Run smoke test on: {output_dir}\n"
                f"Method name: {resolved_method_name}\n"
                f"Data path: {data}\n"
                f"Write results to: {workspace}/test_result.json"
            )
            _track(tester)

            test_path = workspace / "test_result.json"
            test_passed = True
            if test_path.exists():
                try:
                    test = json.loads(test_path.read_text())
                    test_passed = test.get("passed", False)
                except Exception:
                    test_passed = False

            if test_passed:
                print(f"\n[orchestrator] Smoke test passed!")
                break

            if debug_iter < config.max_debug_iterations:
                print(f"\n[orchestrator] Test failed, debugging (attempt {debug_iter + 1})...")
                debugger = Agent("debugger", _debugger_system(config), model=_model("debugger"))
                debugger.run(
                    f"Fix the failing code.\n"
                    f"Test results: {workspace}/test_result.json\n"
                    f"Code at: {output_dir}\n"
                    f"Method name: {resolved_method_name}\n"
                    f"Templates at: {config.template_root}\n"
                    f"Write diagnosis to: {workspace}/debug_diagnosis.json"
                )
                _track(debugger)
            else:
                print(f"\n[orchestrator] Max debug iterations reached.")

    # ── Step 9: Full Training (optional) ──
    if config.enable_training and test_passed:
        data = data_path or config.default_dataset
        trainer = Agent("trainer", _tester_system(config), model=_model("tester"))
        trainer.run(
            f"Run full training on: {output_dir}\n"
            f"Method name: {resolved_method_name}\n"
            f"Data path: {data}\n"
            f"Max iterations: {config.default_max_iters}\n"
            f"Run: ns-train {resolved_method_name} --data {data} "
            f"--max-num-iterations {config.default_max_iters} "
            f"--viewer.quit-on-train-completion True"
        )
        _track(trainer)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Pipeline Complete!")
    print(f"  Run ID:        {run_id}")
    print(f"  Model:         {model}")
    print(f"  Tiered:        {tiered}")
    print(f"  Fast mode:     {config.fast_mode}")
    print(f"  Method:        {resolved_method_name}")
    print(f"  Output:        {output_dir}")
    print(f"  Review:        {'approved' if review_approved else 'not approved'}")
    print(f"  Validation:    {'approved' if validation_approved else 'gaps found'}")
    print(f"  Test:          {'passed' if test_passed else 'failed'}")
    print(f"  Input tokens:  {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"{'=' * 60}")

    result = {
        "run_id": run_id,
        "model": model,
        "method_name": resolved_method_name,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "review_approved": review_approved,
        "validation_approved": validation_approved,
        "test_passed": test_passed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    (workspace / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

MODEL_ALIASES = {
    "claude": "claude-sonnet-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-opus": "claude-opus-4-6",
    "gpt4o": "gpt-4o",
    "gpt4": "gpt-4.1",
    "gemini": "gemini/gemini-2.5-pro",
    "gemini-flash": "gemini/gemini-2.5-flash",
    "llama": "ollama/llama3.3",
    "deepseek": "deepseek/deepseek-chat",
}


def main():
    parser = argparse.ArgumentParser(
        description="Nerfify: Convert NeRF papers to NeRFStudio implementations (multi-model via LiteLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model examples:
  --model claude              Claude Sonnet (default)
  --model claude-opus         Claude Opus
  --model gpt4o               GPT-4o
  --model gemini              Gemini 2.5 Pro
  --model gemini-flash        Gemini 2.5 Flash
  --model llama               Llama 3.3 via Ollama
  --model deepseek            DeepSeek Chat
  --model <any-litellm-id>    Any model LiteLLM supports
""",
    )
    parser.add_argument("--arxiv", type=str, help="arXiv URL or ID")
    parser.add_argument("--url", type=str, help="Direct PDF URL (e.g., https://example.com/paper.pdf)")
    parser.add_argument("--pdf", type=str, help="Path to local PDF")
    parser.add_argument("--method-name", type=str, help="Override method name")
    parser.add_argument("--data", type=str, help="Path to training dataset")
    parser.add_argument("--model", type=str, default="claude",
                        help="LLM model to use (default: claude)")
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--train", action="store_true")
    # Optimization flags
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: skip citation recovery if paper is self-contained")
    parser.add_argument("--tiered", action="store_true",
                        help="Use tiered model routing (cheap/mid/expensive per agent)")

    args = parser.parse_args()
    if not args.arxiv and not args.url and not args.pdf:
        parser.error("Must provide --arxiv, --url, or --pdf")

    # Resolve model alias
    resolved_model = MODEL_ALIASES.get(args.model, args.model)

    config = PipelineConfig()
    config.enable_review = not args.no_review
    config.enable_validation = not args.no_validation
    config.enable_smoke_test = not args.no_test
    config.enable_training = args.train
    config.fast_mode = args.fast
    config.tiered_models = args.tiered
    config.default_max_iters = args.max_iters

    result = run_pipeline(
        arxiv=args.arxiv,
        pdf_url=args.url,
        pdf_path=args.pdf,
        method_name=args.method_name,
        data_path=args.data,
        model=resolved_model,
        config=config,
    )
    print(f"\nOutput: {result['output_dir']}")


if __name__ == "__main__":
    main()
