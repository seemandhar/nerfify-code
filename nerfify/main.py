#!/usr/bin/env python3
"""
Nerfify — Claude Agent SDK Multi-Agent Pipeline

Converts research papers about NeRF methods into complete NeRFStudio implementations
using specialized Claude subagents for each pipeline stage.

Usage:
    # From arXiv URL
    python main.py --arxiv 2308.12345

    # From local PDF
    python main.py --pdf /path/to/paper.pdf

    # With options
    python main.py --arxiv 2308.12345 --method-name my_nerf --no-review --train

    # With custom data path
    python main.py --arxiv 2308.12345 --data /path/to/dataset --train --max-iters 5000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    # Message types
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    # Content block types
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    # Subagent task messages
    TaskStartedMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
)

from config import PipelineConfig
from agents.definitions import build_agent_definitions
from litellm_proxy import proxy_env


# ═══════════════════════════════════════════════════════════════════
# TERMINAL LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════════

# ANSI color codes
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"


def log_header(msg: str):
    print(f"\n{C.BOLD}{C.BG_BLUE}{C.WHITE} {msg} {C.RESET}")


def log_stage(msg: str):
    print(f"\n{C.BOLD}{C.GREEN}>>> {msg}{C.RESET}")


def log_agent(agent: str, msg: str):
    print(f"{C.BOLD}{C.CYAN}[{agent}]{C.RESET} {msg}")


def log_tool(name: str, summary: str):
    print(f"  {C.YELLOW}>> tool:{C.RESET} {C.BOLD}{name}{C.RESET} {C.DIM}{summary}{C.RESET}")


def log_tool_result(summary: str):
    print(f"  {C.DIM}<< {summary}{C.RESET}")


def log_thinking(text: str):
    # Show first few lines of thinking
    lines = text.strip().split("\n")
    preview = lines[0][:120]
    if len(lines) > 1 or len(lines[0]) > 120:
        preview += "..."
    print(f"  {C.MAGENTA}(thinking) {preview}{C.RESET}")


def log_text(text: str, prefix: str = ""):
    """Print agent text output, line by line."""
    for line in text.split("\n"):
        print(f"  {C.WHITE}{prefix}{line}{C.RESET}")


def log_error(msg: str):
    print(f"{C.BOLD}{C.RED}ERROR: {msg}{C.RESET}")


def log_success(msg: str):
    print(f"{C.BOLD}{C.GREEN}OK: {msg}{C.RESET}")


def log_info(msg: str):
    print(f"{C.DIM}{msg}{C.RESET}")


def log_task_event(event_type: str, msg: str):
    print(f"  {C.BLUE}[task:{event_type}]{C.RESET} {msg}")


def _summarize_tool_input(tool_input: dict) -> str:
    """Create a short summary of tool input for logging."""
    if not isinstance(tool_input, dict):
        return str(tool_input)[:80]
    # Common patterns
    if "file_path" in tool_input:
        return tool_input["file_path"]
    if "path" in tool_input:
        return tool_input["path"]
    if "command" in tool_input:
        cmd = tool_input["command"]
        return cmd[:100] + ("..." if len(cmd) > 100 else "")
    if "pattern" in tool_input:
        return f"pattern={tool_input['pattern']}"
    if "prompt" in tool_input:
        p = tool_input["prompt"]
        return p[:80] + ("..." if len(p) > 80 else "")
    if "description" in tool_input:
        return tool_input["description"][:80]
    return str(tool_input)[:80]


def _summarize_tool_result(content) -> str:
    """Create a short summary of a tool result."""
    if isinstance(content, str):
        lines = content.strip().split("\n")
        if len(lines) <= 2:
            return content.strip()[:150]
        return f"{lines[0][:100]}... ({len(lines)} lines)"
    if isinstance(content, list):
        return f"({len(content)} blocks)"
    return str(content)[:100]


# ═══════════════════════════════════════════════════════════════════
# LIVE TOKEN TRACKER
# ═══════════════════════════════════════════════════════════════════

class TokenTracker:
    """Accumulates token usage across all tasks and prints a live status line."""

    def __init__(self):
        self.total_tokens = 0
        self.total_tool_uses = 0
        self.total_api_ms = 0
        self._task_tokens: dict[str, int] = {}  # track per-task to avoid double-counting

    def update_from_progress(self, task_id: str, usage):
        """Update from a TaskProgressMessage (cumulative per-task)."""
        tokens = 0
        tool_uses = 0
        dur_ms = 0
        if usage:
            if isinstance(usage, dict):
                tokens = usage.get("total_tokens", 0) or 0
                tool_uses = usage.get("tool_uses", 0) or 0
                dur_ms = usage.get("duration_ms", 0) or 0
            else:
                tokens = getattr(usage, "total_tokens", 0) or 0
                tool_uses = getattr(usage, "tool_uses", 0) or 0
                dur_ms = getattr(usage, "duration_ms", 0) or 0

        # TaskProgress gives cumulative per-task, so compute delta
        prev = self._task_tokens.get(task_id, 0)
        delta = max(0, tokens - prev)
        self._task_tokens[task_id] = tokens

        self.total_tokens += delta
        # tool_uses and duration are also cumulative per-task; just store latest
        self.total_tool_uses += max(0, tool_uses - 0)  # approximate
        self.total_api_ms = max(self.total_api_ms, dur_ms)

    def update_from_result(self, message):
        """Update from a ResultMessage (final totals)."""
        usage = getattr(message, "usage", None) or {}
        cost = getattr(message, "total_cost_usd", None)
        dur = getattr(message, "duration_ms", 0)
        dur_api = getattr(message, "duration_api_ms", 0)
        turns = getattr(message, "num_turns", 0)
        # Use final usage if available (more accurate than accumulated deltas)
        if isinstance(usage, dict) and usage.get("total_tokens"):
            self.total_tokens = usage["total_tokens"]
        self.cost_usd = cost
        self.duration_ms = dur
        self.duration_api_ms = dur_api
        self.num_turns = turns

    def status_line(self, elapsed: float) -> str:
        """Return a compact status line for live display."""
        parts = [f"{C.DIM}[{elapsed:.0f}s]"]
        parts.append(f"tokens={self.total_tokens:,}")
        if hasattr(self, "cost_usd") and self.cost_usd is not None:
            parts.append(f"cost=${self.cost_usd:.4f}")
        return " ".join(parts) + C.RESET

    def print_status(self, elapsed: float):
        """Print the status line, overwriting the previous one."""
        line = self.status_line(elapsed)
        # Use carriage return to overwrite in-place
        print(f"\r{line}", end="", flush=True)

    def print_final(self, elapsed: float):
        """Print final token summary."""
        print()  # newline after last \r
        cost_str = f"${self.cost_usd:.4f}" if hasattr(self, "cost_usd") and self.cost_usd else "N/A"
        turns_str = str(self.num_turns) if hasattr(self, "num_turns") else "N/A"
        print(f"  {C.BOLD}Tokens:{C.RESET}     {self.total_tokens:,}")
        print(f"  {C.BOLD}Cost:{C.RESET}       {cost_str}")
        print(f"  {C.BOLD}Turns:{C.RESET}      {turns_str}")
        if hasattr(self, "duration_api_ms") and self.duration_api_ms:
            print(f"  {C.BOLD}API time:{C.RESET}   {self.duration_api_ms/1000:.1f}s")


# ═══════════════════════════════════════════════════════════════════
# MESSAGE HANDLER
# ═══════════════════════════════════════════════════════════════════

def handle_message(message, start_time: float, tracker: TokenTracker):
    """Handle and log every message from the agent stream."""
    elapsed = time.time() - start_time

    # ── Result: final output ──
    if isinstance(message, ResultMessage):
        print("\r\033[K", end="", flush=True)
        tracker.update_from_result(message)
        log_header(f"PIPELINE COMPLETE ({elapsed:.1f}s)")
        stop = getattr(message, "stop_reason", "end_turn")
        log_info(f"Stop reason: {stop}")
        if message.result:
            print()
            print(message.result)
        return {"type": "result", "text": message.result}

    # ── System messages ──
    if isinstance(message, SystemMessage):
        subtype = getattr(message, "subtype", "unknown")
        session_id = getattr(message, "session_id", None)
        if session_id:
            log_info(f"[system:{subtype}] session={session_id}")
            return {"type": "system", "session_id": session_id}
        else:
            log_info(f"[system:{subtype}] {_msg_preview(message)}")
        return {"type": "system"}

    # ── Assistant messages (text, thinking, tool calls) ──
    if isinstance(message, AssistantMessage):
        # Clear the status line before printing content
        print("\r\033[K", end="", flush=True)
        content = getattr(message, "content", [])
        for block in content:
            if isinstance(block, ThinkingBlock):
                thinking_text = getattr(block, "thinking", "")
                if thinking_text:
                    log_thinking(thinking_text)

            elif isinstance(block, TextBlock):
                text = getattr(block, "text", "")
                if text:
                    log_text(text)

            elif isinstance(block, ToolUseBlock):
                name = getattr(block, "name", "?")
                tool_input = getattr(block, "input", {})
                summary = _summarize_tool_input(tool_input)

                # Special handling for Agent tool (subagent invocation)
                if name == "Agent":
                    agent_type = tool_input.get("agent_type") or tool_input.get("type") or "subagent"
                    desc = tool_input.get("description", "")
                    prompt_preview = tool_input.get("prompt", "")[:100]
                    log_stage(f"Invoking subagent: {agent_type}")
                    if desc:
                        log_agent(agent_type, desc)
                    if prompt_preview:
                        log_info(f"  prompt: {prompt_preview}...")
                else:
                    log_tool(name, summary)

            else:
                # Other block types
                block_type = getattr(block, "type", type(block).__name__)
                log_info(f"  [{block_type}]")

        return {"type": "assistant"}

    # ── Task started (subagent spawned) ──
    if isinstance(message, TaskStartedMessage):
        print("\r\033[K", end="", flush=True)
        task_id = getattr(message, "task_id", "?")
        agent_type = getattr(message, "agent_type", "subagent")
        log_task_event("started", f"agent={agent_type} task={task_id}")
        return {"type": "task_started"}

    # ── Task progress (subagent working) ──
    if isinstance(message, TaskProgressMessage):
        task_id = getattr(message, "task_id", "?")
        usage = getattr(message, "usage", None)
        tracker.update_from_progress(task_id, usage)
        # Print live status line (overwrites in-place)
        tracker.print_status(elapsed)
        return {"type": "task_progress"}

    # ── Task notification (subagent done) ──
    if isinstance(message, TaskNotificationMessage):
        print("\r\033[K", end="", flush=True)
        task_id = getattr(message, "task_id", "?")
        status = getattr(message, "status", None)
        status_str = ""
        if status:
            status_str = getattr(status, "value", str(status))
        tool_use_id = getattr(message, "tool_use_id", None)
        log_task_event("done", f"status={status_str} task={task_id}")
        return {"type": "task_notification"}

    # ── Catch-all for unknown message types ──
    msg_type = type(message).__name__
    log_info(f"[{msg_type}] {_msg_preview(message)}")
    return {"type": msg_type}


def _msg_preview(msg) -> str:
    """Get a short preview string for any message object."""
    # Try common attributes
    for attr in ("result", "text", "content", "message"):
        val = getattr(msg, attr, None)
        if val and isinstance(val, str):
            return val[:100]
    return ""


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR PROMPT
# ═══════════════════════════════════════════════════════════════════

def build_orchestrator_prompt(
    *,
    arxiv: str | None = None,
    pdf_url: str | None = None,
    pdf_path: str | None = None,
    method_name: str | None = None,
    data_path: str | None = None,
    config: PipelineConfig,
    workspace: Path,
    output_dir: Path,
) -> str:
    """Build the task prompt for the orchestrator agent."""

    # Input source
    if arxiv:
        input_section = f"ArXiv paper: {arxiv}"
    elif pdf_url:
        input_section = f"PDF URL (download this first with wget): {pdf_url}"
    elif pdf_path:
        input_section = f"Local PDF: {pdf_path}"
    else:
        raise ValueError("Must provide --arxiv, --url, or --pdf")

    method_override = ""
    if method_name:
        method_override = f"\nIMPORTANT: Use this exact method name: {method_name}"

    data_section = ""
    if data_path:
        data_section = f"\nTraining data path: {data_path}"

    # Pipeline steps based on config
    optional_steps = []

    # Inline quality check replaces separate reviewer/validator/integrator agents
    quality_check_step = """5. **Quality Check (YOU do this — do NOT spawn reviewer/validator/integrator agents)**:
   After the coder finishes, read ALL generated files yourself and do a FAST inline check:
   a) **Imports**: Every `from` / `import` resolves (no typos, no missing modules)
   b) **METHOD_NAME**: Consistent across all files (__init__, config, model, field, pipeline, datamanager, pyproject.toml)
   c) **Config wiring**: Config references correct Model/Field/Pipeline/DataManager classes
   d) **pyproject.toml**: Entry point correct, dependencies listed
   e) **Key equations**: Spot-check that the paper's main loss/architecture is present in the code

   If you find critical issues (import errors, wrong class names, missing files), call the
   `coder` agent ONE more time with a precise list of fixes. Do NOT loop — one fix pass max.
   If everything looks reasonable, proceed directly to testing."""
    optional_steps.append(quality_check_step)

    if config.enable_smoke_test:
        optional_steps.append(
            f"6. **Test**: Use the `tester` agent. If it fails, use `debugger` to fix "
            f"(up to {config.max_debug_iterations} attempts), then re-test."
        )
    if config.enable_training:
        training_output_dir = config.training_output_dir
        scene = config.scenes[0] if config.scenes else "garden"
        psnr_target = config.get_psnr_target(scene)
        read_tb_script = config.read_tb_script
        eval_script = config.eval_script

        train_step = f"""7. **Train** ({config.default_max_iters} iterations):
   Run full training using the Bash tool in the `{config.conda_env}` conda environment:
   ```
   eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; \\
   cd {output_dir} && pip install -e . && ns-install-cli && \\
   CUDA_VISIBLE_DEVICES={config.training_gpu} ns-train <METHOD_NAME_CLI> \\
     --data {data_path or config.default_dataset} \\
     --vis viewer+tensorboard \\
     --max-num-iterations {config.default_max_iters} \\
     --output-dir {training_output_dir} \\
     --viewer.quit-on-train-completion True
   ```
   Replace <METHOD_NAME_CLI> with the method's CLI name (usually snake_case with hyphens,
   check pyproject.toml entry point).
   The dataparser depends on the data format — check if the data dir has
   transforms_train.json (→ use `blender-data`), or COLMAP (→ use `colmap --eval-mode interval`)."""
        optional_steps.append(train_step)

        if config.enable_psnr_feedback:
            psnr_step = f"""8. **PSNR Review** (up to {config.max_psnr_fix_iterations} iterations):
   After training completes, read the PSNR training curve:

   a) Find the latest timestamp directory:
      ```
      ls -t {training_output_dir}/{scene}/<METHOD_NAME_CLI>/ | head -1
      ```

   b) Read PSNR curve:
      ```
      eval "$(conda shell.bash hook)"; conda activate {config.conda_env}; \\
      python {read_tb_script} {training_output_dir}/{scene}/<METHOD_NAME_CLI>/<LATEST_TIMESTAMP>/ --json
      ```

   c) Analyze the output:
      - **final_psnr**: The PSNR at the end of training
      - **max_psnr**: The peak PSNR during training
      - **issues**: Any detected problems (NaN, drops, still rising)
      - **samples**: The full PSNR curve

   d) Decision:
      - If final_psnr >= {psnr_target} dB AND no critical issues → PASS, proceed to report
      - If final_psnr < {psnr_target} dB OR critical issues found:
        1. Analyze WHY PSNR is low. Common causes:
           - Loss function not converging (check loss weights, gradient flow)
           - Learning rate too high/low
           - Architecture bug (wrong tensor shapes, missing activations)
           - Training instability (NaN losses, exploding gradients)
        2. Read the paper's reported PSNR and compare
        3. Call the `coder` agent with:
           - The PSNR curve data
           - Your diagnosis of what's wrong
           - Specific instructions to fix the code
        4. Re-install and re-train
        5. Read PSNR again. Max {config.max_psnr_fix_iterations} fix cycles."""
            optional_steps.append(psnr_step)

    optional_text = "\n".join(optional_steps)

    # Fast mode: conditional citation recovery skip
    fast_mode_note = ""
    if config.fast_mode:
        fast_mode_note = """
## Fast Mode (ENABLED)
- After parsing, quickly scan the cleaned paper for citation-dependent implementation
  gaps (phrases like "we follow [X]", "as in [Y]" for undescribed details).
- If the paper is self-contained (provides all architecture, loss, training details inline),
  SKIP the citation_recovery agent entirely. Write a short note to
  {workspace}/citation_recovery.md saying "Skipped — paper is self-contained."
- If there ARE gaps, run citation_recovery as normal.
""".format(workspace=workspace)

    # Parallel execution hint for steps 2+3
    parallel_note = """
**OPTIMIZATION**: Steps 2 (Citation Recovery) and 3 (Plan) can be run IN PARALLEL
using the Agent tool with multiple concurrent agent invocations. The planner reads
citation_details.json only if it exists, so it does not strictly depend on citation
recovery finishing first. Launch both agents simultaneously to save time."""

    return f"""## Task
Convert a research paper into a complete NeRFStudio method implementation.

## Input
{input_section}{method_override}{data_section}

## Workspace
- Working directory: {workspace}
- Output directory (write generated code here): {output_dir}
{fast_mode_note}
## SPEED RULES (CRITICAL — follow these to minimize runtime)

1. **EMBED content in agent prompts**: After you read a file (e.g. cleaned_paper.md,
   dag_plan.json, citation_recovery.md), PASTE its full content directly into the
   prompt you send to the next agent. Do NOT tell agents "read file X" — give them
   the content inline. This eliminates redundant file reads across agents.

2. **Do NOT spawn reviewer, validator, or integrator agents.** You do quality checks
   yourself (see step 5 below). This cuts 3 agent round-trips.

3. **One fix pass max.** If the coder's output has issues, call coder ONE more time
   with a precise fix list. Do not loop.

4. **Do NOT re-read files you already have.** Once you read a file, keep it in context.
   Do not read the same file again.

## Pipeline Steps

1. **Parse**: Use the `parser` agent to download (if needed) and extract the paper.
   Tell it to save results in the workspace directory: {workspace}

2. **Citation Recovery**: Use the `citation_recovery` agent to identify and fetch
   implementation details from cited papers.
   Tell it to read from {workspace}/cleaned_paper.md and save results to
   {workspace}/citation_details.json and {workspace}/citation_recovery.md

3. **Plan**: Use the `planner` agent to read the cleaned paper and create a DAG plan.
   Tell it to read from {workspace}/cleaned_paper.md, {workspace}/citation_details.json,
   and the template files.
   Tell it to write the plan to {workspace}/dag_plan.json

{parallel_note}

4. **Generate Code**: Use the `coder` agent to generate the implementation.
   IMPORTANT: Before calling the coder, read these files yourself:
   - {workspace}/cleaned_paper.md
   - {workspace}/dag_plan.json
   - {workspace}/citation_recovery.md
   Then EMBED all three file contents directly in the coder's task prompt, along with:
   - The output directory: {output_dir}
   - The method name (from the plan)
   Do NOT tell the coder to read these files — give it the content inline.
   The coder already has templates and examples embedded in its system prompt.

{optional_text}

## Important Rules
- EMBED file contents in agent prompts instead of telling agents to read files
- Do NOT spawn reviewer, validator, or integrator agents — do checks yourself
- If a step fails, report the error clearly
- The final output should be a complete, installable NeRFStudio method in {output_dir}

## Final Report
After all steps complete, summarize:
- Method name
- Output directory
- Files generated
- Quality check results
- Any issues encountered
"""


# ═══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════

async def run_pipeline(
    *,
    arxiv: str | None = None,
    pdf_url: str | None = None,
    pdf_path: str | None = None,
    method_name: str | None = None,
    data_path: str | None = None,
    model: str | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    """Run the full Nerfify pipeline using Claude Agent SDK."""

    if config is None:
        config = PipelineConfig()
    config.ensure_dirs()

    # Create a unique workspace for this run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    workspace = config.workspace_dir / run_id
    output_dir = config.generated_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_header("Nerfify Multi-Agent Pipeline")
    print(f"  Run ID:    {C.BOLD}{run_id}{C.RESET}")
    print(f"  Workspace: {workspace}")
    print(f"  Output:    {output_dir}")
    if arxiv:
        print(f"  Paper:     arXiv:{arxiv}")
    elif pdf_url:
        print(f"  Paper:     {pdf_url}")
    elif pdf_path:
        print(f"  Paper:     {pdf_path}")
    print()

    # Build agent definitions
    agents = build_agent_definitions(config)
    log_info(f"Loaded {len(agents)} subagent definitions: {', '.join(agents.keys())}")

    # Build orchestrator prompt
    prompt = build_orchestrator_prompt(
        arxiv=arxiv,
        pdf_url=pdf_url,
        pdf_path=pdf_path,
        method_name=method_name,
        data_path=data_path or config.default_dataset,
        config=config,
        workspace=workspace,
        output_dir=output_dir,
    )

    # Save the prompt for debugging
    (workspace / "orchestrator_prompt.txt").write_text(prompt)
    log_info(f"Orchestrator prompt saved to {workspace}/orchestrator_prompt.txt")

    # Build env for model routing (LiteLLM proxy for non-Claude models)
    sdk_env = proxy_env(model or "claude")
    if sdk_env.get("ANTHROPIC_BASE_URL"):
        log_info(f"Using LiteLLM proxy for model: {model}")

    # Run the orchestrator
    start_time = time.time()
    result_text = ""
    session_id = None
    message_count = 0
    tracker = TokenTracker()

    log_stage("Starting Nerfify orchestrator agent...")

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            cwd=str(config.base_dir),
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "Agent", "WebSearch", "WebFetch"],
            agents=agents,
            permission_mode="acceptEdits",
            max_turns=200,
            env=sdk_env,
        ),
    ):
        message_count += 1
        info = handle_message(message, start_time, tracker)

        if info["type"] == "result":
            result_text = info.get("text", "")
        elif info["type"] == "system" and "session_id" in info:
            session_id = info["session_id"]

    # Summary
    elapsed = time.time() - start_time
    log_header("RUN SUMMARY")
    print(f"  Run ID:      {run_id}")
    print(f"  Duration:    {elapsed:.1f}s")
    print(f"  Messages:    {message_count}")
    print(f"  Session:     {session_id or 'N/A'}")
    print(f"  Output:      {output_dir}")
    tracker.print_final(elapsed)
    print()

    # List generated files
    gen_files = sorted(f.relative_to(output_dir) for f in output_dir.rglob("*") if f.is_file())
    if gen_files:
        log_success(f"Generated {len(gen_files)} files:")
        for f in gen_files:
            print(f"    {f}")
    else:
        log_error("No files generated!")

    # Save results
    result = {
        "run_id": run_id,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "session_id": session_id,
        "duration_s": round(elapsed, 1),
        "message_count": message_count,
        "generated_files": [str(f) for f in gen_files],
        "result": result_text,
    }
    (workspace / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )

    return result


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nerfify: Convert NeRF research papers to NeRFStudio implementations using Claude agents"
    )
    parser.add_argument("--arxiv", type=str, help="arXiv URL or ID (e.g., 2308.12345)")
    parser.add_argument("--url", type=str, help="Direct PDF URL (e.g., https://example.com/paper.pdf)")
    parser.add_argument("--pdf", type=str, help="Path to local PDF file")
    parser.add_argument("--method-name", type=str, help="Override method name (snake_case)")
    parser.add_argument("--data", type=str, help="Path to training dataset")
    parser.add_argument("--max-iters", type=int, default=3000, help="Max training iterations")

    # Model (defaults to Claude via native SDK; others use LiteLLM proxy)
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model (default: Claude via native SDK). Non-Claude models use LiteLLM proxy.")

    # Pipeline toggles
    parser.add_argument("--no-review", action="store_true", help="Skip code review step")
    parser.add_argument("--no-validation", action="store_true", help="Skip novelty validation")
    parser.add_argument("--no-test", action="store_true", help="Skip smoke testing")
    parser.add_argument("--train", action="store_true", help="Enable full training + PSNR feedback")
    parser.add_argument("--no-psnr-feedback", action="store_true",
                        help="Disable PSNR feedback loop (train only, no fix cycles)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device for training")
    parser.add_argument("--dataset", type=str, default="mipnerf360",
                        choices=["mipnerf360", "blender", "llff"],
                        help="Dataset for training evaluation")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Scenes to train on (default: garden)")
    parser.add_argument("--expected-psnr", type=float, default=None,
                        help="Override target PSNR (if paper reports a specific value)")
    # Optimization flags
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: skip citation recovery if paper is self-contained")
    parser.add_argument("--tiered", action="store_true",
                        help="Use tiered model routing (cheap/mid/expensive per agent)")

    args = parser.parse_args()

    if not args.arxiv and not args.url and not args.pdf:
        parser.error("Must provide --arxiv, --url, or --pdf")

    # Build config
    config = PipelineConfig()
    config.enable_review = not args.no_review
    config.enable_validation = not args.no_validation
    config.enable_smoke_test = not args.no_test
    config.enable_training = args.train
    config.enable_psnr_feedback = args.train and not args.no_psnr_feedback
    config.training_gpu = args.gpu
    config.dataset = args.dataset
    if args.scenes:
        config.scenes = args.scenes
    config.expected_psnr = args.expected_psnr
    config.fast_mode = args.fast
    config.tiered_models = args.tiered
    config.default_max_iters = args.max_iters

    # Run
    async def _run():
        return await run_pipeline(
            arxiv=args.arxiv,
            pdf_url=args.url,
            pdf_path=args.pdf,
            method_name=args.method_name,
            data_path=args.data,
            model=args.model,
            config=config,
        )

    result = asyncio.run(_run())

    print(f"\n{C.BOLD}Output directory: {result['output_dir']}{C.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
