#!/usr/bin/env python3
"""
Nerfify — Web UI

Flask app with real-time SSE streaming of the multi-agent pipeline.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, jsonify, Response, abort, send_from_directory
from werkzeug.utils import secure_filename

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    TaskStartedMessage,
    TaskProgressMessage,
    TaskNotificationMessage,
)

from config import PipelineConfig
from agents.definitions import build_agent_definitions
from main import build_orchestrator_prompt
from litellm_proxy import proxy_env


# ═══════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nerfify-claude-agents-dev")

JOBS: dict[str, dict[str, Any]] = {}
JOB_QUEUES: dict[str, queue.Queue] = {}
JOB_STOP_EVENTS: dict[str, threading.Event] = {}

DEFAULT_CONFIG = PipelineConfig()
DEFAULT_CONFIG.ensure_dirs()

KNOWN_AGENTS = {"parser", "citation_recovery", "planner", "coder", "reviewer", "validator", "integrator", "tester", "debugger"}

_AGENT_KEYWORDS: dict[str, list[str]] = {
    "parser": ["parse", "extract", "pdf", "markdown", "clean"],
    "citation_recovery": ["citation", "recovery", "reference", "cited", "fetch paper", "dependency"],
    "planner": ["plan", "architect", "dag", "design", "outline"],
    "coder": ["code", "generate", "implement", "write", "nerfstudio"],
    "reviewer": ["review", "check", "error", "lint", "fix"],
    "validator": ["valid", "novelty", "equation", "paper match"],
    "integrator": ["integrat", "cross-file", "consistency", "import"],
    "tester": ["test", "smoke", "run", "train"],
    "debugger": ["debug", "diagnos", "traceback", "error fix"],
}


# ═══════════════════════════════════════════════════════════════════
# EVENT HELPERS
# ═══════════════════════════════════════════════════════════════════

def push_event(job_id: str, event_type: str, data: dict):
    """Push a structured event to the job's SSE queue."""
    q = JOB_QUEUES.get(job_id)
    if q:
        q.put({"type": event_type, **data})


def _match_agent_from_text(text: str) -> str:
    """Fuzzy-match an agent name from description/prompt text."""
    text_lower = text.lower()
    # First: exact agent name match (e.g. "parser", "citation_recovery")
    for agent in KNOWN_AGENTS:
        if agent in text_lower or agent.replace("_", " ") in text_lower:
            return agent
    # Second: keyword-based fuzzy match
    for agent, keywords in _AGENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return agent
    return "subagent"


def _tool_input_summary(tool_input: dict) -> str:
    if not isinstance(tool_input, dict):
        return str(tool_input)[:80]
    for key in ("file_path", "path", "command", "pattern", "prompt", "description", "query"):
        if key in tool_input:
            val = str(tool_input[key])
            return val[:120] + ("..." if len(val) > 120 else "")
    return str(tool_input)[:80]


# ═══════════════════════════════════════════════════════════════════
# PIPELINE RUNNER (background thread)
# ═══════════════════════════════════════════════════════════════════

def _run_pipeline_bg(job_id: str, config: PipelineConfig, **kwargs):
    """Run the Nerfify pipeline in a background thread, pushing events to SSE queue."""
    job = JOBS[job_id]
    workspace = Path(job["workspace"])
    output_dir = Path(job["output_dir"])
    # Track tool_use_id → agent name and task_id → agent name for correlation
    _tool_agent_map: dict[str, str] = {}    # tool_use_id → agent name
    _task_agent_map: dict[str, str] = {}    # task_id → agent name
    _bash_tool_ids: set[str] = set()        # tool_use_ids that are Bash calls
    _cumulative_tokens: list[int] = [0]      # running total across all tasks (list for closure mutation)
    _task_tokens: dict[str, int] = {}       # per-task cumulative (for delta calc)
    stop_event = JOB_STOP_EVENTS.get(job_id)

    agents = build_agent_definitions(config)

    prompt = build_orchestrator_prompt(
        arxiv=kwargs.get("arxiv"),
        pdf_url=kwargs.get("pdf_url"),
        pdf_path=kwargs.get("pdf_path"),
        method_name=kwargs.get("method_name"),
        data_path=kwargs.get("data_path") or config.default_dataset,
        config=config,
        workspace=workspace,
        output_dir=output_dir,
    )

    (workspace / "orchestrator_prompt.txt").write_text(prompt)

    # Build env for model routing (LiteLLM proxy for non-Claude models)
    model = kwargs.get("model")
    api_key = kwargs.get("api_key")
    sdk_env = proxy_env(model or "claude", api_key=api_key)
    if sdk_env:
        push_event(job_id, "system", {"message": f"Model: {model or 'claude'}" + (" (via LiteLLM proxy)" if "ANTHROPIC_BASE_URL" in sdk_env else "")})

    push_event(job_id, "stage", {"stage": "starting", "message": "Nerfify orchestrator agent starting..."})

    start_time = time.time()

    async def _run():
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
            # Check for graceful stop
            if stop_event and stop_event.is_set():
                job["status"] = "stopped"
                push_event(job_id, "stopped", {"message": "Pipeline stopped by user."})
                return

            elapsed = time.time() - start_time

            if isinstance(message, ResultMessage):
                job["status"] = "completed"
                job["result"] = message.result
                # Extract final usage from ResultMessage
                usage = getattr(message, "usage", None) or {}
                cost = getattr(message, "total_cost_usd", None)
                dur = getattr(message, "duration_ms", 0)
                dur_api = getattr(message, "duration_api_ms", 0)
                num_turns = getattr(message, "num_turns", 0)
                # Log raw usage dict for debugging
                print(f"[nerfify] ResultMessage usage={usage}, cost={cost}, dur={dur}, turns={num_turns}")
                # Forward everything — let the frontend pick known keys
                payload = {
                    "total_cost_usd": cost,
                    "num_turns": num_turns,
                    "duration_ms": dur,
                    "duration_api_ms": dur_api,
                }
                if isinstance(usage, dict):
                    payload.update(usage)
                push_event(job_id, "usage_final", payload)
                push_event(job_id, "result", {
                    "message": message.result or "Nerfify pipeline complete.",
                    "elapsed": round(elapsed, 1),
                    "output_dir": str(output_dir),
                })

            # IMPORTANT: Task* subclasses must be checked BEFORE SystemMessage
            # because TaskStartedMessage, TaskProgressMessage, TaskNotificationMessage
            # all inherit from SystemMessage — isinstance would match the parent first.
            elif isinstance(message, TaskStartedMessage):
                task_type = getattr(message, "task_type", None) or ""
                desc = getattr(message, "description", "") or ""
                task_id = getattr(message, "task_id", "") or ""
                tool_use_id = getattr(message, "tool_use_id", "") or ""
                data = getattr(message, "data", {}) or {}
                # Correlate via tool_use_id first, then fuzzy match
                matched = _tool_agent_map.get(tool_use_id, "")
                if not matched or matched == "subagent":
                    match_text = f"{task_type} {desc} {task_id} {data.get('name', '')} {data.get('agent', '')}"
                    matched = _match_agent_from_text(match_text) or matched
                # Store task_id → agent mapping for progress/done correlation
                if task_id and matched:
                    _task_agent_map[task_id] = matched
                print(f"[nerfify] TaskStarted: type={task_type} desc={desc[:80]} tool_use_id={tool_use_id} task_id={task_id} → {matched}")
                push_event(job_id, "agent_active", {
                    "agent": matched,
                    "task": desc,
                    "task_id": task_id,
                    "message": f"{matched} agent started" + (f": {desc}" if desc else ""),
                })

            elif isinstance(message, TaskProgressMessage):
                usage = getattr(message, "usage", None)
                data = getattr(message, "data", None) or {}
                task_id = getattr(message, "task_id", "") or ""
                task_tokens = 0
                tool_uses = 0
                dur_ms = 0
                if usage:
                    if isinstance(usage, dict):
                        task_tokens = usage.get("total_tokens", 0) or 0
                        tool_uses = usage.get("tool_uses", 0) or 0
                        dur_ms = usage.get("duration_ms", 0) or 0
                    else:
                        task_tokens = getattr(usage, "total_tokens", 0) or 0
                        tool_uses = getattr(usage, "tool_uses", 0) or 0
                        dur_ms = getattr(usage, "duration_ms", 0) or 0
                # Accumulate: task_progress gives cumulative per-task, compute delta
                prev = _task_tokens.get(task_id, 0)
                delta = max(0, task_tokens - prev)
                _task_tokens[task_id] = task_tokens
                _cumulative_tokens[0] += delta
                dur_str = f"{dur_ms/1000:.1f}s" if dur_ms else ""
                last_tool = getattr(message, "last_tool_name", None)
                agent = _task_agent_map.get(task_id, "")
                push_event(job_id, "task_progress", {
                    "total_tokens": _cumulative_tokens[0],
                    "task_tokens": task_tokens,
                    "tool_uses": tool_uses,
                    "duration": dur_str,
                    "duration_ms": dur_ms,
                    "last_tool": last_tool,
                    "task_id": task_id,
                    "agent": agent,
                    "message": f"Progress: {_cumulative_tokens[0]:,} tokens, {tool_uses} tools {dur_str}".strip(),
                })

            elif isinstance(message, TaskNotificationMessage):
                status = getattr(message, "status", None)
                status_str = getattr(status, "value", str(status)) if status else "done"
                summary = getattr(message, "summary", "") or ""
                task_id = getattr(message, "task_id", "") or ""
                tool_use_id = getattr(message, "tool_use_id", "") or ""
                data = getattr(message, "data", {}) or {}
                # Correlate via task_id/tool_use_id maps first
                matched = _task_agent_map.get(task_id, "") or _tool_agent_map.get(tool_use_id, "")
                if not matched or matched == "subagent":
                    match_text = f"{summary} {task_id} {data.get('name', '')} {data.get('agent', '')}"
                    matched = _match_agent_from_text(match_text) or matched or "subagent"
                print(f"[nerfify] TaskDone: status={status_str} summary={summary[:80]} task_id={task_id} → {matched}")
                push_event(job_id, "agent_done", {
                    "agent": matched,
                    "status": status_str,
                    "task_id": task_id,
                    "message": f"{matched} agent completed ({status_str})",
                })
                push_event(job_id, "task_done", {
                    "status": status_str,
                    "message": f"Subagent finished: {status_str}",
                })

            elif isinstance(message, SystemMessage):
                sid = getattr(message, "session_id", None)
                if sid:
                    job["session_id"] = sid
                    push_event(job_id, "system", {"message": f"Session: {sid}"})

            elif isinstance(message, AssistantMessage):
                content = getattr(message, "content", [])
                for block in content:
                    if isinstance(block, ThinkingBlock):
                        text = getattr(block, "thinking", "")
                        if text:
                            # Send full thinking so user can see agent reasoning
                            push_event(job_id, "thinking", {
                                "text": text,
                                "message": text[:500] + ("..." if len(text) > 500 else ""),
                            })

                    elif isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(job_id, "text", {"text": text, "message": text})

                    elif isinstance(block, ToolUseBlock):
                        name = getattr(block, "name", "?")
                        tool_id = getattr(block, "id", None)
                        tool_input = getattr(block, "input", {})
                        summary = _tool_input_summary(tool_input)

                        if name == "Agent":
                            # Detect which agent is being invoked
                            desc = tool_input.get("description", "")
                            prompt_text = tool_input.get("prompt", "")
                            subagent_type = tool_input.get("subagent_type", "")
                            agent_name = _match_agent_from_text(f"{subagent_type} {desc} {prompt_text}")
                            # Track tool_use_id → agent for task correlation
                            if tool_id:
                                _tool_agent_map[tool_id] = agent_name
                            print(f"[nerfify] Agent tool: id={tool_id} desc={desc[:60]} → {agent_name}")
                            push_event(job_id, "agent_active", {
                                "agent": agent_name,
                                "task": desc,
                                "message": f"Invoking {agent_name} agent: {desc}",
                            })
                        elif name == "Bash":
                            cmd = tool_input.get("command", "")
                            if tool_id:
                                _bash_tool_ids.add(tool_id)
                            push_event(job_id, "bash", {
                                "id": tool_id,
                                "command": cmd[:1000],
                                "message": f"$ {cmd[:300]}",
                            })
                        else:
                            push_event(job_id, "tool", {
                                "id": tool_id,
                                "name": name,
                                "summary": summary,
                                "input": tool_input if isinstance(tool_input, (dict, list)) else str(tool_input),
                                "message": f"{name}: {summary}",
                            })

                    elif isinstance(block, ToolResultBlock):
                        tool_id = getattr(block, "tool_use_id", None)
                        content_val = getattr(block, "content", "")
                        if isinstance(content_val, list):
                            content_val = " ".join(str(c) for c in content_val)
                        is_bash = tool_id in _bash_tool_ids
                        # More output for bash (3000 chars), normal for others (500)
                        max_len = 3000 if is_bash else 500
                        output_text = str(content_val)[:max_len]
                        # Detect file writes from tool results
                        file_name = None
                        if "Written:" in output_text or "write_file" in output_text:
                            import re as _re
                            m = _re.search(r'(?:Written:|write_file.*?)(\S+\.(?:py|toml|md))', output_text)
                            if m:
                                file_name = m.group(1).split("/")[-1]
                        if is_bash:
                            push_event(job_id, "bash_output", {
                                "id": tool_id,
                                "output": output_text,
                                "message": output_text[:300],
                            })
                        else:
                            push_event(job_id, "tool_result", {
                                "id": tool_id,
                                "output": output_text,
                                "file": file_name,
                            })

    try:
        asyncio.run(_run())
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        push_event(job_id, "error", {"message": str(e)})

    # List generated files
    gen_files = sorted(str(f.relative_to(output_dir)) for f in output_dir.rglob("*") if f.is_file())
    job["generated_files"] = gen_files

    elapsed = time.time() - start_time
    job["duration"] = round(elapsed, 1)

    push_event(job_id, "done", {
        "message": "Nerfify pipeline finished",
        "files": gen_files,
        "duration": round(elapsed, 1),
        "status": job.get("status", "completed"),
    })


# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/auth-status")
def api_auth_status():
    """Check which API credentials are already available in the environment."""
    status = {}
    # Check Anthropic (Claude Agent SDK uses this)
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_anthropic_token = bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    # Check if claude CLI is logged in (has ~/.claude/ config)
    claude_logged_in = Path.home().joinpath(".claude").exists()
    status["anthropic"] = {
        "available": has_anthropic_key or has_anthropic_token or claude_logged_in,
        "source": "api_key" if has_anthropic_key else "oauth" if has_anthropic_token else "claude_cli" if claude_logged_in else None,
    }
    # Check other providers
    status["openai"] = {"available": bool(os.environ.get("OPENAI_API_KEY"))}
    status["gemini"] = {"available": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))}
    status["deepseek"] = {"available": bool(os.environ.get("DEEPSEEK_API_KEY"))}
    # Claude Agent SDK (main.py) always uses Anthropic auth — no extra key needed if logged in
    status["claude_sdk"] = {"available": has_anthropic_key or has_anthropic_token or claude_logged_in}
    return jsonify(status)


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json() or {}
    arxiv = data.get("arxiv", "").strip()
    pdf_url = None
    pdf_path = data.get("pdf_path", "").strip() or None
    method_name = (data.get("method_name") or "").strip() or None
    data_path = (data.get("data_path") or "").strip() or None

    enable_review = data.get("enable_review", True)
    enable_validation = data.get("enable_validation", True)
    enable_smoke_test = data.get("enable_smoke_test", True)
    enable_training = data.get("enable_training", False)
    fast_mode = data.get("fast_mode", False)
    tiered_models = data.get("tiered_models", False)

    # Model & API key
    model = (data.get("model") or "").strip() or None
    auth_mode = (data.get("auth_mode") or "key").strip()
    api_key = (data.get("api_key") or "").strip() or None

    # Set API key in environment for the pipeline to use
    if api_key:
        if auth_mode == "oauth":
            # OAuth token → set as Authorization header via env var
            os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            # Standard API key
            os.environ["ANTHROPIC_API_KEY"] = api_key
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

        # Also set keys for other providers based on model selection
        model_str = model or ""
        if model_str.startswith("gpt") or model_str.startswith("o1") or model_str.startswith("o3"):
            os.environ["OPENAI_API_KEY"] = api_key
        elif model_str.startswith("gemini"):
            os.environ["GEMINI_API_KEY"] = api_key
        elif model_str.startswith("deepseek"):
            os.environ["DEEPSEEK_API_KEY"] = api_key
    else:
        # No API key provided — clear stale keys so SDK falls back to OAuth / CLI auth
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    # Detect if arxiv field contains a direct PDF URL (not arXiv)
    if arxiv and arxiv.startswith("http") and "arxiv.org" not in arxiv:
        pdf_url = arxiv
        arxiv = ""

    # Handle PDF upload
    if not arxiv and not pdf_url and not pdf_path:
        pdf_file = request.files.get("pdf")
        if pdf_file and pdf_file.filename:
            fname = secure_filename(pdf_file.filename)
            upload_dir = DEFAULT_CONFIG.workspace_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = str(upload_dir / fname)
            pdf_file.save(pdf_path)

    if not arxiv and not pdf_url and not pdf_path:
        return jsonify({"error": "Provide an arXiv URL/ID, PDF URL, or PDF path"}), 400

    # Create job
    job_id = uuid.uuid4().hex[:12]
    config = PipelineConfig()
    config.enable_review = enable_review
    config.enable_validation = enable_validation
    config.enable_smoke_test = enable_smoke_test
    config.enable_training = enable_training
    config.fast_mode = fast_mode
    config.tiered_models = tiered_models
    config.ensure_dirs()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + job_id[:6]
    workspace = config.workspace_dir / run_id
    output_dir = config.generated_dir / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "run_id": run_id,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "arxiv": arxiv,
        "pdf_url": pdf_url,
        "model": model,
        "created": datetime.now().isoformat(),
    }
    JOB_QUEUES[job_id] = queue.Queue()
    JOB_STOP_EVENTS[job_id] = threading.Event()

    t = threading.Thread(
        target=_run_pipeline_bg,
        args=(job_id, config),
        kwargs={
            "arxiv": arxiv or None,
            "pdf_url": pdf_url,
            "pdf_path": pdf_path,
            "method_name": method_name,
            "data_path": data_path,
            "model": model,
            "api_key": api_key,
        },
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "run_id": run_id})


@app.route("/api/events/<job_id>")
def api_events(job_id):
    """SSE endpoint for real-time streaming."""
    q = JOB_QUEUES.get(job_id)
    if not q:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
                event_type = event.get("type", "message")
                payload = {k: v for k, v in event.items() if k != "type"}
                yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
                if event_type in ("done", "error", "stopped"):
                    break
            except queue.Empty:
                yield f"event: heartbeat\ndata: {{}}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/jobs")
def api_jobs():
    return jsonify([
        {"id": j["id"], "status": j.get("status"), "arxiv": j.get("arxiv"), "created": j.get("created")}
        for j in JOBS.values()
    ])


@app.route("/api/stop/<job_id>", methods=["POST"])
def api_stop(job_id):
    """Signal a running job to stop gracefully."""
    stop_event = JOB_STOP_EVENTS.get(job_id)
    if not stop_event:
        return jsonify({"error": "Job not found"}), 404
    stop_event.set()
    return jsonify({"message": f"Stop signal sent to job {job_id}"})


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5006"))
    print(f"Nerfify Claude Agents — http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True,use_reloader=False)