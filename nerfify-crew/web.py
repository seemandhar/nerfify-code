#!/usr/bin/env python3
"""
Nerfify-Crew — Web UI

Flask app with real-time SSE streaming of the CrewAI multi-agent pipeline.
"""
from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

from config import PipelineConfig
from agents.definitions import build_agents
from tasks import build_tasks

from crewai import Crew, Process
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionStartedEvent,
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
)
from crewai.events.types.task_events import (
    TaskStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageErrorEvent,
)
import litellm
from crewai.events.types.crew_events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nerfify-crew-dev")

JOBS: dict[str, dict[str, Any]] = {}
JOB_QUEUES: dict[str, queue.Queue] = {}
JOB_STOP_EVENTS: dict[str, threading.Event] = {}

DEFAULT_CONFIG = PipelineConfig()
DEFAULT_CONFIG.ensure_dirs()

KNOWN_AGENTS = {
    "parser", "citation_recovery", "planner", "coder",
    "reviewer", "tester", "debugger",
}

_AGENT_KEYWORDS: dict[str, list[str]] = {
    "parser": ["parse", "extract", "pdf", "markdown", "clean", "paper parser"],
    "citation_recovery": ["citation", "recovery", "reference", "cited", "citation recovery"],
    "planner": ["plan", "architect", "dag", "design", "architecture planner"],
    "coder": ["code", "generate", "implement", "nerfstudio engineer", "senior nerfstudio", "code generator"],
    "reviewer": ["review", "check", "lint", "code reviewer"],
    "tester": ["test", "smoke", "smoke tester"],
    "debugger": ["debug", "diagnos", "traceback", "debug specialist"],
}


def push_event(job_id: str, event_type: str, data: dict):
    q = JOB_QUEUES.get(job_id)
    if q:
        q.put({"type": event_type, **data})


import re as _re

# Lines matching these patterns are noise — skip them in UI
_NOISE_PATTERNS = [
    _re.compile(r"^\s*\d+K\s+\."),                     # wget progress bars
    _re.compile(r"^Fetching \d+ files:"),               # HF hub fetching
    _re.compile(r"^\s*\d+%\|[█▏▎▍▌▋▊▉ ]+\|"),         # tqdm progress bars
    _re.compile(r"^(Resolving|Connecting|Reusing|Saving to|Length:)"),  # wget status
    _re.compile(r"^--\d{4}-\d{2}-\d{2}"),              # wget timestamp lines
    _re.compile(r"^HTTP request sent"),                  # wget
    _re.compile(r"^Will not apply HSTS"),               # wget warnings
    _re.compile(r"^ERROR: could not open HSTS"),        # wget warnings
    _re.compile(r"^Location:.*\[following\]"),          # wget redirect
    _re.compile(r"^\s*$"),                              # blank lines
    _re.compile(r"^[═─━]{5,}"),                         # separator lines
    _re.compile(r"^\.+$"),                              # dots-only lines
    _re.compile(r"^\s*warnings?\.warn\("),              # Python warnings
    _re.compile(r"^/home/.*/site-packages/"),           # file path warnings
]

# Lines matching these are important — always show
_IMPORTANT_PATTERNS = [
    _re.compile(r"(agent|task|tool|action|thought|final answer|observation)", _re.I),
    _re.compile(r"(error|exception|traceback|failed)", _re.I),
    _re.compile(r"^(STDOUT|STDERR|EXIT CODE):"),
    _re.compile(r"(working agent|starting task|delegating)", _re.I),
    _re.compile(r"https?://"),                          # URLs (viewer detection)
    _re.compile(r"(pip install|ns-train|mineru|conda)", _re.I),
    _re.compile(r"(PSNR|loss|training|iteration)", _re.I),
]


def _is_noise(line: str) -> bool:
    """Return True if the line is noise that should not be sent to the UI."""
    for pat in _IMPORTANT_PATTERNS:
        if pat.search(line):
            return False
    for pat in _NOISE_PATTERNS:
        if pat.search(line):
            return True
    # Skip very long lines (raw file content dumps)
    if len(line) > 1000:
        return True
    return False


class _StreamCapture(io.TextIOBase):
    """Captures writes to stdout/stderr and forwards filtered lines as SSE events,
    while also writing to the original stream so terminal output is preserved."""

    def __init__(self, job_id: str, original_stream, stream_name: str = "stdout"):
        self._job_id = job_id
        self._original = original_stream
        self._stream_name = stream_name
        self._buf = ""
        self._viewer_notified = False
        self._lines_sent = 0
        self._lines_skipped = 0

    def write(self, s):
        if self._original:
            self._original.write(s)
        if not s:
            return 0
        # Buffer and flush line-by-line
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue

            # Detect viser viewer URL (always, even if line is "noise")
            if not self._viewer_notified and ("viser" in line.lower() or "viewer" in line.lower()) and "http" in line:
                url_match = _re.search(r'https?://[^\s]+', line)
                if url_match:
                    push_event(self._job_id, "viewer_ready", {
                        "url": url_match.group(0),
                        "message": f"Live viewer: {url_match.group(0)}",
                    })
                    self._viewer_notified = True

            # Filter noise
            if _is_noise(line):
                self._lines_skipped += 1
                continue

            self._lines_sent += 1
            push_event(self._job_id, "console", {
                "stream": self._stream_name,
                "text": line[:500],
                "message": line[:200],
            })
        return len(s)

    def flush(self):
        if self._original:
            self._original.flush()
        if self._buf.strip():
            line = self._buf.strip()
            if not _is_noise(line):
                push_event(self._job_id, "console", {
                    "stream": self._stream_name,
                    "text": line[:500],
                    "message": line[:200],
                })
            self._buf = ""

    def isatty(self):
        return False


def _match_agent_from_text(text: str) -> str:
    text_lower = text.lower()
    for agent in KNOWN_AGENTS:
        if agent in text_lower or agent.replace("_", " ") in text_lower:
            return agent
    for agent, keywords in _AGENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return agent
    return "subagent"


def _run_pipeline_bg(job_id: str, config: PipelineConfig, **kwargs):
    """Run the CrewAI pipeline in a background thread with event streaming."""
    job = JOBS[job_id]
    workspace = Path(job["workspace"])
    output_dir = Path(job["output_dir"])
    stop_event = JOB_STOP_EVENTS.get(job_id)

    # Capture stdout/stderr so all CrewAI verbose output reaches the UI
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _StreamCapture(job_id, old_stdout, "stdout")
    sys.stderr = _StreamCapture(job_id, old_stderr, "stderr")

    push_event(job_id, "stage", {
        "stage": "starting",
        "message": "Building CrewAI agents and tasks...",
    })

    start_time = time.time()

    # Determine paper input
    arxiv = kwargs.get("arxiv")
    pdf_url = kwargs.get("pdf_url")
    pdf_path = kwargs.get("pdf_path")
    if arxiv:
        paper_input = f"arXiv paper: {arxiv}"
    elif pdf_url:
        paper_input = f"PDF URL (download first): {pdf_url}"
    elif pdf_path:
        paper_input = f"Local PDF: {pdf_path}"
    else:
        push_event(job_id, "error", {"message": "No paper input provided"})
        return

    try:
        agents = build_agents(config)
        push_event(job_id, "text", {
            "text": f"Loaded {len(agents)} agents: {', '.join(agents.keys())}",
            "message": f"Loaded {len(agents)} agents",
        })

        tasks = build_tasks(
            agents,
            paper_input=paper_input,
            workspace=workspace,
            output_dir=output_dir,
            method_name=kwargs.get("method_name"),
            data_path=kwargs.get("data_path") or config.default_dataset,
            config=config,
        )
        push_event(job_id, "text", {
            "text": f"Created {len(tasks)} pipeline tasks",
            "message": f"Created {len(tasks)} tasks",
        })

        # Set up CrewAI event handlers for this job via the global event bus.
        # Each handler is registered with @crewai_event_bus.on(EventType) and
        # receives the event object directly.
        handlers = []

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def _on_agent_start(source, event):
            agent_obj = getattr(event, "agent", None)
            agent_role = getattr(agent_obj, "role", "") if agent_obj else ""
            agent_name = _match_agent_from_text(agent_role)
            push_event(job_id, "agent_active", {
                "agent": agent_name,
                "task": agent_role,
                "message": f"{agent_name} agent started ({agent_role})",
            })
        handlers.append((AgentExecutionStartedEvent, _on_agent_start))

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def _on_agent_done(source, event):
            agent_obj = getattr(event, "agent", None)
            agent_role = getattr(agent_obj, "role", "") if agent_obj else ""
            agent_name = _match_agent_from_text(agent_role)
            output = getattr(event, "output", "") or ""
            push_event(job_id, "agent_done", {
                "agent": agent_name,
                "status": "completed",
                "message": f"{agent_name} agent completed",
            })
            if output:
                push_event(job_id, "text", {
                    "text": str(output)[:500],
                    "message": f"{agent_name}: {str(output)[:200]}",
                })
        handlers.append((AgentExecutionCompletedEvent, _on_agent_done))

        @crewai_event_bus.on(AgentExecutionErrorEvent)
        def _on_agent_error(source, event):
            agent_obj = getattr(event, "agent", None)
            agent_role = getattr(agent_obj, "role", "") if agent_obj else ""
            error = getattr(event, "error", "") or ""
            agent_name = _match_agent_from_text(agent_role)
            push_event(job_id, "agent_done", {
                "agent": agent_name,
                "status": "error",
                "message": f"{agent_name} error: {str(error)[:200]}",
            })
        handlers.append((AgentExecutionErrorEvent, _on_agent_error))

        @crewai_event_bus.on(TaskStartedEvent)
        def _on_task_start(source, event):
            task_obj = getattr(event, "task", None)
            desc = ""
            if task_obj:
                desc = getattr(task_obj, "name", "") or getattr(task_obj, "description", "")[:80] or ""
            push_event(job_id, "stage", {
                "stage": "task",
                "message": f"Task started: {desc[:100]}" if desc else "Task started",
            })
        handlers.append((TaskStartedEvent, _on_task_start))

        @crewai_event_bus.on(TaskCompletedEvent)
        def _on_task_done(source, event):
            output = getattr(event, "output", "") or ""
            push_event(job_id, "text", {
                "text": f"Task completed: {str(output)[:300]}",
                "message": "Task completed",
            })
        handlers.append((TaskCompletedEvent, _on_task_done))

        @crewai_event_bus.on(TaskFailedEvent)
        def _on_task_fail(source, event):
            error = getattr(event, "error", "") or ""
            push_event(job_id, "error", {
                "message": f"Task failed: {str(error)[:300]}",
            })
        handlers.append((TaskFailedEvent, _on_task_fail))

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _on_tool_start(source, event):
            tool_name = getattr(event, "tool_name", "") or ""
            raw_args = getattr(event, "tool_args", "") or ""
            # Extract readable args — tool_args is often a dict
            if isinstance(raw_args, dict):
                # For shell commands, show just the command string
                if "command" in raw_args:
                    args_str = raw_args["command"]
                elif "file_path" in raw_args:
                    args_str = raw_args["file_path"]
                else:
                    args_str = ", ".join(f"{k}={v}" for k, v in raw_args.items()
                                        if k != "content" and len(str(v)) < 200)
            else:
                args_str = str(raw_args)
            if "shell" in tool_name.lower() or "bash" in tool_name.lower():
                push_event(job_id, "bash", {
                    "command": args_str[:500],
                    "message": f"$ {args_str[:200]}",
                })
            else:
                push_event(job_id, "tool", {
                    "name": tool_name,
                    "summary": args_str[:200],
                    "message": f"{tool_name}: {args_str[:150]}",
                })
        handlers.append((ToolUsageStartedEvent, _on_tool_start))

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _on_tool_done(source, event):
            tool_name = getattr(event, "tool_name", "") or ""
            output = getattr(event, "output", "") or ""
            if "shell" in tool_name.lower() or "bash" in tool_name.lower():
                push_event(job_id, "bash_output", {
                    "output": str(output)[:2000],
                    "message": str(output)[:200],
                })
        handlers.append((ToolUsageFinishedEvent, _on_tool_done))

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _on_tool_error(source, event):
            error = getattr(event, "error", "") or ""
            push_event(job_id, "error", {
                "message": f"Tool error: {str(error)[:300]}",
            })
        handlers.append((ToolUsageErrorEvent, _on_tool_error))

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def _on_crew_start(source, event):
            push_event(job_id, "stage", {
                "stage": "running",
                "message": "CrewAI pipeline executing...",
            })
        handlers.append((CrewKickoffStartedEvent, _on_crew_start))

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def _on_crew_done(source, event):
            output = getattr(event, "output", "") or ""
            push_event(job_id, "text", {
                "text": f"Pipeline completed: {str(output)[:500]}",
                "message": "Pipeline completed",
            })
        handlers.append((CrewKickoffCompletedEvent, _on_crew_done))

        # ── Live token tracking via litellm callback ──
        # CrewAI uses streaming by default, so LLMCallCompletedEvent.response
        # is a plain string without usage. Instead, use litellm's success_callback
        # which always receives the full ModelResponse with usage data.
        _token_totals = {"prompt": 0, "completion": 0, "total": 0, "requests": 0}

        def _litellm_success_callback(kwargs, completion_response, start_time, end_time):
            usage = getattr(completion_response, "usage", None)
            prompt_tok = 0
            completion_tok = 0
            if usage:
                prompt_tok = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
            _token_totals["prompt"] += prompt_tok
            _token_totals["completion"] += completion_tok
            _token_totals["total"] += prompt_tok + completion_tok
            _token_totals["requests"] += 1
            push_event(job_id, "token_update", {
                "prompt_tokens": _token_totals["prompt"],
                "completion_tokens": _token_totals["completion"],
                "total_tokens": _token_totals["total"],
                "requests": _token_totals["requests"],
            })

        litellm.success_callback.append(_litellm_success_callback)

        # Enable automatic retry with backoff for rate limit errors (429)
        litellm.num_retries = 5
        litellm.retry_after = 60  # wait 60s before retry on 429 (rate limit is per-minute)

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=False,
            full_output=True,
        )

        push_event(job_id, "stage", {
            "stage": "running",
            "message": "CrewAI pipeline starting...",
        })

        result = crew.kickoff()

        job["status"] = "completed"
        job["result"] = str(result)

        # Token usage
        if hasattr(result, "token_usage"):
            usage = result.token_usage
            push_event(job_id, "usage_final", {
                "total_tokens": getattr(usage, "total_tokens", 0),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            })

        elapsed = time.time() - start_time
        push_event(job_id, "result", {
            "message": str(result)[:1000] or "Pipeline complete.",
            "elapsed": round(elapsed, 1),
            "output_dir": str(output_dir),
        })

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        push_event(job_id, "error", {"message": str(e)})
    finally:
        # Restore stdout/stderr
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Unregister event handlers to avoid leaking across jobs
        for event_type, handler in handlers:
            try:
                crewai_event_bus.off(event_type, handler)
            except Exception:
                pass

        # Remove litellm callback
        try:
            litellm.success_callback.remove(_litellm_success_callback)
        except (ValueError, Exception):
            pass

    # List generated files
    gen_files = sorted(
        str(f.relative_to(output_dir))
        for f in output_dir.rglob("*") if f.is_file()
    )
    job["generated_files"] = gen_files
    job["duration"] = round(time.time() - start_time, 1)

    push_event(job_id, "done", {
        "message": "Pipeline finished",
        "files": gen_files,
        "duration": job["duration"],
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
    status = {}
    status["anthropic"] = {"available": bool(os.environ.get("ANTHROPIC_API_KEY"))}
    status["openai"] = {"available": bool(os.environ.get("OPENAI_API_KEY"))}
    status["gemini"] = {"available": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))}
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
    enable_smoke_test = data.get("enable_smoke_test", True)
    enable_training = data.get("enable_training", False)
    fast_mode = data.get("fast_mode", False)
    tiered_models = data.get("tiered_models", False)
    include_examples = data.get("include_examples", True)
    max_iters = data.get("max_iters", 3000)
    gpu = data.get("gpu", 0)
    dataset = data.get("dataset", "mipnerf360")
    scenes = data.get("scenes", None)
    expected_psnr = data.get("expected_psnr", None)

    # Model & API key
    model = (data.get("model") or "").strip() or None
    coder_model = (data.get("coder_model") or "").strip() or None
    api_key = (data.get("api_key") or "").strip() or None

    # Set API key / provider config
    model_str = model or ""
    if "ollama" in model_str.lower():
        # Ollama runs locally — no API key needed
        os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")
    elif api_key:
        if "openai" in model_str or "gpt" in model_str:
            os.environ["OPENAI_API_KEY"] = api_key
        elif "google" in model_str or "gemini" in model_str:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key

    # Detect direct PDF URL
    if arxiv and arxiv.startswith("http") and "arxiv.org" not in arxiv:
        pdf_url = arxiv
        arxiv = ""

    if not arxiv and not pdf_url and not pdf_path:
        return jsonify({"error": "Provide an arXiv URL/ID, PDF URL, or PDF path"}), 400

    # Create job
    job_id = uuid.uuid4().hex[:12]
    config = PipelineConfig()
    config.enable_review = enable_review
    config.enable_smoke_test = enable_smoke_test
    config.enable_training = enable_training
    config.fast_mode = fast_mode
    config.tiered_models = tiered_models
    config.include_examples = include_examples
    config.default_max_iters = max_iters
    config.training_gpu = gpu
    config.dataset = dataset
    if scenes:
        config.scenes = scenes if isinstance(scenes, list) else [scenes]
    if expected_psnr is not None:
        config.expected_psnr = expected_psnr
    if model:
        config.default_model = model
    if coder_model:
        config.coder_model = coder_model
        config._coder_model_explicit = True
    config.sync_models_to_provider()
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
        },
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "run_id": run_id})


@app.route("/api/events/<job_id>")
def api_events(job_id):
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
    stop_event = JOB_STOP_EVENTS.get(job_id)
    if not stop_event:
        return jsonify({"error": "Job not found"}), 404
    stop_event.set()
    return jsonify({"message": f"Stop signal sent to job {job_id}"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5007"))
    print(f"Nerfify-Crew Web UI — http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
