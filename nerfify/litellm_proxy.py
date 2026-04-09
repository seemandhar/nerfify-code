"""
LiteLLM Proxy Manager for Nerfify

Starts a LiteLLM proxy server that translates Anthropic Messages API format
to other providers (OpenAI, Gemini, DeepSeek, Ollama, etc.).

This allows the Claude Agent SDK (which only speaks Anthropic API) to work
with any LLM by pointing ANTHROPIC_BASE_URL at the proxy.

Usage:
    from litellm_proxy import ensure_proxy, proxy_env

    # Start proxy if needed (returns port)
    port = ensure_proxy("gpt-4o")

    # Get env dict for ClaudeAgentOptions
    env = proxy_env("gpt-4o", api_key="sk-...")
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import socket
import time
from pathlib import Path
from typing import Any

_proxy_process: subprocess.Popen | None = None
_proxy_port: int | None = None


def _find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _is_claude_model(model: str) -> bool:
    """Check if the model is a native Claude model (no proxy needed)."""
    m = (model or "").lower()
    return m.startswith("claude") or m.startswith("anthropic/")


def _is_proxy_running(port: int) -> bool:
    """Check if the proxy is responding."""
    try:
        import httpx
        r = httpx.get(f"http://localhost:{port}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_proxy(model: str, *, port: int | None = None, api_key: str | None = None) -> int | None:
    """Start a LiteLLM proxy if the model requires one.

    Returns the proxy port, or None if no proxy is needed (Claude model).
    """
    global _proxy_process, _proxy_port

    if _is_claude_model(model):
        return None

    # Already running?
    if _proxy_port and _is_proxy_running(_proxy_port):
        return _proxy_port

    port = port or _find_free_port()

    # Build the litellm proxy config
    # LiteLLM proxy translates Anthropic API format → target provider
    env = {**os.environ}
    if api_key:
        # Set the right env var based on model prefix
        m = model.lower()
        if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or m.startswith("openai/"):
            env["OPENAI_API_KEY"] = api_key
        elif m.startswith("gemini") or m.startswith("google/"):
            env["GEMINI_API_KEY"] = api_key
        elif m.startswith("deepseek"):
            env["DEEPSEEK_API_KEY"] = api_key
        else:
            # Generic — set both common ones
            env["OPENAI_API_KEY"] = api_key

    # Use the litellm CLI binary from the same Python prefix
    # (python -m litellm fails in newer versions that lack __main__.py)
    litellm_bin = os.path.join(os.path.dirname(sys.executable), "litellm")
    cmd = [
        litellm_bin,
        "--model", model,
        "--port", str(port),
        "--drop_params",  # Silently ignore Anthropic-specific params
        "--num_workers", "1",
    ]

    print(f"[litellm-proxy] Starting proxy for {model} on port {port}...")
    _proxy_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _proxy_port = port

    # Register cleanup
    atexit.register(_stop_proxy)

    # Wait for proxy to be ready (up to 30s)
    for _ in range(60):
        if _proxy_process.poll() is not None:
            stderr = _proxy_process.stderr.read().decode() if _proxy_process.stderr else ""
            raise RuntimeError(f"LiteLLM proxy failed to start:\n{stderr}")
        if _is_proxy_running(port):
            print(f"[litellm-proxy] Ready on http://localhost:{port}")
            return port
        time.sleep(0.5)

    raise RuntimeError("LiteLLM proxy did not become ready within 30s")


def _stop_proxy():
    """Terminate the proxy process."""
    global _proxy_process, _proxy_port
    if _proxy_process:
        _proxy_process.terminate()
        try:
            _proxy_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proxy_process.kill()
        _proxy_process = None
        _proxy_port = None
        print("[litellm-proxy] Stopped.")


def stop_proxy():
    """Public API to stop the proxy."""
    _stop_proxy()


def proxy_env(
    model: str,
    *,
    api_key: str | None = None,
    proxy_port: int | None = None,
) -> dict[str, str]:
    """Build an environment dict for ClaudeAgentOptions.

    For Claude models: returns empty dict (use native API).
    For other models: starts proxy if needed, returns env vars to route
    Claude Agent SDK through the LiteLLM proxy.
    """
    if _is_claude_model(model):
        env = {}
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        return env

    # Ensure proxy is running
    port = proxy_port or ensure_proxy(model, api_key=api_key)
    if port is None:
        return {}

    env = {
        "ANTHROPIC_BASE_URL": f"http://localhost:{port}",
        # LiteLLM proxy uses its own auth; set a dummy key so SDK doesn't complain
        "ANTHROPIC_API_KEY": api_key or os.environ.get("ANTHROPIC_API_KEY", "litellm-proxy"),
    }
    return env
