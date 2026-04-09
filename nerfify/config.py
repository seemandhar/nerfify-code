"""
Configuration for the Nerfify Claude Agents pipeline.

All paths are relative to the nerfify package directory which contains
the base-code templates, PapersAndCode examples, and agent definitions.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ── PSNR Baselines (approximate, for quality gating) ─────────────
# These are baseline PSNR values for standard NeRF methods on common datasets.
# The pipeline uses (baseline - margin) as the minimum acceptable PSNR.
DATASET_PSNR_BASELINES: dict[str, dict[str, float]] = {
    "mipnerf360": {
        "bicycle": 25.25, "bonsai": 32.20, "counter": 28.70, "garden": 27.30,
        "kitchen": 30.80, "room": 31.40, "stump": 26.55, "flowers": 21.70,
        "treehill": 22.50, "_default": 27.30,
    },
    "blender": {
        "chair": 33.00, "drums": 25.01, "ficus": 30.13, "hotdog": 36.18,
        "lego": 32.54, "materials": 29.62, "mic": 32.91, "ship": 28.65,
        "_default": 31.01,
    },
    "llff": {
        "fern": 25.17, "flower": 27.40, "fortress": 31.16, "horns": 27.45,
        "leaves": 20.92, "orchids": 20.36, "room": 32.70, "trex": 26.80,
        "_default": 26.50,
    },
}

class ModelTier(str, Enum):
    """Model tiers for cost/performance tradeoff."""
    CHEAP = "cheap"    # Haiku — fast, low-cost (simple tasks)
    MID = "mid"        # Sonnet — balanced (analysis, review)
    EXPENSIVE = "expensive"  # Opus — highest quality (code generation)


# Default tier assignments per agent
AGENT_DEFAULT_TIERS: dict[str, ModelTier] = {
    "parser": ModelTier.CHEAP,
    "citation_recovery": ModelTier.MID,
    "planner": ModelTier.MID,
    "coder": ModelTier.EXPENSIVE,
    "reviewer": ModelTier.MID,
    "validator": ModelTier.MID,
    "integrator": ModelTier.CHEAP,
    "tester": ModelTier.CHEAP,
    "debugger": ModelTier.EXPENSIVE,
}

# Model IDs per tier (Anthropic defaults; overridden per-provider in main_api.py)
TIER_MODELS: dict[str, dict[ModelTier, str]] = {
    "anthropic": {
        ModelTier.CHEAP: "claude-haiku-4-5-20251001",
        ModelTier.MID: "claude-sonnet-4-20250514",
        ModelTier.EXPENSIVE: "claude-opus-4-6",
    },
    "openai": {
        ModelTier.CHEAP: "gpt-4o-mini",
        ModelTier.MID: "gpt-4o",
        ModelTier.EXPENSIVE: "gpt-4.1",
    },
    "gemini": {
        ModelTier.CHEAP: "gemini/gemini-2.5-flash",
        ModelTier.MID: "gemini/gemini-2.5-pro",
        ModelTier.EXPENSIVE: "gemini/gemini-2.5-pro",
    },
}


def get_model_for_agent(agent_name: str, base_model: str = "claude-sonnet-4-20250514", tiered: bool = False) -> str:
    """Return the model ID to use for a given agent.

    If tiered=False, returns base_model for all agents.
    If tiered=True, maps the agent's tier to the appropriate model from the same provider family.
    """
    if not tiered:
        return base_model

    tier = AGENT_DEFAULT_TIERS.get(agent_name, ModelTier.MID)

    # Detect provider from base_model
    base_lower = base_model.lower()
    if "claude" in base_lower or "anthropic" in base_lower or "haiku" in base_lower or "sonnet" in base_lower or "opus" in base_lower:
        provider = "anthropic"
    elif "gpt" in base_lower or "o1" in base_lower or "o3" in base_lower:
        provider = "openai"
    elif "gemini" in base_lower:
        provider = "gemini"
    else:
        # Unknown provider — just return base_model
        return base_model

    return TIER_MODELS.get(provider, {}).get(tier, base_model)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Base directory (the nerfify package itself)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    # Output directories
    workspace_dir: Path = field(default_factory=lambda: Path(__file__).parent / "workspace")
    generated_dir: Path = field(default_factory=lambda: Path(__file__).parent / "generated")

    # Paths derived from base_dir
    @property
    def template_root(self) -> Path:
        return self.base_dir / "base-code"

    @property
    def papers_and_code(self) -> Path:
        return self.base_dir / "PapersAndCode"

    @property
    def default_dataset(self) -> str:
        return str(Path.home() / "data" / "nerf_synthetic" / "lego")

    # Pipeline options
    enable_review: bool = True
    enable_validation: bool = True
    enable_smoke_test: bool = True
    enable_training: bool = False
    incremental_generation: bool = True
    max_review_iterations: int = 2
    max_debug_iterations: int = 3

    # Optimization options
    fast_mode: bool = False           # Skip citation_recovery if paper is self-contained
    tiered_models: bool = False       # Use different model tiers per agent
    validation_skip_threshold: int = 90  # Skip review if validation coverage >= this %

    # Nerfstudio
    conda_env: str = "nerfstudio"
    default_max_iters: int = 3000

    # Training & PSNR feedback loop
    enable_psnr_feedback: bool = True     # After training, read PSNR and fix if bad
    max_psnr_fix_iterations: int = 2      # Max code-fix → retrain cycles
    psnr_margin: float = 2.0             # Accept if PSNR >= baseline - margin
    dataset: str = "mipnerf360"           # Dataset for training
    scenes: list = field(default_factory=lambda: ["garden"])
    training_gpu: int = 0
    expected_psnr: float | None = None    # Override target PSNR (if paper specifies)

    @property
    def read_tb_script(self) -> Path:
        return self.base_dir / "read_tb.py"

    @property
    def eval_script(self) -> Path:
        return self.base_dir / "eval.py"

    @property
    def training_output_dir(self) -> Path:
        return self.base_dir / "outputs"

    def get_psnr_target(self, scene: str | None = None) -> float:
        """Get minimum acceptable PSNR for the configured dataset/scene."""
        if self.expected_psnr is not None:
            return self.expected_psnr
        baselines = DATASET_PSNR_BASELINES.get(self.dataset, {})
        s = scene or (self.scenes[0] if self.scenes else "_default")
        baseline = baselines.get(s, baselines.get("_default", 25.0))
        return round(baseline - self.psnr_margin, 1)

    # Required output files
    REQUIRED_FILES: list = field(default_factory=lambda: [
        "method_template/__init__.py",
        "method_template/template_config.py",
        "method_template/template_datamanager.py",
        "method_template/template_field.py",
        "method_template/template_model.py",
        "method_template/template_pipeline.py",
        "README.md",
        "pyproject.toml",
    ])

    # Known agent names in the pipeline
    KNOWN_AGENTS: list = field(default_factory=lambda: [
        "parser",
        "citation_recovery",
        "planner",
        "coder",
        "reviewer",
        "validator",
        "integrator",
        "tester",
        "debugger",
    ])

    def ensure_dirs(self):
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
