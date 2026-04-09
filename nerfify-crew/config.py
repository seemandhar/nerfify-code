"""
Configuration for the Nerfify-Crew multi-agent pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ── PSNR Baselines (approximate, for quality gating) ─────────────
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
    CHEAP = "cheap"
    MID = "mid"
    EXPENSIVE = "expensive"


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

# CrewAI model strings per tier and provider
TIER_MODELS: dict[str, dict[ModelTier, str]] = {
    "anthropic": {
        ModelTier.CHEAP: "anthropic/claude-haiku-4-5-20251001",
        ModelTier.MID: "anthropic/claude-sonnet-4-20250514",
        ModelTier.EXPENSIVE: "anthropic/claude-sonnet-4-20250514",
    },
    "openai": {
        ModelTier.CHEAP: "openai/gpt-4o-mini",
        ModelTier.MID: "openai/gpt-4o",
        ModelTier.EXPENSIVE: "openai/gpt-4.1",
    },
    "gemini": {
        ModelTier.CHEAP: "google/gemini-2.5-flash",
        ModelTier.MID: "google/gemini-2.5-pro",
        ModelTier.EXPENSIVE: "google/gemini-2.5-pro",
    },
    "ollama": {
        ModelTier.CHEAP: None,   # same model — no tiers for local
        ModelTier.MID: None,
        ModelTier.EXPENSIVE: None,
    },
}


def get_model_for_agent(
    agent_name: str,
    base_model: str = "anthropic/claude-sonnet-4-20250514",
    tiered: bool = False,
) -> str:
    if not tiered:
        return base_model
    tier = AGENT_DEFAULT_TIERS.get(agent_name, ModelTier.MID)
    base_lower = base_model.lower()
    if "ollama" in base_lower:
        return base_model  # Local: same model for all tiers
    if "anthropic" in base_lower or "claude" in base_lower:
        provider = "anthropic"
    elif "openai" in base_lower or "gpt" in base_lower:
        provider = "openai"
    elif "google" in base_lower or "gemini" in base_lower:
        provider = "gemini"
    else:
        return base_model
    return TIER_MODELS.get(provider, {}).get(tier, base_model)


@dataclass
class PipelineConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    workspace_dir: Path = field(default_factory=lambda: Path(__file__).parent / "workspace")
    generated_dir: Path = field(default_factory=lambda: Path(__file__).parent / "generated")

    @property
    def template_root(self) -> Path:
        return self.base_dir / "base-code"

    @property
    def papers_and_code(self) -> Path:
        return self.base_dir / "PapersAndCode"

    @property
    def default_dataset(self) -> str:
        return str(Path(__file__).parent / "nerf_synthetic" / "chair")

    @property
    def dataparser(self) -> str:
        """Return the ns-train dataparser subcommand for the configured dataset."""
        if self.dataset == "blender":
            return "blender-data"
        elif self.dataset == "llff":
            return "colmap"
        else:  # mipnerf360
            return "colmap"

    # Pipeline toggles
    enable_review: bool = True
    enable_validation: bool = True
    enable_smoke_test: bool = True
    enable_training: bool = False
    max_debug_iterations: int = 3

    # Optimization
    fast_mode: bool = False
    tiered_models: bool = False
    include_examples: bool = True  # Include in-context examples in coder backstory

    # Nerfstudio
    conda_env: str = "nerfstudio"
    default_max_iters: int = 3000
    viewer_port: int = 7007  # Fixed port for ns-train viser viewer

    # Training & PSNR
    enable_psnr_feedback: bool = True
    max_psnr_fix_iterations: int = 2
    psnr_margin: float = 2.0
    dataset: str = "blender"
    scenes: list = field(default_factory=lambda: ["chair"])
    training_gpu: int = 0
    expected_psnr: float | None = None

    # Model configuration
    default_model: str = field(
        default_factory=lambda: os.environ.get(
            "NERFIFY_DEFAULT_MODEL", "anthropic/claude-sonnet-4-20250514"
        )
    )
    coder_model: str = field(
        default_factory=lambda: os.environ.get(
            "NERFIFY_CODER_MODEL", "anthropic/claude-sonnet-4-20250514"
        )
    )
    cheap_model: str = field(
        default_factory=lambda: os.environ.get(
            "NERFIFY_CHEAP_MODEL", "anthropic/claude-haiku-4-5-20251001"
        )
    )

    # Track whether coder/cheap were explicitly set by the user
    _coder_model_explicit: bool = field(default=False, repr=False)
    _cheap_model_explicit: bool = field(default=False, repr=False)

    def sync_models_to_provider(self):
        """Derive coder_model and cheap_model from default_model's provider
        when they weren't explicitly set. Prevents requiring an Anthropic key
        when the user selected an OpenAI or Google model."""
        base = self.default_model.lower()
        if "ollama" in base:
            # Local models: use same model for everything, no tiers
            if not self._coder_model_explicit:
                self.coder_model = self.default_model
            if not self._cheap_model_explicit:
                self.cheap_model = self.default_model
            return
        if "anthropic" in base or "claude" in base:
            provider = "anthropic"
        elif "openai" in base or "gpt" in base or base.startswith("o"):
            provider = "openai"
        elif "google" in base or "gemini" in base:
            provider = "gemini"
        else:
            return
        tiers = TIER_MODELS.get(provider)
        if not tiers:
            return
        if not self._coder_model_explicit:
            self.coder_model = tiers[ModelTier.EXPENSIVE]
        if not self._cheap_model_explicit:
            self.cheap_model = tiers[ModelTier.CHEAP]

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

    @property
    def read_tb_script(self) -> Path:
        return self.base_dir / "read_tb.py"

    def get_psnr_target(self, scene: str | None = None) -> float:
        if self.expected_psnr is not None:
            return self.expected_psnr
        baselines = DATASET_PSNR_BASELINES.get(self.dataset, {})
        s = scene or (self.scenes[0] if self.scenes else "_default")
        baseline = baselines.get(s, baselines.get("_default", 25.0))
        return round(baseline - self.psnr_margin, 1)

    def ensure_dirs(self):
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
