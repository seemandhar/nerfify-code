# Nerfify — Community Implementations

This directory contains NeRFStudio-compatible implementations of NeRF research papers.
Each subdirectory is a complete method that can be installed and trained with nerfstudio.

## Available Implementations

| Method | Directory | Description |
|--------|-----------|-------------|
| AniS-NeRF | `anis_nerf/` | Anisotropic neural radiance fields |
| AR-NeRF | `ar_nerf/` | Attention-based radiance fields |
| ENeRF / ERS | `ers/` | Efficient neural radiance fields with spatial sampling |
| Hybrid-NeRF | `hyb_nerf/` | Hybrid neural radiance fields |
| KeyNeRF | `key_nerf/` | Keyframe-based neural radiance fields |
| LI-NeRF | `li_nerf/` | Light-invariant neural radiance fields |
| MI-NeRF | `mi_nerf/` | Multi-illumination neural radiance fields |
| NeRF-ID | `nerf_id/` | NeRF with identity conditioning |
| Surface Sampling | `surface_sampling/` | Surface-aware sampling for NeRF |
| TV-NeRF | `tvnerf/` | Total variation regularized NeRF |

## Installation

Each implementation follows the standard nerfstudio method pattern:

```bash
cd implementations/<method_name>
pip install -e .
ns-install-cli
ns-train <method-name> --data <path-to-data>
```

## Contributing a New Implementation

We welcome community contributions! To add a new paper implementation:

1. **Fork** this repo and create a branch: `git checkout -b impl/<method-name>`
2. **Create** a directory under `implementations/` named after your method (snake_case)
3. **Include** at minimum these files:

```
implementations/<method_name>/
  method_template/
    __init__.py
    template_config.py
    template_datamanager.py
    template_field.py
    template_model.py
    template_pipeline.py
  pyproject.toml
  README.md
```

4. **Follow** the base template in `base-code/` for API compatibility
5. **Test** that `pip install -e . && ns-train <method> --data <path> --max-num-iterations 10` works
6. **Open a PR** with:
   - Link to the original paper
   - Brief description of what's implemented
   - Test results (even a 10-iteration smoke test is fine)

See `CONTRIBUTING.md` in the repo root for full guidelines.
