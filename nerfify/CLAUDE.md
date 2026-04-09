# Nerfify — Claude Multi-Agent Pipeline for NeRF Paper Reproduction

Nerfify converts NeRF research papers into complete, trainable NeRFStudio method
implementations using specialized Claude subagents orchestrated via Claude Agent SDK.

## Architecture

The orchestrator coordinates subagents with speed optimizations:

1. **parser** — Extracts PDF to markdown using mineru, cleans for implementation
2. **citation_recovery** — Identifies cited papers, fetches implementation details (skippable in fast mode)
3. **planner** — Creates architecture DAG and file generation plan
4. **coder** — Generates complete NeRFStudio code with mandatory 6-point self-check
5. **Quality check** — Orchestrator does inline review (replaces separate reviewer/validator/integrator agents)
6. **tester** — Runs smoke tests in nerfstudio conda environment
7. **debugger** — Diagnoses and fixes errors
8. **Training + PSNR feedback** — Optional train→eval→fix loop

The reviewer, validator, and integrator agents still exist but are bypassed by default
— the orchestrator performs their checks inline for speed.

## Pipeline Flow

```
parser → [citation_recovery] → planner → coder → inline quality check → tester (debug loop) → [train → PSNR check → fix loop]
```

## Speed Optimizations

- File contents (cleaned paper, DAG plan, citation details) are embedded directly in agent prompts — agents don't re-read files
- Reviewer/validator/integrator merged into a single inline orchestrator check
- Coder has a mandatory self-check before finishing (imports, config wiring, forward pass, losses, pyproject.toml)
- Max 1 fix pass after quality check

## PSNR Feedback Loop

When `--train` is passed, after smoke tests pass the pipeline:
1. Installs the method via `pip install -e` + `ns-install-cli`
2. Trains for `--max-iters` iterations (default 3000) with TensorBoard
3. Reads PSNR curve via `read_tb.py` (uses TensorBoard EventAccumulator)
4. Compares final PSNR against target (dataset baseline - 2.0 dB margin)
5. If below target: analyzes issues, calls coder to fix, re-installs, re-trains
6. Up to `max_psnr_fix_iterations` (default 2) fix cycles

PSNR baselines are defined in `config.py` for mipnerf360, blender, and llff datasets.

### Supporting Scripts

- `read_tb.py` — Extracts PSNR from TensorBoard event files, detects curve issues (still rising, dropped from peak, too low, NaN). CLI: `python read_tb.py <logdir> [--json]`
- `eval.py` — Full train+eval harness with multi-GPU support. CLI: `python eval.py --method <name> --scenes garden [--gpu 0,1] [--dataset mipnerf360]`

## Model Tiers

`config.py` defines tiered model assignments per agent (cheap/mid/expensive → Haiku/Sonnet/Opus).
Enable with `--tiered-models`. Supports Anthropic, OpenAI, and Gemini provider families.

## Key Directories

- `base-code/` — NeRFStudio method template (field, model, pipeline, datamanager, config)
- `PapersAndCode/` — In-context examples (paper1-5.md, code1-5.py, VanillaNerfOriginal.py)
- `agents/` — Agent definitions and system prompts
- `workspace/` — Per-run working files (cleaned paper, citation details, plans, results)
- `generated/` — Per-run generated NeRFStudio implementations
- `templates/` — Web UI HTML templates

## Required Output Files

Every generated implementation must contain exactly:
- `method_template/__init__.py`
- `method_template/template_config.py`
- `method_template/template_datamanager.py`
- `method_template/template_field.py`
- `method_template/template_model.py`
- `method_template/template_pipeline.py`
- `README.md`
- `pyproject.toml`

## Nerfstudio Environment

All testing/training commands must run in the `nerfstudio` conda environment:
```bash
eval "$(conda shell.bash hook)"; conda activate nerfstudio; <command>
```

## Usage

```bash
# CLI — basic
python main.py --arxiv 2308.12345
python main.py --pdf /path/to/paper.pdf --method-name my_nerf

# CLI — with training and PSNR feedback
python main.py --arxiv 2308.12345 --train --gpu 0 --scenes garden --dataset mipnerf360
python main.py --pdf paper.pdf --train --expected-psnr 27.0 --no-psnr-feedback

# CLI — skip review (already inline), fast mode
python main.py --arxiv 2308.12345 --no-review --fast

# Standalone eval
python eval.py --method hybrid-nerf --scenes garden bicycle --gpu 0,1
python eval.py --eval-only --method hybrid-nerf --scenes garden

# Web UI
python web.py  # http://localhost:5005
```

## Web UI Features

- Real-time agent graph with active node highlighting
- Live token usage display (input/output/total)
- Live bash/terminal output streaming with terminal-like styling
- SSE-based event streaming (agent_active, agent_done, bash, bash_output, task_progress, etc.)
