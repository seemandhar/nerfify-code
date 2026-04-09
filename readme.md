# NERFIFY: A Multi-Agent Framework for Turning NeRF Papers into Code

### [Project Page](https://jainsee24.github.io/NERFIFY/) | [arXiv](https://arxiv.org/abs/2603.00805)

> Converting research papers into trainable Nerfstudio plugins, from weeks to minutes.

[Seemandhar Jain](https://seemandhar.github.io/), [Keshav Gupta](#), [Kunal Gupta](https://kunalgupta.me/), [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)

University of California, San Diego

**CVPR 2026**

## Overview

NERFIFY is a multi-agent framework that automatically converts NeRF research papers into complete, trainable [NeRFStudio](https://docs.nerf.studio/) implementations. Given an arXiv ID or PDF, the pipeline parses the paper, plans the implementation, generates code, reviews it, tests it, and debugs any failures — all autonomously.

This repository contains **two implementations** of the same pipeline built on different agentic frameworks:

| | [`nerfify/`](nerfify/) | [`nerfify-crew/`](nerfify-crew/) |
|---|---|---|
| **Framework** | [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk) | [CrewAI](https://github.com/crewaiinc/crewai) |
| **Multi-model support** | Via LiteLLM proxy | Native (LiteLLM built-in) |
| **Web search** | Claude WebSearch tool | DuckDuckGo + httpx |
| **Orchestration** | Single orchestrator agent | CrewAI sequential process |
| **Agent communication** | Via Agent tool | CrewAI context passing |
| **Streaming** | SSE + terminal | CrewAI verbose + event bus |

Both versions implement the same 7-agent pipeline and produce identical outputs. Choose whichever framework fits your setup.

## Architecture

The pipeline uses **7 specialized AI agents** orchestrated sequentially:

```
Paper (PDF / arXiv ID / URL)
  │
  ├─ [1] Parser ──────────── Download PDF → Mineru extraction → clean markdown
  │                           (removes narrative, keeps equations/architecture/losses)
  │
  ├─ [2] Citation Recovery ── Fetch missing implementation details from cited papers
  │                           (optional, skipped with --fast if paper is self-contained)
  │
  ├─ [3] Planner ──────────── Design dependency DAG + file generation order
  │                           (chooses Nerfacto vs Vanilla NeRF base architecture)
  │
  ├─ [4] Coder ────────────── Generate complete NeRFStudio method (8 files)
  │                           (uses templates + in-context examples + self-check)
  │
  ├─ [5] Reviewer ─────────── Code review for imports, API mismatches, consistency
  │                           (optional, skip with --no-review)
  │
  ├─ [6] Tester ───────────── Smoke test: pip install → import → 10-iter training
  │
  └─ [7] Debugger ─────────── Diagnose & fix errors (up to 3 iterations)
                               + PSNR debugging if --train enabled
```

### Agent Tools

| Agent | Tools |
|-------|-------|
| Parser | web_search, web_fetch, file_read, file_write, shell, file_glob, clean_paper |
| Citation Recovery | web_search, web_fetch, file_read, file_write |
| Planner | web_search, web_fetch, file_read, file_write |
| Coder | web_search, web_fetch, file_read, file_write, file_glob, shell |
| Reviewer | web_search, web_fetch, file_read, file_write, file_glob |
| Tester | file_read, shell |
| Debugger | web_search, web_fetch, file_read, file_write, file_glob, shell |

## Prerequisites

- Python >= 3.10
- A [NeRFStudio](https://docs.nerf.studio/quickstart/installation.html) installation (with CUDA-enabled PyTorch)
- At least one LLM API key (Anthropic, OpenAI, or Google)
- [Mineru](https://github.com/opendatalab/MinerU) for PDF extraction

## Setup & Running

### Option 1: CrewAI Version (`nerfify-crew/`)

Open-source, multi-model out of the box.

```bash
git clone https://github.com/seemandhar/NERFIFY-code.git
cd NERFIFY-code/nerfify-crew

# Install
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your key(s)

# Run from arXiv
python main.py --arxiv 2308.12345

# Run with options
python main.py --arxiv 2308.12345 --fast --no-review
python main.py --pdf /path/to/paper.pdf --method-name my_nerf
python main.py --arxiv 2308.12345 --train --gpu 0 --dataset blender
python main.py --arxiv 2308.12345 --model openai/gpt-4o --coder-model openai/gpt-4.1
python main.py --arxiv 2308.12345 --tiered

# Web UI
python web.py  # http://127.0.0.1:5000
```

**Dependencies** (from `pyproject.toml`):
- `crewai[tools]>=0.118.0`
- `langchain-anthropic`, `langchain-openai`, `langchain-google-genai`
- `ddgs>=8.0`, `mineru`, `flask>=3.0`, `httpx>=0.25`

### Option 2: Claude Agent SDK Version (`nerfify/`)

Uses the Claude Agent SDK with a single orchestrator agent.

```bash
cd NERFIFY-code/nerfify

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run from arXiv
python main.py --arxiv 2308.12345

# Run with options
python main.py --arxiv 2308.12345 --fast --no-review
python main.py --pdf /path/to/paper.pdf --method-name my_nerf
python main.py --arxiv 2308.12345 --train --gpu 0 --dataset blender

# Web UI
python web.py  # http://127.0.0.1:5000

# API server
python main_api.py
```

### Environment Variables

```bash
# Required: at least one LLM API key
ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=...

# Optional: model overrides
# NERFIFY_CODER_MODEL=anthropic/claude-opus-4-6
# NERFIFY_DEFAULT_MODEL=anthropic/claude-sonnet-4-20250514
# NERFIFY_CHEAP_MODEL=anthropic/claude-haiku-4-5-20251001

# Optional: nerfstudio conda environment name
# NERFSTUDIO_CONDA_ENV=nerfstudio
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--arxiv ID` | arXiv paper ID | — |
| `--pdf PATH` | Local PDF file path | — |
| `--url URL` | PDF download URL | — |
| `--method-name` | Override generated method name | Auto-generated |
| `--model` | Default LLM for all agents | `anthropic/claude-sonnet-4-20250514` |
| `--coder-model` | LLM for code generation (most expensive) | Same as `--model` |
| `--no-review` | Skip the code review stage | `False` |
| `--no-test` | Skip the smoke testing stage | `False` |
| `--fast` | Skip citation recovery if paper is self-contained | `False` |
| `--train` | Run full training + PSNR feedback | `False` |
| `--gpu` | GPU device ID for training | `0` |
| `--data` | Path to training dataset | Built-in synthetic |
| `--dataset` | Dataset type: `blender`, `mipnerf360`, `llff` | `blender` |
| `--tiered` | Use model tiering per agent (CrewAI only) | `False` |

## Model Support

Both versions support multiple LLM providers. The CrewAI version supports any LiteLLM-compatible model natively; the Claude SDK version supports multi-model via a LiteLLM proxy.

| Provider | Model String | Required Env Var |
|----------|-------------|-----------------|
| Anthropic | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Google | `google/gemini-2.5-pro` | `GOOGLE_API_KEY` |

### Tiered Model Routing (CrewAI)

With `--tiered`, agents are assigned different model tiers based on task complexity:

| Tier | Anthropic | OpenAI | Gemini | Assigned To |
|------|-----------|--------|--------|-------------|
| CHEAP | claude-haiku-4-5 | gpt-4o-mini | gemini-2.5-flash | Parser, Tester |
| MID | claude-sonnet-4 | gpt-4o | gemini-2.5-pro | Planner, Reviewer, Citation Recovery |
| EXPENSIVE | claude-sonnet-4 | gpt-4.1 | gemini-2.5-pro | Coder, Debugger |

## Project Structure

```
NERFIFY-code/
├── readme.md
│
├── nerfify/                 # Claude Agent SDK version
│   ├── main.py              # CLI entry point
│   ├── main_api.py          # API server
│   ├── config.py            # Pipeline configuration
│   ├── eval.py              # Evaluation utilities
│   ├── litellm_proxy.py     # LiteLLM proxy for multi-model
│   ├── read_tb.py           # TensorBoard PSNR extraction
│   ├── web.py               # Flask web UI
│   ├── requirements.txt     # Dependencies
│   ├── agents/
│   │   └── definitions.py   # Agent definitions
│   ├── base-code/           # NeRFStudio method templates
│   ├── PapersAndCode/       # In-context examples (paper/code pairs)
│   ├── implementations/     # Example generated implementations
│   ├── templates/           # Web UI frontend
│   └── demo-images/         # Demo screenshots
│
├── nerfify-crew/            # CrewAI version
│   ├── main.py              # CLI entry point
│   ├── crew.py              # CrewAI Crew assembly + execution
│   ├── config.py            # Pipeline configuration + model tiers
│   ├── tasks.py             # 7 task definitions with dependencies
│   ├── read_tb.py           # TensorBoard PSNR extraction
│   ├── web.py               # Flask web UI with SSE streaming
│   ├── pyproject.toml       # Package definition + dependencies
│   ├── .env.example         # API key template
│   ├── agents/
│   │   ├── definitions.py   # Build 7 CrewAI Agent objects
│   │   └── prompts.py       # Agent role/goal/backstory prompts
│   ├── tools/
│   │   ├── web_search.py    # DuckDuckGo search + httpx fetch
│   │   ├── file_ops.py      # File read/write/glob
│   │   ├── shell.py         # Shell command execution
│   │   └── clean_paper.py   # LLM-based paper cleaning
│   ├── base-code/           # NeRFStudio method templates
│   ├── PapersAndCode/       # In-context examples (paper/code pairs)
│   └── templates/           # Web UI frontend
│
├── workspace/               # Per-run intermediate files (auto-generated)
└── generated/               # Per-run generated implementations (auto-generated)
```

## Output

Each run produces a complete NeRFStudio method:

```
generated/<run_id>/
├── method_template/
│   ├── __init__.py              # Exports method_spec entry point
│   ├── template_config.py       # MethodSpecification + hyperparameters
│   ├── template_model.py        # Paper-specific model + loss functions
│   ├── template_field.py        # Custom field architecture (encodings, MLPs)
│   ├── template_datamanager.py  # Data pipeline configuration
│   └── template_pipeline.py     # DDP-aware model/data wiring
├── pyproject.toml               # Dependencies + NeRFStudio entry point
├── README.md                    # Generated method docs + usage
└── training_outputs/            # Training results (if --train enabled)
```

### Using the Generated Method

```bash
cd generated/<run_id>

# Install the generated method
pip install -e .

# Register with NeRFStudio
ns-install-cli

# Train
ns-train <method-name> --data /path/to/dataset
```

## End-to-End Pipeline Flow

1. **User provides** a research paper (arXiv ID, PDF, or URL)
2. **Parser** downloads and extracts the PDF using Mineru, then cleans it into implementation-focused markdown (equations, architecture, losses, hyperparameters)
3. **Citation Recovery** (optional) fetches missing implementation details from cited papers
4. **Planner** designs the NeRFStudio architecture with a dependency DAG and file generation order
5. **Coder** generates all implementation files using templates and in-context examples, with a mandatory self-check
6. **Reviewer** (optional) performs code review for imports, API mismatches, and consistency
7. **Tester** runs smoke tests: `pip install` → import check → 10-iteration training
8. **Debugger** fixes any errors (up to 3 iterations), with optional PSNR debugging
9. **Output**: Complete, installable NeRFStudio package ready to use with `ns-train`

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{jain2026nerfify,
  title     = {NERFIFY: A Multi-Agent Framework for Turning NeRF Papers into Code},
  author    = {Jain, Seemandhar and Gupta, Keshav and Gupta, Kunal and Chandraker, Manmohan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
