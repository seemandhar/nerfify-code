# Nerfify

**Automated NeRF Paper → NeRFStudio Code** using Claude Multi-Agent Pipeline.

Nerfify reads a NeRF research paper (arXiv or PDF), understands the architecture, recovers implementation details from cited papers, and generates a complete, trainable [NeRFStudio](https://docs.nerf.studio/) method implementation — all orchestrated by 9 specialized Claude AI agents.

## Pipeline

```
Paper (PDF/arXiv)
  ↓
Parser Agent ─── extracts & cleans markdown
  ↓
Citation Recovery Agent ─── fetches implementation details from cited papers
  ↓
Planner Agent ─── architecture DAG + file plan
  ↓
Coder Agent ─── generates complete NeRFStudio method (8 files)
  ↓
Reviewer Agent ─── code review loop (API compliance, imports, consistency)
  ↓
Validator Agent ─── novelty/equation gap analysis
  ↓
Integrator Agent ─── cross-file consistency check
  ↓
Tester Agent ─── smoke test (pip install + 10-iter training)
  ↓
Debugger Agent ─── fix errors if any
  ↓
Complete NeRFStudio Method (pip install -e . && ns-train method-name --data <path>)
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk)
- [MinerU](https://github.com/opendatalab/MinerU) (for PDF extraction)
- [NeRFStudio](https://docs.nerf.studio/) conda environment (for testing/training)

### Install

```bash
git clone https://github.com/yourusername/nerfify.git
cd nerfify
pip install -r requirements.txt
```

### CLI Usage

```bash
# From arXiv
python main.py --arxiv 2308.12345

# From local PDF
python main.py --pdf /path/to/paper.pdf

# With options
python main.py --arxiv 2308.12345 --method-name my_nerf --data /path/to/dataset --train

# Skip optional steps
python main.py --arxiv 2308.12345 --no-review --no-test
```

### Web UI

```bash
python web.py
# Open http://localhost:5005
```

The web UI provides a real-time dashboard with:
- Live agent activity monitoring
- Color-coded pipeline log with SSE streaming
- Agent status chips (idle/active/done/error)
- Token usage tracking
- Tool call inspection panel

## Generated Output

Each run produces a complete NeRFStudio method:

```
generated/<run_id>/
├── method_template/
│   ├── __init__.py
│   ├── template_config.py        # TrainerConfig + MethodSpecification
│   ├── template_datamanager.py   # Custom DataManager (if needed)
│   ├── template_field.py         # Neural field (NeRF implicit function)
│   ├── template_model.py         # Model + loss computation
│   └── template_pipeline.py      # Training pipeline
├── README.md
└── pyproject.toml
```

### Install & Train

```bash
cd generated/<run_id>
conda activate nerfstudio
pip install -e .
ns-install-cli
ns-train <method-name> --data /path/to/dataset
```

## Project Structure

```
nerfify/
├── main.py              # CLI orchestrator (Claude Agent SDK)
├── main_api.py          # Alternative orchestrator (Anthropic SDK)
├── web.py               # Flask web UI with SSE streaming
├── config.py            # Pipeline configuration
├── agents/
│   ├── __init__.py
│   └── definitions.py   # 9 agent definitions + system prompts
├── base-code/           # NeRFStudio method template
│   ├── method_template/ # Template Python files
│   ├── pyproject.toml
│   └── README.md
├── PapersAndCode/       # In-context examples for the coder agent
├── templates/
│   └── index.html       # Web UI
├── workspace/           # Per-run working files (gitignored)
└── generated/           # Per-run outputs (gitignored)
```

## Agents

| Agent | Role | Key Tools |
|-------|------|-----------|
| **parser** | PDF → cleaned markdown | Bash, WebSearch |
| **citation_recovery** | Fetch implementation details from cited papers | WebSearch, WebFetch |
| **planner** | Architecture DAG + file plan | Read, Write |
| **coder** | Generate complete NeRFStudio implementation | Read, Write, WebSearch |
| **reviewer** | Code review for API compliance | Read, Glob |
| **validator** | Novelty/equation gap analysis | Read, WebSearch |
| **integrator** | Cross-file consistency check | Read, Grep, Bash |
| **tester** | Smoke test in nerfstudio env | Bash |
| **debugger** | Diagnose and fix errors | Read, Write, Bash, WebSearch |

## Configuration

Edit `config.py` to customize:
- `template_root` — Path to NeRFStudio template files
- `papers_and_code` — Path to in-context examples
- `conda_env` — Conda environment name (default: `nerfstudio`)
- `enable_review` / `enable_validation` / `enable_smoke_test` — Pipeline toggles
- `max_review_iterations` / `max_debug_iterations` — Loop limits

## License

MIT
