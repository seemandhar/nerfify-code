# Contributing to Nerfify

Thanks for your interest in contributing! Nerfify aims to be the central repo for NeRF paper implementations on nerfstudio.

## Adding a Paper Implementation

### Quick Start

1. Fork and clone the repo
2. Create a branch: `git checkout -b impl/<method-name>`
3. Copy the template: `cp -r base-code/ implementations/<method_name>/`
4. Implement the paper in the template files
5. Test: `pip install -e . && ns-train <method> --data <path> --max-num-iterations 10`
6. Open a PR

### File Structure

Every implementation must follow this structure:

```
implementations/<method_name>/
  method_template/
    __init__.py           # exports method_spec
    template_config.py    # TrainerConfig + MethodSpecification
    template_datamanager.py
    template_field.py     # neural field (density + color)
    template_model.py     # model (losses, rendering)
    template_pipeline.py  # pipeline (training loop)
  pyproject.toml          # installable package with nerfstudio entry point
  README.md               # paper link, description, usage
```

### Requirements

- Must be installable: `pip install -e .`
- Must register with nerfstudio: `ns-install-cli` should detect it
- Must run: `ns-train <method-name> --data <path>` should start training
- Follow the base template APIs (extend NerfactoModel/VanillaNeRF, use nerfstudio datamanagers)
- Include a README with:
  - Paper title and link
  - What is implemented
  - How to train

### PR Template

```markdown
## Paper
- Title: ...
- arXiv: ...

## What's implemented
- [ ] Core architecture
- [ ] Loss functions
- [ ] Custom field/model
- [ ] Training protocol

## Testing
- [ ] `pip install -e .` succeeds
- [ ] `ns-train <method> --max-num-iterations 10` completes
- [ ] (optional) Full training PSNR on a standard scene
```

## Other Contributions

- **Bug fixes**: Open an issue first, then PR
- **Agent improvements**: Changes to `agents/definitions.py` prompts
- **UI improvements**: Changes to `templates/index.html`
- **New agents**: Add to `agents/definitions.py` and register in `config.py`
