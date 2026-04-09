# Nerfify Template

A comprehensive nerfstudio method template derived from 6+ tested NeRF implementations (hyb_nerf, tvnerf, ar_nerf, li_nerf, mi_nerf, anis_nerf) with proven hyperparameters.

## Quick Start

```bash
# Install as editable package
cd final_nerfify_template
pip install -e .

# Train on your data
ns-train my-method --data /path/to/your/data
```

## Architecture

```
method_template/
├── __init__.py              # Package exports
├── template_config.py       # MethodSpecification + TrainerConfig (entry point)
├── template_pipeline.py     # Pipeline: wires datamanager + model, handles DDP
├── template_datamanager.py  # DataManager: ray sampling, batch augmentation
├── template_model.py        # Model: field setup, loss computation, rendering
└── template_field.py        # Field: position/direction encoding, density/color MLPs
```

## Two Backbone Strategies

### 1. Nerfacto (Default)
Modern proposal-network-based sampling with hash encoding. Best for most new methods.
- Used by: hyb_nerf, li_nerf, anis_nerf
- Parent class: `NerfactoModel` / `NerfactoModelConfig`

### 2. VanillaNeRF (Alternative)
Classic coarse/fine hierarchical sampling. Use when your paper builds on vanilla NeRF.
- Used by: mi_nerf, tvnerf, ar_nerf
- Parent class: `NeRFModel` / `VanillaModelConfig`
- See commented alternative in `template_model.py`

## How to Adapt for Your Paper

1. **Rename**: Replace `my-method` in `pyproject.toml` and `template_config.py`
2. **Config**: Add paper-specific hyperparameters to `TemplateModelConfig`
3. **Field**: Modify encoding/MLP architecture in `template_field.py`
4. **Model**: Add custom losses in `TemplateModel.get_loss_dict()`
5. **DataManager**: Add batch augmentation in `TemplateDataManager.next_train()` (if needed)

## Proven Hyperparameters

These defaults are tested across multiple implementations:

| Parameter | Value | Source |
|-----------|-------|--------|
| train_num_rays_per_batch | 4096 | All methods |
| eval_num_rays_per_batch | 4096 | All methods |
| hash_num_levels | 16 | hyb_nerf, li_nerf |
| hash_log2_hashmap_size | 19 | hyb_nerf, li_nerf |
| geo_feat_dim | 15 | hyb_nerf, li_nerf |
| hidden_dim (density/color) | 64 | hyb_nerf, li_nerf |
| SH levels (direction) | 4 | All hash-based methods |
| Fields optimizer | RAdam lr=5e-4, eps=1e-15 | All methods |
| Proposal optimizer | Adam lr=5e-4, eps=1e-15 | All methods |
| Camera optimizer | Adam lr=1e-3, eps=1e-15 | All methods |
| Fields lr_final | 1e-4, max_steps=50000 | All methods |
| Proposal lr_final | 1e-4, max_steps=200000 | All methods |
| l2_appearance_mult | 1e-4 (when enabled) | hyb_nerf, anis_nerf |
