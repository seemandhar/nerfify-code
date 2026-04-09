# NeRFPlayer: A Streamable Dynamic Scene Representation with Decomposed Neural Radiance Fields

## Problem Setup and Assumptions

* Goal: free-viewpoint rendering of dynamic 4D scenes from RGB images (single moving camera or multi-camera).
* Decompose spatiotemporal space into three mutually exclusive areas with probabilities per point $(\pmb p,t)$:

  * **Static**: time-invariant geometry and low-frequency appearance.
  * **Deforming**: objects present across the sequence with (rigid/non-rigid) motion.
  * **New**: content that emerges over time.
* Expected point feature: $\begin{array}{r}{\pmb v=\sum_{*}P_{*},\pmb v_{*}}\end{array}$ where $*\in{\text{static},\text{deform},\text{new}}$; a lightweight, view-conditioned radiance decoder predicts $(\sigma,\pmb c)$ from $\pmb v$.

## Rendering Preliminaries and Notation

* Camera ray color:
  $$
  C ( \pmb { r } ) = \int_{ i_n }^{ i_f } e^{ - \int_{ i_n }^{ i } \sigma ( \pmb { p } _ { j } ) d j } , \sigma !\big( \pmb { p } _ { i } \big) , c !\big( \pmb { p } _ { i } , d \big) , d i ,
  $$
* Reconstruction loss:
  $$
  L _ { \mathrm { r e c } } = \sum _ { \pmb { r } \in \mathcal { R } } | \pmb { C } ( \pmb { r } ) - \pmb { C } _ { \mathrm { g t } } ( \pmb { r } ) | _ { 2 } ^ { 2 } ,
  $$

## Decomposed Fields and Decoders

* **Decomposition field** $f:(\pmb p,t)\mapsto(P_{\text{static}},P_{\text{deform}},P_{\text{new}})$

  * Implemented as streamed explicit features $V_f$ + small MLP $D_f$.
  * Self-supervised via parsimony regularization (no manual labels).
* **Stationary field** $s$

  * Explicit feature volume $V_s$ (spatial) + tiny MLP (inputs: $t$ and $V_s(\pmb p)$) to capture low-frequency illumination changes.
* **Deformation field** $d$

  * MLP mapping $(\pmb p,t)\mapsto\Delta\pmb p$; query $(\pmb p+\Delta\pmb p)$ in canonical/static space.
* **Newness field** $n$

  * Explicit spatiotemporal features $V_n(\pmb p,t)$.

## Streamable Hybrid Representation (Backbone-Agnostic)

* Use hybrid backbones with explicit features (e.g., voxel grids, hash tables, tensor bases). Treat feature **channels** as time-dependent and **stream** them with a sliding window:

  * If each per-frame feature has dimension $F$ and $k$ new channels are introduced per new frame, then an entry stores $F+k(T-1)$ channels for $T$ frames; at time $t$, use channels $[kt,,kt+F]$ (with rearrangement to keep shared channels aligned).
  * Temporal interpolation between frames $t_s$ and $t_{s+1}$ via feature blending:
    $v _ { p , t } = \begin{array}{r}{ \frac{t-t_s}{t_{s+1}-t_s}, \pmb v_{p,t_{s+1}} + \frac{t_{s+1}-t}{t_{s+1}-t_s},\pmb v_{p,t_s} }\end{array}$
* Benefits: reduced model size (shared channels), streaming-friendly (only load new channels when advancing $t$).

## Training Objective and Regularization

* Batch of rays $\mathcal R$; sampled points $\mathcal R_p$.
* Average probabilities $\overline{P_*}\doteq \frac{1}{|\mathcal R_p|}\sum_{p\in\mathcal R_p}P_*(p)$.
* Parsimony loss encourages static modeling, penalizing deform/new:
  $$
  {\cal L}*{\mathrm{reg}}=\alpha ,\overline{P*{\mathrm{deform}}}+\overline{P_{\mathrm{new}}} ,
  $$
* Total loss:
  $$
  {\cal L}={\cal L}*{\mathrm{rec}}+\lambda,{\cal L}*{\mathrm{reg}} .
  $$

## Rendering Efficiency

* Skip low-probability branches: with threshold $\tau$, if $P_*<\tau$ set $\pmb v_*= \pmb 0$ and **do not** forward that field; default $\tau=0.001$.

## Implementation Details (Reference Configuration)

* Framework: PyTorch; backbones: **InstantNGP** [52] and **TensoRF** [11].
* Networks:

  * Deformation MLP: 4 layers, width 256.
  * Stationary MLP (appearance head): 2 layers, width 64.
  * Radiance decoder $r$: 4 layers, width 64 (mirrors backbone decoder).
* Backbone hyperparameters:

  * InstantNGP: 8 levels; 4 features/entry.
  * TensoRF: same as authors’ real forward-facing (LLFF [50]) settings.
* Streaming/channel parameters: $k=1$ (new channels per next frame).
* Loss weights: $\lambda=0.1$, $\alpha=0.01$.
* Practical notes:

  * Multi-camera datasets with high FPS: deformation often omitted by default (newness suffices for smooth interpolation).
  * Long sequences are split into 90-frame clips for memory.
  * Metrics: PSNR, SSIM [83].

## Datasets and Splits (for Reproduction)

* **Immersive Video** [6] (multi-camera; 46 synchronized 4K fisheyes). Use 7 scenes with similar imaging; downsample to $1280\times960$. Camera ID 0 for validation; others for training.
* **Plenoptic Video** [36] (multi-camera; indoor). Downsample to $1352\times1014$; follow official train/val split; 6 public scenes.
* **HyperNeRF** [58,59] (single-camera). Quantitative: $960\times540$; qualitative: $1920\times1080$.

  * Settings: “vrig” (stereo; train on one camera, validate on the other) and “interp” (monocular moving camera).

## Training & Evaluation Protocol (Concise)

* **Inputs**: RGB frames + known camera poses.
* **Backbone**: choose InstantNGP (speed) or TensoRF (quality/compactness); enable streaming channels.
* **Optimization**:

  * Use backbone’s default schedule; train per 90-frame clip.
  * Start with larger $\alpha$ to suppress excessive deformation; gradually reduce to allow necessary deforms.
* **Ablations that affect implementation**:

  * **Decomposition necessity**: removing “new” blurs emergent content; removing “deform” blurs moving objects; removing “static” induces flicker in backgrounds (especially at novel times/views).
  * **Low FPS or large motion**: include deformation field to avoid disappear/reappear artifacts during interpolation.
* **Rendering**:

  * Apply $\tau=0.001$ branch-skipping at inference.
  * Adjust speed/quality via backbone controls (e.g., hash table size $T$, ray-march stepping).

## Minimal Hyperparameter Checklist

* Backbone: InstantNGP(8 levels, 4 feats/entry) **or** TensoRF(LLFF-style).
* MLPs: $d$ (4×256), $s$ head (2×64), $r$ (4×64).
* Streaming: $k=1$; clip length 90 frames.
* Loss: $\lambda=0.1$, $\alpha=0.01$; threshold $\tau=0.001$.
* Resolutions/splits as above; metrics: PSNR, SSIM.