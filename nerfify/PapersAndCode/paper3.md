# Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields

## Goal and Scope

* Objective: combine mip-NeRF 360’s scale-aware rendering with iNGP’s grid speed to prevent spatial aliasing (jaggies/missing content) and $z$-aliasing along rays.
* Keep: multisample prefiltering of iNGP features, scale-aware downweighting, anti-aliased interlevel loss, spatial contraction, normalized distance transform, training/sample schedules, weight decay on grid features.

## Spatial Anti-Aliasing with Prefiltered iNGP Features

* For each conical frustum interval $[t_0,t_1)$ with pixel cone radius $\dot r t$:

  * Construct 6-point hexagonal multisamples with angles
    $$
    \pmb\theta=\left[0,\tfrac{2\pi}{3},\tfrac{4\pi}{3},\pi,\tfrac{5\pi}{3},\tfrac{\pi}{3}\right].
    $$
  * Depths $t_j$ concentrate mass near the far end:
    $$
    t_j=t_0+\frac{t_\delta!\left(t_1^2+2t_\mu^2+\frac{3}{\sqrt7}!\left(\frac{2j}{5}-1\right)!\sqrt{(t_\delta^2-t_\mu^2)^2+4t_\mu^4}\right)}{t_\delta^2+3t_\mu^2}.
    $$
  * Local coordinates (before rotation to world frame):
    $$
    \left[\tfrac{\dot r,t_j\cos\theta_j}{\sqrt2},\ \tfrac{\dot r,t_j\sin\theta_j}{\sqrt2},\ t_j\right].
    $$
  * Each point is an isotropic Gaussian with $\sigma_j=\dot r,t_j/\sqrt2$ scaled by 0.5 (fixed hyperparameter).
  * Training: random rotate/flip patterns; inference: deterministic $30^\circ$ alternation.
* Downweighting per iNGP level $\ell$ (grid size $n_\ell$) for each multisample:
  $$
  \omega_{j,\ell}=\operatorname{erf}!\Big(\tfrac{1}{\sqrt{8},\sigma_j,n_\ell}\Big),
  \qquad
  \mathrm{erf}(x)\approx \operatorname{sign}(x)\sqrt{1-e^{-(4/\pi)x^2}}.
  $$
* Prefiltered feature at level $\ell$:
  $$
  \mathbf f_\ell=\mathrm{mean}*j!\left(\omega*{j,\ell}\cdot \mathrm{trilerp}(n_\ell\mathbf x_j;\ V_\ell)\right),
  $$
  and concatenate a featurized mean of ${\omega_{j,\ell}}$ with ${\mathbf f_\ell}$ as MLP input.
* Spatial contraction for unbounded scenes:
  $$
  \mathcal C(\mathbf x)=
  \begin{cases}
  \mathbf x,&|\mathbf x|\le 1\
  \left(2-\tfrac{1}{|\mathbf x|}\right)\tfrac{\mathbf x}{|\mathbf x|},&|\mathbf x|>1
  \end{cases}
  $$
  Apply to means and scale the (isotropic) $\sigma_j$ via Jacobian determinant:
  $$
  \sigma'*j=\sigma_j\ \big|\det(J*{\mathcal C}(\mathbf x_j))\big|^{1/3}.
  $$
* Normalized weight decay on grids/hashes to encourage zero-mean features:
  $$
  \sum_\ell \mathrm{mean}!\big(V_\ell^2\big)\quad (\text{multiplier }0.1).
  $$

## Anti-Aliased Interlevel (Proposal) Loss to Prevent $z$-Aliasing

* Given NeRF histogram $(\mathbf s,\mathbf w)$ and proposal histogram $(\hat{\mathbf s},\hat{\mathbf w})$:

  1. Convert $\mathbf w$ to a piecewise-constant PDF by dividing by interval sizes in $\mathbf s$ (integrates to $\le 1$).
  2. Convolve with a rectangular pulse of radius $r$ to obtain a piecewise-linear PDF.
  3. Integrate to a piecewise-quadratic CDF and sample at $\hat{\mathbf s}$.
  4. Take adjacent differences to get resampled weights $\mathbf w^{\hat{\mathbf s}}$.
  5. Proposal supervision (element-wise, smooth w.r.t. translation):
     $$
     \mathcal L_{\mathrm{prop}}(\mathbf s,\mathbf w,\hat{\mathbf s},\hat{\mathbf w})
     =\sum_i \frac{1}{\hat w_i},\max!\big(0,\ \mathcal A(\mathrm{sg}(w_i^{\hat{\mathbf s}}))-\hat w_i\big)^2.
     $$
* Normalized distance along the ray via power transform $\mathcal P$:
  $$
  \mathcal P(x,\lambda)=\frac{|\lambda-1|}{\lambda}!\left[\left(\frac{x}{|\lambda-1|}+1\right)^\lambda-1\right],
  \qquad
  g(x)=\mathcal P(2x,-1.5).
  $$

## Model/Training Configuration (Reference)

* Codebase: JAX; architecture follows mip-NeRF 360 with iNGP pyramid (voxel grids + hashes) replacing large MLP.
* Sampling: 2 proposal rounds × 64 samples, then 32 NeRF samples.
* Anti-aliased interlevel loss on both proposal rounds:

  * Pulse widths: $r=0.03$ (round 1), $r=0.003$ (round 2).
  * Loss multiplier: $0.01$.
* Use separate proposal iNGPs/MLPs per round; larger view-dependent MLP than iNGP baseline.
* Rendering uses metric $t$; proposal supervision/resampling uses normalized $s=g(t)$.

## Implementation Checklist

* Hexagonal 6-point multisampling per frustum; $\sigma_j=\dot r,t_j/\sqrt2\times 0.5$.
* Contraction $\mathcal C$ on means and scales via $|\det J_{\mathcal C}|^{1/3}$.
* Prefiltered iNGP features with per-level downweighting $\omega_{j,\ell}$; concatenate scale features.
* Normalized grid weight decay (mult.=0.1).
* Two-stage proposal sampling with anti-aliased $\mathcal L_{\mathrm{prop}}$; $r={0.03,0.003}$, loss mult.=0.01.
* Distance normalization $g(x)=\mathcal P(2x,-1.5)$.
* Final NeRF sampling count: 32.
