# SeaThru-NeRF: Neural Radiance Fields in Scattering Media

## Problem Setup and Assumptions

* Goal: novel-view synthesis in scattering media (haze, fog, underwater) by jointly modeling opaque objects and semi-transparent medium.
* Assumptions per camera ray $\mathbf r$: single opaque object; medium is semi-transparent; medium parameters are constant along the ray (direction-dependent) and may differ per color channel.
* Outputs are decomposed into **object** and **medium (backscatter/attenuation)** components that sum to the rendered color.

## NeRF Rendering (Reference)

If $\mathbf r(t)=\mathbf o+t\mathbf d$, with near/far $t_n,t_f$:
$$
C(\mathbf r)=\int_{t_n}^{t_f}T(t),\sigma(t),\mathbf c(t),dt
$$
$$
T(t)=\exp!\left(-\int_{t_n}^{t}\sigma(s),ds\right)
$$
Discrete rendering with intervals $[s_i,s_{i+1}]$:
$$
\hat C(\mathbf r)=\sum_{i=1}^{N}C_i(\mathbf r),\quad
C_i(\mathbf r)=T(s_i)\Big(1-e^{-\sigma_i\delta_i}\Big)\mathbf c_i
$$
$$
T(s_i)=\exp!\Big(-\sum_{j=0}^{i-1}\sigma_j\delta_j\Big)
$$
Reconstruction loss on linear images:
$$
L=\sum_{\mathbf r\in R}\lVert\hat C(\mathbf r)-C(\mathbf r)\rVert^2
$$

## Rendering in Scattering Media (Object + Medium)

Continuous form:
$$
C(\mathbf r)=\int_{t_n}^{t_f}T(t)\Big(\sigma^{\mathrm{obj}}(t)\mathbf c^{\mathrm{obj}}(t)+\sigma^{\mathrm{med}}(t)\mathbf c^{\mathrm{med}}(t)\Big),dt
$$
$$
T(t)=\exp!\left(-\int_{t_n}^{t}\big(\sigma^{\mathrm{obj}}(s)+\sigma^{\mathrm{med}}(s)\big),ds\right)
$$
Discrete form:
$$
C_i(\mathbf r)=T(s_i)!\left(1-e^{-(\sigma_i^{\mathrm{obj}}+\sigma_i^{\mathrm{med}})\delta_i}\right)\frac{\sigma_i^{\mathrm{obj}}\mathbf c_i^{\mathrm{obj}}+\sigma_i^{\mathrm{med}}\mathbf c_i^{\mathrm{med}}}{\sigma_i^{\mathrm{obj}}+\sigma_i^{\mathrm{med}}}
$$
$$
T(s_i)=\exp!\Big(-\sum_{j=0}^{i-1}(\sigma_j^{\mathrm{obj}}+\sigma_j^{\mathrm{med}})\delta_j\Big)
$$
With medium parameters constant per ray and an opaque object, the contributions reduce to:
$$
\begin{array}{l}
\hat C_i^{\mathrm{obj}}(\mathbf r)=T_i\cdot\big(1-e^{-\sigma_i^{\mathrm{obj}}\delta_i}\big)\cdot\mathbf c_i^{\mathrm{obj}}[2pt]
\hat C_i^{\mathrm{med}}(\mathbf r)=T_i\cdot\big(1-e^{-\sigma^{\mathrm{med}}\delta_i}\big)\cdot\mathbf c^{\mathrm{med}}[2pt]
T_i=\exp!\Big(-\sum_{j=0}^{i-1}\sigma_j^{\mathrm{obj}}\delta_j\Big)\cdot\exp!\big(-\sigma^{\mathrm{med}}s_i\big)
\end{array}
$$
Image-formation equivalence (opaque object at depth $z$):
$$
\hat C(\mathbf r)\approx e^{-\mathbf{\sigma}^{\mathrm{med}} z}\cdot\mathbf c_k^{\mathrm{obj}}+\big(1-e^{-\mathbf{\sigma}^{\mathrm{med}} z}\big)\cdot\mathbf c^{\mathrm{med}}
$$

## Final Model (Different Attenuations for Direct/Backscatter)

$$
\begin{array}{rl}
\hat C_i^{\mathrm{obj}}(\mathbf r)&=\mathcal T_i^{\mathrm{obj}}\cdot \exp!\big(-\mathbf a^{\mathrm{attn}} s_i\big)\cdot\big(1-\exp(-\sigma_i^{\mathrm{obj}}\delta_i)\big)\cdot\mathbf c_i^{\mathrm{obj}}\
\hat C_i^{\mathrm{med}}(\mathbf r)&=\mathcal T_i^{\mathrm{obj}}\cdot \exp!\big(-\mathbf a^{\mathrm{bs}} s_i\big)\cdot\big(1-\exp(-\mathbf a^{\mathrm{bs}}\delta_i)\big)\cdot\mathbf c^{\mathrm{med}}\
\mathcal T_i^{\mathrm{obj}}&=\exp!\Big(-\sum_{j=0}^{i-1}\sigma_j^{\mathrm{obj}}\delta_j\Big)
\end{array}
$$

## Architecture

* **Object path (as in NeRF):** $\sigma^{\mathrm{obj}}(\mathbf x)$ depends on position; $\mathbf c^{\mathrm{obj}}(\mathbf x,\mathbf d)$ depends on position and view.
* **Medium path (direction-only):** MLP with 6 linear layers (256 units, softplus), branching to:

  * $\mathbf c^{\mathrm{med}}$ via sigmoid,
  * $\mathbf a^{\mathrm{attn}}$ and $\mathbf a^{\mathrm{bs}}$ via softplus.
* Medium parameters are constant along a ray (direction-conditioned), potentially distinct per color channel.

## Losses

Let samples be $\mathbf s={s_i}_{i=0}^{N}$, object weights $w_i^{\mathrm{obj}}=\mathcal T_i^{\mathrm{obj}}\big(1-\exp(-\sigma_i^{\mathrm{obj}}\delta_i)\big)$, and ground-truth color $C^*$.

* **Reconstruction (linear domain; RawNeRF-style):**
  $$
  \mathcal L_{\mathrm{recon}}(\hat C,C^*)=\left(\frac{\hat C-C^*}{\mathrm{sg}(\hat C)+\epsilon}\right)^2,\quad \epsilon=10^{-3}
  $$
* **Proposal distribution matching (MipNeRF-360):** $\mathcal L_{\mathrm{prop}}(\mathbf s,\mathbf w)$.
* **Opaque-object prior on $\mathcal T_i^{\mathrm{obj}}$ (mixture of Laplacians):**
  $$
  \mathbb P(x)\propto e^{-|x|/0.1}+e^{-|1-x|/0.1},\quad
  \mathcal L_{\mathrm{acc}}(\mathbf w)=-\log\mathbb P(\mathcal T_i^{\mathrm{obj}})
  $$
* **Total:**
  $$
  \mathcal L=\mathcal L_{\mathrm{recon}}+\mathcal L_{\mathrm{prop}}+\lambda,\mathcal L_{\mathrm{acc}},\quad \lambda=0.0001
  $$

## Implementation Details

* Base code: Mip-NeRF-360 (forward-facing, NDC). Disable the “infinite-far-delta background color” so the medium explains residual color along rays.
* Training: 250k iterations; batch size 16,384 rays; optimizer and LR schedule as in Mip-NeRF-360. Compute losses/metrics on linear outputs before any post-processing.

## Datasets, Preprocessing, Simulation

* **Real underwater sets:** Red Sea (Eilat), Curaçao (Caribbean), Panama (Pacific); RAW Nikon D850 with dome port; ~20 images per site with 3 for validation; downsample to $\approx900\times1400$; per-channel white balance with 0.5% clipping; camera poses via COLMAP.
* **Simulation (LLFF Fern):** underwater via Eq. (7) with $\beta^{D}=[1.3,1.2,0.9]$, $\beta^{B}=[0.95,0.85,0.7]$, $B^{\infty}=[0.07,0.2,0.39]$; fog with $\beta=1.2$.
* **Photofinishing:** applied only for visualization using a standard camera pipeline; PSNR/SSIM/LPIPS computed on linear images.

## Evaluation Protocol and Ablations (Affecting Implementation)

* Metrics: PSNR, SSIM, LPIPS on validation images.
* **Ablations:**

  * Medium parameterization: 6-parameter per-channel model outperforms 1- or 3-parameter simplifications.
  * Rendering equations: final model (separate $\mathbf a^{\mathrm{attn}}$ and $\mathbf a^{\mathrm{bs}}$) outperforms the basic shared-attenuation variant.

