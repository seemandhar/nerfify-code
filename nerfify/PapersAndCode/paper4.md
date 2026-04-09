# K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

## Problem setup & representation

* Goal: explicit, scalable radiance fields for arbitrary dimension (d); use (k=\binom{d}{2}) 2D planes that cover all dimension pairs. For 4D ((x,y,z,t)): six planes ({xy,xz,yz,xt,yt,zt}). 
* Each plane stores an (N\times N\times M) feature grid (bilinear interpolation (\psi)). For a 4D point (\mathbf q=(i,j,k,\tau)):
  [
  f(\mathbf q)*c=\psi(\mathbf P_c,\pi_c(\mathbf q)),\quad
  f(\mathbf q)=\prod*{c\in C} f(\mathbf q)_c .
  ]
  (Hadamard product across planes.) 
* Multiscale spatial planes to encourage smoothness/coherence and reduce high-res storage; e.g., spatial scales ({64,128,256,512}); time is single-scale. Concatenate per-scale features before decoding. 
* Why multiply (Hadamard) rather than add plane features: multiplication selects localized intersections; empirically improves explicit models vs. addition. 

## Decoders

* **Linear learned color basis (explicit model):** tiny MLP maps view direction (\mathbf d) to basis vectors (b_R(\mathbf d),b_G(\mathbf d),b_B(\mathbf d)\in\mathbb R^M).
  [
  c(\mathbf q,\mathbf d)=\bigcup_{i\in{R,G,B}}, \mathbf f(\mathbf q)\cdot \mathbf b_i(\mathbf d),\quad
  \sigma(\mathbf q)=\mathbf f(\mathbf q)\cdot \mathbf b_\sigma .
  ]
  Apply sigmoid to color and (truncated-grad) exponential to density. 
* **Hybrid MLP decoder:** optional MLP replaces linear basis while keeping the same plane features (used when desired; linear decoder suffices given Hadamard combination). 
* **Variable appearance:** add a global per-image appearance code only to the color decoder so geometry is unchanged. 

## Priors & regularization (applied to plane features)

* **Spatial total variation (L2):**
  [
  \mathcal L_{TV}(\mathbf P)=\frac{1}{|C|,n^2}\sum_{c,i,j}\Big(\lVert \mathbf P_c^{i,j}-\mathbf P_c^{i-1,j}\rVert_2^2+\lVert \mathbf P_c^{i,j}-\mathbf P_c^{i,j-1}\rVert_2^2\Big).
  ]
  Use 2D TV on space-only planes and 1D TV along spatial axes of space-time planes. 
* **Temporal smoothness (Laplacian in time, space-time planes only):**
  [
  \mathcal L_{\text{smooth}}(\mathbf P)=\frac{1}{|C|,n^2}\sum_{c,i,t}\big\lVert \mathbf P_c^{i,t-1}-2\mathbf P_c^{i,t}+\mathbf P_c^{i,t+1}\big\rVert_2^2 .
  ]

* **Static–dynamic separation prior:** initialize time planes near the multiplicative identity (ones) and penalize deviations so unchanged regions remain static:
  [
  \mathcal L_{\text{sep}}(\mathbf P)=\sum_{c\in{xt,yt,zt}}\lVert \mathbf 1-\mathbf P_c\rVert_1 .
  ]


## Rendering & training essentials

* Use standard volumetric rendering of (\sigma) and (c) along camera rays (as in NeRF); plane features are queried per sample via bilinear interpolation at all used spatial scales, multiplied across planes, then decoded to density/color. (Follows from the equations and representation above.) 

## Recommended implementation choices (from ablations/usage)

* Combine planes via Hadamard product (explicit models benefit substantially over addition). 
* Practical multiscale setup for static scenes shown in ablations: three spatial scales ({128,256,512}) with (M=32) features per scale (illustrative config that balanced quality/size). 

## Evaluation protocol notes (only if reproducing variable-appearance experiments)

* Dataset: Phototourism; train on (\sim)thousands of photos; evaluate on a clean test split; optimize a per-image appearance code and score metrics on held-out regions, following NeRF-W. 
