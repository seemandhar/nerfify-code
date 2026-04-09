# BioNeRF: Biologically Plausible Neural Radiance Fields for View Synthesis

# 3. Biologically Plausible Neural Radiance Fields

This paper introduces BioNeRF, which combines features extracted from both the camera position and direction into a memory mechanism inspired by cortical circuits. Further, such memory is employed as a context to leverage the information flow into deeper layers. The architecture spans four main modules: (i) Positional Feature Extraction, (ii) Cognitive Filtering, (iii) Memory Updating, and (iv) Contextual Inference.

Positional Feature Extraction. The first step consists of feeding two neural models simultaneously, namely $M_{\Delta}$ and $M_{c}$, such that $\Theta_{\Delta}=(W_{\Delta},b_{\Delta})$ and $\boldsymbol{\Theta}*{c}=(W*{c},b_{c})$ denote their respective set of parameters, i.e., neural weights $(W)$ and biases $(b)$. The output of these models, i.e., $h_{\Delta}$ and $h_{c}$, encodes positional information from the input image. Although the input is the same, the neural models do not share weights and follow a different flow in the next steps.

Cognitive Filtering. This step performs a series of operations from now on called filters that work on the embeddings coming from the previous step. There are four filters this step derives: (i) density $(f_{\Delta})$, color $(f_{c})$, memory $(f_{\psi})$, and modulation $(f_{\mu})$, computed as follows:

$$
\begin{array}{r}
{\pmb f}*{\Delta}=\sigma\left(W*{\Delta}^{f}h_{\Delta}+b_{\Delta}^{f}\right),
\end{array}
$$

$$
\pmb{f}*{c}=\sigma\left(\pmb{W}*{c}^{f}h_{c}+\pmb{b}_{c}^{f}\right),
$$

$$
\begin{array}{r}
\pmb{f}*{\psi}=\sigma\left(\pmb{W}*{\psi}^{f}[\pmb{h}*{\Delta},\pmb{h}*{c}]+\pmb{b}_{\psi}^{f}\right),
\end{array}
$$

and

$$
\pmb{f}*{\mu}=\sigma\left(W*{\mu}^{f}[\pmb{h}*{\Delta},\pmb{h}*{c}]+b_{\mu}^{f}\right),
$$

where $\pmb{W}*{\Delta}^{f}$ $*{\Delta}^{f},W_{c}^{f},W_{\psi}^{f}$ and $\mathbf{}W_{\mu}^{f}$ correspond to the weight matrices for the density, color, memory, and modulation filters, respectively. Additionally, $b_{\Delta}^{f},\dot{b}*{c}^{f},,b*{\psi}^{f}$ and $b_{\mu}^{f}$ stand for their respective biases. Moreover, $[h_{\Delta},h_{c}]$ represents the concatenation of embeddings $h_{\Delta}$ and $h_{c}$, while $\sigma$ denotes a sigmoid function. The pre-modulation $\gamma$ is computed as follows:

$$
\gamma=tanh\left(W_{\gamma}[h_{\Delta},h_{c}]+b_{\gamma}\right),
$$

where $tanh(\cdot)$ is the hyperbolic tangent function, while $W_{\gamma}$ and $b_{\gamma}$ are the pre-modulation weight matrix and bias, respectively.

Memory Updating. Updating the memory requires the implementation of a mechanism capable of obliterating trivial information, which is performed using the memory filter $f_{\psi}$. First, one needs to compute the modulation $\pmb{\mu}$, where $\otimes$ represents the dot product:

$$
\pmb{\mu}=\pmb{f}_{\mu}\otimes\pmb{\gamma}.
$$

New experiences are introduced in the memory $\Psi$ through the modulating variable $\pmb{\mu}$ using a tanh function:

$$
\Psi=tanh\left(W_{\Psi}\left(\pmb{\mu}+(\pmb{f}*{\psi}\otimes\Psi)\right)+b*{\Psi}\right),
$$

where $W_{\Psi}$ and $b_{\Psi}$ are the memory weight matrix and bias, respectively.

Contextual Inference. This step is responsible for adding contextual information to BioNeRF. We generate two new embeddings $h_{\Delta}^{\prime}$ and $\pmb{h}*{c}^{\prime}$ based on filters $f*{\Delta}$ and $f_{c}$, respectively, which further feed two neural models, i.e., $M_{\Delta}^{\prime}=(W_{\Delta}^{\prime},b_{\Delta}^{\prime})$ and $M_{c}^{\prime}=(W_{c}^{\prime},W_{c}^{\prime})$, accordingly:

$$
{h}*{\Delta}^{\prime}=[\Psi\otimes f*{\Delta},]],
$$

and

$$
\begin{array}{r}
{h}*{c}^{\prime}=[\Psi\otimes f*{c},\mathbf{d}].
\end{array}
$$

Subsequently, $M_{\Delta}^{\prime}$ outputs the volume density $\Delta$, while color information $^c$ is predicted by $M_{c}^{\prime}$, ending up in the final predicted pixel information $(\Delta,c)$, further used to compute the loss function.

Loss Function. Let $r:\mathfrak{R}\times\mathfrak{R}^{3}\to\mathfrak{R}^{3}$ be a volume rendering technique that computes the pixel color given the volume density and the color. The BioNeRF loss function is defined as follows:

$$
\mathcal{L}=MSE\left(r(\Delta,\pmb{c}),\pmb{g}\right),
$$

where $MSE(\cdot)$ is the mean squared error function and $\textbf{{g}}$ corresponds to the ground truth pixel color.

The error is then back-propagated to the model’s previous layers/steps to update its set of weights $W=\overset{\cdot}{{W_{\Delta},W_{c},W_{\Delta},W_{c}^{f},W_{\psi}^{f},W_{\psi}^{f},W_{\mu}^{f},W_{\gamma},W_{\Psi},W_{\Delta}^{'},W_{c}^{\prime}}}$ and biases $\boldsymbol{b}={b_{\Delta},b_{c},b_{\Delta}^{f},b_{c}^{f},b_{\psi}^{f},b_{\mu}^{f},b_{\gamma},b_{\Psi},b_{\Delta}^{\prime},b_{c}^{\prime}}.$

# 4. Methodology

This section provides information regarding the datasets employed in this work and the configuration adopted in the experimental setup.

# 4.1. Datasets

We conducted experiments over two well-known datasets concerning view synthesis, i.e., Blender and LLFF.

# 4.1.1. Blender

Also known as NeRF Synthetic, the Blender comprises eight object scenes with intricate geometry and realistic non-Lambertian materials. Six of these objects are rendered from viewpoints tested on the upper hemisphere, while the remaining two come from viewpoints sampled on a complete sphere.

# 4.1.2. Local Light Field Fusion

This paper considers 8 scenes from the LLFF Real dataset, which comprises 24 scenes captured from handheld cellphones with $20$–$30$ images each. The authors used a COLMAP structure-from-motion implementation to compute the poses.

# 4.2. Experimental Setup

The experiments conducted in this work aim to evaluate the behavior of BioNeRF in the context of scene-view synthesis against established methods.

* Two parallel MLP blocks with ReLU activations process camera $3$D coordinates; each block has 3 layers with $h=256$ hidden units.
* Memory $\Psi\in\mathbb{R}^{z\times h}$ is updated and used as context; $z=8,192$ is the number of directional rays processed in parallel.
* $M_{\Delta}^{\prime}$: two dense layers (256 units) plus a $1$-unit output for $\Delta$ (volume density).
* $M_{c}^{\prime}$: one dense layer (128 units) plus a $3$-unit output for RGB color; inputs are memory context concatenated with $(\theta,\phi)$.
* Optimizer: Adam, learning rate $5e-4$, for $400k$ updates.
* Metrics: PSNR (higher is better), SSIM (higher is better), and LPIPS (lower is better).
* System: Ubuntu 18, dual Intel Xeon Bronze 3104, 62 GB RAM, NVIDIA Tesla T4 GPU.
* Implementation: Python with PyTorch.

# 5.4. Ablation study

This section provides an evaluation of the BioNeRF effectiveness over the Blender dataset considering three implemented versions, based on the standard NeRF, NeRFacto, and TensoRF, which will be referred here as BioNeRF, BioNeRFacto, and BioTensoRF, respectively.

* The biologically plausible module is implemented in a substructure of NeRF’s pipeline called the field.
* BioNeRF changes are implemented in the Coarse and Fine blocks of NeRF’s pipeline.
* For BioNeRFacto and BioTensoRF, analogous changes are implemented in the nerfacto field and TensoRF field, respectively.
* After $50k$ iterations on Blender scenes, BioNeRF and BioTensoRF perform comparably on PSNR/SSIM, with BioNeRF showing a pronounced advantage on LPIPS.
