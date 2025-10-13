# E3 Equivariant Diffusion for Molecules: Theoretical Foundation

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [E(3) Equivariance](#e3-equivariance)
4. [Diffusion Models](#diffusion-models)
5. [EGNN Architecture](#egnn-architecture)
6. [Crystal Extension Theory](#crystal-extension-theory)
7. [Conditioning Theory](#conditioning-theory)
8. [Loss Functions](#loss-functions)
9. [Sampling Theory](#sampling-theory)

---

## Introduction

This document provides a rigorous theoretical foundation for the E(3) Equivariant Diffusion Model (EDM) for 3D molecule and crystal generation. The model generates molecular structures by learning to reverse a diffusion process while maintaining E(3) equivariance—a fundamental symmetry property of physical systems.

### Core Principles

1. **E(3) Equivariance**: All operations respect 3D rotations, translations, and reflections
2. **Diffusion-based Generation**: Uses score-based generative modeling
3. **No Fallback Heuristics**: All operations are theoretically grounded with no approximations
4. **Physical Validity**: Generated structures satisfy chemical and physical constraints

---

## Mathematical Foundation

### Notation

- **Molecular Graph**: $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with nodes $\mathcal{V}$ and edges $\mathcal{E}$
- **Node Features**: $h_i \in \mathbb{R}^{d_h}$ for node $i$
- **Node Positions**: $\mathbf{x}_i \in \mathbb{R}^3$ for node $i$
- **Edge Features**: $e_{ij} \in \mathbb{R}^{d_e}$ for edge $(i,j)$

### Data Representation

A molecule is represented as a tuple $(h, \mathbf{x})$ where:
- $h = \{h_1, ..., h_N\}$ are node features (atom types, charges)
- $\mathbf{x} = \{\mathbf{x}_1, ..., \mathbf{x}_N\}$ are 3D coordinates

**Invariants**:
- Number of atoms: $N$
- Atom types: encoded as one-hot vectors in $h$
- Charges: encoded as integers in $h$

---

## E(3) Equivariance

### Definition

A function $f: (\mathbf{x}, h) \rightarrow (\mathbf{x}', h')$ is E(3) equivariant if for all rotations $R \in SO(3)$ and translations $t \in \mathbb{R}^3$:

$$f(R\mathbf{x} + t, h) = (R\mathbf{x}' + t, h')$$

where the transformation applies element-wise to coordinates.

### Implications

**E(3) Equivariance ensures**:
1. **Translation Invariance**: Model output independent of molecule position
2. **Rotation Invariance**: Model output independent of molecule orientation
3. **Reflection Invariance**: Properly handles chiral structures
4. **Physical Validity**: Generated structures are physically meaningful

### Coordinate Constraints

To ensure well-defined training, we enforce:

$$\sum_{i=1}^N \mathbf{x}_i = \mathbf{0}$$

This **center-of-mass constraint** removes translational degrees of freedom.

---

## Diffusion Models

### Forward Process

The forward diffusion process adds Gaussian noise over timesteps $t \in [0, T]$:

$$q(z_t | z_0) = \mathcal{N}(z_t; \alpha_t z_0, \sigma_t^2 I)$$

where:
- $z_0 = (\mathbf{x}_0, h_0)$ is the clean data
- $z_t = (\mathbf{x}_t, h_t)$ is the noisy data at time $t$
- $\alpha_t = \sqrt{\bar{\alpha}_t}$ is the signal scale
- $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$ is the noise scale

### Noise Schedule

We use a polynomial noise schedule:

$$\gamma(t) = -\log(\alpha_t^2 / \sigma_t^2) = -\log \text{SNR}(t)$$

where SNR is the signal-to-noise ratio. Common schedules:
- **polynomial_2**: $\gamma(t) = (t/T)^2$
- **cosine**: $\gamma(t) = -2\log(\cos(\pi t / 2T))$

### Reverse Process

The reverse process learns to denoise:

$$p_\theta(z_{t-\Delta t} | z_t) = \mathcal{N}(z_{t-\Delta t}; \mu_\theta(z_t, t), \sigma_{t|t-\Delta t}^2 I)$$

where:
- $\mu_\theta$ is the learned mean predictor
- $\sigma_{t|t-\Delta t}$ is the conditional noise scale

### Denoising Objective

The model learns to predict the noise $\epsilon$ added at each step:

$$\epsilon_\theta(z_t, t) \approx \epsilon \quad \text{where } z_t = \alpha_t z_0 + \sigma_t \epsilon$$

---

## EGNN Architecture

### Message Passing

The E(n) Equivariant Graph Neural Network (EGNN) updates node features and positions through message passing:

**Edge Messages**:
$$m_{ij} = \phi_e(h_i, h_j, \|\mathbf{x}_i - \mathbf{x}_j\|^2, e_{ij})$$

**Node Feature Update**:
$$h_i' = \phi_h(h_i, \sum_{j \in \mathcal{N}(i)} m_{ij})$$

**Coordinate Update**:
$$\mathbf{x}_i' = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} (\mathbf{x}_i - \mathbf{x}_j) \phi_x(m_{ij})$$

where:
- $\phi_e, \phi_h, \phi_x$ are MLPs
- $\mathcal{N}(i)$ denotes neighbors of node $i$
- $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ ensures rotation invariance

### Attention Mechanism

Optional attention weights:
$$a_{ij} = \text{softmax}_j(\phi_a(m_{ij}))$$

modulate message contributions:
$$m_{ij}^{att} = a_{ij} \cdot m_{ij}$$

### Sinusoidal Distance Embedding

Distances are embedded using sinusoidal functions:

$$\text{emb}(d) = [\sin(2\pi f_1 d), \cos(2\pi f_1 d), ..., \sin(2\pi f_k d), \cos(2\pi f_k d)]$$

where $f_k = 2^k / d_{\max}$ for $k = 0, ..., K-1$.

This provides:
- **Multi-scale** distance representation
- **Continuous** distance encoding
- **Periodic** structure for bond lengths

---

## Crystal Extension Theory

### Periodic Boundary Conditions

Crystals are infinite periodic systems. We work with:
- **Unit cell**: Finite representative volume
- **Lattice vectors**: $\mathbf{a}, \mathbf{b}, \mathbf{c}$ define periodicity
- **Cell parameters**: $(a, b, c, \alpha, \beta, \gamma)$

### Minimum Image Convention

For periodic systems, the distance between atoms $i$ and $j$ is:

$$\mathbf{r}_{ij}^{\text{min}} = \min_{n_1, n_2, n_3} \|\mathbf{r}_{ij} + n_1\mathbf{a} + n_2\mathbf{b} + n_3\mathbf{c}\|$$

where $n_1, n_2, n_3 \in \mathbb{Z}$ enumerate periodic images.

**Algorithm**:
1. Convert to fractional coordinates: $\mathbf{f} = \mathbf{M}^{-1} \mathbf{x}$
2. Apply PBC: $\mathbf{f}' = \mathbf{f} - \lfloor \mathbf{f} + 0.5 \rfloor$
3. Convert back: $\mathbf{x}' = \mathbf{M} \mathbf{f}'$

where $\mathbf{M} = [\mathbf{a}, \mathbf{b}, \mathbf{c}]$ is the cell matrix.

### Fractional Coordinates

To handle changing cell parameters during diffusion:

$$\mathbf{x}_{\text{frac}} = \mathbf{M}^{-1} \mathbf{x}_{\text{cart}}$$

**Properties**:
- **Invariant** to cell parameter changes
- Values in $[0, 1)$ with periodic wrapping
- Natural for periodic systems

### Lattice Diffusion

Cell parameters undergo their own diffusion process:

$$c_t = \alpha_t^c c_0 + \sigma_t^c \epsilon^c$$

where $c = (a, b, c, \alpha, \beta, \gamma)$.

**Normalization**:
- Lengths: $\log(a), \log(b), \log(c)$ for scale invariance
- Angles: radians, enforced in valid ranges

**Physical Constraints**:
- $a, b, c > 0$: Positive cell dimensions
- $0 < \alpha, \beta, \gamma < \pi$: Valid angles
- $\alpha + \beta > \gamma$ (and cyclic): Triangle inequality

---

## Conditioning Theory

### Molecular Conditioning

Given a single molecule, extract features to guide crystal generation:

**Feature Extraction**:
1. **EGNN Features**: $\mathbf{f}_{\text{mol}} = \text{EGNN}(h_{\text{mol}}, \mathbf{x}_{\text{mol}})$
2. **Geometric Properties**:
   - Molecular size: $s = \max_i \|\mathbf{x}_i - \bar{\mathbf{x}}\|$
   - Volume: $V = \frac{4}{3}\pi s^3$ (approximate)
   - Principal axes: $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ from PCA

**Conditioning Vector**:
$$\mathbf{c}_{\text{mol}} = \text{MLP}([\mathbf{f}_{\text{mol}}, s, V, \mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3])$$

**Usage**: Concatenated to node features at each diffusion step.

### Space Group Conditioning

Space groups define crystal symmetry. There are 230 unique space groups.

**Embedding**:
$$\mathbf{c}_{\text{sg}} = \text{Embed}(g) \quad g \in \{1, ..., 230\}$$

followed by MLP projection.

**Validation**:
- Input must be integer in $[1, 230]$
- No fallback to default value
- Raises error for invalid input

### Density Conditioning

Crystal density $\rho$ (g/cm³) provides packing information:

**Normalization**:
$$\rho_{\text{norm}} = \frac{\rho - \rho_{\min}}{\rho_{\max} - \rho_{\min}}$$

with $\rho_{\min} = 0.5$, $\rho_{\max} = 5.0$ (typical molecular crystals).

**Embedding**:
$$\mathbf{c}_{\rho} = \text{MLP}(\rho_{\text{norm}})$$

### Combined Conditioning

Multiple conditioning sources are fused:

$$\mathbf{c} = w_{\text{mol}} \mathbf{c}_{\text{mol}} + w_{\text{sg}} \mathbf{c}_{\text{sg}} + w_{\rho} \mathbf{c}_{\rho}$$

where weights $w$ are learned or fixed.

**Requirement**: Molecular conditioning is **mandatory** as the primary information source.

---

## Loss Functions

### Variational Lower Bound (VLB)

The model maximizes the evidence lower bound:

$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_{q(z_0)} \left[ -\log p_\theta(z_0) \right]$$

Decomposed as:
$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_t \left[ \mathcal{L}_t \right] + \mathcal{L}_0 + \mathcal{L}_T$$

where:
- $\mathcal{L}_t$: Denoising loss at time $t$
- $\mathcal{L}_0$: Reconstruction loss
- $\mathcal{L}_T$: Prior matching loss

### L2 Loss (Simplified)

For efficiency, we use the denoising score matching loss:

$$\mathcal{L}_{\text{L2}} = \mathbb{E}_{z_0, t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right]$$

where $\epsilon \sim \mathcal{N}(0, I)$ is the noise.

### Position Loss

For coordinates, we enforce center-of-mass constraint:

$$\mathcal{L}_{\mathbf{x}} = \|\epsilon_{\mathbf{x}} - \epsilon_\theta^{\mathbf{x}}(z_t, t)\|^2 \quad \text{s.t.} \quad \sum_i \epsilon_\theta^{\mathbf{x}}_i = 0$$

### Feature Loss

For discrete features (atom types):

$$\mathcal{L}_h = \text{CrossEntropy}(h_0, \hat{h}_0(z_t, t))$$

where $\hat{h}_0$ is the predicted clean features.

### Cell Loss (Crystals)

For cell parameters:

$$\mathcal{L}_c = \|\epsilon_c - \epsilon_\theta^c(z_t, t)\|^2 + \lambda_{\text{reg}} \mathcal{R}(c)$$

where $\mathcal{R}(c)$ enforces physical constraints:
- Positive lengths: $\mathcal{R}_a = \max(0, -a)^2$
- Valid angles: $\mathcal{R}_\alpha = \max(0, \alpha - \pi)^2 + \max(0, -\alpha)^2$
- Triangle inequality: Ensures valid cell

---

## Sampling Theory

### Reverse Diffusion

Starting from noise $z_T \sim \mathcal{N}(0, I)$, iteratively denoise:

$$z_{t-\Delta t} = \mu_\theta(z_t, t) + \sigma_{t|t-\Delta t} \epsilon \quad \epsilon \sim \mathcal{N}(0, I)$$

where:
$$\mu_\theta(z_t, t) = \frac{1}{\alpha_{t|t-\Delta t}} \left( z_t - \frac{\sigma_{t|t-\Delta t}^2}{\sigma_t} \epsilon_\theta(z_t, t) \right)$$

### Predictor-Corrector

For better sample quality, alternate:
1. **Predictor**: Take a reverse diffusion step
2. **Corrector**: Langevin dynamics step

$$z_{t-\Delta t}^{(k+1)} = z_{t-\Delta t}^{(k)} + \eta \nabla_{z} \log p_\theta(z_{t-\Delta t}^{(k)}) + \sqrt{2\eta} \epsilon$$

### Conditional Sampling

For conditional generation with property $y$:

$$p(z_0 | y) \propto p(z_0) p(y | z_0)$$

Implemented via:
1. **Conditioning during training**: Include $y$ in model input
2. **Classifier guidance**: Scale by $\nabla_z \log p(y|z)$

### Crystal Sampling

For crystals, sample both positions and cell:

1. Initialize: $z_T \sim \mathcal{N}(0, I)$, $c_T \sim \mathcal{N}(c_{\text{prior}}, \sigma_c^2 I)$
2. For $t = T, T-1, ..., 1$:
   - Predict: $\epsilon_\theta(z_t, c_t, t)$
   - Update positions: $z_{t-1} = \mu_\theta^z(z_t, c_t, t) + \sigma \epsilon_z$
   - Update cell: $c_{t-1} = \mu_\theta^c(z_t, c_t, t) + \sigma \epsilon_c$
3. Convert to Cartesian: $\mathbf{x}_0 = \mathbf{M}(c_0) \mathbf{f}_0$

**Note**: Positions are in fractional coordinates during diffusion.

---

## Theoretical Guarantees

### Consistency

Under assumptions:
1. Model capacity sufficient
2. Training converged
3. Noise schedule appropriate

The model satisfies:
$$p_\theta(z_0) \approx p_{\text{data}}(z_0)$$

### E(3) Equivariance Preservation

All operations maintain:
$$p_\theta(R\mathbf{x} + t, h) = p_\theta(\mathbf{x}, h)$$

ensuring physical validity.

### Uniqueness

The combination of:
1. Center-of-mass constraint
2. E(3) equivariance
3. Proper conditioning

ensures unique, valid molecular structures.

---

## References

1. **EDM**: Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D" (2022)
2. **EGNN**: Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
3. **Score-based Models**: Song et al., "Score-Based Generative Modeling through SDEs" (2021)
4. **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
5. **Periodic Systems**: Allen & Tildesley, "Computer Simulation of Liquids" (2017)
6. **Crystallography**: International Tables for Crystallography (2016)

---

## Mathematical Notation Summary

| Symbol | Description |
|--------|-------------|
| $\mathbf{x}_i$ | Position of atom $i$ in $\mathbb{R}^3$ |
| $h_i$ | Feature vector of atom $i$ |
| $z_t$ | Noisy data at time $t$ |
| $\alpha_t, \sigma_t$ | Signal and noise scales |
| $\epsilon_\theta$ | Learned denoising function |
| $\mathbf{M}$ | Cell matrix $[\mathbf{a}, \mathbf{b}, \mathbf{c}]$ |
| $(a,b,c,\alpha,\beta,\gamma)$ | Cell parameters |
| $\mathbf{c}$ | Conditioning vector |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian distribution |
| $SO(3)$ | 3D rotation group |

---

**Version**: 1.0  
**Date**: 2025-10-13  
**Status**: Complete theoretical documentation for PR#124-130
