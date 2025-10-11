# 単分子情報を用いた分子性結晶生成の理論説明書
# Theoretical Documentation for Single-Molecule Conditioned Crystal Generation

## 概要 (Overview)

本文書は、単分子の幾何学的・化学的情報を条件として用いた分子性結晶生成の理論的基礎を提供します。既存の無条件拡散モデルを拡張し、特定の分子から形成されるホモ結晶（同一分子から構成される結晶）を生成します。

This document provides the theoretical foundation for molecular crystal generation conditioned on single molecule information. We extend the existing unconditional diffusion model to generate homo-crystals (crystals composed of identical molecules) formed by a specific molecule.

---

## 1. 問題設定 (Problem Formulation)

### 1.1 データセット構造 (Dataset Structure)

訓練データは以下の対応するデータセットで構成されます：

**Training data consists of paired datasets:**

$$
\mathcal{D} = \{(\mathbf{M}_i, \mathbf{C}_i)\}_{i=1}^N
$$

where:
- $\mathbf{M}_i$: 単分子の構造情報 (Single molecule structure)
- $\mathbf{C}_i$: 対応する分子性結晶の構造情報 (Corresponding molecular crystal structure)
- $N$: データセットサイズ (Dataset size)

### 1.2 単分子の表現 (Molecular Representation)

単分子 $\mathbf{M}$ は以下で記述されます：

**A single molecule $\mathbf{M}$ is described by:**

$$
\mathbf{M} = \{(\mathbf{r}_j^{(mol)}, \mathbf{h}_j^{(mol)})\}_{j=1}^{n_{mol}}
$$

where:
- $n_{mol}$: 分子内の原子数 (Number of atoms in the molecule)
- $\mathbf{r}_j^{(mol)} \in \mathbb{R}^3$: $j$ 番目の原子の3次元座標 (3D coordinates of the $j$-th atom)
- $\mathbf{h}_j^{(mol)} \in \mathbb{R}^d$: $j$ 番目の原子の特徴ベクトル (Feature vector of the $j$-th atom)
  - 原子種のone-hot表現
  - 部分電荷
  - その他の原子特性

### 1.3 結晶の表現 (Crystal Representation)

分子性結晶 $\mathbf{C}$ は以下で記述されます：

**A molecular crystal $\mathbf{C}$ is described by:**

$$
\mathbf{C} = \{\mathbf{X}, \mathbf{H}, \mathbf{L}\}
$$

where:
- $\mathbf{X} = \{\mathbf{x}_i\}_{i=1}^{n_{atoms}}$: 単位格子内の全原子座標 (All atomic coordinates in the unit cell)
- $\mathbf{H} = \{\mathbf{h}_i\}_{i=1}^{n_{atoms}}$: 単位格子内の全原子特徴 (All atomic features in the unit cell)
- $\mathbf{L} \in \mathbb{R}^{3 \times 3}$: 格子ベクトル行列 (Lattice vector matrix)

$$
\mathbf{L} = \begin{pmatrix}
| & | & | \\
\mathbf{a} & \mathbf{b} & \mathbf{c} \\
| & | & |
\end{pmatrix}
$$

または格子パラメータ形式：

**Or in lattice parameter form:**

$$
\mathbf{l} = (a, b, c, \alpha, \beta, \gamma) \in \mathbb{R}^6
$$

### 1.4 ホモ結晶制約 (Homo-Crystal Constraint)

生成される結晶は、入力分子 $\mathbf{M}$ と同一の分子 $Z$ 個から構成されます：

**The generated crystal consists of $Z$ identical copies of the input molecule $\mathbf{M}$:**

$$
\mathbf{C} = \bigcup_{k=1}^{Z} \mathbf{M}_k, \quad \mathbf{M}_k \cong \mathbf{M}
$$

where:
- $Z$: 単位格子内の分子数 (Number of molecules per unit cell)
- $\mathbf{M}_k \cong \mathbf{M}$: $k$ 番目の分子は入力分子 $\mathbf{M}$ と同型 (The $k$-th molecule is isomorphic to $\mathbf{M}$)

---

## 2. 拡散過程の定式化 (Diffusion Process Formulation)

### 2.1 前方拡散過程 (Forward Diffusion Process)

前方過程では、結晶構造 $\mathbf{C}_0$ にノイズを徐々に加えます：

**In the forward process, we gradually add noise to the crystal structure $\mathbf{C}_0$:**

#### 原子座標の拡散 (Atomic Coordinate Diffusion)

$$
q(\mathbf{X}_t | \mathbf{X}_0) = \mathcal{N}(\mathbf{X}_t; \sqrt{\bar{\alpha}_t} \mathbf{X}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

#### 原子特徴の拡散 (Atomic Feature Diffusion)

$$
q(\mathbf{H}_t | \mathbf{H}_0) = \mathcal{N}(\mathbf{H}_t; \sqrt{\bar{\alpha}_t} \mathbf{H}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

#### 格子パラメータの拡散 (Lattice Parameter Diffusion)

$$
q(\mathbf{L}_t | \mathbf{L}_0) = \mathcal{N}(\mathbf{L}_t; \sqrt{\bar{\alpha}_t} \mathbf{L}_0, (1 - \bar{\alpha}_t) \mathbf{\Sigma}_L)
$$

where:
- $t \in [0, T]$: 拡散時間ステップ (Diffusion time step)
- $\alpha_t$: ノイズスケジュール (Noise schedule)
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$: 累積ノイズ (Cumulative noise)
- $\mathbf{\Sigma}_L$: 格子パラメータの共分散行列 (Covariance matrix for lattice parameters)

### 2.2 逆拡散過程（条件付き）(Reverse Diffusion Process with Conditioning)

逆過程では、単分子情報 $\mathbf{M}$ を条件として結晶を生成します：

**In the reverse process, we generate the crystal conditioned on the single molecule information $\mathbf{M}$:**

$$
p_\theta(\mathbf{C}_{t-1} | \mathbf{C}_t, \mathbf{M}) = \mathcal{N}(\mathbf{C}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{C}_t, t, \mathbf{M}), \boldsymbol{\Sigma}_\theta(\mathbf{C}_t, t, \mathbf{M}))
$$

平均関数は以下のように分解されます：

**The mean function is decomposed as:**

$$
\boldsymbol{\mu}_\theta(\mathbf{C}_t, t, \mathbf{M}) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{C}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{C}_t, t, \mathbf{M}) \right)
$$

where $\boldsymbol{\epsilon}_\theta(\mathbf{C}_t, t, \mathbf{M})$ はノイズ予測ネットワーク (noise prediction network)。

---

## 3. 分子条件付けの数学的定式化 (Mathematical Formulation of Molecular Conditioning)

### 3.1 分子エンコーダ (Molecule Encoder)

単分子 $\mathbf{M}$ を固定長のコンテキストベクトルにエンコードします：

**Encode the single molecule $\mathbf{M}$ into a fixed-length context vector:**

$$
\mathbf{c}_{mol} = \text{Enc}(\mathbf{M}) = \text{Enc}(\{\mathbf{r}_j^{(mol)}, \mathbf{h}_j^{(mol)}\}_{j=1}^{n_{mol}})
$$

#### 3.1.1 E(3)等変エンコーダ (E(3) Equivariant Encoder)

分子構造の回転・並進対称性を保持するため、E(3)等変なエンコーダを使用：

**To preserve rotational and translational symmetry, we use an E(3) equivariant encoder:**

$$
\text{Enc}(\mathbf{M}) = \text{EGNN}(\mathbf{M}) = \text{Pooling}\left(\{\mathbf{h}_j^{(L)}\}_{j=1}^{n_{mol}}\right)
$$

where $\mathbf{h}_j^{(L)}$ は $L$ 層のEGNNを通した後の特徴ベクトル。

**EGNN層の更新式：**

**EGNN layer update equations:**

$$
\begin{align}
\mathbf{m}_{ij} &= \phi_e(\mathbf{h}_i, \mathbf{h}_j, \|\mathbf{r}_{ij}\|^2, \mathbf{e}_{ij}) \\
\mathbf{r}_i' &= \mathbf{r}_i + \sum_{j \neq i} \mathbf{r}_{ij} \cdot \phi_x(\mathbf{m}_{ij}) \\
\mathbf{h}_i' &= \phi_h(\mathbf{h}_i, \sum_{j \neq i} \mathbf{m}_{ij})
\end{align}
$$

where:
- $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$: 相対位置ベクトル (Relative position vector)
- $\phi_e, \phi_x, \phi_h$: MLP関数 (MLP functions)
- $\mathbf{e}_{ij}$: エッジ特徴 (Edge features)

#### 3.1.2 グローバルプーリング (Global Pooling)

分子全体の情報を集約：

**Aggregate information from the entire molecule:**

$$
\mathbf{c}_{mol} = \frac{1}{n_{mol}} \sum_{j=1}^{n_{mol}} \mathbf{h}_j^{(L)} + \max_{j=1}^{n_{mol}} \mathbf{h}_j^{(L)}
$$

または、より高度なアテンション機構：

**Or using a more sophisticated attention mechanism:**

$$
\begin{align}
\mathbf{w}_j &= \text{softmax}(\mathbf{q}^\top \mathbf{h}_j^{(L)}) \\
\mathbf{c}_{mol} &= \sum_{j=1}^{n_{mol}} \mathbf{w}_j \mathbf{h}_j^{(L)}
\end{align}
$$

where $\mathbf{q}$ は学習可能なクエリベクトル (learnable query vector)。

### 3.2 条件付きノイズ予測 (Conditional Noise Prediction)

ノイズ予測ネットワークは、分子コンテキストベクトルを入力に含めます：

**The noise prediction network includes the molecular context vector as input:**

$$
\boldsymbol{\epsilon}_\theta(\mathbf{C}_t, t, \mathbf{M}) = \boldsymbol{\epsilon}_\theta(\mathbf{X}_t, \mathbf{H}_t, \mathbf{L}_t, t, \mathbf{c}_{mol})
$$

#### 3.2.1 条件付け方法 (Conditioning Methods)

**方法1: 加算的条件付け (Additive Conditioning)**

$$
\mathbf{h}_i^{(cond)} = \mathbf{h}_i + \text{MLP}(\mathbf{c}_{mol})
$$

**方法2: FiLM条件付け (Feature-wise Linear Modulation)**

$$
\begin{align}
\boldsymbol{\gamma}, \boldsymbol{\beta} &= \text{MLP}(\mathbf{c}_{mol}) \\
\mathbf{h}_i^{(cond)} &= \boldsymbol{\gamma} \odot \mathbf{h}_i + \boldsymbol{\beta}
\end{align}
$$

where $\odot$ は要素ごとの積 (element-wise product)。

**方法3: クロスアテンション (Cross-Attention)**

$$
\begin{align}
\mathbf{Q} &= \mathbf{H}_t \mathbf{W}_Q \\
\mathbf{K} &= \mathbf{c}_{mol} \mathbf{W}_K \\
\mathbf{V} &= \mathbf{c}_{mol} \mathbf{W}_V \\
\mathbf{H}_{cond} &= \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
\end{align}
$$

---

## 4. 周期境界条件下の条件付け (Conditioning under Periodic Boundary Conditions)

### 4.1 周期的距離計算 (Periodic Distance Calculation)

結晶構造では、最小イメージ規約を用いた距離計算が必要：

**For crystal structures, we need distance calculation using the minimum image convention:**

$$
\mathbf{r}_{ij}^{(pbc)} = \mathbf{L} \cdot \left[\text{frac}(\mathbf{L}^{-1} (\mathbf{r}_j - \mathbf{r}_i)) - \text{round}(\text{frac}(\mathbf{L}^{-1} (\mathbf{r}_j - \mathbf{r}_i)))\right]
$$

where:
- $\text{frac}(\cdot)$: 分数座標への変換 (Conversion to fractional coordinates)
- $\text{round}(\cdot)$: 最も近い整数への丸め (Rounding to nearest integer)
- $\mathbf{L}$: 格子ベクトル行列 (Lattice vector matrix)

距離は以下で計算：

**Distance is calculated as:**

$$
d_{ij}^{(pbc)} = \|\mathbf{r}_{ij}^{(pbc)}\|_2
$$

### 4.2 周期的E(3)等変性 (Periodic E(3) Equivariance)

周期境界条件下では、並進対称性が離散的になります：

**Under periodic boundary conditions, translational symmetry becomes discrete:**

$$
\mathbf{r}_i' = \mathbf{r}_i + \mathbf{n}_1 \mathbf{a} + \mathbf{n}_2 \mathbf{b} + \mathbf{n}_3 \mathbf{c}, \quad \mathbf{n} = (n_1, n_2, n_3) \in \mathbb{Z}^3
$$

モデルは以下の性質を満たす必要があります：

**The model must satisfy the following properties:**

$$
f(\mathbf{X} + \mathbf{T}_{\mathbf{n}}, \mathbf{L}) = f(\mathbf{X}, \mathbf{L}) + \mathbf{T}_{\mathbf{n}}
$$

where $\mathbf{T}_{\mathbf{n}} = \mathbf{n}_1 \mathbf{a} + \mathbf{n}_2 \mathbf{b} + \mathbf{n}_3 \mathbf{c}$ は格子並進 (lattice translation)。

---

## 5. 学習目的関数 (Training Objective)

### 5.1 変分下限 (Variational Lower Bound)

条件付き生成の学習目的関数は以下の変分下限を最大化：

**The training objective for conditional generation maximizes the following variational lower bound:**

$$
\begin{align}
\mathcal{L}_{VLB}(\theta) &= \mathbb{E}_{(\mathbf{M}, \mathbf{C}_0) \sim \mathcal{D}} \left[ \log p_\theta(\mathbf{C}_0 | \mathbf{M}) \right] \\
&\geq \mathbb{E}_{(\mathbf{M}, \mathbf{C}_0) \sim \mathcal{D}} \left[ \mathbb{E}_{q} \left[ -\sum_{t=1}^T D_{KL}(q(\mathbf{C}_{t-1} | \mathbf{C}_t, \mathbf{C}_0) \| p_\theta(\mathbf{C}_{t-1} | \mathbf{C}_t, \mathbf{M})) \right] \right]
\end{align}
$$

### 5.2 簡略化された目的関数 (Simplified Objective)

実際には、以下の簡略化されたノイズ予測損失を使用：

**In practice, we use the following simplified noise prediction loss:**

$$
\mathcal{L}_{simple}(\theta) = \mathbb{E}_{(\mathbf{M}, \mathbf{C}_0) \sim \mathcal{D}, t \sim \mathcal{U}(1, T), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{C}_t, t, \mathbf{M})\|^2 \right]
$$

### 5.3 複合損失関数 (Composite Loss Function)

結晶構造の各要素に対する損失を組み合わせ：

**Combine losses for each component of the crystal structure:**

$$
\mathcal{L}_{total} = \lambda_X \mathcal{L}_X + \lambda_H \mathcal{L}_H + \lambda_L \mathcal{L}_L + \lambda_{consist} \mathcal{L}_{consist}
$$

where:
- $\mathcal{L}_X$: 原子座標の損失 (Atomic coordinate loss)
- $\mathcal{L}_H$: 原子特徴の損失 (Atomic feature loss)
- $\mathcal{L}_L$: 格子パラメータの損失 (Lattice parameter loss)
- $\mathcal{L}_{consist}$: 分子一致性損失 (Molecular consistency loss)

#### 5.3.1 分子一致性損失 (Molecular Consistency Loss)

生成された結晶内の分子が入力分子と一致することを保証：

**Ensure that molecules in the generated crystal match the input molecule:**

$$
\mathcal{L}_{consist} = \sum_{k=1}^{Z} d_{mol}(\mathbf{M}_k^{(gen)}, \mathbf{M})
$$

where:
- $\mathbf{M}_k^{(gen)}$: 生成された結晶から抽出された $k$ 番目の分子 (The $k$-th molecule extracted from the generated crystal)
- $d_{mol}(\cdot, \cdot)$: 分子間距離メトリック (Intermolecular distance metric)

分子間距離は以下で定義：

**Intermolecular distance is defined as:**

$$
d_{mol}(\mathbf{M}_1, \mathbf{M}_2) = \min_{\mathbf{R} \in SO(3), \mathbf{t} \in \mathbb{R}^3, \pi \in S_{n_{mol}}} \sum_{j=1}^{n_{mol}} \|\mathbf{r}_j^{(1)} - (\mathbf{R} \mathbf{r}_{\pi(j)}^{(2)} + \mathbf{t})\|^2
$$

where:
- $\mathbf{R} \in SO(3)$: 回転行列 (Rotation matrix)
- $\mathbf{t} \in \mathbb{R}^3$: 並進ベクトル (Translation vector)
- $\pi \in S_{n_{mol}}$: 原子の並び替え (Permutation of atoms)

---

## 6. サンプリング手順 (Sampling Procedure)

### 6.1 条件付きサンプリング (Conditional Sampling)

与えられた単分子 $\mathbf{M}$ から結晶を生成：

**Generate a crystal from a given single molecule $\mathbf{M}$:**

**アルゴリズム: 条件付き結晶生成**

**Algorithm: Conditional Crystal Generation**

```
Input: 単分子 M, ノイズスケジュール {α_t}_{t=1}^T
Output: 生成された結晶 C_0

1. c_mol ← Enc(M)                    // 分子エンコード
2. C_T ~ N(0, I)                     // ランダムノイズで初期化
3. for t = T down to 1:
4.     ε_θ ← NoisePredictor(C_t, t, c_mol)  // ノイズ予測
5.     μ_θ ← (1/√α_t)(C_t - (1-α_t)/√(1-ᾱ_t) ε_θ)  // 平均計算
6.     if t > 1:
7.         z ~ N(0, I)               // ランダムノイズ
8.         C_{t-1} ← μ_θ + √(1-α_t) z
9.     else:
10.        C_0 ← μ_θ
11. return C_0
```

### 6.2 ガイダンススケール (Guidance Scale)

条件付けの強度を調整するため、classifier-free guidanceを使用：

**To adjust the strength of conditioning, use classifier-free guidance:**

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{C}_t, t, \mathbf{M}) = (1 + w) \boldsymbol{\epsilon}_\theta(\mathbf{C}_t, t, \mathbf{M}) - w \boldsymbol{\epsilon}_\theta(\mathbf{C}_t, t, \emptyset)
$$

where:
- $w$: ガイダンススケール (Guidance scale)
- $\emptyset$: 無条件（分子情報なし）(Unconditional, no molecule information)

大きな $w$ は入力分子への忠実度を高めますが、多様性を低下させます。

**Larger $w$ increases fidelity to the input molecule but reduces diversity.**

### 6.3 分子配置の初期化 (Initial Molecular Placement)

効率的な生成のため、初期ノイズに分子情報を部分的に埋め込み：

**For efficient generation, partially embed molecular information in the initial noise:**

$$
\mathbf{C}_T = \alpha_{init} \cdot \text{TileM olecule}(\mathbf{M}, Z) + (1 - \alpha_{init}) \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

where:
- $\text{TileMolecule}(\mathbf{M}, Z)$: 分子を $Z$ 回複製して配置 (Replicate and place the molecule $Z$ times)
- $\alpha_{init} \in [0, 1]$: 初期化の強度 (Initialization strength)

---

## 7. 評価指標 (Evaluation Metrics)

### 7.1 分子一致度 (Molecular Consistency)

生成された結晶から分子を抽出し、入力分子との一致度を測定：

**Extract molecules from the generated crystal and measure consistency with the input molecule:**

$$
\text{MolConsistency} = \frac{1}{N} \sum_{i=1}^N \frac{1}{Z} \sum_{k=1}^Z \mathbb{I}[d_{mol}(\mathbf{M}_k^{(i)}, \mathbf{M}^{(i)}) < \tau]
$$

where:
- $\mathbb{I}[\cdot]$: 指示関数 (Indicator function)
- $\tau$: 一致判定の閾値 (Threshold for consistency)

### 7.2 構造的類似度 (Structural Similarity)

参照結晶との構造的類似度：

**Structural similarity with reference crystals:**

**格子パラメータの誤差 (Lattice Parameter Error):**

$$
\text{LatticeMAE} = \frac{1}{6} \sum_{p \in \{a,b,c,\alpha,\beta,\gamma\}} |\hat{p} - p|
$$

**動径分布関数の類似度 (Radial Distribution Function Similarity):**

$$
\text{RDF-Similarity} = 1 - \frac{\int |g_{gen}(r) - g_{ref}(r)| dr}{\int (g_{gen}(r) + g_{ref}(r)) dr}
$$

### 7.3 物理的妥当性 (Physical Validity)

**最小原子間距離 (Minimum Interatomic Distance):**

$$
d_{min} = \min_{i \neq j} d_{ij}^{(pbc)}(\mathbf{X}, \mathbf{L})
$$

妥当な結晶: $d_{min} > d_{threshold}$ (例: 0.7 Å)

**Valid crystal: $d_{min} > d_{threshold}$ (e.g., 0.7 Å)**

**パッキング効率 (Packing Efficiency):**

$$
\eta = \frac{V_{molecules}}{V_{cell}} = \frac{Z \cdot V_{mol}}{|\det(\mathbf{L})|}
$$

where:
- $V_{mol}$: 単一分子の体積 (Volume of a single molecule)
- $|\det(\mathbf{L})|$: 単位格子の体積 (Unit cell volume)

---

## 8. 実装の詳細 (Implementation Details)

### 8.1 ネットワークアーキテクチャ (Network Architecture)

**分子エンコーダ:**

**Molecule Encoder:**

```
Input: M = {(r_j, h_j)}_{j=1}^{n_mol}
↓
EGNN Layers (L=6 layers, hidden_dim=256)
↓
Global Pooling (mean + max)
↓
MLP (256 → 128 → 128)
↓
Output: c_mol ∈ R^{128}
```

**条件付き結晶ダイナミクス:**

**Conditional Crystal Dynamics:**

```
Input: C_t, t, c_mol
↓
Time Embedding (t → t_emb ∈ R^{128})
↓
Molecular Conditioning (FiLM: c_mol → γ, β)
↓
Periodic EGNN (L=9 layers, hidden_dim=256)
  ├─ Node features: h_i' = γ ⊙ h_i + β
  ├─ Message passing with periodic boundary
  └─ Coordinate update with minimum image
↓
Lattice MLP (for lattice parameter evolution)
↓
Output: ε_θ(C_t, t, M)
```

### 8.2 学習設定 (Training Configuration)

**ハイパーパラメータ:**

**Hyperparameters:**

- バッチサイズ (Batch size): 32
- 学習率 (Learning rate): $1 \times 10^{-4}$
- 拡散ステップ数 (Diffusion steps): $T = 1000$
- ノイズスケジュール (Noise schedule): $\beta_t = 0.0001 + 0.02 \cdot (t/T)^2$
- 損失重み (Loss weights): $\lambda_X = 1.0, \lambda_H = 1.0, \lambda_L = 0.1, \lambda_{consist} = 0.5$
- ガイダンススケール (Guidance scale): $w = 2.0$

### 8.3 データ拡張 (Data Augmentation)

**分子の回転・並進:**

**Molecular Rotation and Translation:**

訓練時、各分子をランダムに回転・並進させて拡張：

**During training, augment each molecule with random rotation and translation:**

$$
\mathbf{M}_{aug} = \{\mathbf{R} \mathbf{r}_j^{(mol)} + \mathbf{t}\}_{j=1}^{n_{mol}}, \quad \mathbf{R} \sim SO(3), \mathbf{t} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})
$$

これによりモデルの回転・並進不変性を強化。

**This enhances the model's rotational and translational invariance.**

---

## 9. 理論的保証 (Theoretical Guarantees)

### 9.1 E(3)等変性の保持 (Preservation of E(3) Equivariance)

**定理1:** 分子エンコーダがE(3)等変であり、結晶ダイナミクスが周期的E(3)等変である場合、全体のモデルは回転・並進に対して共変です。

**Theorem 1:** If the molecule encoder is E(3) equivariant and the crystal dynamics is periodically E(3) equivariant, the overall model is covariant with respect to rotation and translation.

**証明スケッチ:**

**Proof Sketch:**

分子 $\mathbf{M}$ を回転 $\mathbf{R}$ と並進 $\mathbf{t}$ で変換：

**Transform molecule $\mathbf{M}$ by rotation $\mathbf{R}$ and translation $\mathbf{t}$:**

$$
\mathbf{M}' = \{(\mathbf{R} \mathbf{r}_j + \mathbf{t}, \mathbf{h}_j)\}_{j=1}^{n_{mol}}
$$

E(3)等変性により：

**By E(3) equivariance:**

$$
\text{Enc}(\mathbf{M}') = \text{Enc}(\mathbf{M}) = \mathbf{c}_{mol}
$$

（回転・並進は不変な表現に写像される）

**(Rotation and translation are mapped to an invariant representation)**

生成された結晶も対応する変換を受けます：

**The generated crystal also undergoes the corresponding transformation:**

$$
\mathbf{C}' = \{(\mathbf{R} \mathbf{x}_i + \mathbf{t}, \mathbf{h}_i), \mathbf{R} \mathbf{L}\}
$$

### 9.2 収束性 (Convergence)

**定理2:** 十分な訓練データとモデル容量があれば、提案手法は真の条件付き分布 $p(\mathbf{C} | \mathbf{M})$ に収束します。

**Theorem 2:** With sufficient training data and model capacity, the proposed method converges to the true conditional distribution $p(\mathbf{C} | \mathbf{M})$.

これはスコアベース生成モデルの理論的保証に基づきます。

**This is based on the theoretical guarantees of score-based generative models.**

---

## 10. 制限事項と今後の課題 (Limitations and Future Work)

### 10.1 現在の制限事項 (Current Limitations)

1. **Z値の固定 (Fixed Z value)**
   - 現在、単位格子あたりの分子数 $Z$ は訓練データから推定
   - 異なる $Z$ 値での生成には追加の条件付けが必要

2. **空間群制約なし (No space group constraints)**
   - Phase 1では空間群対称性を明示的に強制しない
   - 生成後に対称性を適用可能

3. **分子の変形 (Molecular deformation)**
   - 結晶環境での分子の微小な変形を許容
   - 完全な剛体制約は課さない

### 10.2 今後の拡張 (Future Extensions)

1. **多分子系への拡張 (Extension to multi-molecule systems)**
   $$
   \mathbf{C} = \bigcup_{m=1}^{M} \bigcup_{k=1}^{Z_m} \mathbf{M}_m
   $$
   複数種類の分子から成る共結晶の生成

2. **対称性制約下での生成 (Generation under symmetry constraints)**
   $$
   p_\theta(\mathbf{C} | \mathbf{M}, \text{space\_group}) = p_\theta(\mathbf{C} | \mathbf{M}) \cdot \mathbb{I}[\mathbf{C} \in \mathcal{G}]
   $$
   where $\mathcal{G}$ は指定された空間群に属する構造の集合

3. **物性予測との統合 (Integration with property prediction)**
   $$
   \mathbf{M}^* = \arg\max_{\mathbf{M}} \mathbb{E}_{\mathbf{C} \sim p_\theta(\cdot | \mathbf{M})} [f_{property}(\mathbf{C})]
   $$
   目的物性を最大化する分子の逆設計

---

## 11. 結論 (Conclusion)

本理論説明書では、単分子情報を条件とした分子性結晶生成の数学的定式化を提供しました。主な貢献は以下の通りです：

**This theoretical documentation provided a mathematical formulation for molecular crystal generation conditioned on single molecule information. The main contributions are:**

1. **条件付き拡散過程の定式化**
   - 単分子から結晶への写像
   - E(3)等変な分子エンコーダ
   - 周期境界条件下での拡散

2. **学習目的関数の導出**
   - 変分下限
   - 分子一致性損失
   - 複合損失関数

3. **サンプリング手順の確立**
   - 条件付きサンプリングアルゴリズム
   - ガイダンススケールの導入
   - 効率的な初期化手法

4. **評価指標の定義**
   - 分子一致度
   - 構造的類似度
   - 物理的妥当性

この理論的基盤に基づき、実装された手法は単分子から物理的に妥当なホモ結晶を生成できます。

**Based on this theoretical foundation, the implemented method can generate physically valid homo-crystals from single molecules.**

---

## 参考文献 (References)

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.

2. Hoogeboom, E., et al. (2022). Equivariant diffusion for molecule generation in 3D. *ICML*.

3. Satorras, V. G., et al. (2021). E(n) equivariant graph neural networks. *ICML*.

4. Jiao, R., et al. (2023). Crystal diffusion variational autoencoder for periodic material generation. *arXiv*.

5. Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *NeurIPS Workshop*.

6. Allen, F. H. (2002). The Cambridge Structural Database. *Acta Crystallographica Section B*.

---

## 付録: 数式記法一覧 (Appendix: Mathematical Notation)

| 記号 | 説明 |
|------|------|
| $\mathbf{M}$ | 単分子構造 (Single molecule structure) |
| $\mathbf{C}$ | 結晶構造 (Crystal structure) |
| $\mathbf{X}, \mathbf{H}, \mathbf{L}$ | 原子座標、特徴、格子ベクトル (Atomic coordinates, features, lattice vectors) |
| $\mathbf{c}_{mol}$ | 分子コンテキストベクトル (Molecular context vector) |
| $Z$ | 単位格子内の分子数 (Number of molecules per unit cell) |
| $t$ | 拡散時間ステップ (Diffusion time step) |
| $\boldsymbol{\epsilon}_\theta$ | ノイズ予測ネットワーク (Noise prediction network) |
| $\alpha_t, \bar{\alpha}_t$ | ノイズスケジュール (Noise schedule) |
| $\text{Enc}(\cdot)$ | 分子エンコーダ (Molecule encoder) |
| $d_{mol}(\cdot, \cdot)$ | 分子間距離 (Intermolecular distance) |
| $d^{(pbc)}$ | 周期境界条件下の距離 (Distance under periodic boundary conditions) |
| $\mathbf{R} \in SO(3)$ | 3次元回転群 (3D rotation group) |
| $\mathcal{N}(\mu, \sigma^2)$ | 正規分布 (Normal distribution) |
| $\mathbb{E}[\cdot]$ | 期待値 (Expectation) |
| $D_{KL}(\cdot \| \cdot)$ | KLダイバージェンス (KL divergence) |

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-11
