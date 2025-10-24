# 物性値を条件とした分子性結晶生成の詳細理論説明書

## 文書情報

- **文書タイトル**: 物性値を条件とした分子性結晶生成の詳細理論説明書
- **作成日**: 2025-10-24
- **バージョン**: 1.0
- **対象読者**: 研究者、開発者、データサイエンティスト
- **参照**: PR#143、doc/molecular_crystal_generation_spec.md Q4

---

## 目次

1. [概要](#概要)
2. [理論的背景](#理論的背景)
3. [数学的定式化](#数学的定式化)
4. [条件付き拡散モデルの理論](#条件付き拡散モデルの理論)
5. [物性値条件付けの理論](#物性値条件付けの理論)
6. [等変性の保証](#等変性の保証)
7. [結論](#結論)

---

## 概要

### 背景

分子性結晶は、医薬品、有機半導体、エネルギー貯蔵材料など、多くの産業応用において重要な役割を果たしています。これらの応用において、結晶の物性値（バンドギャップ、融点、機械的性質など）は、材料の機能を決定する重要な要因です。

従来の結晶設計では、以下の課題がありました：

1. **試行錯誤的なアプローチ**: 目標物性値を達成するために、多数の結晶を合成・評価する必要がある
2. **逆問題の困難性**: 「どのような結晶構造が目標物性値を実現するか」という逆問題を解くことは極めて困難
3. **計算コスト**: 第一原理計算による物性値予測は計算コストが高く、大規模探索には不向き

### 本研究の目的

本研究では、E(3)等変拡散モデルを用いて、**物性値を直接的な条件として分子性結晶を生成する手法**を提案します。これにより、以下を実現します：

- **目標指向設計**: 目標物性値を満たす結晶を直接生成
- **エンドツーエンド学習**: 分子構造、結晶構造、物性値の関係を深層学習で獲得
- **高速生成**: 訓練後は高速に多様な結晶候補を生成可能

### 主要な貢献

1. **物性値条件付け機構の導入**: PropertyConditioningモジュールによる物性値の条件付け
2. **多重条件の統合**: 分子条件、結晶条件、物性値条件を統一的に扱うフレームワーク
3. **理論的正当性**: ヒューリスティックなfallbackを使用せず、理論的に正しい生成を保証
4. **等変性の保証**: E(3)対称性を保ちながら物性値条件付けを実現

---

## 理論的背景

### 拡散モデルの基礎理論

#### 順拡散過程（Forward Diffusion Process）

データ分布 $p_{\text{data}}(\mathbf{x})$ から出発し、徐々にノイズを加えていく過程を考えます：

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

ここで：
- $\mathbf{x}_0$: 元のデータ（結晶構造）
- $\mathbf{x}_t$: 時刻 $t$ でのノイズ付きデータ
- $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$: ノイズスケジュール
- $\beta_t$: 時刻 $t$ でのノイズ量

#### 逆拡散過程（Reverse Diffusion Process）

ノイズから元のデータを復元する過程：

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

ここで、$\boldsymbol{\mu}_\theta$ はニューラルネットワークでパラメータ化された平均関数です。

#### スコアマッチングによる訓練

拡散モデルは、スコア関数 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ を学習します：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

ここで：
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 真のノイズ
- $\boldsymbol{\epsilon}_\theta$: ネットワークで予測されたノイズ
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$

### 条件付き拡散モデル

#### 条件付き生成の定式化

条件 $\mathbf{c}$ が与えられたときの条件付き分布を学習します：

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{c})
$$

条件付きスコア関数は：

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}_t | \mathbf{c}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}} \log p(\mathbf{c} | \mathbf{x}_t)
$$

#### 条件付き訓練目的関数

$$
\mathcal{L}_{\text{cond}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}, \mathbf{c}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right\|^2 \right]
$$

### E(3)等変性

#### E(3)群

E(3)はユークリッド空間における等長変換の群で、以下を含みます：

- **回転**: $\mathbf{R} \in SO(3)$
- **並進**: $\mathbf{t} \in \mathbb{R}^3$
- **反転**: 鏡映変換

変換 $g = (\mathbf{R}, \mathbf{t}) \in E(3)$ による作用：

$$
g \cdot \mathbf{x} = \mathbf{R} \mathbf{x} + \mathbf{t}
$$

#### 等変性の定義

関数 $f: \mathcal{X} \rightarrow \mathcal{Y}$ が E(3) 等変であるとは：

$$
f(g \cdot \mathbf{x}) = g \cdot f(\mathbf{x}), \quad \forall g \in E(3)
$$

物理的には、座標系の選択に依存しない性質を表します。

#### 結晶構造における等変性

結晶構造 $\mathbf{X} = \{\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N\}$ に対して：

$$
f(\mathbf{R} \mathbf{X} + \mathbf{t}) = \mathbf{R} f(\mathbf{X}) + \mathbf{t}
$$

これにより、座標系によらない一貫した予測が可能になります。

---

## 数学的定式化

### 問題設定

#### 入力

1. **分子条件** $\mathbf{c}_{\text{mol}}$:
   - 分子量: $M_w \in \mathbb{R}_+$
   - π共役比率: $\pi_r \in [0, 1]$
   - 原子タイプエンコーディング: $\mathbf{a} \in \{0, 1\}^{|\mathcal{A}|}$
   - 官能基エンコーディング: $\mathbf{f} \in \{0, 1\}^{|\mathcal{F}|}$

2. **結晶条件** $\mathbf{c}_{\text{crys}}$:
   - 空間群: $\text{SG} \in \{1, 2, \ldots, 230\}$
   - 密度: $\rho \in \mathbb{R}_+$
   - 格子パラメータ: $\mathbf{L} = (a, b, c, \alpha, \beta, \gamma) \in \mathbb{R}^6$

3. **物性値条件** $\mathbf{c}_{\text{prop}}$:
   - バンドギャップ: $E_g \in \mathbb{R}_+$
   - 融点: $T_m \in \mathbb{R}_+$
   - その他の物性値: $\mathbf{p} \in \mathbb{R}^d$

#### 出力

分子性結晶構造 $\mathbf{X}$:
- 原子座標: $\mathbf{R} = \{\mathbf{r}_1, \mathbf{r}_2, \ldots, \mathbf{r}_N\} \in \mathbb{R}^{N \times 3}$
- 原子タイプ: $\mathbf{Z} = \{z_1, z_2, \ldots, z_N\} \in \mathbb{Z}^N$
- 格子ベクトル: $\mathbf{H} = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3] \in \mathbb{R}^{3 \times 3}$

### 条件付き分布

目標は、以下の条件付き分布からサンプリングすることです：

$$
p(\mathbf{X} | \mathbf{c}_{\text{mol}}, \mathbf{c}_{\text{crys}}, \mathbf{c}_{\text{prop}})
$$

これを拡散モデルで近似します：

$$
p_\theta(\mathbf{X} | \mathbf{c}) = \int p_\theta(\mathbf{X}_{0:T} | \mathbf{c}) d\mathbf{X}_{1:T}
$$

ここで、$\mathbf{c} = (\mathbf{c}_{\text{mol}}, \mathbf{c}_{\text{crys}}, \mathbf{c}_{\text{prop}})$ です。

### 条件付けベクトルの構築

各条件を埋め込みベクトルに変換し、統合します：

#### 1. 分子条件の埋め込み

分子の3D構造からEGNN特徴量を抽出：

$$
\mathbf{h}_{\text{mol}} = \text{MolecularEncoder}(\mathbf{X}_{\text{mol}})
$$

ここで：
- $\mathbf{X}_{\text{mol}}$: 分子の原子座標とタイプ
- $\mathbf{h}_{\text{mol}} \in \mathbb{R}^{d_{\text{mol}}}$: 分子特徴ベクトル

#### 2. 結晶条件の埋め込み

空間群、密度、格子パラメータをそれぞれ埋め込み：

$$
\mathbf{h}_{\text{crys}} = \text{CrystalConditioning}(\text{SG}, \rho, \mathbf{L})
$$

具体的には：

$$
\begin{align}
\mathbf{h}_{\text{sg}} &= \text{SpaceGroupEmbedding}(\text{SG}) \\
\mathbf{h}_{\rho} &= \text{DensityConditioning}(\rho) \\
\mathbf{h}_{\mathbf{L}} &= \text{LatticeConditioning}(\mathbf{L})
\end{align}
$$

#### 3. 物性値条件の埋め込み（新規）

物性値ベクトルを条件付けベクトルに変換：

$$
\mathbf{h}_{\text{prop}} = \text{PropertyConditioning}(\mathbf{c}_{\text{prop}})
$$

具体的には、多層パーセプトロン（MLP）で変換：

$$
\begin{align}
\mathbf{c}_{\text{prop}}^{\text{norm}} &= \frac{\mathbf{c}_{\text{prop}} - \boldsymbol{\mu}_{\text{prop}}}{\boldsymbol{\sigma}_{\text{prop}} + \epsilon} \\
\mathbf{h}_{\text{prop}} &= \text{MLP}_{\text{prop}}(\mathbf{c}_{\text{prop}}^{\text{norm}})
\end{align}
$$

ここで：
- $\boldsymbol{\mu}_{\text{prop}}, \boldsymbol{\sigma}_{\text{prop}}$: 訓練データから計算された平均と標準偏差
- $\epsilon = 10^{-8}$: 数値安定性のための小さな定数

#### 4. 統合条件付けベクトル

すべての条件を統合：

$$
\mathbf{h}_{\text{cond}} = \text{CombineConditionings}(\mathbf{h}_{\text{mol}}, \mathbf{h}_{\text{crys}}, \mathbf{h}_{\text{prop}})
$$

具体的には、連結後にMLPで処理：

$$
\mathbf{h}_{\text{cond}} = \text{MLP}_{\text{combine}}([\mathbf{h}_{\text{mol}} \,||\, \mathbf{h}_{\text{crys}} \,||\, \mathbf{h}_{\text{prop}}])
$$

ここで、$||$ は連結演算子です。

---

## 条件付き拡散モデルの理論

### 結晶構造の表現

結晶構造は以下で表現されます：

$$
\mathbf{X} = (\mathbf{R}, \mathbf{Z}, \mathbf{H})
$$

ここで：
- $\mathbf{R} \in \mathbb{R}^{N \times 3}$: 分数座標（fractional coordinates）
- $\mathbf{Z} \in \{1, 2, \ldots, Z_{\max}\}^N$: 原子番号
- $\mathbf{H} \in \mathbb{R}^{3 \times 3}$: 格子ベクトル

### 拡散過程

#### 原子座標の拡散

分数座標に対する拡散：

$$
\mathbf{R}_t = \sqrt{\bar{\alpha}_t} \mathbf{R}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_R
$$

ここで、$\boldsymbol{\epsilon}_R \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{N \times 3})$ です。

**重要**: 分数座標は周期境界条件下で定義されるため、$\mathbf{R}_t \mod 1$ の操作が必要です。

#### 格子ベクトルの拡散

格子ベクトルは対称正定値行列として扱います：

$$
\mathbf{G} = \mathbf{H}^T \mathbf{H} \in \mathbb{R}^{3 \times 3}
$$

これに対する拡散：

$$
\mathbf{G}_t = \sqrt{\bar{\alpha}_t} \mathbf{G}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_G
$$

### 条件付きスコアネットワーク

時刻 $t$ とノイズ付き構造 $\mathbf{X}_t$ から、ノイズを予測：

$$
\boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t, \mathbf{h}_{\text{cond}}) = (\boldsymbol{\epsilon}_R, \boldsymbol{\epsilon}_G)
$$

ネットワーク構造：

$$
\begin{align}
\mathbf{h}_t &= \text{TimeEmbedding}(t) \\
\mathbf{h}_{\text{node}} &= \text{EGNN}_{\text{encoder}}(\mathbf{R}_t, \mathbf{Z}, \mathbf{H}_t, \mathbf{h}_t, \mathbf{h}_{\text{cond}}) \\
\boldsymbol{\epsilon}_R &= \text{EGNN}_{\text{decoder}}^R(\mathbf{h}_{\text{node}}) \\
\boldsymbol{\epsilon}_G &= \text{EGNN}_{\text{decoder}}^G(\mathbf{h}_{\text{node}})
\end{align}
$$

### 訓練目的関数

完全な訓練目的関数：

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{X}_0, \boldsymbol{\epsilon}_R, \boldsymbol{\epsilon}_G, \mathbf{c}} \left[ \lambda_R \left\| \boldsymbol{\epsilon}_R - \boldsymbol{\epsilon}_\theta^R(\mathbf{X}_t, t, \mathbf{c}) \right\|^2 + \lambda_G \left\| \boldsymbol{\epsilon}_G - \boldsymbol{\epsilon}_\theta^G(\mathbf{X}_t, t, \mathbf{c}) \right\|^2 \right]
$$

ここで：
- $\lambda_R, \lambda_G$: 各項の重み
- $\mathbf{c} = (\mathbf{c}_{\text{mol}}, \mathbf{c}_{\text{crys}}, \mathbf{c}_{\text{prop}})$: 統合条件

### サンプリングアルゴリズム

逆拡散によるサンプリング（DDPM）：

**入力**: 条件 $\mathbf{c}$、訓練済みモデル $\boldsymbol{\epsilon}_\theta$  
**出力**: 結晶構造 $\mathbf{X}_0$

1. $\mathbf{X}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ をサンプリング
2. 条件付けベクトルを計算: $\mathbf{h}_{\text{cond}} = \text{Conditioning}(\mathbf{c})$
3. $t = T, T-1, \ldots, 1$ について：
   - ノイズを予測: $\boldsymbol{\epsilon} = \boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t, \mathbf{h}_{\text{cond}})$
   - 平均を計算:
     $$
     \boldsymbol{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{X}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)
     $$
   - 分散を計算: $\boldsymbol{\sigma}_t^2 = \beta_t$
   - サンプリング: $\mathbf{X}_{t-1} \sim \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\sigma}_t^2 \mathbf{I})$
4. $\mathbf{X}_0$ を返す

---

## 物性値条件付けの理論

### 物性値と結晶構造の関係

物性値は、結晶構造の複雑な関数です：

$$
\mathbf{p} = \mathcal{P}(\mathbf{X})
$$

ここで：
- $\mathbf{p} = (E_g, T_m, \ldots)$: 物性値ベクトル
- $\mathcal{P}$: 物性値関数（通常は複雑で非線形）

**課題**: $\mathcal{P}$ は通常、第一原理計算や実験でしか得られず、解析的に表現できない。

### PropertyConditioningの理論的根拠

PropertyConditioningモジュールは、以下の写像を学習します：

$$
\phi_{\text{prop}}: \mathbb{R}^{d_{\text{prop}}} \rightarrow \mathbb{R}^{d_{\text{cond}}}
$$

これにより、物性値空間から条件付けベクトル空間への埋め込みを実現します。

#### 学習プロセス

訓練データセット $\mathcal{D} = \{(\mathbf{X}_i, \mathbf{p}_i)\}_{i=1}^N$ が与えられたとき：

1. **順方向**: 結晶構造 $\mathbf{X}_i$ から物性値 $\mathbf{p}_i$ を計算（または測定）
2. **逆方向（学習）**: 物性値 $\mathbf{p}_i$ から条件付けベクトル $\mathbf{h}_{\text{prop},i}$ を生成
3. **生成**: $\mathbf{h}_{\text{prop},i}$ を条件として結晶 $\mathbf{X}_i'$ を生成
4. **検証**: $\mathcal{P}(\mathbf{X}_i') \approx \mathbf{p}_i$ となるように学習

#### 条件付き分布の近似

理想的には、以下を近似したい：

$$
p(\mathbf{X} | \mathbf{p}) = \frac{p(\mathbf{p} | \mathbf{X}) p(\mathbf{X})}{p(\mathbf{p})}
$$

拡散モデルでは、条件付きスコア関数を学習：

$$
\nabla_{\mathbf{X}} \log p(\mathbf{X}_t | \mathbf{p}) \approx \boldsymbol{\epsilon}_\theta(\mathbf{X}_t, t, \phi_{\text{prop}}(\mathbf{p}))
$$

### 多重条件の統合理論

複数の条件 $\mathbf{c} = (\mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_K)$ が与えられたとき、同時条件付き分布は：

$$
p(\mathbf{X} | \mathbf{c}_1, \mathbf{c}_2, \ldots, \mathbf{c}_K)
$$

**独立性仮定**下では：

$$
\log p(\mathbf{X} | \mathbf{c}_1, \ldots, \mathbf{c}_K) = \log p(\mathbf{X}) + \sum_{k=1}^K \log p(\mathbf{c}_k | \mathbf{X}) - \log Z
$$

スコア関数：

$$
\nabla_{\mathbf{X}} \log p(\mathbf{X} | \mathbf{c}_1, \ldots, \mathbf{c}_K) = \nabla_{\mathbf{X}} \log p(\mathbf{X}) + \sum_{k=1}^K \nabla_{\mathbf{X}} \log p(\mathbf{c}_k | \mathbf{X})
$$

**実装**: ExtendedCombinedConditioningは、各条件の埋め込みを統合し、スコアネットワークに渡します。

### 正規化の重要性

物性値は異なるスケールを持つため、正規化が不可欠です：

$$
\mathbf{c}_{\text{prop}}^{\text{norm}} = \frac{\mathbf{c}_{\text{prop}} - \boldsymbol{\mu}}{\boldsymbol{\sigma} + \epsilon}
$$

**理論的理由**:
1. **勾配の安定性**: 正規化により、すべての物性値が同等の影響を持つ
2. **学習効率**: 正規化により、最適化が高速化
3. **一般化性能**: 正規化により、訓練範囲外の物性値に対する外挿が改善

統計情報は訓練データから計算：

$$
\begin{align}
\boldsymbol{\mu} &= \frac{1}{N} \sum_{i=1}^N \mathbf{p}_i \\
\boldsymbol{\sigma}^2 &= \frac{1}{N} \sum_{i=1}^N (\mathbf{p}_i - \boldsymbol{\mu})^2
\end{align}
$$

---

## 等変性の保証

### E(3)等変性の必要性

結晶構造生成において、E(3)等変性は以下の理由で重要です：

1. **物理的一貫性**: 座標系の選択に依存しない予測
2. **データ効率**: 対称性を利用することで、少ないデータで学習可能
3. **一般化性能**: 訓練していない配置に対しても正しく予測

### 等変性の保証方法

#### 1. EGNN（E(n) Equivariant Graph Neural Network）

EGNNは、以下の不変量と等変量を使用：

**不変量**（座標系に依存しない）：
- 距離: $d_{ij} = \|\mathbf{r}_i - \mathbf{r}_j\|$
- 内積: $\langle \mathbf{h}_i, \mathbf{h}_j \rangle$

**等変量**（座標系の変換に従う）：
- 相対位置: $\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j$

メッセージ伝播：

$$
\begin{align}
\mathbf{m}_{ij} &= \phi_m(\mathbf{h}_i, \mathbf{h}_j, d_{ij}^2, \mathbf{e}_{ij}) \\
\mathbf{h}_i' &= \phi_h\left(\mathbf{h}_i, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}\right) \\
\mathbf{r}_i' &= \mathbf{r}_i + \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \phi_r(\mathbf{h}_i, \mathbf{h}_j) \mathbf{r}_{ij}
\end{align}
$$

ここで、$\phi_m, \phi_h, \phi_r$ はMLPです。

**等変性の証明**:

変換 $g = (\mathbf{R}, \mathbf{t})$ に対して：

$$
\begin{align}
d_{ij}(g \cdot \mathbf{r}) &= \|\mathbf{R}(\mathbf{r}_i - \mathbf{r}_j)\| = d_{ij}(\mathbf{r}) \quad \text{(不変)} \\
\mathbf{r}_{ij}(g \cdot \mathbf{r}) &= \mathbf{R}(\mathbf{r}_i - \mathbf{r}_j) = \mathbf{R} \mathbf{r}_{ij}(\mathbf{r}) \quad \text{(等変)}
\end{align}
$$

したがって：

$$
\mathbf{r}_i'(g \cdot \mathbf{r}) = g \cdot \mathbf{r}_i'(\mathbf{r})
$$

#### 2. 条件付けベクトルの扱い

**スカラー条件**（物性値、密度など）は**不変量**として扱います：

$$
\mathbf{h}_{\text{prop}}(g \cdot \mathbf{X}) = \mathbf{h}_{\text{prop}}(\mathbf{X})
$$

これは、物性値が座標系に依存しないためです。

**分子特徴量**は、MolecularEncoderで不変特徴を抽出：

$$
\mathbf{h}_{\text{mol}}(g \cdot \mathbf{X}_{\text{mol}}) = \mathbf{h}_{\text{mol}}(\mathbf{X}_{\text{mol}})
$$

#### 3. 周期境界条件下での等変性

結晶構造では、並進対称性も考慮する必要があります。

最小イメージ規約（Minimum Image Convention）：

$$
\mathbf{r}_{ij}^{\text{mic}} = \mathbf{H} \cdot \text{round}(\mathbf{H}^{-1} (\mathbf{r}_i - \mathbf{r}_j))
$$

ここで、$\text{round}$ は最近接整数への丸め操作です。

これにより、周期境界を越えた相互作用を正しく計算できます。

### 格子ベクトルの扱い

格子ベクトル $\mathbf{H}$ は、座標変換に対して以下のように変換されます：

$$
\mathbf{H}' = \mathbf{R} \mathbf{H}
$$

したがって、グラム行列 $\mathbf{G} = \mathbf{H}^T \mathbf{H}$ は**不変量**です：

$$
\mathbf{G}' = (\mathbf{R} \mathbf{H})^T (\mathbf{R} \mathbf{H}) = \mathbf{H}^T \mathbf{R}^T \mathbf{R} \mathbf{H} = \mathbf{H}^T \mathbf{H} = \mathbf{G}
$$

モデルは $\mathbf{G}$ を予測し、生成時にCholesky分解で $\mathbf{H}$ を復元します：

$$
\mathbf{H} = \text{Cholesky}(\mathbf{G})
$$

---

## 結論

### 理論的貢献のまとめ

本理論説明書では、物性値を条件とした分子性結晶生成の理論的基盤を示しました：

1. **条件付き拡散モデルの拡張**
   - 従来の分子・結晶条件に加え、物性値条件を統合
   - PropertyConditioningによる物性値空間から条件付けベクトル空間への写像

2. **多重条件の統一的扱い**
   - ExtendedCombinedConditioningによる複数条件の統合
   - 理論的には独立性仮定下でのスコア関数の加法性

3. **E(3)等変性の保証**
   - EGNNによる座標系に依存しない予測
   - 周期境界条件下での等変性の維持

4. **正規化による安定化**
   - 異なるスケールの物性値を統一的に扱う
   - 学習の安定性と一般化性能の向上

### 理論的正当性

本手法は、以下の点で理論的に正当です：

- **ヒューリスティックなfallbackを使用しない**: すべての処理が理論的に定義され、根拠がある
- **等変性の厳密な保証**: E(3)対称性を数学的に証明
- **条件付き分布の近似**: 拡散モデルによる厳密な確率的定式化

### 今後の理論的展開

以下の理論的拡張が考えられます：

1. **ベイズ的不確実性定量化**
   - 生成された結晶の物性値の不確実性を推定
   - アンサンブル手法による信頼区間の計算

2. **逆問題の理論**
   - 物性値から結晶構造への逆写像の一意性と安定性
   - 正則化手法の理論的解析

3. **多目的最適化**
   - 複数の物性値を同時に最適化する理論
   - パレート最適解の探索

4. **転移学習の理論**
   - 少ないデータでの物性値条件付けの学習
   - ドメイン適応の理論的基盤

---

## 参考文献

### 拡散モデル

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
2. Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS.
3. Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. NeurIPS.

### 等変ニューラルネットワーク

4. Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. ICML.
5. Köhler, J., Klein, L., & Noé, F. (2020). Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities. ICML.

### 分子・結晶生成

6. Hoogeboom, E., et al. (2022). Equivariant Diffusion for Molecule Generation in 3D. ICML.
7. Xu, M., et al. (2021). GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation. ICLR.
8. Jiao, R., et al. (2023). Crystal Structure Prediction by Joint Equivariant Diffusion. NeurIPS.

### 物性予測

9. Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. Physical Review Letters.
10. Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. NeurIPS.

---

**文書作成者**: E(3)等変拡散モデル研究チーム  
**最終更新日**: 2025-10-24  
**バージョン**: 1.0  
**ステータス**: 承認済み
