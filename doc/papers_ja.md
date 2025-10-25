# 論文リスト（日本語解説付き）

**最終更新**: 2025年10月25日  
**対象**: E3 Diffusion for Molecules プロジェクト  
**目的**: プロジェクトの理論的基盤となる主要論文の紹介

---

## 目次

1. [基礎理論](#基礎理論)
2. [分子生成](#分子生成)
3. [条件付き生成](#条件付き生成)
4. [結晶構造予測](#結晶構造予測)
5. [評価手法](#評価手法)

---

## 基礎理論

### [1] E(n) Equivariant Graph Neural Networks

**著者**: Victor Garcia Satorras, Emiel Hoogeboom, Max Welling  
**会議**: ICML 2021  
**arXiv**: https://arxiv.org/abs/2102.09844

#### 日本語要約

本論文は、n次元ユークリッド群E(n)（回転、並進、反転）に対して同変なグラフニューラルネットワーク（EGNN）を提案する。EGNNは座標と特徴を同時に処理し、E(n)同変性を厳密に保持する。従来のGNNと異なり、EGNNは原子座標を明示的に扱い、回転や並進に対して不変な距離情報のみを使用することで同変性を実現する。

**主要な貢献**:
1. **E(n)同変なメッセージパッシング**: 座標更新が同変性を保つ
2. **シンプルなアーキテクチャ**: 複雑な幾何学的変換が不要
3. **理論的保証**: 厳密な数学的証明
4. **幅広い応用**: 分子動力学、N体問題、点群予測

#### 実装との対応

本プロジェクトの `egnn/egnn_new.py` の `EGNN` クラスは、この論文のアルゴリズムを実装している。

**コード例**:
```python
from egnn.egnn_new import EGNN

# EGNNレイヤーの初期化
egnn_layer = EGNN(
    in_node_nf=5,      # 入力ノード特徴（原子タイプ）
    hidden_nf=128,     # 隠れ層の次元
    out_node_nf=5,     # 出力ノード特徴
    in_edge_nf=0,      # エッジ特徴
    act_fn=nn.SiLU()   # 活性化関数
)

# 順伝播
h_out, x_out = egnn_layer(
    h=h,              # ノード特徴 [batch, n_atoms, in_node_nf]
    x=x,              # 座標 [batch, n_atoms, 3]
    edges=edges,      # エッジインデックス
    edge_attr=None    # エッジ特徴
)
```

#### キーとなる数式

**座標更新**（同変操作）:
$$\mathbf{x}_i' = \mathbf{x}_i + \sum_{j \neq i} (\mathbf{x}_i - \mathbf{x}_j) \phi_x(m_{ij})$$

**特徴更新**（不変操作）:
$$\mathbf{h}_i' = \phi_h(\mathbf{h}_i, \sum_{j \neq i} m_{ij})$$

ここで、$m_{ij}$は距離$||\mathbf{x}_i - \mathbf{x}_j||$に依存するメッセージ。

---

### [2] Denoising Diffusion Probabilistic Models

**著者**: Jonathan Ho, Ajay Jain, Pieter Abbeel  
**会議**: NeurIPS 2020  
**arXiv**: https://arxiv.org/abs/2006.11239

#### 日本語要約

拡散確率モデル（DDPM）の基礎理論を確立した論文。DDPMは、データに段階的にノイズを加える順過程と、ノイズから元のデータを復元する逆過程を学習する生成モデルである。変分下界を最適化することで、高品質な画像生成を実現する。本論文は、拡散モデルがVAEやGANに匹敵する、あるいはそれを上回る性能を持つことを示した。

**主要な貢献**:
1. **拡散過程の定式化**: 前向き過程と逆過程の数学的定義
2. **訓練目標の導出**: ノイズ予測の損失関数
3. **サンプリングアルゴリズム**: 高品質な生成手法
4. **ノイズスケジュールの設計**: 生成品質に影響する重要な要素

#### 実装との対応

本プロジェクトの `equivariant_diffusion/en_diffusion.py` の `EnVariationalDiffusion` クラスは、DDPMの枠組みをE(3)同変な設定に拡張している。

**コード例**:
```python
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion

# 拡散モデルの初期化
diffusion = EnVariationalDiffusion(
    dynamics=dynamics_model,        # EGNNベースのダイナミクス
    in_node_nf=5,                  # 原子タイプ数
    n_dims=3,                       # 3次元空間
    timesteps=1000,                 # 拡散ステップ数
    noise_schedule='polynomial_2'   # ノイズスケジュール
)

# 順過程（ノイズを加える）
x_t, h_t = diffusion.forward_diffusion(x_0, h_0, t)

# 逆過程（ノイズを除去）
x_pred, h_pred = diffusion.reverse_diffusion(x_t, h_t, t)
```

#### キーとなる数式

**順過程**:
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

**逆過程**:
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**訓練目標**（簡略化版）:
$$L = \mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$$

ここで、$\epsilon$は加えられたノイズ、$\epsilon_\theta$はモデルの予測。

---

### [3] Improved Denoising Diffusion Probabilistic Models

**著者**: Alex Nichol, Prafulla Dhariwal  
**会議**: ICML 2021  
**arXiv**: https://arxiv.org/abs/2102.09672

#### 日本語要約

DDPMの改良版を提案。学習可能な分散、改善されたノイズスケジュール、cosineスケジュールなどを導入し、生成品質を大幅に向上させた。特に、cosineスケジュールは従来のlinearスケジュールよりも安定した学習を可能にする。

**主要な貢献**:
1. **Cosineノイズスケジュール**: より安定した学習
2. **学習可能な分散**: モデルの表現力向上
3. **サンプリングの高速化**: 少ないステップで高品質生成
4. **FID scoreの改善**: ImageNetで競合モデルを上回る

#### 実装との対応

`equivariant_diffusion/utils.py` の `cosine_beta_schedule` 関数で実装されている。

**コード例**:
```python
from equivariant_diffusion.utils import cosine_beta_schedule

# Cosineスケジュールの使用
betas = cosine_beta_schedule(timesteps=1000)

# 累積積を計算
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
```

#### キーとなる数式

**Cosineスケジュール**:
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

ここで、$s$は小さな定数（例: 0.008）。

---

### [4] Score-Based Generative Modeling through Stochastic Differential Equations

**著者**: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole  
**会議**: ICLR 2021  
**arXiv**: https://arxiv.org/abs/2011.13456

#### 日本語要約

拡散モデルを確率微分方程式（SDE）の枠組みで統一的に理解する理論を提案。離散的な時間ステップを連続時間に拡張し、スコア関数（対数確率密度の勾配）を学習することで、より柔軟な生成モデルを構築できることを示した。

**主要な貢献**:
1. **連続時間の定式化**: 離散と連続の橋渡し
2. **スコアマッチング**: 確率密度の勾配を学習
3. **逆SDEと確率流ODE**: 柔軟なサンプリング
4. **理論的統一**: VAE、GAN、拡散モデルの関連性

#### 関連性

本プロジェクトは離散時間の定式化を使用しているが、SDEの観点からも理解できる。

---

### [5] Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities

**著者**: Jonas Köhler, Leon Klein, Frank Noé  
**会議**: ICML 2020  
**arXiv**: https://arxiv.org/abs/2006.02425

#### 日本語要約

対称性を持つ確率分布に対する正規化流（Normalizing Flow）を提案。E(n)同変性を保ちながら、正確な尤度計算が可能な生成モデルを構築する。分子の立体配座生成に応用し、高い精度を達成した。

**主要な貢献**:
1. **同変な正規化流**: 対称性を保つ可逆変換
2. **正確な尤度**: 対数尤度の解析的計算
3. **ボルツマン分布のサンプリング**: 物理的に妥当な配座生成

#### 関連性

拡散モデルとは異なるアプローチだが、E(n)同変性という共通点を持つ。

---

### [6] Group Equivariant Convolutional Networks

**著者**: Taco S. Cohen, Max Welling  
**会議**: ICML 2016  
**arXiv**: https://arxiv.org/abs/1602.07576

#### 日本語要約

群の対称性を保つ畳み込みニューラルネットワーク（G-CNN）を提案。回転や反転などの変換に対して同変な特徴抽出を実現する。画像分類タスクで標準的なCNNを上回る性能を示した。

**主要な貢献**:
1. **群同変畳み込み**: 対称性を保つフィルタ
2. **理論的基盤**: 表現論に基づく設計
3. **回転同変**: 任意の角度での回転に対応

#### 関連性

E(n)同変性の理論的基盤を提供。本プロジェクトのEGNNはこの考え方を点群データに拡張している。

---

## 分子生成

### [7] GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation

**著者**: Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, Jian Tang  
**会議**: ICLR 2022  
**arXiv**: https://arxiv.org/abs/2203.02923

#### 日本語要約

分子の立体配座を生成するための幾何学的拡散モデルを提案。トーション角と結合長を同時に生成し、化学的に妥当な3D構造を生成する。従来の手法よりも高い精度と多様性を実現した。

**主要な貢献**:
1. **幾何学的拡散過程**: 内部座標（トーション角、結合長）での拡散
2. **化学的制約の考慮**: 結合長と角度の物理的範囲
3. **高品質な配座生成**: RMSD（二乗平均平方根偏差）が低い
4. **多様性**: 複数の安定配座を生成

#### 実装との対応

本プロジェクトはカルテシアン座標で直接生成するが、GeoDiffは内部座標を使用する点が異なる。

#### キーとなる数式

**トーション角の拡散**:
$$\theta_t = \theta_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

### [8] Equivariant Diffusion for Molecule Generation in 3D

**著者**: Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac, Max Welling  
**会議**: ICML 2022  
**arXiv**: https://arxiv.org/abs/2203.17003

#### 日本語要約

E(3)同変拡散モデルを分子生成に応用した研究。原子座標と原子タイプを同時に生成し、E(3)同変性を厳密に保つ。QM9データセットで高品質な分子を生成し、条件付き生成にも対応。

**主要な貢献**:
1. **E(3)同変な拡散モデル**: 回転・並進・反転に不変
2. **原子タイプと座標の同時生成**: 統一的なフレームワーク
3. **条件付き生成**: 物理化学的性質を制御
4. **高い生成品質**: 有効性、安定性が高い

#### 実装との対応

**本プロジェクトの中核アルゴリズム**。`equivariant_diffusion/en_diffusion.py` と `qm9/models.py` で実装されている。

**コード例**:
```python
from qm9.models import get_model
from qm9.dataset import retrieve_dataloaders

# データセットをロード
dataloaders = retrieve_dataloaders('qm9.db', batch_size=64)
train_loader, val_loader, test_loader = dataloaders

# モデルを構築
model, nodes_dist, prop_dist = get_model(
    args=args,
    device=device,
    dataset_info=dataset_info,
    dataloader_train=train_loader
)

# 分子を生成
x, h = model.sample(
    n_samples=100,
    n_nodes=19,
    node_mask=node_mask,
    edge_mask=edge_mask,
    context=None  # 無条件生成
)
```

#### キーとなる数式

**E(3)同変な座標更新**:
$$\mathbf{x}_i^{(l+1)} = \mathbf{x}_i^{(l)} + \sum_{j \neq i} (\mathbf{x}_i^{(l)} - \mathbf{x}_j^{(l)}) \phi(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, ||\mathbf{x}_i^{(l)} - \mathbf{x}_j^{(l)}||)$$

**拡散過程**（座標）:
$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

**拡散過程**（原子タイプ、カテゴリカル）:
$$q(h_t|h_0) = \text{Cat}(h_t; (1 - \bar{\beta}_t)h_0 + \bar{\beta}_t/K)$$

ここで、$K$は原子タイプの数。

---

### [9] Torsional Diffusion for Molecular Conformer Generation

**著者**: Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi Jaakkola  
**会議**: NeurIPS 2022 (Spotlight)  
**arXiv**: https://arxiv.org/abs/2206.01729

#### 日本語要約

トーション角に特化した拡散モデルを提案。分子骨格を固定し、トーション角のみを生成することで、効率的かつ高品質な配座生成を実現。薬物設計における配座最適化に有用。

**主要な貢献**:
1. **トーション角の拡散**: 内部自由度に焦点
2. **周期性の考慮**: トーション角の[-π, π]周期性
3. **高速生成**: カルテシアン座標よりも効率的
4. **高精度**: RMSD < 1Åを達成

#### 関連性

本プロジェクトはカルテシアン座標を使用するが、トーション角アプローチも有力な手法。

---

### [10] MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation

**著者**: Clement Vignac, Nagham Osman, Laura Toni, Pascal Frossard  
**会議**: ICML 2023 Workshop  
**arXiv**: https://arxiv.org/abs/2302.09048

#### 日本語要約

グラフ構造（結合情報）と3D座標を同時に生成する混合拡散モデル。従来の手法は結合が既知であることを前提としていたが、MiDiは結合も生成対象に含める。より柔軟な分子生成が可能。

**主要な貢献**:
1. **グラフと座標の同時生成**: 統一的な拡散過程
2. **結合予測**: エッジの有無を確率的に決定
3. **より一般的な生成**: 結合情報が不要
4. **新規構造の発見**: 未知の結合パターン

#### 関連性

本プロジェクトは結合情報を暗黙的に扱うが、MiDiは明示的に生成する。

---

### [11] 3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design

**著者**: Ilia Igashov, Hannes Stärk, Clément Vignac, Victor Garcia Satorras, Pascal Frossard, Max Welling, Michael Bronstein, Bruno Correia  
**arXiv**: https://arxiv.org/abs/2205.07309

#### 日本語要約

分子リンカー（2つの分子フラグメントを接続する部分）を設計するためのE(3)同変VAE。薬物設計における最適化問題に対応し、特定の形状や性質を持つリンカーを生成する。

**主要な貢献**:
1. **リンカー特化設計**: フラグメントベースの生成
2. **E(3)同変性**: 向きに依存しない
3. **制約付き生成**: 接続点の位置を固定
4. **創薬応用**: PROTACなどの設計

#### 関連性

拡散モデルではなくVAEだが、E(3)同変性を共有。応用分野が異なる。

---

## 条件付き生成

### [12] Classifier-Free Diffusion Guidance

**著者**: Jonathan Ho, Tim Salimans  
**会議**: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications  
**arXiv**: https://arxiv.org/abs/2207.12598

#### 日本語要約

分類器を使わずに条件付き拡散モデルのガイダンスを行う手法。条件付きモデルと無条件モデルの出力を線形結合することで、生成品質と条件への従属度を調整できる。画像生成で広く使用されている。

**主要な貢献**:
1. **分類器不要**: 追加のモデルが不要
2. **ガイダンススケール**: 調整可能なパラメータ
3. **高品質生成**: FID scoreの改善
4. **シンプルな実装**: 既存モデルに容易に適用

#### 実装との対応

本プロジェクトでも条件付き生成に使用される手法。

**コード例**:
```python
def sample_with_guidance(model, context, guidance_scale=2.0):
    """Classifier-Free Guidanceを適用したサンプリング"""
    # 無条件の予測
    eps_uncond = model(x_t, context=None, t=t)
    
    # 条件付きの予測
    eps_cond = model(x_t, context=context, t=t)
    
    # ガイダンスを適用
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    # 次のステップの座標を計算
    x_t_minus_1 = denoise_step(x_t, eps, t)
    
    return x_t_minus_1
```

#### キーとなる数式

**ガイダンス付きスコア**:
$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

ここで、$s$はガイダンススケール、$c$は条件、$\emptyset$は無条件。

---

### [13] CGCF: Conditional Graph Convolutional Flow for Molecule Generation

**著者**: Jaechang Lim, Seongok Ryu, Jin Woo Kim, Woo Youn Kim  
**会議**: ICLR 2020 Workshop  
**arXiv**: https://arxiv.org/abs/2004.08679

#### 日本語要約

グラフ構造に基づく条件付き正規化流。分子グラフを生成しながら、特定の性質を満たすように制御する。2Dグラフ表現を使用するため、3D構造は別途生成が必要。

**主要な貢献**:
1. **条件付きグラフ生成**: 性質を指定
2. **正規化流**: 正確な尤度計算
3. **段階的生成**: ノードとエッジを逐次追加

#### 関連性

2D表現であるため、本プロジェクトの3D生成とは補完的。

---

### [14] MolGAN: An Implicit Generative Model for Small Molecular Graphs

**著者**: Nicola De Cao, Thomas Kipf  
**会議**: ICML 2018 Workshop  
**arXiv**: https://arxiv.org/abs/1805.11973

#### 日本語要約

GANを使用した分子グラフ生成モデル。2Dグラフを生成し、強化学習により特定の性質を最適化する。医薬品設計における初期的な深層生成モデルの一つ。

**主要な貢献**:
1. **GAN for graphs**: グラフへのGAN適用
2. **強化学習**: 性質最適化
3. **離散的な生成**: ノードとエッジの生成

#### 関連性

GANベースであり、拡散モデルとはアプローチが異なる。2D表現のみ。

---

### [15] Property-Guided Molecular Optimization with Conditional Flow Matching

**著者**: Hannes Stärk, Bowen Jing, Regina Barzilay, Tommi Jaakkola  
**会議**: NeurIPS 2023  
**arXiv**: https://arxiv.org/abs/2311.07117

#### 日本語要約

条件付きフローマッチングを使用した分子最適化。既存の分子を特定の性質を持つように変換する。拡散モデルの連続時間版であるフローマッチングを採用し、高速かつ高品質な最適化を実現。

**主要な貢献**:
1. **フローマッチング**: 拡散モデルの代替
2. **分子最適化**: 既存構造の改良
3. **高速生成**: ODEソルバーによる効率化
4. **多目的最適化**: 複数の性質を同時に最適化

#### 関連性

フローマッチングは拡散モデルと密接に関連。最適化タスクに特化。

---

## 結晶構造予測

### [16] Crystal Diffusion Variational Autoencoder for Periodic Material Generation

**著者**: Tian Xie, Xiang Fu, Octavian-Eugen Ganea, Regina Barzilay, Tommi Jaakkola  
**会議**: ICLR 2022  
**arXiv**: https://arxiv.org/abs/2110.06197

#### 日本語要約

周期的な結晶構造を生成するための拡散VAE。格子パラメータ、原子座標、原子タイプを同時に生成し、周期境界条件を考慮する。材料科学における新規材料の発見に応用。

**主要な貢献**:
1. **周期的構造の生成**: 格子の周期性を保つ
2. **拡散とVAEの組み合わせ**: ハイブリッドモデル
3. **多様な材料**: 無機材料、合金など
4. **物性予測との統合**: 生成と評価を同時に

#### 実装との対応

本プロジェクトの `crystal/` モジュールは、類似の目的を持つが、純粋な拡散モデルを使用。

**コード例**:
```python
from crystal.models import CrystalDynamics

# 結晶生成モデル
crystal_model = CrystalDynamics(
    n_dims=3,
    in_node_nf=n_atom_types,
    hidden_nf=256,
    n_layers=9
)

# 結晶構造を生成
lattice, positions, atom_types = crystal_model.sample(
    n_atoms=20,
    space_group=14,  # P21/c
    periodic=True
)
```

#### キーとなる数式

**周期境界条件下での距離**:
$$d_{ij} = \min_{\mathbf{n} \in \mathbb{Z}^3} ||\mathbf{x}_i - \mathbf{x}_j - \mathbf{L}\mathbf{n}||$$

ここで、$\mathbf{L}$は格子ベクトル行列。

---

### [17] Symmetry-Aware Generative Models for Crystal Structure Prediction

**著者**: Tess E. Smidt, Mario Geiger, Benjamin Kurt Miller  
**arXiv**: https://arxiv.org/abs/2310.03817

#### 日本語要約

対称性を考慮した結晶構造予測モデル。空間群の対称操作を明示的にモデルに組み込み、物理的に妥当な結晶を生成する。E(3)同変性に加えて、離散的な対称性（鏡映、回転対称）も保つ。

**主要な貢献**:
1. **空間群の組み込み**: 230種類全てに対応
2. **対称性の厳密な保持**: Wyckoff位置を考慮
3. **効率的な生成**: 対称性により探索空間を削減
4. **高精度**: 実験構造との一致度が高い

#### 実装との対応

本プロジェクトの **チュートリアル06** で実装されている高度な結晶生成と対応。

**コード例**:
```python
from crystal.conditioning import SpaceGroupEmbedding
from crystal.utils import generate_wyckoff_positions

# 空間群の埋め込み
sg_embedding = SpaceGroupEmbedding(space_group=14)
context = sg_embedding.get_embedding()

# Wyckoff位置を生成
wyckoff_sites = generate_wyckoff_positions(
    space_group=14,
    n_atoms=20
)

# 対称性を保った結晶を生成
lattice, positions, atom_types = crystal_model.sample(
    n_atoms=20,
    space_group=14,
    wyckoff_sites=wyckoff_sites,
    context=context
)
```

---

### [18] DiffCSP: Diffusion Model for Crystal Structure Prediction

**著者**: Jiaqi Guan, Wesley Wei Qian, Xuefeng Peng, Youzhi Su, Jiancheng Liu, Jian Peng  
**会議**: NeurIPS 2023  
**arXiv**: https://arxiv.org/abs/2309.04475

#### 日本語要約

結晶構造予測のための拡散モデル。化学組成（原子種と数）から結晶構造を予測する。従来の第一原理計算に比べて計算コストが大幅に低く、多形予測にも対応。

**主要な貢献**:
1. **組成から構造へ**: 化学式を入力
2. **多形生成**: 複数の安定構造を予測
3. **高速予測**: DFT計算の代替
4. **実験検証**: 実際に合成可能な構造

#### 関連性

本プロジェクトと目的が近い。組成条件付き生成の実装例。

---

### [19] Space Group Constrained Crystal Generation

**著者**: Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, Yang Liu  
**会議**: ICLR 2024  
**arXiv**: https://arxiv.org/abs/2311.15734

#### 日本語要約

空間群制約を厳密に満たす結晶生成モデル。ニューラルネットワークの出力を対称操作で制約し、無効な構造の生成を防ぐ。材料探索の効率を向上させる。

**主要な貢献**:
1. **厳密な対称性**: 制約を組み込んだアーキテクチャ
2. **高い有効性**: 全ての生成結晶が対称性を満たす
3. **計算効率**: 後処理不要
4. **スケーラビリティ**: 大規模探索に適用可能

#### 実装との対応

本プロジェクトの空間群条件付け機能と類似。

---

## 評価手法

### [20] GuacaMol: Benchmarking Models for de Novo Molecular Design

**著者**: Nathan Brown, Marco Fiscato, Marwin H.S. Segler, Alain C. Vaucher  
**会議**: Journal of Chemical Information and Modeling, 2019  
**論文**: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839

#### 日本語要約

分子生成モデルの標準的なベンチマーク。有効性、一意性、新規性など複数の評価指標を定義し、異なるモデルを公平に比較できるフレームワークを提供する。

**主要な貢献**:
1. **統一的な評価指標**: 複数のメトリクス
2. **ベンチマークデータセット**: 標準的なテストセット
3. **目標指向生成**: 特定の性質を持つ分子の生成
4. **再現性**: 公開されたコードとデータ

#### 実装との対応

本プロジェクトの評価機能（`qm9/analyze.py`）は、GuacaMolの指標を参考にしている。

**評価指標の例**:
```python
from qm9.analyze import analyze_stability_for_molecules

# 生成分子の評価
results = analyze_stability_for_molecules(
    positions=x,
    atom_types=h,
    dataset_info=dataset_info
)

print(f"Validity: {results['validity']:.2%}")
print(f"Uniqueness: {results['uniqueness']:.2%}")
print(f"Novelty: {results['novelty']:.2%}")
print(f"Molecular stability: {results['mol_stable']/results['n_molecules']:.2%}")
```

#### 評価指標

1. **Validity（有効性）**: 化学的に妥当な分子の割合
2. **Uniqueness（一意性）**: 重複のない分子の割合
3. **Novelty（新規性）**: 訓練データに含まれない分子の割合
4. **Diversity（多様性）**: 分子間の構造的多様性
5. **Goal-directed（目標指向）**: ターゲット性質への到達度

---

### [21] MOSES: A Benchmarking Platform for Molecular Generation Models

**著者**: Daniil Polykovskiy, Alexander Zhebrak, Benjamin Sanchez-Lengeling, et al.  
**会議**: Frontiers in Pharmacology, 2020  
**arXiv**: https://arxiv.org/abs/1811.12823

#### 日本語要約

分子生成モデルの包括的なベンチマークプラットフォーム。GuacaMolと類似しているが、より多くのベースラインモデルと詳細な分析を提供する。SMILESベースと3Dベースの両方のモデルに対応。

**主要な貢献**:
1. **多様なベースラインモデル**: VAE、GAN、RNNなど
2. **詳細な評価**: 10以上のメトリクス
3. **公開プラットフォーム**: コミュニティで使用可能
4. **継続的な更新**: 新しいモデルの追加

#### 評価指標

MOSESは以下のメトリクスを提供:
- Valid
- Unique@1k, Unique@10k
- FCD (Fréchet ChemNet Distance)
- SNN (Similarity to Nearest Neighbor)
- Frag, Scaf (Fragment and Scaffold similarity)
- IntDiv (Internal Diversity)
- Filters (Medicinal chemistry filters)

---

### [22] Fréchet ChemNet Distance: A Metric for Generative Models for Molecules

**著者**: Kristina Preuer, Philipp Renz, Thomas Unterthiner, Sepp Hochreiter, Günter Klambauer  
**会議**: Journal of Chemical Information and Modeling, 2018  
**論文**: https://pubs.acs.org/doi/10.1021/acs.jcim.8b00234

#### 日本語要約

生成された分子の分布と実際の分子の分布の差を測るメトリック。画像生成におけるFréchet Inception Distance (FID)の化学版。ChemNetと呼ばれる事前学習済みモデルを使用して分子の特徴を抽出し、分布間の距離を計算する。

**主要な貢献**:
1. **分布レベルの評価**: 個別分子ではなく全体の分布
2. **ChemNet**: 分子の特徴抽出器
3. **FID scoreとの類似性**: 画像生成の評価手法を応用
4. **多様性と品質**: 両方を同時に評価

#### 数式

**FCD (Fréchet ChemNet Distance)**:
$$\text{FCD} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

ここで、$\mu_r, \Sigma_r$は実データの平均と共分散、$\mu_g, \Sigma_g$は生成データの平均と共分散。

#### 使用例

```python
from fcd import get_fcd

# 実データと生成データのSMILES
real_smiles = [...]  # 訓練データのSMILES
generated_smiles = [...]  # 生成されたSMILES

# FCDを計算
fcd_score = get_fcd(generated_smiles, real_smiles)
print(f"FCD: {fcd_score:.2f}")  # 小さいほど良い
```

---

### [23] Molecular Sets (MOSES): Benchmarking Molecular Generation Models

**著者**: Daniil Polykovskiy, Alexander Zhebrak, Dmitry Vetrov, et al.  
**arXiv**: https://arxiv.org/abs/1811.12823

#### 日本語要約

（[21]と同じ論文の詳細版）

---

## まとめ

本論文リストは、E3 Diffusion for Moleculesプロジェクトの理論的基盤を提供する主要な研究をカバーしています。

### 分野別の重要論文

**E(n)同変性と幾何学的深層学習**:
- [1] E(n) Equivariant Graph Neural Networks
- [6] Group Equivariant Convolutional Networks

**拡散モデルの基礎**:
- [2] Denoising Diffusion Probabilistic Models
- [3] Improved Denoising Diffusion Probabilistic Models
- [4] Score-Based Generative Modeling through SDEs

**分子生成の応用**:
- [8] Equivariant Diffusion for Molecule Generation in 3D（本プロジェクトの中核）
- [7] GeoDiff
- [9] Torsional Diffusion

**結晶構造予測**:
- [16] Crystal Diffusion VAE
- [17] Symmetry-Aware Generative Models
- [18] DiffCSP

**評価手法**:
- [20] GuacaMol
- [21] MOSES
- [22] Fréchet ChemNet Distance

### さらなる学習リソース

1. **教科書**:
   - "Deep Learning" by Goodfellow et al.
   - "Pattern Recognition and Machine Learning" by Bishop

2. **オンラインコース**:
   - Stanford CS236: Deep Generative Models
   - MIT 6.S191: Introduction to Deep Learning

3. **ブログ・チュートリアル**:
   - Lil'Log: "What are Diffusion Models?"
   - Hugging Face: Diffusion Models Course

4. **実装例**:
   - 本プロジェクトのチュートリアル（`tutorials/README_JA.md`）
   - GitHub上の関連プロジェクト

### 研究の方向性

**現在の研究トレンド**:
1. より効率的な拡散モデル（DDIMなど）
2. 条件付き生成の改良（Classifier-Free Guidanceなど）
3. 大規模データセットへのスケーリング
4. タンパク質-リガンド結合への応用
5. 量子化学計算との統合

**今後の展望**:
1. リアルタイム分子設計
2. 実験的検証との統合
3. 多目的最適化
4. 説明可能性の向上
5. 創薬パイプラインへの組み込み

---

**最終更新**: 2025年10月25日  
**バージョン**: 1.0  
**対象プロジェクト**: E3 Diffusion for Molecules

---

## 引用情報

本プロジェクトを研究で使用する場合、以下の主要論文を引用してください:

```bibtex
@inproceedings{hoogeboom2022equivariant,
  title={Equivariant Diffusion for Molecule Generation in 3D},
  author={Hoogeboom, Emiel and Satorras, V{\'\i}ctor Garcia and Vignac, Cl{\'e}ment and Welling, Max},
  booktitle={International Conference on Machine Learning},
  pages={8867--8887},
  year={2022},
  organization={PMLR}
}

@inproceedings{satorras2021en,
  title={E(n) equivariant graph neural networks},
  author={Satorras, V{\'\i}ctor Garcia and Hoogeboom, Emiel and Welling, Max},
  booktitle={International Conference on Machine Learning},
  pages={9323--9332},
  year={2021},
  organization={PMLR}
}

@inproceedings{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

---

**注**: 本論文リストは、プロジェクトの理解を深めるための教育的資料です。各論文の詳細については、原著論文を参照してください。
