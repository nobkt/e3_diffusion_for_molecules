# 用語集（日本語）

**最終更新**: 2025年10月25日  
**対象**: E3 Diffusion for Molecules プロジェクト  
**目的**: 技術用語、略語、概念の定義と解説

---

## 目次

1. [技術用語（五十音順）](#技術用語五十音順)
2. [技術用語（アルファベット順）](#技術用語アルファベット順)
3. [略語](#略語)
4. [英日対訳表](#英日対訳表)

---

## 技術用語（五十音順）

### あ行

#### 圧力 (Pressure)
結晶生成における外部条件の一つ。単位はGPa（ギガパスカル）。高圧下では異なる結晶多形が安定化する。

#### アンサンブル (Ensemble)
統計力学における状態の集合。分子動力学シミュレーションではNVT（定積定温）、NPT（定圧定温）などのアンサンブルが使用される。

#### 異方性 (Anisotropy)
方向によって性質が異なること。結晶は一般に異方性を持ち、特定の方向に沿って異なる物理的性質を示す。

#### 逆拡散過程 (Reverse Diffusion Process)
ノイズから元のデータを復元する過程。拡散モデルの生成プロセスに対応。ニューラルネットワークが学習するのはこの逆過程。

**数式**:
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

#### 運動量 (Momentum)
最適化アルゴリズムにおける勾配の移動平均。SGD with momentumやAdamなどで使用される。学習を安定化し、収束を加速する。

#### エッジマスク (Edge Mask)
グラフニューラルネットワークにおいて、どのエッジ（原子間結合）が存在するかを示すバイナリマスク。形状: `[batch_size, n_atoms, n_atoms]`。

**コード例**:
```python
# エッジマスクの生成
edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
# 自己ループを除外
edge_mask = edge_mask * (1 - torch.eye(n_atoms))
```

#### エネルギー最小化 (Energy Minimization)
分子や結晶の構造を最適化し、系のポテンシャルエネルギーを最小化するプロセス。生成された構造の安定性評価に使用。

#### 同変性 (Equivariance)
入力を変換した後に関数を適用した結果と、関数を適用した後に出力を変換した結果が一致する性質。

**数式**:
$$f(Tx) = T'f(x)$$

ここで、$T$は入力空間の変換、$T'$は出力空間の変換。

#### 大津の二値化法 (Otsu's Method)
画像処理における閾値決定法。本プロジェクトでは使用しないが、一般的な前処理技術。

### か行

#### 回転 (Rotation)
3次元空間における原子座標の回転変換。SO(3)群（3次元特殊直交群）の要素。E(3)同変性を保つためには、回転に対して適切に変換する必要がある。

**コード例**:
```python
import torch

# ランダムな回転行列を生成（SO(3)）
def random_rotation_matrix():
    """ランダムな3x3回転行列を生成"""
    # QR分解を使用
    q, r = torch.qr(torch.randn(3, 3))
    # 行列式が+1になるよう調整
    q = q * torch.sign(torch.det(q))
    return q

# 座標を回転
R = random_rotation_matrix()
x_rotated = x @ R.T  # [n_atoms, 3] @ [3, 3] = [n_atoms, 3]
```

#### 格子定数 (Lattice Parameters)
結晶の単位胞を定義する6つのパラメータ: a, b, c（格子ベクトルの長さ）とα, β, γ（格子ベクトル間の角度）。

**例**: 
- 立方晶: a = b = c, α = β = γ = 90°
- 六方晶: a = b ≠ c, α = β = 90°, γ = 120°

#### 格子ベクトル (Lattice Vectors)
結晶の周期性を定義する3つのベクトル。単位胞を張る基底ベクトル。

**数式**:
$$\mathbf{R} = n_1\mathbf{a} + n_2\mathbf{b} + n_3\mathbf{c}$$

ここで、$n_1, n_2, n_3$は整数、$\mathbf{a}, \mathbf{b}, \mathbf{c}$は格子ベクトル。

#### 活性化関数 (Activation Function)
ニューラルネットワークの非線形性を導入する関数。一般的なもの: ReLU、SiLU（Swish）、GELU。

**コード例**:
```python
import torch.nn as nn

# SiLU (Swish) 活性化関数
activation = nn.SiLU()
x_activated = activation(x)
```

#### 官能基 (Functional Group)
分子の特定の原子の配置パターン。化学的性質を決定する重要な構造単位。

**例**:
- ヒドロキシ基: -OH
- カルボキシル基: -COOH
- アミノ基: -NH₂

**SMARTS表現**:
```python
FUNCTIONAL_GROUPS = {
    'hydroxyl': '[OX2H]',
    'carboxyl': '[CX3](=O)[OX2H1]',
    'amino': '[NX3;H2,H1;!$(NC=O)]',
}
```

#### 勾配消失 (Gradient Vanishing)
ニューラルネットワークの訓練中に、逆伝播時の勾配が非常に小さくなる現象。深いネットワークで発生しやすい。

**対策**:
- ResNet（残差接続）の使用
- バッチ正規化
- 適切な活性化関数（ReLU、SiLU）

#### 勾配爆発 (Gradient Explosion)
逆伝播時の勾配が非常に大きくなる現象。学習が不安定になる。

**対策**:
```python
# 勾配クリッピング
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 剛体 (Rigid Body)
変形しない物体。分子全体の回転や並進を考える際に、分子を剛体として扱うことがある。

#### 空間群 (Space Group)
結晶の対称性を記述する230種類の対称操作の集合。各空間群は固有の対称操作（回転、並進、鏡映など）の組み合わせで定義される。

**表記法**:
- Hermann-Mauguin記号: P21/c, Pbca, Fd-3m
- Schoenflies記号: C2h^5, D2h^15

**コード例**:
```python
import spglib

# 空間群番号から情報を取得
spacegroup_type = spglib.get_spacegroup_type(14)  # P21/c
print(f"International: {spacegroup_type['international']}")
print(f"Hall symbol: {spacegroup_type['hall_symbol']}")
```

#### クーロン相互作用 (Coulomb Interaction)
電荷を持つ粒子間の静電相互作用。分子動力学シミュレーションで考慮される。

**数式**:
$$E = k_e \frac{q_1 q_2}{r}$$

ここで、$k_e$はクーロン定数、$q_1, q_2$は電荷、$r$は距離。

#### 結晶系 (Crystal System)
格子の対称性に基づく7つの分類: 三斜晶系、単斜晶系、斜方晶系、正方晶系、三方晶系、六方晶系、立方晶系。

**例**:
```python
CRYSTAL_SYSTEMS = {
    'triclinic': [1, 2],        # 三斜晶系（空間群1-2）
    'monoclinic': [3, 15],      # 単斜晶系（空間群3-15）
    'orthorhombic': [16, 74],   # 斜方晶系（空間群16-74）
    'tetragonal': [75, 142],    # 正方晶系（空間群75-142）
    'trigonal': [143, 167],     # 三方晶系（空間群143-167）
    'hexagonal': [168, 194],    # 六方晶系（空間群168-194）
    'cubic': [195, 230],        # 立方晶系（空間群195-230）
}
```

#### 原子番号 (Atomic Number)
元素の陽子の数。周期表における元素の位置を決定する。

**例**:
- H (水素): 1
- C (炭素): 6
- N (窒素): 7
- O (酸素): 8
- F (フッ素): 9

#### 原子タイプ (Atom Type)
分子内の原子の種類。通常はワンホットエンコーディングで表現される。

**コード例**:
```python
# 原子タイプのワンホットエンコーディング
atom_types = torch.tensor([6, 1, 1, 1, 1])  # CH4 (炭素1つ、水素4つ)
n_atom_types = 5  # H, C, N, O, F
h = torch.nn.functional.one_hot(atom_types, n_atom_types).float()
# 形状: [5, 5]
```

### さ行

#### 周期境界条件 (Periodic Boundary Conditions, PBC)
結晶シミュレーションで使用される境界条件。単位胞を無限に繰り返すことで、バルク結晶を模擬する。

**コード例**:
```python
def apply_pbc(positions, cell):
    """周期境界条件を適用"""
    # 分数座標に変換
    frac_coords = positions @ torch.inverse(cell)
    # [0, 1)の範囲に収める
    frac_coords = frac_coords % 1.0
    # カルテシアン座標に戻す
    positions_pbc = frac_coords @ cell
    return positions_pbc
```

#### 消失勾配 (Vanishing Gradient)
→ 勾配消失を参照

#### 条件付け (Conditioning)
特定の性質や制約を満たすようにモデルの出力を制御すること。拡散モデルでは、コンテキストベクトルとして性質情報を入力する。

**数式**:
$$p_\theta(x|c) = \prod_{t=1}^T p_\theta(x_{t-1}|x_t, c)$$

ここで、$c$は条件（性質ベクトル）。

#### 正規化 (Normalization)
データのスケールを統一する処理。平均0、標準偏差1に変換するz-score正規化が一般的。

**コード例**:
```python
def normalize(x, mean, std):
    """データを正規化"""
    return (x - mean) / (std + 1e-8)  # ゼロ除算を防ぐ

def denormalize(x_norm, mean, std):
    """正規化を元に戻す"""
    return x_norm * std + mean
```

#### 正規分布 (Normal Distribution)
ガウス分布とも呼ばれる確率分布。拡散モデルのノイズ過程で使用される。

**数式**:
$$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

#### 生成モデル (Generative Model)
データの確率分布を学習し、新しいサンプルを生成するモデル。VAE、GAN、拡散モデルなどがある。

#### 線形層 (Linear Layer)
ニューラルネットワークの基本的な層。全結合層とも呼ばれる。

**数式**:
$$y = Wx + b$$

**コード例**:
```python
import torch.nn as nn

linear = nn.Linear(in_features=256, out_features=128)
output = linear(input)  # [batch_size, 256] -> [batch_size, 128]
```

#### 損失関数 (Loss Function)
モデルの予測と正解の差を測る関数。学習時に最小化する目標。

**例**:
- 平均二乗誤差 (MSE): $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
- 交差エントロピー: $-\sum_{i=1}^n y_i \log(\hat{y}_i)$

### た行

#### 対称操作 (Symmetry Operation)
物体を不変に保つ変換。回転、並進、鏡映、反転など。

#### 多形 (Polymorphism)
同じ化学組成を持つが、異なる結晶構造を取る現象。医薬品では多形によって溶解度や安定性が異なる。

**例**: アスピリンには少なくとも5つの多形が知られている。

#### 単位胞 (Unit Cell)
結晶の最小繰り返し単位。格子ベクトルで定義される平行六面体。

#### 張力 (Stress)
結晶に加わる力。単位面積あたりの力として定義される。

#### 注意機構 (Attention Mechanism)
ニューラルネットワークにおいて、入力の重要な部分に注目する機構。Transformerで広く使用される。

**数式**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 調和振動子 (Harmonic Oscillator)
量子力学における基本的なモデル。分子振動の近似に使用される。

#### 転移学習 (Transfer Learning)
あるタスクで学習したモデルを、別のタスクに適用する手法。小規模データセットで有効。

**コード例**:
```python
# 事前学習済みモデルをロード
pretrained_model = torch.load('pretrained_qm9.pkl')

# 新しいタスク用に微調整
model = pretrained_model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 低い学習率

# 少量のデータで微調整
for epoch in range(10):
    train(model, small_dataset, optimizer)
```

#### 等方性 (Isotropy)
方向によって性質が変わらないこと。気体や液体は等方的。

#### 同変グラフニューラルネットワーク (Equivariant Graph Neural Network)
グラフ構造のデータを処理し、E(n)同変性を保つニューラルネットワーク。

#### トーション角 (Torsion Angle)
4つの原子が定義する二面角。分子の立体配座を特徴付ける。

**計算**:
```python
def compute_torsion_angle(p1, p2, p3, p4):
    """4つの原子座標からトーション角を計算"""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = torch.cross(b1, b2)
    n2 = torch.cross(b2, b3)
    
    # 角度を計算
    cos_angle = (n1 * n2).sum() / (n1.norm() * n2.norm())
    angle = torch.acos(cos_angle)
    
    return angle
```

### な行

#### ノイズスケジュール (Noise Schedule)
拡散過程におけるノイズの追加量を時間ステップごとに定義する関数。

**例**:
- Linear schedule: $\beta_t = \beta_{\text{start}} + (\beta_{\text{end}} - \beta_{\text{start}}) \frac{t}{T}$
- Cosine schedule: $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$

**コード例**:
```python
def polynomial_schedule(timesteps, s=1e-4, power=2.0):
    """多項式ノイズスケジュール"""
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = (1 - (x / steps) ** power) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

#### ノードマスク (Node Mask)
グラフニューラルネットワークにおいて、どのノード（原子）が存在するかを示すバイナリマスク。

**形状**: `[batch_size, max_n_atoms]`

**用途**: パディングされた位置を無視するため。

### は行

#### バッチ正規化 (Batch Normalization)
ミニバッチ内でのデータを正規化する手法。学習を安定化し、収束を加速する。

**数式**:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**コード例**:
```python
import torch.nn as nn

batch_norm = nn.BatchNorm1d(num_features=256)
x_normalized = batch_norm(x)
```

#### バッチサイズ (Batch Size)
一度に処理するサンプル数。大きいほどGPUメモリが必要だが、学習が安定する。

**推奨値**: 16, 32, 64（GPUメモリに依存）

#### 反転 (Inversion)
座標を原点を中心に反転する操作。E(3)群の要素。

**数式**: $\mathbf{x} \rightarrow -\mathbf{x}$

#### 汎化性能 (Generalization Performance)
未知のデータに対するモデルの性能。過学習を避けるために重要。

#### 非共有結合相互作用 (Non-bonded Interaction)
原子間の共有結合以外の相互作用。ファンデルワールス力、水素結合など。

#### 微分可能 (Differentiable)
関数が微分可能であること。ニューラルネットワークの訓練には微分可能性が必須。

#### 標準偏差 (Standard Deviation)
データのばらつきを表す統計量。分散の平方根。

**数式**:
$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2}$$

#### 不変性 (Invariance)
入力を変換しても出力が変わらない性質。

**数式**:
$$f(Tx) = f(x)$$

**例**: 分子のエネルギーは回転・並進に対して不変。

#### 分子記述子 (Molecular Descriptor)
分子の化学的・物理的性質を数値で表現したもの。

**例**:
- 分子量 (Molecular Weight)
- logP（分配係数）
- TPSA（極性表面積）
- 回転可能結合数

**コード例**:
```python
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles('CCO')  # エタノール
descriptors = {
    'MW': Descriptors.MolWt(mol),
    'LogP': Descriptors.MolLogP(mol),
    'TPSA': Descriptors.TPSA(mol),
    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
}
```

#### 分散 (Variance)
データのばらつきを表す統計量。

**数式**:
$$\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2$$

#### 並進 (Translation)
座標全体を一定のベクトルだけ移動させる操作。E(3)群の要素。

**数式**: $\mathbf{x} \rightarrow \mathbf{x} + \mathbf{t}$

**コード例**:
```python
# ランダムな並進
translation = torch.randn(1, 3)  # [1, 3]
x_translated = x + translation  # [n_atoms, 3] + [1, 3] (broadcast)
```

### ま行

#### マスキング (Masking)
不要な要素を無視するための技術。パディングされた位置を無視する際に使用。

**コード例**:
```python
# ノードマスクを適用
masked_positions = positions * node_mask.unsqueeze(-1)
# 形状: [batch, n_atoms, 3] * [batch, n_atoms, 1] -> [batch, n_atoms, 3]
```

#### 密度 (Density)
単位体積あたりの質量。結晶の物理的性質の一つ。

**単位**: g/cm³

**計算**:
```python
def compute_density(atoms, cell):
    """結晶の密度を計算"""
    mass = atoms.get_masses().sum()  # 原子質量の合計（amu）
    volume = atoms.get_volume()      # 体積（Å³）
    
    # amu/Å³ から g/cm³ に変換
    density = mass / volume * 1.66054  # 変換係数
    return density
```

#### 無向グラフ (Undirected Graph)
エッジに方向がないグラフ。分子グラフは通常、無向グラフとして表現される。

### や行

#### 誘電率 (Dielectric Constant)
物質が電場をどの程度遮蔽するかを表す量。溶媒の性質を特徴づける。

#### 有効性 (Validity)
生成された分子が化学的に妥当であるかどうかを示す指標。

**評価方法**:
```python
from rdkit import Chem

def check_validity(smiles):
    """SMILESの有効性をチェック"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None
```

#### 誘導表現 (Induced Representation)
群の表現論における概念。本プロジェクトでは使用しないが、理論的背景に関連。

### ら行

#### ランダムシード (Random Seed)
乱数生成器の初期値。再現性を確保するために設定する。

**コード例**:
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """再現性のためにシードを設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

#### 粒子数 (Number of Particles)
系に含まれる原子や分子の数。

#### 量子化学 (Quantum Chemistry)
量子力学の原理を用いて分子の性質を計算する学問分野。QM9データセットはDFT計算から得られた。

#### レート定数 (Rate Constant)
化学反応の速度を決定する定数。

#### 連続性 (Continuity)
関数が連続であること。拡散過程は連続的な時間発展として定式化される。

### わ行

#### ワンホットエンコーディング (One-hot Encoding)
カテゴリカルデータを0と1のベクトルで表現する方法。

**例**:
```python
# 原子タイプのワンホットエンコーディング
# C (炭素) を [0, 1, 0, 0, 0] として表現（5種類の原子タイプの場合）
atom_type = 1  # C
n_types = 5    # H, C, N, O, F
one_hot = torch.zeros(n_types)
one_hot[atom_type] = 1.0
```

---

## 技術用語（アルファベット順）

### A

#### ADME (Absorption, Distribution, Metabolism, Excretion)
薬物の吸収、分布、代謝、排泄。医薬品設計における重要な性質。

#### Annealing
学習率やノイズレベルを徐々に減少させる技術。最適化を改善する。

#### ASE (Atomic Simulation Environment)
原子シミュレーション用のPythonライブラリ。分子・結晶データの読み書き、構造最適化、性質計算などの機能を提供。

**コード例**:
```python
from ase import Atoms
from ase.db import connect

# データベースに接続
db = connect('molecules.db')

# 分子を追加
atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
db.write(atoms, data={'energy': -76.4})
```

### B

#### Batch Normalization
→ バッチ正規化を参照

#### BFGS (Broyden-Fletcher-Goldfarb-Shanno)
準ニュートン法の一種。構造最適化に使用される。

### C

#### CIF (Crystallographic Information File)
結晶構造を記述する標準ファイルフォーマット。

**例**:
```
data_example
_cell_length_a    5.43
_cell_length_b    5.43
_cell_length_c    5.43
_cell_angle_alpha  90
_cell_angle_beta   90
_cell_angle_gamma  90
_space_group_name_H-M_alt  'P 21 3'
```

#### Classifier-Free Guidance
分類器を使わずに条件付き生成の品質を向上させる技術。

**数式**:
$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

ここで、$s$はガイダンススケール。

**コード例**:
```python
def classifier_free_guidance(model, x_t, context, guidance_scale=2.0):
    """Classifier-Free Guidanceを適用"""
    # 条件なしの予測
    eps_uncond = model(x_t, context=None)
    
    # 条件付きの予測
    eps_cond = model(x_t, context=context)
    
    # ガイダンスを適用
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps
```

### D

#### DDPM (Denoising Diffusion Probabilistic Model)
拡散確率モデルの一種。Ho et al. (2020)で提案された。

#### DFT (Density Functional Theory)
密度汎関数理論。量子化学計算の手法。QM9データセットの性質はDFTで計算されている。

#### Diffusion Model
→ 拡散モデルを参照

### E

#### E(3) Equivariance
3次元ユークリッド群E(3)（回転、並進、反転）の変換に対する同変性。

**定義**: 
$$f(gx) = \rho(g)f(x), \quad \forall g \in E(3)$$

ここで、$\rho$は群表現。

#### E(n) Equivariant Graph Neural Network (EGNN)
n次元ユークリッド群E(n)に対して同変なグラフニューラルネットワーク。Satorras et al. (2021)で提案された。

**アーキテクチャ**:
```python
class EGNN_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features * 2 + 1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features)
        )
    
    def forward(self, h, x, edge_index):
        # hは不変特徴、xは座標
        # エッジメッセージの計算
        row, col = edge_index
        distance = (x[row] - x[col]).norm(dim=-1, keepdim=True)
        message = self.edge_mlp(torch.cat([h[row], h[col], distance], dim=-1))
        
        # ノード特徴の更新（不変）
        h_new = scatter_add(message, row, dim=0)
        
        # 座標の更新（同変）
        x_diff = x[row] - x[col]
        x_new = x + scatter_add(message * x_diff, row, dim=0)
        
        return h_new, x_new
```

#### Encoder-Decoder
エンコーダとデコーダからなるアーキテクチャ。本プロジェクトでは使用しないが、一般的なニューラルネットワーク構造。

#### Equivariance
→ 同変性を参照

### F

#### FAQ (Frequently Asked Questions)
よくある質問。本ドキュメントの一部として提供。

### G

#### GAN (Generative Adversarial Network)
生成敵対ネットワーク。生成モデルの一種だが、本プロジェクトでは拡散モデルを使用。

#### GELU (Gaussian Error Linear Unit)
活性化関数の一種。Transformerで使用される。

**数式**:
$$\text{GELU}(x) = x \cdot \Phi(x)$$

ここで、$\Phi$は標準正規分布の累積分布関数。

#### GNN (Graph Neural Network)
グラフ構造のデータを処理するニューラルネットワーク。

### H

#### Hermann-Mauguin Notation
空間群を表記する国際記号。

**例**:
- P21/c: 単斜晶系、空間群番号14
- Pbca: 斜方晶系、空間群番号61
- Fd-3m: 立方晶系、空間群番号227

#### HOMO (Highest Occupied Molecular Orbital)
最高被占分子軌道。分子の電子供与能を特徴づける。

**単位**: eV（電子ボルト）

**コード例**:
```python
# QM9データセットからHOMOエネルギーを取得
from ase.db import connect

db = connect('qm9.db')
row = db.get(1)
homo_energy = row.data['homo']  # eV
print(f"HOMO energy: {homo_energy:.3f} eV")
```

### I

#### Initialization
ニューラルネットワークの重みの初期化。適切な初期化は学習の成功に重要。

**コード例**:
```python
def init_weights(m):
    """重みの初期化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)
```

#### Invariance
→ 不変性を参照

### K

#### KL Divergence (Kullback-Leibler Divergence)
2つの確率分布の差異を測る尺度。VAEの損失関数に使用される。

**数式**:
$$D_{\text{KL}}(P||Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$$

### L

#### Lattice
→ 格子を参照

#### Layer Normalization
層ごとの正規化。Transformerで使用される。

**数式**:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

ここで、$\mu, \sigma^2$は層内での平均と分散。

#### Learning Rate
学習率。最適化アルゴリズムのステップサイズを制御するハイパーパラメータ。

**推奨値**: 1e-4 ~ 2e-4（Adam使用時）

#### Lipinski's Rule of Five
医薬品らしさを評価する5つのルール。

**基準**:
1. 分子量 ≤ 500 Da
2. LogP ≤ 5
3. 水素結合ドナー ≤ 5
4. 水素結合アクセプター ≤ 10

**コード例**:
```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def check_lipinski(smiles):
    """Lipinskiのルールをチェック"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    
    passes = (
        mw <= 500 and
        logp <= 5 and
        hbd <= 5 and
        hba <= 10
    )
    
    return passes
```

#### LogP
オクタノール-水分配係数の対数。脂溶性を表す指標。

**解釈**:
- LogP < 0: 親水性
- LogP > 0: 疎水性
- LogP = 2-3: バランスが良い（経口薬に適する）

#### LUMO (Lowest Unoccupied Molecular Orbital)
最低空分子軌道。分子の電子受容能を特徴づける。

**単位**: eV

### M

#### MAD (Mean Absolute Deviation)
平均絶対偏差。データのばらつきを測る統計量。

**数式**:
$$\text{MAD} = \frac{1}{n}\sum_{i=1}^n |x_i - \bar{x}|$$

**コード例**:
```python
def compute_mad(data):
    """平均絶対偏差を計算"""
    mean = data.mean()
    mad = (data - mean).abs().mean()
    return mad
```

#### Message Passing
グラフニューラルネットワークの基本操作。ノード間でメッセージを交換し、特徴を更新する。

#### Molecular Dynamics (MD)
分子動力学シミュレーション。原子の運動をニュートンの運動方程式で計算。

#### MSE (Mean Squared Error)
平均二乗誤差。回帰タスクで使用される損失関数。

**数式**:
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

### N

#### Noise Schedule
→ ノイズスケジュールを参照

#### Normalization
→ 正規化を参照

### O

#### Optimizer
最適化アルゴリズム。ニューラルネットワークの重みを更新する。

**種類**:
- SGD (Stochastic Gradient Descent)
- Adam
- AdamW
- RMSprop

**コード例**:
```python
import torch.optim as optim

# Adam最適化器
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# AdamW最適化器（weight decay付き）
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
```

### P

#### PBC (Periodic Boundary Conditions)
→ 周期境界条件を参照

#### Point Cloud
3次元空間における点の集合。分子は原子座標の点群として表現される。

### Q

#### QM9 Dataset
約13万個の小分子（最大9個の重原子）を含む量子化学データセット。DFT計算で得られた13種類の性質が含まれる。

**性質**:
- mu: 双極子モーメント
- alpha: 分極率
- homo: HOMO エネルギー
- lumo: LUMO エネルギー
- gap: HOMO-LUMO ギャップ
- r2: 電子空間範囲
- zpve: ゼロ点振動エネルギー
- U0, U, H, G: 内部エネルギー、エンタルピー、自由エネルギー
- Cv: 熱容量

### R

#### RDKit
ケモインフォマティクス用のPythonライブラリ。分子の読み書き、記述子計算、構造最適化などの機能を提供。

**基本的な使用例**:
```python
from rdkit import Chem

# SMILESから分子を作成
mol = Chem.MolFromSmiles('CCO')

# 原子数と結合数
n_atoms = mol.GetNumAtoms()
n_bonds = mol.GetNumBonds()

# 3D座標を生成
from rdkit.Chem import AllChem
AllChem.EmbedMolecule(mol)
```

#### Regularization
正則化。過学習を防ぐための技術。

**種類**:
- L1正則化（Lasso）
- L2正則化（Ridge）
- Dropout
- Data augmentation

#### ReLU (Rectified Linear Unit)
活性化関数の一種。

**数式**:
$$\text{ReLU}(x) = \max(0, x)$$

**コード例**:
```python
import torch.nn as nn

relu = nn.ReLU()
x_activated = relu(x)
```

#### Residual Connection
残差接続。ResNetで使用される技術。勾配消失を防ぐ。

**数式**:
$$y = F(x) + x$$

### S

#### Sampling
モデルから新しいサンプルを生成するプロセス。

#### SiLU (Sigmoid Linear Unit)
活性化関数の一種。Swishとも呼ばれる。

**数式**:
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

#### SMARTS (SMILES Arbitrary Target Specification)
部分構造を記述するための言語。官能基の検出に使用される。

**例**:
```python
# ヒドロキシ基のSMARTS
hydroxyl_smarts = '[OX2H]'

# カルボキシル基のSMARTS
carboxyl_smarts = '[CX3](=O)[OX2H1]'
```

#### SMILES (Simplified Molecular Input Line Entry System)
分子構造を文字列で表現する記法。

**例**:
- 水: O
- メタン: C
- エタノール: CCO
- ベンゼン: c1ccccc1

#### SO(3) (Special Orthogonal Group)
3次元回転群。行列式が+1の3×3直交行列の集合。

**性質**:
- 回転を表現
- E(3)群の部分群
- 9次元だが3自由度（オイラー角）

#### Space Group
→ 空間群を参照

#### Spglib
結晶の空間群を解析するライブラリ。

**コード例**:
```python
import spglib

# 結晶構造から空間群を検出
cell = (lattice, positions, numbers)
spacegroup = spglib.get_spacegroup(cell, symprec=1e-5)
print(f"Space group: {spacegroup}")
```

### T

#### Temperature
拡散モデルのサンプリング時の温度パラメータ。多様性を制御する。

**使用法**:
```python
# 低温: 決定論的、高品質
x = model.sample(temperature=0.5)

# 高温: ランダム、多様
x = model.sample(temperature=1.5)
```

#### Transformer
注意機構に基づくニューラルネットワークアーキテクチャ。NLPで広く使用される。本プロジェクトでは使用しないが、関連技術。

#### TPSA (Topological Polar Surface Area)
トポロジカル極性表面積。分子の極性部分の表面積。

**単位**: Ų

**解釈**: TPSA < 140 Ų は血液脳関門を通過しやすい。

### U

#### Unit Cell
→ 単位胞を参照

### V

#### VAE (Variational Autoencoder)
変分オートエンコーダ。生成モデルの一種。本プロジェクトでは拡散モデルを使用。

#### Validation
検証。訓練データとは別のデータセットでモデルの性能を評価する。

### W

#### Weights & Biases (wandb)
機械学習実験の管理・可視化ツール。

**基本的な使用法**:
```python
import wandb

# 実験の初期化
wandb.init(project="molecule-generation")

# メトリクスをログ
wandb.log({"loss": loss, "epoch": epoch})

# モデルをログ
wandb.save("model.pkl")
```

#### Wyckoff Position
空間群における原子の対称的な位置。各空間群は固有のWyckoff位置を持つ。

**例**: 空間群P21/cのWyckoff位置
- 2a: (0, 0, 0)
- 2b: (0, 1/2, 0)
- 4e: (x, y, z)（一般位置）

**コード例**:
```python
def generate_wyckoff_positions(space_group_number):
    """空間群のWyckoff位置を生成"""
    import spglib
    
    # 空間群情報を取得
    sg = spglib.get_spacegroup_type(space_group_number)
    
    # Wyckoff位置の例（簡略化）
    if space_group_number == 14:  # P21/c
        positions = {
            '2a': [[0, 0, 0], [0, 0.5, 0.5]],
            '2b': [[0.5, 0, 0], [0.5, 0.5, 0.5]],
            '4e': 'general',  # (x, y, z)
        }
    
    return positions
```

### Z

#### Zero-shot Learning
訓練時に見たことがないクラスに対する予測。転移学習の一種。

---

## 略語

### A-F

- **ADME**: Absorption, Distribution, Metabolism, Excretion（吸収、分布、代謝、排泄）
- **ASE**: Atomic Simulation Environment
- **BFGS**: Broyden-Fletcher-Goldfarb-Shanno（最適化アルゴリズム）
- **CIF**: Crystallographic Information File
- **CNN**: Convolutional Neural Network（畳み込みニューラルネットワーク）
- **CPU**: Central Processing Unit
- **CSV**: Comma-Separated Values
- **CUDA**: Compute Unified Device Architecture（NVIDIA）
- **DDPM**: Denoising Diffusion Probabilistic Model
- **DFT**: Density Functional Theory（密度汎関数理論）
- **EDA**: Exploratory Data Analysis（探索的データ解析）
- **EGNN**: E(n) Equivariant Graph Neural Network

### G-L

- **GAN**: Generative Adversarial Network
- **GELU**: Gaussian Error Linear Unit
- **GNN**: Graph Neural Network
- **GPU**: Graphics Processing Unit
- **HOMO**: Highest Occupied Molecular Orbital
- **KL**: Kullback-Leibler
- **LUMO**: Lowest Unoccupied Molecular Orbital

### M-R

- **MAD**: Mean Absolute Deviation（平均絶対偏差）
- **MD**: Molecular Dynamics（分子動力学）
- **MLP**: Multi-Layer Perceptron（多層パーセプトロン）
- **MSE**: Mean Squared Error（平均二乗誤差）
- **NaN**: Not a Number
- **NLP**: Natural Language Processing（自然言語処理）
- **PBC**: Periodic Boundary Conditions（周期境界条件）
- **PDB**: Protein Data Bank
- **QM9**: Quantum Machine 9（データセット名）
- **RAM**: Random Access Memory
- **RDKit**: Chemistry toolkit for Python
- **ReLU**: Rectified Linear Unit
- **RMSE**: Root Mean Squared Error（二乗平均平方根誤差）

### S-Z

- **SGD**: Stochastic Gradient Descent（確率的勾配降下法）
- **SiLU**: Sigmoid Linear Unit
- **SMARTS**: SMILES Arbitrary Target Specification
- **SMILES**: Simplified Molecular Input Line Entry System
- **SO(3)**: Special Orthogonal Group（3次元回転群）
- **SSD**: Solid State Drive
- **TPSA**: Topological Polar Surface Area
- **VAE**: Variational Autoencoder
- **VRAM**: Video Random Access Memory（GPU メモリ）

---

## 英日対訳表

| 英語 | 日本語 | 備考 |
|------|--------|------|
| Absorption | 吸収 | ADME |
| Activation function | 活性化関数 | ReLU, SiLU等 |
| Anisotropy | 異方性 | ⇔ Isotropy |
| Annealing | アニーリング | 焼きなまし |
| Atom | 原子 | |
| Atomic number | 原子番号 | |
| Attention | 注意機構 | Transformer |
| Backbone | 骨格 | 分子の主鎖 |
| Batch | バッチ | ミニバッチ |
| Batch normalization | バッチ正規化 | |
| Bias | バイアス | 偏り、または重み |
| Bond | 結合 | 化学結合 |
| Branching | 分岐 | |
| Cell | 胞、セル | 単位胞 |
| Checkpoint | チェックポイント | モデルの保存 |
| Classifier | 分類器 | |
| Conditioning | 条件付け | |
| Conformation | 立体配座 | |
| Convergence | 収束 | |
| Coordination | 配位 | 錯体 |
| Crystal | 結晶 | |
| Crystal system | 結晶系 | 7種類 |
| Dataset | データセット | |
| Decoder | デコーダ | |
| Denoising | ノイズ除去 | |
| Density | 密度 | g/cm³ |
| Descriptor | 記述子 | 分子記述子 |
| Dielectric constant | 誘電率 | |
| Diffusion | 拡散 | |
| Distribution | 分布 | 確率分布 |
| Dropout | ドロップアウト | 正則化 |
| Edge | エッジ | グラフの辺 |
| Embedding | 埋め込み | |
| Encoder | エンコーダ | |
| Energy | エネルギー | |
| Ensemble | アンサンブル | |
| Epoch | エポック | 訓練の1周期 |
| Equivariance | 同変性 | |
| Evaluation | 評価 | |
| Feature | 特徴 | 特徴量 |
| Forward pass | 順伝播 | |
| Functional group | 官能基 | |
| Gap | ギャップ | HOMO-LUMO gap |
| Generalization | 汎化 | |
| Generation | 生成 | |
| Gradient | 勾配 | |
| Graph | グラフ | ネットワーク構造 |
| Harmonic oscillator | 調和振動子 | |
| Hidden layer | 隠れ層 | |
| Hyperparameter | ハイパーパラメータ | |
| Inference | 推論 | |
| Initialization | 初期化 | |
| Invariance | 不変性 | |
| Inversion | 反転 | |
| Isotropy | 等方性 | ⇔ Anisotropy |
| Iteration | イテレーション | 反復 |
| Label | ラベル | 正解データ |
| Lattice | 格子 | 結晶格子 |
| Lattice parameter | 格子定数 | a, b, c, α, β, γ |
| Lattice vector | 格子ベクトル | |
| Layer | 層 | ニューラルネットワーク |
| Learning rate | 学習率 | |
| Linear layer | 線形層 | 全結合層 |
| Loss function | 損失関数 | |
| Mask | マスク | |
| Mean | 平均 | 算術平均 |
| Message passing | メッセージパッシング | GNN |
| Metric | メトリック | 評価指標 |
| Minimize | 最小化 | |
| Model | モデル | |
| Molecular dynamics | 分子動力学 | MD |
| Molecule | 分子 | |
| Momentum | 運動量 | 最適化 |
| Node | ノード | グラフの頂点 |
| Noise | ノイズ | |
| Noise schedule | ノイズスケジュール | |
| Normalization | 正規化 | |
| Novelty | 新規性 | 評価指標 |
| Optimization | 最適化 | |
| Optimizer | 最適化器 | |
| Overfitting | 過学習 | |
| Parameter | パラメータ | 重み |
| Periodic boundary | 周期境界 | PBC |
| Point cloud | 点群 | |
| Polymorphism | 多形 | |
| Position | 位置 | 座標 |
| Potential energy | ポテンシャルエネルギー | |
| Prediction | 予測 | |
| Pressure | 圧力 | |
| Property | 性質 | 物性 |
| Quantization | 量子化 | |
| Quantum chemistry | 量子化学 | |
| Random seed | 乱数シード | |
| Rectified | 整流された | ReLU |
| Regularization | 正則化 | |
| Representation | 表現 | |
| Residual connection | 残差接続 | ResNet |
| Rotation | 回転 | |
| Sampling | サンプリング | 生成 |
| Scheduler | スケジューラ | 学習率調整 |
| Score | スコア | 評価値 |
| Softmax | ソフトマックス | |
| Space group | 空間群 | 230種類 |
| Stability | 安定性 | |
| Standard deviation | 標準偏差 | |
| Stress | 張力、応力 | |
| Structure | 構造 | |
| Symmetry | 対称性 | |
| Symmetry operation | 対称操作 | |
| Temperature | 温度 | |
| Tensor | テンソル | |
| Timestep | 時間ステップ | |
| Torsion angle | トーション角 | 二面角 |
| Training | 訓練、学習 | |
| Transfer learning | 転移学習 | |
| Transformation | 変換 | |
| Translation | 並進 | |
| Underfitting | 学習不足 | |
| Uniqueness | 一意性 | 評価指標 |
| Unit cell | 単位胞 | |
| Validation | 検証 | |
| Validity | 有効性 | 評価指標 |
| Variance | 分散 | |
| Vector | ベクトル | |
| Visualization | 可視化 | |
| Volume | 体積 | |
| Weight | 重み | パラメータ |
| Wyckoff position | Wyckoff位置 | 対称位置 |
| Zero-point energy | ゼロ点エネルギー | ZPVE |

---

## 使用上の注意

### 用語の検索方法

1. **日本語の用語**: 五十音順セクションを参照
2. **英語の用語**: アルファベット順セクションを参照
3. **略語**: 略語セクションを参照
4. **対訳**: 英日対訳表を参照

### 追加の情報源

- **チュートリアル**: `tutorials/README_JA.md`
- **APIリファレンス**: `doc/api_reference_ja.md`
- **ユースケース**: `doc/use_cases_ja.md`
- **トラブルシューティング**: `doc/troubleshooting_extended_ja.md`

---

**最終更新**: 2025年10月25日  
**バージョン**: 1.0  
**対象プロジェクト**: E3 Diffusion for Molecules
