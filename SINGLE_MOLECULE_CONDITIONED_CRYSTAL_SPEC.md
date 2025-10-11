# 単分子条件付き分子性結晶生成の詳細仕様書
# Detailed Specification for Single-Molecule Conditioned Molecular Crystal Generation

## 目次 (Table of Contents)

1. [概要](#1-概要-overview)
2. [データセット仕様](#2-データセット仕様-dataset-specification)
3. [システムアーキテクチャ](#3-システムアーキテクチャ-system-architecture)
4. [機能仕様](#4-機能仕様-functional-specification)
5. [インターフェース仕様](#5-インターフェース仕様-interface-specification)
6. [性能要件](#6-性能要件-performance-requirements)
7. [実装要件](#7-実装要件-implementation-requirements)
8. [テスト仕様](#8-テスト仕様-testing-specification)
9. [使用例](#9-使用例-usage-examples)

---

## 1. 概要 (Overview)

### 1.1 目的 (Purpose)

本仕様書は、単分子の構造情報を条件として用いて分子性結晶を生成するシステムの詳細仕様を定義します。

**This specification defines the detailed requirements for a system that generates molecular crystals conditioned on single molecule structural information.**

### 1.2 スコープ (Scope)

**対象範囲:**
- 単分子XYZ座標と対応する分子性結晶のペアデータセット
- ASE DBフォーマットでのデータ管理
- E(3)等変拡散モデルによる条件付き生成
- ホモ結晶（同一分子から構成される結晶）の生成

**Out of scope (Phase 1):**
- 共結晶（異なる分子種を含む結晶）の生成
- 空間群対称性の明示的な強制
- 結晶多形の網羅的生成

### 1.3 主要な特徴 (Key Features)

1. **ペアデータセットのサポート**: 分子-結晶の対応関係を保持
2. **E(3)等変な分子エンコーディング**: 回転・並進不変な分子表現
3. **周期境界条件下での拡散**: 結晶特有の周期性を考慮
4. **分子一致性の保証**: 生成される結晶が入力分子から構成されることを保証
5. **既存機能との互換性**: 単一分子生成機能を維持

---

## 2. データセット仕様 (Dataset Specification)

### 2.1 データセット構造 (Dataset Structure)

#### 2.1.1 全体構造 (Overall Structure)

システムは2つのASEデータベースを使用：

**The system uses two ASE databases:**

```
data/
├── molecules.db        # 単分子データセット
└── crystals.db         # 対応する結晶データセット
```

または、単一のデータベースに統合：

**Or consolidated into a single database:**

```
data/
└── paired_data.db      # 統合データセット
```

#### 2.1.2 分子データベース (Molecule Database)

**スキーマ:**

| フィールド | 型 | 必須 | 説明 |
|-----------|---|-----|-----|
| `id` | Integer | Yes | 分子のユニークID |
| `symbols` | String Array | Yes | 原子種のリスト |
| `positions` | Float Array (N×3) | Yes | 原子座標 (Å) |
| `numbers` | Integer Array | Yes | 原子番号 |
| `formula` | String | Yes | 化学式 |
| `mol_weight` | Float | No | 分子量 (g/mol) |
| `smiles` | String | No | SMILES文字列 |
| `inchi` | String | No | InChI文字列 |

**データ例:**

```python
from ase import Atoms
from ase.db import connect

# 単分子の例 (ベンゼン)
molecule = Atoms(
    symbols=['C']*6 + ['H']*6,
    positions=[
        [0.000, 1.400, 0.000],  # C
        [1.212, 0.700, 0.000],  # C
        [1.212, -0.700, 0.000], # C
        [0.000, -1.400, 0.000], # C
        [-1.212, -0.700, 0.000],# C
        [-1.212, 0.700, 0.000], # C
        [0.000, 2.480, 0.000],  # H
        # ... 残りのH原子
    ]
)

db = connect('molecules.db')
db.write(molecule, smiles='c1ccccc1')
```

#### 2.1.3 結晶データベース (Crystal Database)

**スキーマ:**

| フィールド | 型 | 必須 | 説明 |
|-----------|---|-----|-----|
| `id` | Integer | Yes | 結晶のユニークID |
| `molecule_id` | Integer | Yes | 対応する分子のID |
| `symbols` | String Array | Yes | 単位格子内の全原子種 |
| `positions` | Float Array (N×3) | Yes | 単位格子内の原子座標 (Å) |
| `cell` | Float Array (3×3) | Yes | 格子ベクトル行列 |
| `pbc` | Boolean Array (3) | Yes | 周期境界条件 [True, True, True] |
| `space_group` | Integer | No | 空間群番号 (1-230) |
| `density` | Float | No | 密度 (g/cm³) |
| `Z` | Integer | No | 単位格子あたりの分子数 |
| `temperature` | Float | No | 測定温度 (K) |

**データ例:**

```python
# 対応する結晶構造
crystal = Atoms(
    symbols=['C']*24 + ['H']*24,  # 4分子 × 12原子
    positions=[...],  # 単位格子内の全原子座標
    cell=[
        [7.39, 0.0, 0.0],
        [0.0, 9.42, 0.0],
        [0.0, 0.0, 6.81]
    ],
    pbc=[True, True, True]
)

db = connect('crystals.db')
db.write(
    crystal,
    molecule_id=1,  # molecules.dbの対応するID
    space_group=14,  # P21/c
    Z=4,
    density=1.15
)
```

### 2.2 データの整合性要件 (Data Integrity Requirements)

#### 2.2.1 分子-結晶対応 (Molecule-Crystal Correspondence)

**REQ-DATA-001**: 各結晶データは `molecule_id` フィールドを持ち、対応する分子を参照しなければならない。

**REQ-DATA-002**: 結晶内の分子組成は、参照される単分子の化学式と一致しなければならない（$Z$ 倍）。

$$
\text{Formula}_{crystal} = Z \times \text{Formula}_{molecule}
$$

**REQ-DATA-003**: 結晶内の原子総数は、分子内の原子数の整数倍でなければならない。

$$
N_{atoms}^{crystal} = Z \times N_{atoms}^{molecule}
$$

#### 2.2.2 品質フィルタリング (Quality Filtering)

**REQ-DATA-004**: 以下の条件を満たさないデータは除外される：

1. **原子座標の妥当性:**
   - 最小原子間距離 > 0.5 Å
   - 最大原子間距離 < 10 Å (分子内)

2. **格子パラメータの妥当性:**
   - 格子定数: 3 Å < a, b, c < 50 Å
   - 格子角: 30° < α, β, γ < 150°
   - 体積: 50 Å³ < V < 10000 Å³

3. **データの完全性:**
   - 原子座標にNaN/Infを含まない
   - 格子ベクトルが退化していない (det(L) ≠ 0)

### 2.3 データ分割 (Data Split)

**REQ-DATA-005**: データセットは以下の比率で分割される：

- 訓練データ (Training): 80%
- 検証データ (Validation): 10%
- テストデータ (Testing): 10%

**REQ-DATA-006**: 分割は分子単位で行われ、同一分子の結晶多形は同じセットに含まれる。

```python
# 分子IDでグループ化して分割
molecule_ids = list(set([row.molecule_id for row in crystal_db.select()]))
train_mol_ids, temp = train_test_split(molecule_ids, test_size=0.2, random_seed=42)
valid_mol_ids, test_mol_ids = train_test_split(temp, test_size=0.5, random_seed=42)
```

---

## 3. システムアーキテクチャ (System Architecture)

### 3.1 全体アーキテクチャ (Overall Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                      入力層 (Input Layer)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐           ┌──────────────────┐        │
│  │  Molecule DB     │           │  Crystal DB      │        │
│  │  Loader          │◀─────────▶│  Loader          │        │
│  └──────────────────┘           └──────────────────┘        │
│         │                                │                   │
│         │  Single Molecule               │  Crystal          │
│         ▼                                ▼  Structure        │
└─────────────────────────────────────────────────────────────┘
         │                                │
         ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   エンコーディング層 (Encoding Layer)          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐               │
│  │  Molecule Encoder (E(3)-EGNN)            │               │
│  │  - Atom positions → Molecular embedding  │               │
│  │  - Rotation/Translation equivariant      │               │
│  └──────────────────────────────────────────┘               │
│                       │                                      │
│                       ▼  c_mol ∈ R^d                        │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              拡散モデル層 (Diffusion Model Layer)             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐               │
│  │  Conditional Crystal Dynamics             │               │
│  │  ┌─────────────────────────────┐         │               │
│  │  │ Periodic E(3)-EGNN          │         │               │
│  │  │ - Message passing           │         │               │
│  │  │ - Minimum image convention  │         │               │
│  │  │ - Molecular conditioning    │         │               │
│  │  └─────────────────────────────┘         │               │
│  │  ┌─────────────────────────────┐         │               │
│  │  │ Lattice Diffusion           │         │               │
│  │  │ - Cell parameter evolution  │         │               │
│  │  │ - Molecular conditioning    │         │               │
│  │  └─────────────────────────────┘         │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 後処理層 (Post-processing Layer)              │
├─────────────────────────────────────────────────────────────┤
│  - Molecular consistency check                              │
│  - Structure validation                                     │
│  - Symmetry analysis (optional)                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 主要コンポーネント (Main Components)

#### 3.2.1 MolecularCrystalDataset

**責任:** ペアデータセットの管理とバッチ生成

**インターフェース:**

```python
class MolecularCrystalDataset(Dataset):
    def __init__(
        self,
        molecule_db_path: str,
        crystal_db_path: str,
        indices: List[int],
        remove_h: bool = False,
        use_fractional_coords: bool = True,
    ):
        """
        Args:
            molecule_db_path: 分子データベースのパス
            crystal_db_path: 結晶データベースのパス
            indices: 使用するデータのインデックス
            remove_h: 水素原子を除去するか
            use_fractional_coords: 分数座標を使用するか
        """
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'molecule': {
                    'positions': Tensor[n_mol_atoms, 3],
                    'atom_types': Tensor[n_mol_atoms],
                    'one_hot': Tensor[n_mol_atoms, n_atom_types],
                    'num_atoms': Tensor[1],
                },
                'crystal': {
                    'positions': Tensor[n_crystal_atoms, 3],
                    'atom_types': Tensor[n_crystal_atoms],
                    'one_hot': Tensor[n_crystal_atoms, n_atom_types],
                    'cell': Tensor[3, 3],
                    'cell_params': Tensor[6],
                    'pbc': Tensor[3],
                    'num_atoms': Tensor[1],
                    'Z': Tensor[1],  # molecules per unit cell
                },
                'metadata': {
                    'molecule_id': int,
                    'space_group': Tensor[1],  # optional
                    'density': Tensor[1],  # optional
                }
            }
        """
```

#### 3.2.2 MoleculeEncoder

**責任:** 単分子をE(3)等変に固定長ベクトルにエンコード

**インターフェース:**

```python
class MoleculeEncoder(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int = 256,
        out_nf: int = 128,
        n_layers: int = 6,
        attention: bool = True,
        normalization_factor: float = 100.0,
        aggregation_method: str = 'mean_max',
    ):
        """
        Args:
            in_node_nf: 入力ノード特徴量の次元
            hidden_nf: 隠れ層の次元
            out_nf: 出力埋め込みの次元
            n_layers: EGNNレイヤー数
            attention: アテンションを使用するか
            normalization_factor: 座標正規化係数
            aggregation_method: 'mean', 'max', 'mean_max', 'attention'
        """
        
    def forward(
        self,
        h: torch.Tensor,  # [batch, n_atoms, in_node_nf]
        x: torch.Tensor,  # [batch, n_atoms, 3]
        node_mask: torch.Tensor,  # [batch, n_atoms]
    ) -> torch.Tensor:
        """
        Returns:
            c_mol: [batch, out_nf] 分子埋め込みベクトル
        """
```

#### 3.2.3 ConditionalCrystalDynamics

**責任:** 分子条件付き結晶構造の拡散ダイナミクス

**インターフェース:**

```python
class ConditionalCrystalDynamics(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        context_node_nf: int = 128,  # from MoleculeEncoder
        hidden_nf: int = 256,
        n_layers: int = 9,
        attention: bool = True,
        condition_time: bool = True,
        tanh: bool = False,
        mode: str = 'egnn_dynamics',
        norm_constant: float = 1.0,
        sin_embedding: bool = False,
        normalization_factor: float = 100.0,
        aggregation_method: str = 'sum',
        conditioning_method: str = 'film',  # 'film', 'add', 'cross_attention'
    ):
        """
        Args:
            in_node_nf: ノード特徴量の入力次元
            context_node_nf: 分子コンテキストの次元
            hidden_nf: 隠れ層の次元
            n_layers: ネットワークのレイヤー数
            conditioning_method: 条件付けの方法
                - 'film': Feature-wise Linear Modulation
                - 'add': Additive conditioning
                - 'cross_attention': Cross-attention conditioning
        """
        
    def forward(
        self,
        t: torch.Tensor,  # [batch]
        xh: Tuple[torch.Tensor, torch.Tensor],  # (x, h)
        cell: torch.Tensor,  # [batch, 3, 3]
        pbc: torch.Tensor,  # [batch, 3]
        node_mask: torch.Tensor,  # [batch, n_atoms]
        edge_mask: torch.Tensor,  # [batch, n_edges]
        context: torch.Tensor,  # [batch, context_node_nf] from MoleculeEncoder
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            velocity_x: [batch, n_atoms, 3]
            velocity_h: [batch, n_atoms, in_node_nf]
            velocity_cell: [batch, 3, 3]
        """
```

---

## 4. 機能仕様 (Functional Specification)

### 4.1 訓練機能 (Training Functionality)

#### 4.1.1 基本訓練フロー (Basic Training Flow)

**FUNC-TRAIN-001**: 訓練ループは以下の手順を実行する：

```python
for epoch in range(n_epochs):
    for batch in train_loader:
        # 1. データ取得
        molecule_data = batch['molecule']
        crystal_data = batch['crystal']
        
        # 2. 分子エンコーディング
        c_mol = molecule_encoder(
            h=molecule_data['one_hot'],
            x=molecule_data['positions'],
            node_mask=molecule_data['node_mask']
        )
        
        # 3. ランダムな時間ステップ
        t = torch.randint(0, diffusion_steps, (batch_size,))
        
        # 4. ノイズ追加
        noise = torch.randn_like(crystal_data['positions'])
        x_t = sqrt(alpha_t) * crystal_data['positions'] + sqrt(1 - alpha_t) * noise
        
        # 5. ノイズ予測
        pred_noise = dynamics(
            t=t,
            xh=(x_t, crystal_data['one_hot']),
            cell=crystal_data['cell'],
            pbc=crystal_data['pbc'],
            node_mask=crystal_data['node_mask'],
            context=c_mol
        )
        
        # 6. 損失計算
        loss = compute_loss(pred_noise, noise, crystal_data, molecule_data)
        
        # 7. 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 4.1.2 損失関数 (Loss Function)

**FUNC-TRAIN-002**: 総損失は以下の要素から構成される：

$$
\mathcal{L}_{total} = \lambda_X \mathcal{L}_X + \lambda_H \mathcal{L}_H + \lambda_L \mathcal{L}_L + \lambda_{consist} \mathcal{L}_{consist}
$$

**座標損失 (Coordinate Loss):**

```python
def compute_coordinate_loss(pred_noise_x, noise_x, node_mask):
    """
    L_X = ||ε_pred - ε_true||^2
    """
    loss = ((pred_noise_x - noise_x) ** 2).sum(dim=-1)  # [batch, n_atoms]
    loss = (loss * node_mask).sum() / node_mask.sum()
    return loss
```

**特徴損失 (Feature Loss):**

```python
def compute_feature_loss(pred_noise_h, noise_h, node_mask):
    """
    L_H = ||ε_h_pred - ε_h_true||^2
    """
    loss = ((pred_noise_h - noise_h) ** 2).sum(dim=-1)
    loss = (loss * node_mask).sum() / node_mask.sum()
    return loss
```

**格子損失 (Lattice Loss):**

```python
def compute_lattice_loss(pred_noise_L, noise_L):
    """
    L_L = ||ε_L_pred - ε_L_true||^2
    """
    loss = ((pred_noise_L - noise_L) ** 2).mean()
    return loss
```

**分子一致性損失 (Molecular Consistency Loss):**

```python
def compute_consistency_loss(generated_crystal, input_molecule, Z):
    """
    L_consist = Σ_k d_mol(M_k^gen, M_input)
    
    生成された結晶から分子を抽出し、入力分子との距離を計算
    """
    # 結晶から分子を抽出
    molecules = extract_molecules_from_crystal(generated_crystal, Z)
    
    # 各分子と入力分子の距離
    distances = []
    for mol_k in molecules:
        # 最適な配置での距離 (Kabsch algorithm)
        dist = kabsch_distance(mol_k, input_molecule)
        distances.append(dist)
    
    loss = sum(distances) / Z
    return loss
```

#### 4.1.3 Classifier-Free Guidance訓練 (CFG Training)

**FUNC-TRAIN-003**: 訓練時、確率 $p_{uncond}$ で条件付けをドロップ：

```python
def forward_with_cfg(self, ..., context, drop_prob=0.1):
    # ランダムに条件をドロップ
    mask = torch.rand(batch_size) > drop_prob
    context_masked = context * mask.unsqueeze(-1)
    
    # 通常の forward
    return self.forward(..., context=context_masked)
```

### 4.2 サンプリング機能 (Sampling Functionality)

#### 4.2.1 基本サンプリング (Basic Sampling)

**FUNC-SAMPLE-001**: 与えられた単分子から結晶を生成：

```python
def sample_crystal_from_molecule(
    molecule: Dict[str, torch.Tensor],
    model: ConditionalCrystalDynamics,
    molecule_encoder: MoleculeEncoder,
    n_samples: int = 1,
    guidance_scale: float = 2.0,
    Z: int = 4,  # molecules per unit cell
    initial_cell: Optional[torch.Tensor] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Args:
        molecule: 単分子データ
        model: 訓練済みモデル
        molecule_encoder: 分子エンコーダ
        n_samples: 生成するサンプル数
        guidance_scale: ガイダンスの強度
        Z: 単位格子あたりの分子数
        initial_cell: 初期格子（Noneの場合はランダム）
        
    Returns:
        generated_crystals: 生成された結晶のリスト
    """
    # 1. 分子エンコーディング
    with torch.no_grad():
        c_mol = molecule_encoder(
            h=molecule['one_hot'],
            x=molecule['positions'],
            node_mask=molecule['node_mask']
        )
    
    # 2. 初期ノイズ生成
    n_atoms = molecule['num_atoms'] * Z
    x_T = torch.randn(n_samples, n_atoms, 3)
    h_T = torch.randn(n_samples, n_atoms, hidden_nf)
    
    if initial_cell is None:
        cell_T = initialize_random_cell()
    else:
        cell_T = initial_cell.unsqueeze(0).expand(n_samples, -1, -1)
    
    # 3. 逆拡散プロセス
    x_t, h_t, cell_t = x_T, h_T, cell_T
    
    for t in reversed(range(diffusion_steps)):
        # ノイズ予測 (conditional)
        noise_cond = model(
            t=torch.full((n_samples,), t),
            xh=(x_t, h_t),
            cell=cell_t,
            pbc=torch.ones(n_samples, 3, dtype=torch.bool),
            context=c_mol.expand(n_samples, -1)
        )
        
        # ノイズ予測 (unconditional) for CFG
        if guidance_scale > 0:
            noise_uncond = model(
                t=torch.full((n_samples,), t),
                xh=(x_t, h_t),
                cell=cell_t,
                pbc=torch.ones(n_samples, 3, dtype=torch.bool),
                context=torch.zeros_like(c_mol).expand(n_samples, -1)
            )
            
            # Classifier-free guidance
            noise = (1 + guidance_scale) * noise_cond - guidance_scale * noise_uncond
        else:
            noise = noise_cond
        
        # 1ステップ更新
        x_t, h_t, cell_t = diffusion_step(x_t, h_t, cell_t, noise, t)
    
    # 4. 後処理
    crystals = []
    for i in range(n_samples):
        crystal = {
            'positions': x_t[i],
            'atom_types': torch.argmax(h_t[i], dim=-1),
            'cell': cell_t[i],
            'pbc': torch.ones(3, dtype=torch.bool),
            'Z': Z,
        }
        crystals.append(crystal)
    
    return crystals
```

#### 4.2.2 初期化戦略 (Initialization Strategy)

**FUNC-SAMPLE-002**: 効率的なサンプリングのため、入力分子情報を初期化に利用：

```python
def initialize_from_molecule(molecule, Z, cell_estimate=None):
    """
    分子をZ回複製して初期配置を生成
    """
    n_mol_atoms = molecule['num_atoms']
    n_total_atoms = n_mol_atoms * Z
    
    # 1. 分子を複製
    mol_positions = molecule['positions']  # [n_mol_atoms, 3]
    
    # 2. 格子内にランダム配置
    if cell_estimate is None:
        # 分子サイズから格子を推定
        mol_size = mol_positions.max(dim=0)[0] - mol_positions.min(dim=0)[0]
        cell_size = mol_size * (Z ** (1/3)) * 1.5
        cell = torch.diag(cell_size)
    else:
        cell = cell_estimate
    
    # 3. 各分子をランダムに回転・配置
    positions = []
    for k in range(Z):
        # ランダム回転
        R = random_rotation_matrix()
        rotated = (R @ mol_positions.T).T
        
        # ランダム並進（分数座標で0~1）
        frac_trans = torch.rand(3)
        cart_trans = (cell @ frac_trans.unsqueeze(-1)).squeeze(-1)
        translated = rotated + cart_trans
        
        positions.append(translated)
    
    positions = torch.cat(positions, dim=0)  # [n_total_atoms, 3]
    
    # 4. ノイズとブレンド
    noise = torch.randn_like(positions)
    alpha_init = 0.3  # initialization strength
    x_T = alpha_init * positions + (1 - alpha_init) * noise
    
    return x_T, cell
```

### 4.3 評価機能 (Evaluation Functionality)

#### 4.3.1 分子一致性評価 (Molecular Consistency Evaluation)

**FUNC-EVAL-001**: 生成された結晶から分子を抽出し、入力分子との一致度を評価：

```python
def evaluate_molecular_consistency(
    generated_crystals: List[Dict],
    input_molecules: List[Dict],
    tolerance: float = 0.5,  # Angstrom
) -> Dict[str, float]:
    """
    Args:
        generated_crystals: 生成された結晶のリスト
        input_molecules: 対応する入力分子のリスト
        tolerance: 一致判定の閾値（Å）
        
    Returns:
        metrics: {
            'consistency_rate': float,  # 0-1
            'avg_rmsd': float,  # Angstrom
            'valid_molecule_rate': float,  # 0-1
        }
    """
    total_molecules = 0
    consistent_molecules = 0
    valid_molecules = 0
    rmsd_list = []
    
    for crystal, molecule in zip(generated_crystals, input_molecules):
        # 結晶から分子を抽出
        extracted_mols = extract_molecules(crystal)
        
        for mol in extracted_mols:
            total_molecules += 1
            
            # 分子の妥当性チェック
            if is_valid_molecule(mol):
                valid_molecules += 1
            
            # 入力分子とのRMSD計算
            rmsd = compute_molecular_rmsd(mol, molecule)
            rmsd_list.append(rmsd)
            
            if rmsd < tolerance:
                consistent_molecules += 1
    
    return {
        'consistency_rate': consistent_molecules / total_molecules,
        'avg_rmsd': np.mean(rmsd_list),
        'valid_molecule_rate': valid_molecules / total_molecules,
    }
```

#### 4.3.2 結晶品質評価 (Crystal Quality Evaluation)

**FUNC-EVAL-002**: 生成された結晶の物理的妥当性と品質を評価：

```python
def evaluate_crystal_quality(
    generated_crystals: List[Dict],
    reference_crystals: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """
    Returns:
        metrics: {
            'validity_rate': float,
            'min_distance_mean': float,
            'packing_efficiency_mean': float,
            'lattice_mae': float,  # if reference provided
            'density_mae': float,  # if reference provided
        }
    """
    valid_count = 0
    min_distances = []
    packing_efficiencies = []
    
    for crystal in generated_crystals:
        # 妥当性チェック
        is_valid = check_crystal_validity(crystal)
        if is_valid:
            valid_count += 1
        
        # 最小原子間距離
        min_dist = compute_minimum_distance(crystal)
        min_distances.append(min_dist)
        
        # パッキング効率
        packing_eff = compute_packing_efficiency(crystal)
        packing_efficiencies.append(packing_eff)
    
    metrics = {
        'validity_rate': valid_count / len(generated_crystals),
        'min_distance_mean': np.mean(min_distances),
        'packing_efficiency_mean': np.mean(packing_efficiencies),
    }
    
    # 参照データとの比較
    if reference_crystals is not None:
        lattice_errors = []
        density_errors = []
        
        for gen, ref in zip(generated_crystals, reference_crystals):
            # 格子パラメータの誤差
            gen_params = cell_vectors_to_params(gen['cell'])
            ref_params = cell_vectors_to_params(ref['cell'])
            lattice_error = torch.abs(gen_params - ref_params).mean()
            lattice_errors.append(lattice_error.item())
            
            # 密度の誤差
            gen_density = compute_density(gen)
            ref_density = compute_density(ref)
            density_error = abs(gen_density - ref_density)
            density_errors.append(density_error)
        
        metrics['lattice_mae'] = np.mean(lattice_errors)
        metrics['density_mae'] = np.mean(density_errors)
    
    return metrics
```

---

## 5. インターフェース仕様 (Interface Specification)

### 5.1 コマンドラインインターフェース (Command Line Interface)

#### 5.1.1 訓練コマンド (Training Command)

```bash
python main_crystal.py \
    --exp_name crystal_from_molecule \
    --dataset ase_paired_db \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --molecule_conditioning True \
    --conditioning_method film \
    --guidance_scale 2.0 \
    --molecule_encoder_layers 6 \
    --molecule_encoder_hidden 256 \
    --molecule_encoding_dim 128 \
    --n_layers 9 \
    --nf 256 \
    --n_epochs 3000 \
    --batch_size 32 \
    --lr 1e-4 \
    --diffusion_steps 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --loss_weights 1.0 1.0 0.1 0.5 \
    --save_model outputs/models \
    --wandb_project crystal_generation
```

#### 5.1.2 サンプリングコマンド (Sampling Command)

```bash
python sample_crystal.py \
    --model_path outputs/models/crystal_from_molecule \
    --molecule_db_path data/molecules.db \
    --molecule_ids 1,2,3,4,5 \
    --n_samples_per_molecule 10 \
    --guidance_scale 2.0 \
    --Z 4 \
    --output_dir outputs/generated_crystals \
    --output_format ase_db \
    --save_cif True \
    --visualize True
```

#### 5.1.3 評価コマンド (Evaluation Command)

```bash
python eval_crystal.py \
    --model_path outputs/models/crystal_from_molecule \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --test_indices test_split.txt \
    --n_samples 1000 \
    --guidance_scale 2.0 \
    --evaluate_consistency True \
    --evaluate_quality True \
    --consistency_tolerance 0.5 \
    --output_report outputs/evaluation_report.json
```

### 5.2 Python API

#### 5.2.1 高レベルAPI (High-Level API)

```python
from crystal_generation import (
    load_paired_dataset,
    train_conditional_model,
    generate_crystals_from_molecules,
    evaluate_generated_crystals,
)

# データセット読み込み
datasets, dataset_info = load_paired_dataset(
    molecule_db_path='data/molecules.db',
    crystal_db_path='data/crystals.db',
    split_ratios=[0.8, 0.1, 0.1],
)

# モデル訓練
model, molecule_encoder = train_conditional_model(
    datasets=datasets,
    dataset_info=dataset_info,
    config={
        'molecule_encoding_dim': 128,
        'hidden_nf': 256,
        'n_layers': 9,
        'n_epochs': 3000,
        'batch_size': 32,
        'lr': 1e-4,
    }
)

# 結晶生成
test_molecules = datasets['test'].get_molecules(indices=[0, 1, 2])
generated_crystals = generate_crystals_from_molecules(
    molecules=test_molecules,
    model=model,
    molecule_encoder=molecule_encoder,
    n_samples_per_molecule=10,
    guidance_scale=2.0,
)

# 評価
metrics = evaluate_generated_crystals(
    generated_crystals=generated_crystals,
    reference_crystals=datasets['test'].get_crystals(indices=[0, 1, 2]),
    input_molecules=test_molecules,
)

print(metrics)
```

#### 5.2.2 低レベルAPI (Low-Level API)

```python
from crystal.data import MolecularCrystalDataset, collate_molecular_crystal_batch
from crystal.models import MoleculeEncoder, ConditionalCrystalDynamics
from crystal.training import train_step, validation_step
from crystal.sampling import sample_conditional
from crystal.evaluation import compute_metrics

# カスタム訓練ループ
dataset = MolecularCrystalDataset(...)
dataloader = DataLoader(dataset, collate_fn=collate_molecular_crystal_batch)

molecule_encoder = MoleculeEncoder(...)
dynamics = ConditionalCrystalDynamics(...)

for epoch in range(n_epochs):
    for batch in dataloader:
        loss, metrics = train_step(
            batch=batch,
            molecule_encoder=molecule_encoder,
            dynamics=dynamics,
            optimizer=optimizer,
        )
        
    val_metrics = validation_step(...)
```

---

## 6. 性能要件 (Performance Requirements)

### 6.1 訓練性能 (Training Performance)

**PERF-TRAIN-001**: 単一GPUでの訓練速度

- **バッチサイズ32**: 最小 5 iterations/秒
- **メモリ使用量**: 最大 16 GB (VRAM)
- **収束時間**: 3000エポック、最大 72時間

**PERF-TRAIN-002**: 分子エンコーディング時間

- **単一分子**: 最大 10 ms
- **バッチ32分子**: 最大 300 ms

### 6.2 サンプリング性能 (Sampling Performance)

**PERF-SAMPLE-001**: 単一結晶の生成時間

- **1000ステップ拡散**: 最大 10秒 (GPU)
- **バッチ10サンプル**: 最大 30秒 (GPU)

**PERF-SAMPLE-002**: スケーラビリティ

- **並列生成**: 100サンプルを10分以内

### 6.3 評価性能 (Evaluation Performance)

**PERF-EVAL-001**: 評価メトリクスの計算時間

- **1000サンプルの評価**: 最大 5分

---

## 7. 実装要件 (Implementation Requirements)

### 7.1 依存関係 (Dependencies)

**REQ-IMPL-001**: 以下のライブラリが必要：

```python
# requirements.txt
torch>=1.10.0
ase>=3.22.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
wandb>=0.12.0
rdkit>=2021.09.1
pytest>=6.2.0
```

### 7.2 コード構造 (Code Structure)

**REQ-IMPL-002**: 以下のディレクトリ構造を維持：

```
crystal/
├── data/
│   ├── molecular_crystal_loader.py    # NEW
│   ├── paired_dataset.py               # NEW
│   └── ...
├── models/
│   ├── molecule_encoder.py            # NEW
│   ├── conditional_crystal_dynamics.py # NEW
│   └── ...
├── conditioning/
│   ├── molecular_conditioning.py       # NEW
│   └── ...
├── training/
│   ├── train_conditional.py           # NEW
│   └── ...
├── sampling/
│   ├── conditional_sampling.py        # NEW
│   └── ...
└── evaluation/
    ├── molecular_consistency.py        # NEW
    └── ...
```

### 7.3 後方互換性 (Backward Compatibility)

**REQ-IMPL-003**: 既存の単一分子生成機能は維持される

```python
# 既存機能（変更なし）
python main_qm9.py --dataset qm9 --exp_name molecule_gen

# 新機能
python main_crystal.py --molecule_conditioning True --exp_name crystal_from_mol
```

---

## 8. テスト仕様 (Testing Specification)

### 8.1 ユニットテスト (Unit Tests)

**TEST-UNIT-001**: データローダーのテスト

```python
def test_molecular_crystal_dataset():
    """ペアデータセットの読み込みと整合性をテスト"""
    dataset = MolecularCrystalDataset(...)
    
    sample = dataset[0]
    
    # 分子と結晶の原子数の整合性
    assert sample['crystal']['num_atoms'] == sample['molecule']['num_atoms'] * sample['crystal']['Z']
    
    # 原子種の一致
    mol_formula = get_formula(sample['molecule'])
    crystal_formula = get_formula(sample['crystal'])
    assert crystal_formula == mol_formula * sample['crystal']['Z']
```

**TEST-UNIT-002**: 分子エンコーダーのテスト

```python
def test_molecule_encoder_equivariance():
    """E(3)等変性をテスト"""
    encoder = MoleculeEncoder(...)
    
    # 元の分子
    h, x = ...,  # molecule data
    c_mol = encoder(h, x, node_mask)
    
    # 回転・並進
    R = random_rotation_matrix()
    t = torch.randn(3)
    x_transformed = (R @ x.T).T + t
    c_mol_transformed = encoder(h, x_transformed, node_mask)
    
    # 埋め込みが不変
    assert torch.allclose(c_mol, c_mol_transformed, atol=1e-5)
```

**TEST-UNIT-003**: 条件付けメカニズムのテスト

```python
def test_conditioning():
    """分子条件付けが正しく機能するかテスト"""
    dynamics = ConditionalCrystalDynamics(...)
    
    # 異なる分子で異なる出力
    output1 = dynamics(..., context=c_mol1)
    output2 = dynamics(..., context=c_mol2)
    
    assert not torch.allclose(output1, output2)
    
    # 同じ分子で同じ出力
    output1_a = dynamics(..., context=c_mol1)
    output1_b = dynamics(..., context=c_mol1)
    
    assert torch.allclose(output1_a, output1_b)
```

### 8.2 統合テスト (Integration Tests)

**TEST-INTEG-001**: エンドツーエンド訓練テスト

```python
def test_end_to_end_training():
    """小規模データセットで訓練が完了するかテスト"""
    # 10サンプルで10エポック
    train_conditional_model(
        datasets=small_datasets,
        config={'n_epochs': 10, 'batch_size': 2},
    )
    # エラーなく完了することを確認
```

**TEST-INTEG-002**: サンプリングテスト

```python
def test_sampling():
    """サンプリングが妥当な結晶を生成するかテスト"""
    generated = generate_crystals_from_molecules(
        molecules=[test_molecule],
        model=trained_model,
        n_samples_per_molecule=5,
    )
    
    for crystal in generated:
        # 基本的な妥当性チェック
        assert check_crystal_validity(crystal)
        assert crystal['Z'] > 0
        assert torch.all(crystal['cell'].det() > 0)  # non-degenerate cell
```

### 8.3 性能テスト (Performance Tests)

**TEST-PERF-001**: 訓練速度テスト

```python
def test_training_speed():
    """訓練速度が要件を満たすかテスト"""
    import time
    
    start = time.time()
    for _ in range(100):  # 100 iterations
        train_step(...)
    elapsed = time.time() - start
    
    iterations_per_sec = 100 / elapsed
    assert iterations_per_sec >= 5  # 最小 5 iter/s
```

**TEST-PERF-002**: メモリ使用量テスト

```python
def test_memory_usage():
    """メモリ使用量が要件を満たすかテスト"""
    import torch
    
    torch.cuda.reset_peak_memory_stats()
    train_step(batch_size=32, ...)
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    
    assert peak_memory <= 16.0  # 最大 16 GB
```

---

## 9. 使用例 (Usage Examples)

### 9.1 基本的な使用例 (Basic Usage)

#### 9.1.1 データセットの準備

```python
# ステップ1: 分子データベースの作成
from ase import Atoms
from ase.db import connect

mol_db = connect('molecules.db')

# ベンゼン分子
benzene = Atoms(
    symbols=['C']*6 + ['H']*6,
    positions=[...],  # XYZ coordinates
)
mol_id = mol_db.write(benzene, smiles='c1ccccc1')

# ステップ2: 対応する結晶データベースの作成
crystal_db = connect('crystals.db')

benzene_crystal = Atoms(
    symbols=['C']*24 + ['H']*24,  # 4 molecules
    positions=[...],  # Unit cell atomic coordinates
    cell=[[7.39, 0, 0], [0, 9.42, 0], [0, 0, 6.81]],
    pbc=[True, True, True],
)
crystal_db.write(
    benzene_crystal,
    molecule_id=mol_id,
    Z=4,
    space_group=14,
)
```

#### 9.1.2 モデルの訓練

```python
# ステップ3: 訓練
from crystal_generation import train_conditional_model, load_paired_dataset

datasets, dataset_info = load_paired_dataset(
    molecule_db_path='molecules.db',
    crystal_db_path='crystals.db',
)

model, mol_encoder = train_conditional_model(
    datasets=datasets,
    dataset_info=dataset_info,
    config={
        'exp_name': 'my_crystal_gen',
        'n_epochs': 3000,
        'batch_size': 32,
        'lr': 1e-4,
        'molecule_encoding_dim': 128,
        'hidden_nf': 256,
        'guidance_scale': 2.0,
    }
)

# モデル保存
torch.save({
    'model_state_dict': model.state_dict(),
    'mol_encoder_state_dict': mol_encoder.state_dict(),
    'dataset_info': dataset_info,
}, 'model.pt')
```

#### 9.1.3 結晶の生成

```python
# ステップ4: 新しい分子から結晶を生成
from crystal_generation import generate_crystals_from_molecules

# チェックポイント読み込み
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
mol_encoder.load_state_dict(checkpoint['mol_encoder_state_dict'])

# 新しい分子（例: トルエン）
toluene = Atoms(symbols=['C']*7 + ['H']*8, positions=[...])

# 結晶生成
crystals = generate_crystals_from_molecules(
    molecules=[toluene],
    model=model,
    molecule_encoder=mol_encoder,
    n_samples_per_molecule=10,
    guidance_scale=2.0,
    Z=4,
)

# 保存
for i, crystal in enumerate(crystals):
    crystal_ase = dict_to_ase_atoms(crystal)
    crystal_ase.write(f'generated_crystal_{i}.cif')
```

### 9.2 高度な使用例 (Advanced Usage)

#### 9.2.1 カスタム条件付け

```python
# 密度も条件として追加
from crystal.conditioning import DensityConditioning

class CustomConditionalDynamics(ConditionalCrystalDynamics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.density_conditioner = DensityConditioning(embedding_dim=64)
    
    def forward(self, ..., context_mol, target_density):
        # 分子条件
        context = context_mol
        
        # 密度条件を追加
        density_emb = self.density_conditioner(target_density)
        context = torch.cat([context, density_emb], dim=-1)
        
        return super().forward(..., context=context)

# 使用
crystals = generate_crystals_with_density(
    molecule=toluene,
    target_density=1.2,  # g/cm³
    model=custom_model,
    ...
)
```

#### 9.2.2 バッチ生成の並列化

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def generate_for_molecule(args):
    molecule_id, molecule_data, config = args
    crystals = generate_crystals_from_molecules(
        molecules=[molecule_data],
        **config,
    )
    return molecule_id, crystals

# 並列生成
molecules = load_molecules('test_molecules.db')
config = {'model': model, 'n_samples_per_molecule': 10, ...}

with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    tasks = [(i, mol, config) for i, mol in enumerate(molecules)]
    results = list(executor.map(generate_for_molecule, tasks))

# 結果の集約
all_crystals = {mol_id: crystals for mol_id, crystals in results}
```

#### 9.2.3 多形生成（異なる結晶構造）

```python
# 異なる初期化とガイダンスで多形を生成
polymorphs = []

for seed in range(5):
    torch.manual_seed(seed)
    
    # 異なる格子初期化
    initial_cells = [
        generate_cubic_cell(volume=500),
        generate_orthorhombic_cell(a=8, b=10, c=12),
        generate_monoclinic_cell(a=9, b=9, c=10, beta=100),
    ]
    
    for init_cell in initial_cells:
        crystal = generate_crystals_from_molecules(
            molecules=[molecule],
            initial_cell=init_cell,
            guidance_scale=2.0,
            n_samples_per_molecule=1,
        )[0]
        
        polymorphs.append(crystal)

# 重複除去
unique_polymorphs = remove_duplicate_structures(polymorphs, threshold=0.1)

print(f"Found {len(unique_polymorphs)} unique polymorphs")
```

---

## 10. まとめ (Summary)

本仕様書は、単分子情報を条件とした分子性結晶生成システムの詳細な要件を定義しました。

**This specification document defined the detailed requirements for a molecular crystal generation system conditioned on single molecule information.**

### 主要な成果物 (Key Deliverables)

1. **ペアデータセット管理**: 分子-結晶の対応関係を保持
2. **E(3)等変分子エンコーダ**: 回転・並進不変な分子表現
3. **条件付き結晶ダイナミクス**: 分子情報を統合した拡散モデル
4. **包括的な評価フレームワーク**: 分子一致性と結晶品質の評価
5. **柔軟なインターフェース**: CLI、Python API、カスタマイズ可能

### 次のステップ (Next Steps)

1. **詳細設計**: 各コンポーネントの実装詳細を設計
2. **プロトタイプ実装**: 小規模データセットでの概念実証
3. **大規模実験**: 実データセットでの訓練と評価
4. **最適化**: 性能とメモリ使用量の最適化
5. **ドキュメント**: ユーザーガイドとチュートリアルの作成

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-11  
**Status:** Final
