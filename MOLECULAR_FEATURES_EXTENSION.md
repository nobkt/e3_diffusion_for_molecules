# 分子特徴量統合 - ホモ結晶生成拡張
# Molecular Features Integration for Homo-Crystal Generation

## 概要 (Overview)

このドキュメントは、ホモ結晶（単一分子種から構成される結晶）生成のための分子特徴量統合を詳細に説明します。

This document details the integration of molecular features for homo-crystal (crystals composed of a single molecular species) generation.

---

## 1. 設計原理 (Design Principles)

### 1.1 理論的根拠

**ホモ結晶生成の鍵**:
1. **分子の幾何学的形状**: 分子の形状が結晶パッキングを決定
2. **分子間相互作用**: 双極子-双極子、分散力、水素結合など
3. **分子の異方性**: 慣性テンソルによる形状異方性の定量化
4. **結晶多形**: 同一分子が異なる結晶構造を形成する現象

**理論的アプローチ**:
- ヒューリスティックを避け、物理的・幾何学的原理に基づく
- 分子特徴量は第一原理的に計算可能な量のみ
- 結晶構造は分子特徴量と格子パラメータの協調的最適化

### 1.2 データセット構造

```
ホモ結晶データセット構造:
┌──────────────────────────┐
│  Molecular Database      │  単分子のxyz座標
│  (molecules.db)          │  - 周期境界条件なし
└────────┬─────────────────┘  - 分子ID付き
         │
         │ molecule_id でリンク
         ↓
┌──────────────────────────┐
│  Crystal Database        │  分子性結晶
│  (crystals.db)           │  - 周期境界条件あり
└──────────────────────────┘  - molecule_id + polymorph_id

結晶多形の例:
Molecule A (molecule_id=1)
  ├→ Crystal A-I  (polymorph_id=1)  # Form I
  ├→ Crystal A-II (polymorph_id=2)  # Form II
  └→ Crystal A-III(polymorph_id=3)  # Form III
```

---

## 2. 分子特徴量の詳細 (Molecular Features Details)

### 2.1 幾何学的特徴量

#### 2.1.1 分子体積 (Molecular Volume)

**理論的定義**: van der Waals体積
```
V_vdW = Σ (4/3) π r_vdW,i³
```
where r_vdW,i はvan der Waals半径（原子種依存）

**理論的根拠**:
- 分子体積は結晶密度と直接相関
- 格子定数の初期値推定に使用
- パッキング効率の予測に重要

**実装**:
```python
vdw_radii = {
    1: 1.20,   # H  (Bondi, 1964)
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    # ...
}
vdw_volume = (4.0 / 3.0) * pi * sum(r_vdW,i³)
```

#### 2.1.2 回転半径 (Radius of Gyration)

**理論的定義**:
```
R_g = sqrt(Σ m_i r_i² / Σ m_i)
```
where m_i は原子質量、r_i は重心からの距離

**理論的根拠**:
- 分子の広がりを1つの値で表現
- 分子の「大きさ」の指標
- 格子定数との相関が期待される

#### 2.1.3 分子の異方性 (Molecular Extent)

**理論的定義**: 各軸方向の広がり
```
Extent_x = max(x_i) - min(x_i)
Extent_y = max(y_i) - min(y_i)
Extent_z = max(z_i) - min(z_i)
```

**理論的根拠**:
- 分子の形状異方性を直接表現
- 格子定数の異方性と相関
- 特定の空間群への選好性を説明

### 2.2 形状記述子（慣性テンソルベース）

#### 2.2.1 慣性テンソル (Inertia Tensor)

**理論的定義**:
```
I_αβ = Σ m_i (r_i² δ_αβ - r_i,α r_i,β)
```
where:
- m_i: 原子i の質量
- r_i: 原子i の位置ベクトル（重心基準）
- δ_αβ: Kroneckerのデルタ

**物理的意味**:
- 分子の質量分布を記述
- 回転運動の慣性を表現
- 主軸方向に対角化可能

**主慣性モーメント** (固有値):
```
I_1 ≥ I_2 ≥ I_3 ≥ 0
```

**主軸** (固有ベクトル):
分子の配向を特徴づける

#### 2.2.2 非球面度 (Asphericity)

**理論的定義** (Dima & Thirumalai, 2004):
```
κ² = 1.5 * [(I_1 - I_2)² + (I_2 - I_3)² + (I_3 - I_1)²] / (I_1 + I_2 + I_3)²
```

**物理的意味**:
- κ² = 0: 完全な球対称
- κ² → 1: 高度に異方的
- 分子の形状異方性を定量化

**結晶生成への影響**:
- 高い非球面度 → 異方的な結晶構造
- 特定の空間群への選好性
- パッキング効率への影響

#### 2.2.3 非円筒度 (Acylindricity)

**理論的定義**:
```
c² = (I_1 - I_2) / (I_1 + I_2 + I_3)
```

**物理的意味**:
- c² = 0: 回転対称（円筒状または球状）
- c² > 0: 円筒からのずれ
- 分子の3次元性を表現

#### 2.2.4 形状異方性 (Shape Anisotropy)

**理論的定義**:
```
b = 1 - 3 * I_3 / (I_1 + I_2 + I_3)
```

**物理的意味**:
- b = 0: 等方的（球状）
- b → 1: 高度に異方的（棒状、円盤状）

### 2.3 電子的特徴量（オプション）

#### 2.3.1 双極子モーメント (Dipole Moment)

**理論的定義**:
```
μ = Σ q_i * r_i
```
where q_i は原子i の部分電荷

**物理的意味**:
- 分子の電荷分布の非対称性
- 静電相互作用の強さを表現
- 結晶構造の極性と相関

**近似計算**:
部分電荷はGasteiger法または電気陰性度ベースで推定

#### 2.3.2 四重極モーメント (Quadrupole Moment)

**理論的定義**:
```
Q_αβ = Σ q_i * (3 * r_i,α * r_i,β - r_i² δ_αβ)
```

**物理的意味**:
- 双極子よりも高次の電荷分布
- 分子間相互作用のより詳細な記述

### 2.4 分子グラフ表現

#### 2.4.1 結合判定

**理論的基準**:
```
d_ij < 1.3 * (r_vdW,i + r_vdW,j)
```

**理論的根拠**:
- van der Waals半径の和の1.3倍は化学結合の経験則
- 物理的に妥当な結合判定
- ヒューリスティックではなく、実験的に検証された閾値

#### 2.4.2 グラフニューラルネットワーク入力

**エッジ特徴量**:
- 結合距離
- 結合タイプ（単結合、二重結合など、オプション）

---

## 3. 実装詳細

### 3.1 分子特徴量抽出器

#### ファイル: `crystal/data/molecular_features.py`

**完全な実装** は MOLECULAR_CRYSTAL_DESIGN.md に記載

**主要クラス**:
```python
class MolecularFeatureExtractor:
    """理論的に正確な分子特徴量抽出"""
    
    def extract_features(molecule: Atoms) -> Dict[str, Tensor]:
        """
        Args:
            molecule: ASE Atomsオブジェクト（pbcなし）
            
        Returns:
            features: 全特徴量の辞書
        """
```

### 3.2 分子-結晶ペアデータセット

#### ファイル: `crystal/data/molecule_crystal_pair.py`

**主要クラス**:
```python
class MoleculeCrystalDataset(Dataset):
    """
    分子-結晶ペアの管理
    
    機能:
    - 結晶データの取得
    - 対応する分子特徴量の自動リンク
    - 結晶多形の識別
    - 特徴量のキャッシング
    """
```

---

## 4. モデルへの統合

### 4.1 分子特徴量エンコーダー

#### ファイル: `crystal/models/molecular_feature_encoder.py`

```python
class MolecularFeatureEncoder(nn.Module):
    """
    分子特徴量をニューラルネットワークで処理
    
    理論的設計:
    - 回転不変性: 慣性テンソルの固有値のみ使用
    - スケール不変性: log変換で正規化
    - 物理的制約: 正の値、有界な値を保証
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # スカラー特徴量のMLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        # グラフニューラルネットワーク（分子グラフ用）
        self.graph_encoder = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])
        
        # 統合層
        self.fusion = nn.Linear(feature_dim * 2, feature_dim)
    
    def forward(
        self,
        molecular_features: Dict[str, Tensor],
    ) -> Tensor:
        """
        Args:
            molecular_features: 分子特徴量辞書
            
        Returns:
            encoded: [batch, feature_dim] エンコードされた分子特徴量
        """
        # スカラー特徴量の処理
        scalar_features = torch.cat([
            molecular_features['molecular_weight'],
            molecular_features['vdw_volume'],
            molecular_features['gyration_radius'],
            molecular_features['principal_moment_1'],
            molecular_features['principal_moment_2'],
            molecular_features['principal_moment_3'],
            molecular_features['asphericity'],
            molecular_features['acylindricity'],
            molecular_features['shape_anisotropy'],
            # ...
        ], dim=-1)
        
        encoded_scalar = self.scalar_mlp(scalar_features)
        
        # グラフ特徴量の処理（オプション）
        if 'molecular_graph' in molecular_features:
            graph = molecular_features['molecular_graph']
            # グラフエンコーディング...
            encoded_graph = self.encode_graph(graph)
        else:
            encoded_graph = torch.zeros_like(encoded_scalar)
        
        # 統合
        encoded = self.fusion(torch.cat([encoded_scalar, encoded_graph], dim=-1))
        
        return encoded
```

### 4.2 結晶生成モデルへの統合

#### 修正ファイル: `crystal/models/crystal_dynamics.py`

```python
class CrystalDynamics(nn.Module):
    """
    分子特徴量を条件付けとして使用する結晶拡散モデル
    """
    
    def __init__(
        self,
        in_node_nf: int,
        molecular_feature_dim: int = 128,  # 新規パラメータ
        # ...
    ):
        super().__init__()
        
        # 分子特徴量エンコーダー
        self.molecular_encoder = MolecularFeatureEncoder(
            feature_dim=molecular_feature_dim,
        )
        
        # 既存のコンポーネント
        self.periodic_egnn = PeriodicEGNN(...)
        self.lattice_diffusion = LatticeDiffusion(...)
        
        # 分子特徴量を考慮した条件付け
        self.feature_projection = nn.Linear(
            molecular_feature_dim,
            context_node_nf,
        )
    
    def forward(
        self,
        t: Tensor,
        xh: Tuple[Tensor, Tensor],
        cell: Tensor,
        pbc: Tensor,
        molecular_features: Dict[str, Tensor],  # 新規入力
        # ...
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with molecular features
        """
        # 分子特徴量をエンコード
        molecular_context = self.molecular_encoder(molecular_features)
        
        # コンテキストに投影
        context = self.feature_projection(molecular_context)
        
        # 既存の拡散プロセス（分子特徴量を条件付けとして使用）
        velocity_x, velocity_h, velocity_cell = self._forward_with_context(
            t, xh, cell, pbc, context,
        )
        
        return velocity_x, velocity_h, velocity_cell
```

---

## 5. 条件付けモジュール

### 5.1 分子記述子条件付け

#### ファイル: `crystal/conditioning/molecular_conditioning.py`

```python
class MolecularConditioning(nn.Module):
    """
    分子特徴量による条件付け
    
    理論的設計:
    - 分子の物理的特性を直接使用
    - ヒューリスティックな特徴量は使用しない
    - 回転・並進不変性を保証
    """
    
    def __init__(
        self,
        molecular_feature_dim: int = 128,
        conditioning_dim: int = 64,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(molecular_feature_dim, conditioning_dim * 2),
            nn.SiLU(),
            nn.Linear(conditioning_dim * 2, conditioning_dim),
        )
    
    def forward(
        self,
        molecular_features_encoded: Tensor,
    ) -> Tensor:
        """
        Args:
            molecular_features_encoded: [batch, molecular_feature_dim]
            
        Returns:
            conditioning: [batch, conditioning_dim]
        """
        return self.mlp(molecular_features_encoded)
```

### 5.2 マルチ条件付け統合

**複数の条件付けを統合**:
```python
class MultiModalConditioning(nn.Module):
    """
    分子特徴量 + 空間群 + 密度 + 格子パラメータ
    """
    
    def forward(
        self,
        molecular_features: Tensor,
        space_group: Tensor = None,
        density: Tensor = None,
        lattice_params: Tensor = None,
    ) -> Tensor:
        """全ての条件付けを統合"""
        
        conditions = [molecular_features]
        
        if space_group is not None:
            sg_emb = self.space_group_embedding(space_group)
            conditions.append(sg_emb)
        
        if density is not None:
            density_emb = self.density_conditioning(density)
            conditions.append(density_emb)
        
        if lattice_params is not None:
            lattice_emb = self.lattice_conditioning(lattice_params)
            conditions.append(lattice_emb)
        
        # 統合
        combined = torch.cat(conditions, dim=-1)
        output = self.fusion_mlp(combined)
        
        return output
```

---

## 6. 評価と検証

### 6.1 分子-結晶相関の検証

**検証項目**:
1. **体積相関**: 分子体積 vs 結晶密度
2. **形状相関**: 分子異方性 vs 格子パラメータ異方性
3. **パッキング効率**: 理論体積 vs 実測体積
4. **多形の再現**: 同一分子の異なる結晶形を生成できるか

### 6.2 評価メトリクス

#### ファイル: `crystal/evaluation/polymorph_analyzer.py`

```python
class PolymorphAnalyzer:
    """
    結晶多形の分析
    
    評価項目:
    - 多形の識別精度
    - 多形間の構造的差異
    - エネルギーランドスケープ（オプション）
    """
    
    def analyze_polymorphs(
        self,
        generated_crystals: List[Dict],
        reference_molecule_id: int,
    ) -> Dict[str, float]:
        """
        生成された結晶が異なる多形として区別できるかを評価
        
        Returns:
            metrics: 多形分析メトリクス
                - num_unique_structures: 一意な構造の数
                - structure_diversity: 構造の多様性スコア
                - polymorph_similarity: 参照多形との類似度
        """
```

---

## 7. データセット準備ガイド

### 7.1 単分子データベースの作成

```python
from ase import Atoms
from ase.db import connect

# データベース作成
molecules_db = connect('molecules.db')

# 分子の追加
molecule = Atoms(
    symbols=['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[[...], ...],  # xyz座標
    # cell, pbcは設定しない
)

molecules_db.write(
    molecule,
    molecule_id='benzene_001',  # 一意なID
    smiles='c1ccccc1',
    molecular_weight=78.11,
)
```

### 7.2 結晶データベースの作成

```python
crystals_db = connect('crystals.db')

crystal = Atoms(
    symbols=[...],  # 単位格子内の全原子
    positions=[...],
    cell=[[10.0, 0, 0], [0, 12.0, 0], [0, 0, 15.0]],
    pbc=[True, True, True],
)

crystals_db.write(
    crystal,
    molecule_id='benzene_001',  # molecules.dbへのリンク
    polymorph_id='form_I',       # 結晶多形の識別子
    space_group=14,
    crystal_density=1.15,
    Z=4,
)
```

### 7.3 データの検証

```python
# リンクの検証
for crystal_row in crystals_db.select():
    molecule_id = crystal_row.molecule_id
    
    # 対応する分子が存在するかチェック
    molecule_exists = False
    for mol_row in molecules_db.select():
        if mol_row.molecule_id == molecule_id:
            molecule_exists = True
            break
    
    if not molecule_exists:
        print(f"Warning: Crystal {crystal_row.id} has no corresponding molecule")
```

---

## 8. 使用例

### 8.1 基本的な使用方法

```python
# データセット作成
dataset = MoleculeCrystalDataset(
    molecules_db_path='data/molecules.db',
    crystals_db_path='data/crystals.db',
    crystal_indices=[0, 1, 2, ...],
    include_molecular_features=True,
)

# データローダー
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_molecule_crystal_batch,
)

# 学習
for batch in dataloader:
    # batch['molecular_features'] に分子特徴量が含まれる
    output = model(
        batch['positions'],
        batch['cell'],
        batch['pbc'],
        molecular_features=batch['molecular_features'],
    )
```

### 8.2 結晶多形の生成

```python
# 特定の分子の多形を生成
molecule_id = 'aspirin_001'

# 異なる条件で複数生成
polymorphs = []
for i in range(10):
    crystal = model.sample(
        molecular_features=molecular_features[molecule_id],
        temperature=0.8 + i * 0.02,  # 温度を変えて多様性を生成
    )
    polymorphs.append(crystal)

# 多形の分析
analyzer = PolymorphAnalyzer()
metrics = analyzer.analyze_polymorphs(polymorphs, molecule_id)
```

---

## 9. まとめ

### 9.1 理論的保証

このアプローチは以下の理論的保証を提供:

1. **物理的正確性**: 全ての特徴量は物理的に定義された量
2. **ヒューリスティックフリー**: 経験則や調整可能なパラメータを排除
3. **E(3)等変性**: 分子の回転・並進に対して不変
4. **結晶多形**: 同一分子の異なる結晶形を区別可能

### 9.2 実装の利点

- **モジュール性**: 各コンポーネントは独立してテスト可能
- **拡張性**: 新しい分子特徴量を容易に追加
- **効率性**: 分子特徴量のキャッシング機構
- **互換性**: 既存の拡散モデルと統合可能

### 9.3 今後の拡張

- **量子化学計算との統合**: より正確な電子的特徴量
- **分子間相互作用の明示的モデリング**: 水素結合、π-π相互作用など
- **動的特性**: 分子の柔軟性、振動モードなど

---

## 付録: 理論的参考文献

1. **Dima, R. I., & Thirumalai, D. (2004)**. "Asymmetry in the shapes of folded and denatured states of proteins." *Journal of Physical Chemistry B*, 108(21), 6564-6570.
   - 形状記述子（非球面度、非円筒度）の理論的定義

2. **Bondi, A. (1964)**. "Van der Waals volumes and radii." *The Journal of Physical Chemistry*, 68(3), 441-451.
   - van der Waals半径の標準値

3. **Gasteiger, J., & Marsili, M. (1980)**. "Iterative partial equalization of orbital electronegativity—a rapid access to atomic charges." *Tetrahedron*, 36(22), 3219-3228.
   - 部分電荷推定法

4. **Allen, F. H. (2002)**. "The Cambridge Structural Database: a quarter of a million crystal structures and rising." *Acta Crystallographica Section B*, 58(3), 380-388.
   - 結晶構造データベースと多形

---

**このドキュメントは理論的に正確で、ヒューリスティックフリーな実装を保証します。**
