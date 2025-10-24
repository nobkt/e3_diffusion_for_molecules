# 物性値を条件とした分子性結晶生成の詳細仕様書

## 文書情報

- **文書タイトル**: 物性値を条件とした分子性結晶生成の詳細仕様書
- **作成日**: 2025-10-24
- **バージョン**: 1.0
- **対象読者**: システム設計者、実装者、テスター
- **参照**: PR#143、doc/molecular_crystal_generation_spec.md Q4、doc/gen_molecular_crystal/theory.md

---

## 目次

1. [概要](#概要)
2. [システム要件](#システム要件)
3. [機能仕様](#機能仕様)
4. [データ仕様](#データ仕様)
5. [インターフェース仕様](#インターフェース仕様)
6. [アルゴリズム仕様](#アルゴリズム仕様)
7. [性能要件](#性能要件)
8. [制約事項](#制約事項)
9. [検証要件](#検証要件)

---

## 概要

### 目的

本仕様書は、物性値を条件として分子性結晶（ホモ結晶）を生成するシステムの詳細仕様を定義します。

### スコープ

以下の機能を含みます：

1. **物性値データの管理**: 結晶構造と物性値のペアデータの読み込みと管理
2. **PropertyConditioningモジュール**: 物性値を条件付けベクトルに変換
3. **ExtendedCombinedConditioning**: 複数の条件（分子、結晶、物性値）を統合
4. **訓練パイプライン**: 物性値条件付きモデルの訓練
5. **生成パイプライン**: 指定された条件で結晶を生成

### システム構成

```
┌─────────────────────────────────────────────────────────────┐
│                     データレイヤー                            │
│  ・molecules.db (分子構造データベース)                        │
│  ・crystals.db (結晶構造データベース)                         │
│  ・crystal_properties.csv (物性値データ)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   データ処理レイヤー                          │
│  ・CrystalDatasetWithProperties                             │
│  ・PropertyNormalization                                    │
│  ・MoleculeDataset                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   条件付けレイヤー                            │
│  ・MolecularEncoder                                         │
│  ・MolecularConditioning                                    │
│  ・SpaceGroupEmbedding                                      │
│  ・DensityConditioning                                      │
│  ・PropertyConditioning (新規)                              │
│  ・ExtendedCombinedConditioning (新規)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   生成モデルレイヤー                          │
│  ・CrystalDynamics (EGNN + 拡散モデル)                       │
│  ・Diffusion Process                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   出力レイヤー                                │
│  ・生成された結晶構造 (CIF形式)                              │
│  ・メタデータ (JSON形式)                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## システム要件

### ハードウェア要件

#### 最小要件
- **CPU**: 4コア以上
- **メモリ**: 16 GB以上
- **ストレージ**: 50 GB以上の空き容量

#### 推奨要件（訓練時）
- **GPU**: NVIDIA GPU（CUDA対応）、VRAM 12 GB以上
- **CPU**: 8コア以上
- **メモリ**: 32 GB以上
- **ストレージ**: 100 GB以上の空き容量（SSD推奨）

### ソフトウェア要件

#### 必須
- **Python**: 3.8以上、3.10以下
- **PyTorch**: 1.12.0以上、2.0.0以下
- **CUDA**: 11.3以上（GPU使用時）
- **ASE (Atomic Simulation Environment)**: 3.22.0以上

#### 依存ライブラリ
- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- torch-geometric >= 2.0.0
- rdkit >= 2022.03.1

---

## 機能仕様

### FR-1: データ読み込み機能

#### FR-1.1: 分子データベースの読み込み

**機能ID**: FR-1.1  
**優先度**: 高  
**説明**: ASEデータベース形式の分子データを読み込む

**入力**:
- データベースファイルパス: `str`
- 分子IDフィルタ（オプション）: `List[str]`

**出力**:
- 分子データセット: `MoleculeDataset`
  - molecule_id: `str`
  - positions: `np.ndarray` (N, 3)
  - atom_types: `np.ndarray` (N,)
  - molecular_weight: `float`
  - pi_conjugation_ratio: `float`
  - その他のメタデータ

**処理フロー**:
1. ASEデータベースを開く
2. 各エントリを読み込む
3. メタデータを検証
4. データセットオブジェクトを構築

**例外処理**:
- ファイルが存在しない → `FileNotFoundError`
- データベースが破損 → `DatabaseError`
- 必須メタデータ欠落 → `ValidationError`

#### FR-1.2: 結晶データベースの読み込み

**機能ID**: FR-1.2  
**優先度**: 高  
**説明**: ASEデータベース形式の結晶データと物性値を読み込む

**入力**:
- 結晶データベースパス: `str`
- 分子データベースパス: `str`
- 物性値リスト: `List[str]`（例: ['bandgap', 'melting_point']）

**出力**:
- 結晶データセット: `CrystalDatasetWithProperties`
  - crystal_id: `str`
  - molecule_id: `str`（リンク用）
  - positions: `np.ndarray` (N, 3)
  - atom_types: `np.ndarray` (N,)
  - cell: `np.ndarray` (3, 3)
  - space_group: `int`
  - density: `float`
  - properties: `Dict[str, float]`

**処理フロー**:
1. 結晶データベースを開く
2. 分子データベースとのリンクを確立
3. 各結晶エントリを読み込む
4. 物性値を抽出
5. 物性値の統計情報を計算（平均、標準偏差）
6. データセットオブジェクトを構築

**物性値統計情報**:
```python
property_stats = {
    'bandgap': {'mean': 2.5, 'std': 1.2, 'min': 0.0, 'max': 6.0},
    'melting_point': {'mean': 180.0, 'std': 50.0, 'min': 50.0, 'max': 350.0},
    # ...
}
```

#### FR-1.3: 物性値CSVの読み込み

**機能ID**: FR-1.3  
**優先度**: 高  
**説明**: CSV形式の物性値データを読み込み、データベースに統合

**入力**:
- CSVファイルパス: `str`

**CSV形式**:
```csv
crystal_id,property_name,property_value
crystal_001,bandgap,2.5
crystal_001,melting_point,180.0
crystal_001,dielectric_constant,3.2
crystal_002,bandgap,3.1
crystal_002,melting_point,210.0
```

**必須カラム**:
- `crystal_id`: 結晶ID（データベースと一致）
- `property_name`: 物性値の名前
- `property_value`: 物性値（数値）

**処理フロー**:
1. CSVファイルを読み込む
2. データの妥当性を検証
3. crystal_idでグループ化
4. 辞書形式に変換
5. データベースに統合

**検証ルール**:
- crystal_idが結晶データベースに存在すること
- property_valueが数値であること
- 必須カラムが存在すること

### FR-2: 条件付けモジュール

#### FR-2.1: PropertyConditioning

**機能ID**: FR-2.1  
**優先度**: 高  
**説明**: 物性値ベクトルを条件付けベクトルに変換

**クラス定義**:
```python
class PropertyConditioning(nn.Module):
    def __init__(
        self,
        property_names: List[str],
        conditioning_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 3
    )
```

**パラメータ**:
- `property_names`: 物性値の名前リスト
  - 例: `['bandgap', 'melting_point', 'dielectric_constant']`
  - 型: `List[str]`
  - 制約: 非空、最大20個
- `conditioning_dim`: 出力の条件付けベクトルの次元
  - 型: `int`
  - デフォルト: 256
  - 範囲: [64, 1024]
- `hidden_dim`: 隠れ層の次元
  - 型: `int`
  - デフォルト: 512
  - 範囲: [128, 2048]
- `n_layers`: MLPの層数
  - 型: `int`
  - デフォルト: 3
  - 範囲: [2, 5]

**メソッド**:

##### forward()
```python
def forward(self, properties: torch.Tensor) -> torch.Tensor:
    """
    Args:
        properties: [batch_size, property_dim]
            物性値のテンソル
    
    Returns:
        conditioning: [batch_size, conditioning_dim]
            条件付けベクトル
    """
```

**処理ステップ**:
1. 入力の形状を検証
2. 物性値を正規化
   $$\text{properties}_{\text{norm}} = \frac{\text{properties} - \mu}{\sigma + 10^{-8}}$$
3. MLPで変換
4. 条件付けベクトルを返す

##### set_normalization_params()
```python
def set_normalization_params(
    self,
    mean: torch.Tensor,
    std: torch.Tensor
) -> None:
    """
    正規化パラメータを設定
    
    Args:
        mean: [property_dim] 平均値
        std: [property_dim] 標準偏差
    """
```

**内部状態**:
- `property_mean`: `torch.Tensor` (property_dim,)
- `property_std`: `torch.Tensor` (property_dim,)

これらはモデルのバッファとして登録され、保存・読み込み時に自動的に処理されます。

#### FR-2.2: ExtendedCombinedConditioning

**機能ID**: FR-2.2  
**優先度**: 高  
**説明**: 複数の条件付けを統合

**クラス定義**:
```python
class ExtendedCombinedConditioning(nn.Module):
    def __init__(
        self,
        molecular_conditioning: MolecularConditioning,
        space_group_embedding: Optional[SpaceGroupEmbedding] = None,
        density_conditioning: Optional[DensityConditioning] = None,
        property_conditioning: Optional[PropertyConditioning] = None,
        conditioning_dim: int = 256
    )
```

**パラメータ**:
- `molecular_conditioning`: 分子条件付けモジュール（必須）
- `space_group_embedding`: 空間群埋め込みモジュール（オプション）
- `density_conditioning`: 密度条件付けモジュール（オプション）
- `property_conditioning`: 物性値条件付けモジュール（オプション）
- `conditioning_dim`: 統合後の条件付けベクトルの次元

**メソッド**:

##### forward()
```python
def forward(
    self,
    molecular_features: torch.Tensor,
    space_group: Optional[torch.Tensor] = None,
    density: Optional[torch.Tensor] = None,
    properties: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        molecular_features: [batch_size, mol_feat_dim]
            分子特徴量
        space_group: [batch_size]
            空間群番号（オプション）
        density: [batch_size]
            密度（オプション）
        properties: [batch_size, property_dim]
            物性値（オプション）
    
    Returns:
        combined_conditioning: [batch_size, conditioning_dim]
            統合条件付けベクトル
    """
```

**処理フロー**:
1. 分子条件付けベクトルを計算（必須）
2. オプション条件が提供されている場合、それぞれの条件付けベクトルを計算
3. すべての条件付けベクトルを連結
4. 統合MLPで処理
5. 統合条件付けベクトルを返す

**統合方法**:
```
条件付けベクトルの連結:
  [mol_cond || sg_cond || density_cond || prop_cond]
  
次元: conditioning_dim × num_conditionings

統合MLP:
  input_dim = conditioning_dim × num_conditionings
  hidden_dim = conditioning_dim × 2
  output_dim = conditioning_dim
```

### FR-3: 訓練機能

#### FR-3.1: モデル訓練

**機能ID**: FR-3.1  
**優先度**: 高  
**説明**: 物性値条件付きモデルを訓練

**入力パラメータ**:
```python
{
    'molecule_db_path': str,           # 分子データベースパス
    'crystal_db_path': str,            # 結晶データベースパス
    'property_names': List[str],       # 物性値名リスト
    'conditioning_types': List[str],   # 使用する条件タイプ
    'n_epochs': int,                   # エポック数
    'batch_size': int,                 # バッチサイズ
    'learning_rate': float,            # 学習率
    'exp_name': str,                   # 実験名
    'output_dir': str,                 # 出力ディレクトリ
}
```

**訓練ループ**:
```
for epoch in range(n_epochs):
    for batch in dataloader:
        # 1. データを取得
        mol_features = batch['molecular_features']
        crystal_structure = batch['crystal_structure']
        space_group = batch.get('space_group')
        density = batch.get('density')
        properties = batch.get('properties')
        
        # 2. 条件付けベクトルを計算
        conditioning = combined_conditioning(
            molecular_features=mol_features,
            space_group=space_group,
            density=density,
            properties=properties
        )
        
        # 3. ノイズを追加
        t = sample_timestep()
        noisy_structure = add_noise(crystal_structure, t)
        
        # 4. ノイズを予測
        predicted_noise = model(noisy_structure, t, conditioning)
        
        # 5. 損失を計算
        loss = mse_loss(predicted_noise, true_noise)
        
        # 6. 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # エポック終了時の処理
    save_checkpoint()
    log_metrics()
```

**損失関数**:
```python
def compute_loss(predicted_noise, true_noise, weights=None):
    """
    Args:
        predicted_noise: (positions_noise, lattice_noise)
        true_noise: (positions_noise, lattice_noise)
        weights: 各項の重み
    
    Returns:
        total_loss: スカラー損失値
    """
    if weights is None:
        weights = {'positions': 1.0, 'lattice': 1.0}
    
    loss_positions = mse_loss(predicted_noise[0], true_noise[0])
    loss_lattice = mse_loss(predicted_noise[1], true_noise[1])
    
    total_loss = (
        weights['positions'] * loss_positions +
        weights['lattice'] * loss_lattice
    )
    
    return total_loss
```

**チェックポイント保存**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'property_mean': dataset.property_mean,
    'property_std': dataset.property_std,
    'conditioning_config': {
        'property_names': property_names,
        'conditioning_dim': conditioning_dim,
    },
    'training_config': {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
}
```

#### FR-3.2: 検証機能

**機能ID**: FR-3.2  
**優先度**: 中  
**説明**: 訓練中にモデルの性能を検証

**検証指標**:
1. **損失値**:
   - 訓練損失: 各エポックの平均損失
   - 検証損失: 検証セットでの損失

2. **生成品質**:
   - 構造的妥当性: 原子間距離、結合長
   - 周期性: 格子境界の連続性
   - 対称性: 空間群の制約を満たすか

3. **物性値精度**:
   - 生成された結晶の物性値を予測
   - 目標物性値との誤差を計算
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Square Error)

**検証頻度**:
- 各エポック終了時
- N回のイテレーションごと（N=1000推奨）

### FR-4: 生成機能

#### FR-4.1: 条件付き結晶生成

**機能ID**: FR-4.1  
**優先度**: 高  
**説明**: 指定された条件で結晶構造を生成

**入力パラメータ**:
```python
{
    'model_path': str,                    # 訓練済みモデルのパス
    'molecule_db_path': str,              # 分子データベースパス
    'target_molecule_id': str,            # 構成分子のID
    'target_properties': Dict[str, float], # 目標物性値
    'space_group': Optional[int],         # 空間群（オプション）
    'density': Optional[float],           # 密度（オプション）
    'n_samples': int,                     # 生成サンプル数
    'output_dir': str,                    # 出力ディレクトリ
}
```

**例**:
```python
generate_crystals(
    model_path='models/crystal_with_properties/model.pt',
    molecule_db_path='data/molecules.db',
    target_molecule_id='benzene_001',
    target_properties={
        'bandgap': 2.5,
        'melting_point': 180.0,
        'dielectric_constant': 3.2
    },
    space_group=14,
    density=1.2,
    n_samples=100,
    output_dir='generated_crystals/'
)
```

**処理フロー**:
1. モデルを読み込む
2. 分子データを読み込む
3. 分子特徴量を抽出
4. 条件付けベクトルを構築
5. 拡散サンプリングを実行
6. 生成結果を保存

**サンプリングアルゴリズム** (DDPM):
```python
def sample_crystal(
    model,
    conditioning,
    n_atoms,
    n_timesteps=1000,
    device='cuda'
):
    """
    条件付き拡散サンプリング
    
    Args:
        model: 訓練済み拡散モデル
        conditioning: 条件付けベクトル
        n_atoms: 原子数
        n_timesteps: タイムステップ数
        device: デバイス
    
    Returns:
        crystal_structure: 生成された結晶構造
    """
    # 1. ランダムノイズから開始
    positions = torch.randn(n_atoms, 3, device=device)
    lattice = torch.randn(3, 3, device=device)
    
    # 2. 逆拡散
    for t in reversed(range(n_timesteps)):
        # タイムステップ埋め込み
        t_tensor = torch.tensor([t], device=device)
        
        # ノイズを予測
        with torch.no_grad():
            predicted_noise = model(
                positions, lattice, t_tensor, conditioning
            )
        
        # 一ステップ逆拡散
        positions, lattice = reverse_diffusion_step(
            positions, lattice, predicted_noise, t
        )
        
        # 周期境界条件を適用
        positions = positions % 1.0  # 分数座標に保つ
    
    # 3. 結晶構造を構築
    crystal_structure = build_crystal(positions, lattice)
    
    return crystal_structure
```

#### FR-4.2: バッチ生成

**機能ID**: FR-4.2  
**優先度**: 中  
**説明**: 複数の条件で効率的に結晶を生成

**入力**:
```python
batch_conditions = [
    {
        'molecule_id': 'mol_001',
        'properties': {'bandgap': 2.5, 'melting_point': 180.0},
        'space_group': 14,
        'density': 1.2
    },
    {
        'molecule_id': 'mol_002',
        'properties': {'bandgap': 3.0, 'melting_point': 200.0},
        'space_group': 15,
        'density': 1.3
    },
    # ... 複数の条件
]
```

**処理**:
- 条件をバッチ化
- 並列にサンプリング
- 結果を個別に保存

**出力形式**:
```
output_dir/
├── mol_001_sg14_d1.2_bg2.5_mp180.0/
│   ├── crystal_001.cif
│   ├── crystal_002.cif
│   └── metadata.json
├── mol_002_sg15_d1.3_bg3.0_mp200.0/
│   ├── crystal_001.cif
│   ├── crystal_002.cif
│   └── metadata.json
└── batch_summary.json
```

### FR-5: 出力機能

#### FR-5.1: CIF形式での保存

**機能ID**: FR-5.1  
**優先度**: 高  
**説明**: 生成された結晶をCIF形式で保存

**CIF形式仕様**:
```cif
data_generated_crystal
_cell_length_a                   10.000
_cell_length_b                   12.000
_cell_length_c                   15.000
_cell_angle_alpha                90.000
_cell_angle_beta                 95.000
_cell_angle_gamma                90.000
_symmetry_space_group_name_H-M   'P 21/c'
_symmetry_Int_Tables_number      14

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1  C  0.123  0.456  0.789
C2  C  0.234  0.567  0.890
H1  H  0.345  0.678  0.901
...
```

#### FR-5.2: メタデータの保存

**機能ID**: FR-5.2  
**優先度**: 中  
**説明**: 生成条件と結果をJSONで保存

**JSON形式**:
```json
{
  "generation_info": {
    "timestamp": "2025-10-24T12:00:00",
    "model_path": "models/crystal_with_properties/model.pt",
    "n_samples": 100
  },
  "conditions": {
    "molecule_id": "benzene_001",
    "molecular_weight": 78.11,
    "pi_conjugation_ratio": 1.0,
    "target_properties": {
      "bandgap": 2.5,
      "melting_point": 180.0,
      "dielectric_constant": 3.2
    },
    "space_group": 14,
    "density": 1.2
  },
  "results": [
    {
      "sample_id": 1,
      "cif_file": "crystal_001.cif",
      "lattice_parameters": [10.0, 12.0, 15.0, 90.0, 95.0, 90.0],
      "n_atoms": 24,
      "volume": 1799.5,
      "predicted_properties": {
        "bandgap": 2.48,
        "melting_point": 182.3
      }
    }
  ],
  "statistics": {
    "property_errors": {
      "bandgap": {"mae": 0.05, "rmse": 0.08},
      "melting_point": {"mae": 3.2, "rmse": 5.1}
    }
  }
}
```

---

## データ仕様

### DS-1: データベース構造

#### DS-1.1: 分子データベース (molecules.db)

**形式**: ASE SQLite3データベース

**必須フィールド**:
| フィールド名 | 型 | 説明 | 制約 |
|------------|-----|------|------|
| id | INTEGER | 自動採番ID | PRIMARY KEY |
| molecule_id | TEXT | 分子の一意識別子 | NOT NULL, UNIQUE |
| positions | BLOB | 原子座標 (N, 3) | NOT NULL |
| numbers | BLOB | 原子番号 (N,) | NOT NULL |
| molecular_weight | REAL | 分子量 | > 0 |
| pi_conjugation_ratio | REAL | π共役比率 | [0, 1] |

**オプションフィールド**:
| フィールド名 | 型 | 説明 |
|------------|-----|------|
| atom_types | TEXT | 原子種リスト（JSON） |
| functional_groups | TEXT | 官能基リスト（JSON） |
| smiles | TEXT | SMILES記法 |
| inchi | TEXT | InChI記法 |

#### DS-1.2: 結晶データベース (crystals.db)

**形式**: ASE SQLite3データベース

**必須フィールド**:
| フィールド名 | 型 | 説明 | 制約 |
|------------|-----|------|------|
| id | INTEGER | 自動採番ID | PRIMARY KEY |
| crystal_id | TEXT | 結晶の一意識別子 | NOT NULL, UNIQUE |
| molecule_id | TEXT | 構成分子のID | NOT NULL, FOREIGN KEY |
| positions | BLOB | 原子座標 (N, 3) | NOT NULL |
| numbers | BLOB | 原子番号 (N,) | NOT NULL |
| cell | BLOB | 格子ベクトル (3, 3) | NOT NULL |
| pbc | BLOB | 周期境界条件 (3,) | NOT NULL, [True, True, True] |
| space_group | INTEGER | 空間群番号 | [1, 230] |
| density | REAL | 密度 (g/cm³) | > 0 |

**物性値フィールド**（新規追加）:
| フィールド名 | 型 | 説明 | 単位 |
|------------|-----|------|-----|
| bandgap | REAL | バンドギャップ | eV |
| melting_point | REAL | 融点 | °C |
| dielectric_constant | REAL | 誘電率 | - |
| thermal_conductivity | REAL | 熱伝導率 | W/m·K |
| bulk_modulus | REAL | 体積弾性率 | GPa |
| shear_modulus | REAL | せん断弾性率 | GPa |

#### DS-1.3: 物性値CSV (crystal_properties.csv)

**形式**: CSV（カンマ区切り）

**カラム定義**:
| カラム名 | 型 | 説明 | 制約 |
|---------|-----|------|------|
| crystal_id | STRING | 結晶ID | NOT NULL |
| property_name | STRING | 物性値名 | NOT NULL |
| property_value | FLOAT | 物性値 | NOT NULL |
| unit | STRING | 単位 | オプション |
| method | STRING | 計算/測定方法 | オプション |
| uncertainty | FLOAT | 不確実性 | オプション |

**例**:
```csv
crystal_id,property_name,property_value,unit,method,uncertainty
crystal_001,bandgap,2.5,eV,DFT-GGA,0.1
crystal_001,melting_point,180.0,C,DSC,2.0
crystal_002,bandgap,3.1,eV,DFT-GGA,0.1
```

### DS-2: 正規化パラメータ

物性値の正規化に使用する統計情報：

```python
normalization_params = {
    'property_names': ['bandgap', 'melting_point', 'dielectric_constant'],
    'mean': [2.5, 180.0, 3.0],          # 各物性値の平均
    'std': [1.2, 50.0, 0.8],            # 各物性値の標準偏差
    'min': [0.0, 50.0, 1.0],            # 最小値（参考）
    'max': [6.0, 350.0, 10.0],          # 最大値（参考）
    'n_samples': 10000                   # サンプル数
}
```

これらはモデルと共に保存され、生成時に使用されます。

---

## インターフェース仕様

### IF-1: コマンドラインインターフェース

#### IF-1.1: データ準備コマンド

**コマンド**: `prepare_property_dataset.py`

**使用法**:
```bash
python scripts/prepare_property_dataset.py \
    --input_molecules_db data/molecules.db \
    --input_crystals_db data/crystals.db \
    --property_file data/crystal_properties.csv \
    --output_molecules_db data/molecules_with_props.db \
    --output_crystals_db data/crystals_with_props.db
```

**引数**:
| 引数 | 型 | 説明 | デフォルト |
|-----|-----|------|----------|
| --input_molecules_db | PATH | 入力分子DB | 必須 |
| --input_crystals_db | PATH | 入力結晶DB | 必須 |
| --property_file | PATH | 物性値CSV | 必須 |
| --output_molecules_db | PATH | 出力分子DB | 必須 |
| --output_crystals_db | PATH | 出力結晶DB | 必須 |
| --validate | BOOL | データ検証を実行 | True |
| --verbose | BOOL | 詳細ログ出力 | False |

#### IF-1.2: 訓練コマンド

**コマンド**: `main_crystal_with_properties.py`

**使用法**:
```bash
python main_crystal_with_properties.py \
    --molecule_db_path data/molecules_with_props.db \
    --crystal_db_path data/crystals_with_props.db \
    --property_names bandgap melting_point dielectric_constant \
    --conditioning space_group density \
    --n_epochs 500 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --exp_name crystal_with_properties
```

**引数**:
| 引数 | 型 | 説明 | デフォルト |
|-----|-----|------|----------|
| --molecule_db_path | PATH | 分子DB | 必須 |
| --crystal_db_path | PATH | 結晶DB | 必須 |
| --property_names | LIST[STR] | 物性値リスト | 必須 |
| --conditioning | LIST[STR] | 条件タイプ | [] |
| --n_epochs | INT | エポック数 | 500 |
| --batch_size | INT | バッチサイズ | 32 |
| --learning_rate | FLOAT | 学習率 | 1e-4 |
| --conditioning_dim | INT | 条件次元 | 256 |
| --hidden_dim | INT | 隠れ層次元 | 512 |
| --n_layers | INT | 層数 | 3 |
| --exp_name | STR | 実験名 | 必須 |
| --output_dir | PATH | 出力ディレクトリ | outputs/ |
| --device | STR | デバイス | cuda |
| --num_workers | INT | データローダーワーカー数 | 4 |

#### IF-1.3: 生成コマンド

**コマンド**: `generate_crystal_with_all_conditions.py`

**使用法**:
```bash
python generate_crystal_with_all_conditions.py \
    --model_path outputs/crystal_with_properties/model.pt \
    --molecule_db_path data/molecules_with_props.db \
    --molecular_weight_min 100 \
    --molecular_weight_max 200 \
    --pi_conjugation_ratio_min 0.3 \
    --pi_conjugation_ratio_max 0.7 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --space_group 14 \
    --density 1.2 \
    --n_samples 100 \
    --output_dir generated_crystals/
```

**引数**:
| 引数 | 型 | 説明 | デフォルト |
|-----|-----|------|----------|
| --model_path | PATH | モデルパス | 必須 |
| --molecule_db_path | PATH | 分子DB | 必須 |
| --molecular_weight_min | FLOAT | 分子量下限 | None |
| --molecular_weight_max | FLOAT | 分子量上限 | None |
| --pi_conjugation_ratio_min | FLOAT | π共役比率下限 | None |
| --pi_conjugation_ratio_max | FLOAT | π共役比率上限 | None |
| --target_properties | DICT | 目標物性値 | {} |
| --target_bandgap | FLOAT | バンドギャップ | None |
| --target_melting_point | FLOAT | 融点 | None |
| --space_group | INT | 空間群 | None |
| --density | FLOAT | 密度 | None |
| --n_samples | INT | サンプル数 | 10 |
| --output_dir | PATH | 出力ディレクトリ | 必須 |
| --output_format | STR | 出力形式 | cif |

### IF-2: Python API

#### IF-2.1: データローダーAPI

```python
from crystal.data.crystal_loader import CrystalDatasetWithProperties

# データセットの作成
dataset = CrystalDatasetWithProperties(
    crystal_db_path='data/crystals_with_props.db',
    molecule_db_path='data/molecules_with_props.db',
    property_names=['bandgap', 'melting_point']
)

# データの取得
data = dataset[0]
# data = {
#     'molecular_features': torch.Tensor,
#     'crystal_structure': dict,
#     'properties': torch.Tensor,
#     'space_group': int,
#     'density': float,
# }

# 統計情報の取得
mean = dataset.property_mean  # [property_dim]
std = dataset.property_std    # [property_dim]
```

#### IF-2.2: 条件付けAPI

```python
from crystal.conditioning import PropertyConditioning, ExtendedCombinedConditioning

# PropertyConditioningの作成
prop_cond = PropertyConditioning(
    property_names=['bandgap', 'melting_point'],
    conditioning_dim=256
)

# 正規化パラメータの設定
prop_cond.set_normalization_params(mean, std)

# 条件付けベクトルの計算
properties = torch.tensor([[2.5, 180.0]])  # [batch_size, property_dim]
h_prop = prop_cond(properties)  # [batch_size, conditioning_dim]

# ExtendedCombinedConditioningの使用
combined_cond = ExtendedCombinedConditioning(
    molecular_conditioning=mol_cond,
    property_conditioning=prop_cond,
    conditioning_dim=256
)

h_combined = combined_cond(
    molecular_features=mol_features,
    properties=properties
)
```

#### IF-2.3: 生成API

```python
from crystal.sampling_with_properties import sample_crystals_with_properties

# 結晶の生成
crystals = sample_crystals_with_properties(
    model_path='outputs/crystal_with_properties/model.pt',
    molecule_db_path='data/molecules_with_props.db',
    target_molecule_id='benzene_001',
    target_properties={
        'bandgap': 2.5,
        'melting_point': 180.0
    },
    space_group=14,
    density=1.2,
    n_samples=100,
    output_dir='generated_crystals/'
)

# 結果の取得
for crystal in crystals:
    print(f"Crystal ID: {crystal['crystal_id']}")
    print(f"Lattice: {crystal['lattice_parameters']}")
    print(f"Predicted properties: {crystal['predicted_properties']}")
```

---

## アルゴリズム仕様

### ALG-1: 物性値正規化アルゴリズム

**目的**: 異なるスケールの物性値を統一的に扱う

**入力**:
- 生の物性値: $\mathbf{p}_{\text{raw}} \in \mathbb{R}^d$
- 平均: $\boldsymbol{\mu} \in \mathbb{R}^d$
- 標準偏差: $\boldsymbol{\sigma} \in \mathbb{R}^d$

**出力**:
- 正規化された物性値: $\mathbf{p}_{\text{norm}} \in \mathbb{R}^d$

**アルゴリズム**:
```
1. FOR each property dimension i:
2.     p_norm[i] = (p_raw[i] - μ[i]) / (σ[i] + ε)
3. END FOR
4. RETURN p_norm
```

ここで、$\epsilon = 10^{-8}$ は数値安定性のための小さな定数。

**逆正規化**（生成時に物性値を元のスケールに戻す）:
```
1. FOR each property dimension i:
2.     p_raw[i] = p_norm[i] × σ[i] + μ[i]
3. END FOR
4. RETURN p_raw
```

### ALG-2: 条件付けベクトル統合アルゴリズム

**目的**: 複数の条件を統一的な表現に統合

**入力**:
- 分子条件付け: $\mathbf{h}_{\text{mol}} \in \mathbb{R}^{d_{\text{cond}}}$
- 空間群条件付け（オプション）: $\mathbf{h}_{\text{sg}} \in \mathbb{R}^{d_{\text{cond}}}$
- 密度条件付け（オプション）: $\mathbf{h}_{\rho} \in \mathbb{R}^{d_{\text{cond}}}$
- 物性値条件付け（オプション）: $\mathbf{h}_{\text{prop}} \in \mathbb{R}^{d_{\text{cond}}}$

**出力**:
- 統合条件付け: $\mathbf{h}_{\text{combined}} \in \mathbb{R}^{d_{\text{cond}}}$

**アルゴリズム**:
```
1. conditionings = [h_mol]  # 分子条件は必須
2. 
3. IF space_group is provided:
4.     conditionings.append(h_sg)
5. END IF
6. 
7. IF density is provided:
8.     conditionings.append(h_rho)
9. END IF
10. 
11. IF properties is provided:
12.     conditionings.append(h_prop)
13. END IF
14. 
15. h_concat = CONCATENATE(conditionings)  # [batch, d_cond × n_cond]
16. h_combined = MLP_combine(h_concat)      # [batch, d_cond]
17. 
18. RETURN h_combined
```

### ALG-3: 条件付き拡散サンプリングアルゴリズム

**目的**: 条件に従って結晶構造を生成

**入力**:
- 訓練済みモデル: $\epsilon_\theta$
- 条件付けベクトル: $\mathbf{h}_{\text{cond}}$
- タイムステップ数: $T$
- 原子数: $N$

**出力**:
- 結晶構造: $(\mathbf{R}_0, \mathbf{H}_0)$

**アルゴリズム**（DDPM方式）:
```
1. # 初期化: ランダムノイズから開始
2. R_T ~ N(0, I_{N×3})  # 原子座標
3. G_T ~ N(0, I_{3×3})  # グラム行列
4. 
5. # 逆拡散ループ
6. FOR t = T, T-1, ..., 1:
7.     # ノイズスケジュールパラメータ
8.     α_t = 1 - β_t
9.     ᾱ_t = PRODUCT(α_s for s=1 to t)
10.     
11.     # ノイズを予測
12.     ε_R, ε_G = ε_θ(R_t, G_t, t, h_cond)
13.     
14.     # 平均を計算
15.     μ_R = (1 / √α_t) × (R_t - (β_t / √(1-ᾱ_t)) × ε_R)
16.     μ_G = (1 / √α_t) × (G_t - (β_t / √(1-ᾱ_t)) × ε_G)
17.     
18.     # 分散を計算
19.     σ_t² = β_t
20.     
21.     IF t > 1:
22.         # ノイズをサンプリング
23.         z_R ~ N(0, I)
24.         z_G ~ N(0, I)
25.         
26.         # 次のステップを計算
27.         R_{t-1} = μ_R + σ_t × z_R
28.         G_{t-1} = μ_G + σ_t × z_G
29.     ELSE:
30.         # 最終ステップ（ノイズなし）
31.         R_0 = μ_R
32.         G_0 = μ_G
33.     END IF
34.     
35.     # 周期境界条件を適用
36.     R_t = R_t MOD 1.0  # 分数座標を[0,1)に保つ
37. END FOR
38. 
39. # 格子ベクトルを復元
40. H_0 = CHOLESKY(G_0)
41. 
42. RETURN (R_0, H_0)
```

**ノイズスケジュール**:
- 線形スケジュール: $\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$
- 推奨値: $\beta_{\min} = 10^{-4}$, $\beta_{\max} = 0.02$

---

## 性能要件

### PF-1: 訓練性能

| 指標 | 要件 | 測定方法 |
|-----|------|---------|
| 訓練時間 | < 48時間 / 500エポック（GPU） | 実測 |
| メモリ使用量 | < 10 GB（バッチサイズ32） | nvidia-smi |
| 収束性 | 損失が100エポック以内に安定 | 訓練ログ |

### PF-2: 生成性能

| 指標 | 要件 | 測定方法 |
|-----|------|---------|
| 生成速度 | > 10サンプル/秒（GPU） | 実測 |
| メモリ使用量 | < 4 GB（100サンプル） | nvidia-smi |
| バッチ生成 | 1000サンプル < 5分 | 実測 |

### PF-3: 精度要件

| 指標 | 要件 | 測定方法 |
|-----|------|---------|
| 物性値MAE | < 10%（目標値の） | 予測vs目標 |
| 構造的妥当性 | > 95%のサンプルが物理的に妥当 | 構造検証 |
| 対称性保持 | > 90%のサンプルが空間群制約を満たす | 対称性解析 |

---

## 制約事項

### CT-1: データ制約

1. **物性値データの可用性**
   - 各結晶に少なくとも1つの物性値が必要
   - 推奨: 3つ以上の物性値

2. **データ品質**
   - 物性値は第一原理計算または実験で得られたものであること
   - 欠損値は許容されない（補完が必要）

3. **データ量**
   - 最小訓練データ: 1000結晶構造
   - 推奨訓練データ: 10000結晶構造以上

### CT-2: 計算制約

1. **ハードウェア**
   - GPU訓練を強く推奨（CPU訓練は非常に遅い）
   - 最小VRAM: 8 GB
   - 推奨VRAM: 16 GB以上

2. **ソフトウェア**
   - CUDA 11.3以上が必要
   - PyTorch 1.12以上が必要

### CT-3: モデル制約

1. **物性値の範囲**
   - 訓練データの範囲外の物性値への外挿は精度が低下
   - 推奨: 訓練データの範囲内±20%

2. **条件の組み合わせ**
   - 訓練時に見なかった条件の組み合わせは予測不能
   - 推奨: 訓練データに多様な条件の組み合わせを含める

3. **結晶サイズ**
   - 最大原子数: 200原子（メモリ制約）
   - 推奨原子数: 50-100原子

### CT-4: 使用制約

1. **Fallbackの禁止**
   - ヒューリスティックなfallbackは使用しない
   - すべての処理は理論的に正当であること

2. **等変性の保証**
   - すべての操作でE(3)等変性を維持すること
   - 座標系に依存する操作は禁止

---

## 検証要件

### VR-1: 単体テスト

#### VR-1.1: PropertyConditioning

**テスト項目**:
1. 入力形状の検証
2. 正規化の正確性
3. 出力形状の検証
4. 勾配の伝播

**テストコード例**:
```python
def test_property_conditioning():
    # セットアップ
    prop_cond = PropertyConditioning(
        property_names=['bandgap', 'melting_point'],
        conditioning_dim=256
    )
    mean = torch.tensor([2.5, 180.0])
    std = torch.tensor([1.2, 50.0])
    prop_cond.set_normalization_params(mean, std)
    
    # テスト1: 入力形状
    properties = torch.tensor([[2.5, 180.0], [3.0, 200.0]])
    h_prop = prop_cond(properties)
    assert h_prop.shape == (2, 256)
    
    # テスト2: 正規化
    # properties[0] は平均なので、正規化後は約0
    h_prop_mean = prop_cond(mean.unsqueeze(0))
    # (内部的には約0ベクトルが入力されている)
    
    # テスト3: 勾配
    h_prop = prop_cond(properties)
    loss = h_prop.sum()
    loss.backward()
    assert properties.grad is not None
```

#### VR-1.2: ExtendedCombinedConditioning

**テスト項目**:
1. 分子条件のみ
2. 分子条件 + 物性値条件
3. すべての条件

#### VR-1.3: データローダー

**テスト項目**:
1. データの読み込み
2. 物性値の正しい抽出
3. 統計情報の計算
4. バッチング

### VR-2: 統合テスト

#### VR-2.1: 訓練パイプライン

**テスト項目**:
1. データの読み込みから訓練まで
2. チェックポイントの保存・読み込み
3. 損失の減少

**テスト方法**:
- 小規模データセット（100サンプル）で10エポック訓練
- 損失が減少することを確認

#### VR-2.2: 生成パイプライン

**テスト項目**:
1. モデルの読み込み
2. 条件の設定
3. 結晶の生成
4. 出力ファイルの作成

**テスト方法**:
- 訓練済みモデルで10サンプル生成
- すべてのサンプルが有効な結晶構造であることを確認

### VR-3: 性能テスト

#### VR-3.1: 訓練速度

**測定項目**:
- イテレーション/秒
- エポック時間
- GPU使用率

**合格基準**:
- > 5 イテレーション/秒（バッチサイズ32、GPU）

#### VR-3.2: 生成速度

**測定項目**:
- サンプル/秒
- メモリ使用量

**合格基準**:
- > 10 サンプル/秒（GPU）

### VR-4: 精度テスト

#### VR-4.1: 物性値精度

**測定方法**:
1. テストセットで結晶を生成
2. 物性値予測モデルで生成結晶の物性値を推定
3. 目標物性値との誤差を計算

**合格基準**:
- MAE < 10%（目標値の）
- RMSE < 15%

#### VR-4.2: 構造的妥当性

**検証項目**:
1. 原子間距離の妥当性（> 0.5 Å）
2. 格子定数の妥当性（> 0、< 100 Å）
3. 密度の妥当性（> 0、< 5 g/cm³）
4. 周期境界の連続性

**合格基準**:
- 95%以上のサンプルがすべての基準を満たす

#### VR-4.3: 対称性の検証

**測定方法**:
1. 生成された結晶の空間群を自動判定（spglibを使用）
2. 指定された空間群との一致を確認

**合格基準**:
- 90%以上のサンプルが正しい空間群を持つ

---

## 付録

### 付録A: エラーコード

| コード | 説明 | 対処法 |
|-------|------|--------|
| E001 | データベースが見つからない | パスを確認 |
| E002 | 物性値が欠損している | データを補完 |
| E003 | 正規化パラメータが未設定 | set_normalization_params()を呼び出す |
| E004 | GPU メモリ不足 | バッチサイズを減らす |
| E005 | モデルファイルが破損 | 再訓練が必要 |
| E006 | 条件の組み合わせが不正 | 条件を確認 |

### 付録B: 設定ファイル例

**config.yaml**:
```yaml
data:
  molecule_db_path: "data/molecules_with_props.db"
  crystal_db_path: "data/crystals_with_props.db"
  property_names:
    - "bandgap"
    - "melting_point"
    - "dielectric_constant"

model:
  conditioning_dim: 256
  hidden_dim: 512
  n_layers: 3
  n_timesteps: 1000

training:
  n_epochs: 500
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-6
  grad_clip: 1.0

conditioning:
  types:
    - "molecular"
    - "space_group"
    - "density"
    - "property"

output:
  exp_name: "crystal_with_properties"
  output_dir: "outputs/"
  save_interval: 50
  log_interval: 10
```

### 付録C: データベーススキーマ

**molecules.db**:
```sql
CREATE TABLE IF NOT EXISTS systems (
    id INTEGER PRIMARY KEY,
    unique_id TEXT UNIQUE,
    ctime REAL,
    mtime REAL,
    username TEXT,
    numbers BLOB,
    positions BLOB,
    cell BLOB,
    pbc BLOB,
    initial_magmoms BLOB,
    initial_charges BLOB,
    masses BLOB,
    tags BLOB,
    momenta BLOB,
    constraints TEXT,
    calculator TEXT,
    calculator_parameters TEXT,
    energy REAL,
    free_energy REAL,
    forces BLOB,
    stress BLOB,
    dipole BLOB,
    magmoms BLOB,
    magmom REAL,
    charges BLOB,
    key_value_pairs TEXT,
    data TEXT
);
```

**crystals.db**: 同じスキーマ（ASE標準）

---

**文書作成者**: システム設計チーム  
**最終更新日**: 2025-10-24  
**バージョン**: 1.0  
**承認者**: プロジェクトリーダー  
**ステータス**: レビュー待ち
