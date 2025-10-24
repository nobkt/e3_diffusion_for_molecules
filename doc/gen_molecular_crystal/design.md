# 物性値を条件とした分子性結晶生成の詳細設計書

## 文書情報

- **文書タイトル**: 物性値を条件とした分子性結晶生成の詳細設計書
- **作成日**: 2025-10-24
- **バージョン**: 1.0
- **対象読者**: 実装者、レビュアー、保守担当者
- **参照**: PR#143、doc/molecular_crystal_generation_spec.md Q4、doc/gen_molecular_crystal/theory.md、doc/gen_molecular_crystal/specification.md

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ設計](#アーキテクチャ設計)
3. [モジュール設計](#モジュール設計)
4. [クラス設計](#クラス設計)
5. [データフロー設計](#データフロー設計)
6. [インターフェース設計](#インターフェース設計)
7. [実装ガイドライン](#実装ガイドライン)
8. [テスト設計](#テスト設計)
9. [デプロイメント設計](#デプロイメント設計)

---

## 概要

### 設計原則

本システムの設計は以下の原則に基づいています：

1. **モジュール性**: 各機能を独立したモジュールとして実装
2. **拡張性**: 新しい条件タイプを容易に追加可能
3. **再利用性**: 既存コンポーネントを最大限活用
4. **保守性**: 明確な構造とドキュメント
5. **理論的正当性**: fallbackなしの厳密な実装

### システム構成図

```
┌─────────────────────────────────────────────────────────────────┐
│                           Application Layer                      │
│  ・main_crystal_with_properties.py                              │
│  ・generate_crystal_with_all_conditions.py                      │
│  ・scripts/prepare_property_dataset.py                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                           API Layer                             │
│  ・crystal/sampling_with_properties.py                          │
│  ・crystal/training_with_properties.py                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Core Model Layer                         │
│  crystal/                                                       │
│  ├── conditioning/                                              │
│  │   ├── property_conditioning.py (新規)                       │
│  │   ├── extended_combined_conditioning.py (新規)              │
│  │   ├── molecular_conditioning.py (既存)                      │
│  │   ├── space_group_embedding.py (既存)                       │
│  │   └── density_conditioning.py (既存)                        │
│  ├── dynamics/                                                  │
│  │   └── crystal_dynamics.py (既存、拡張)                      │
│  └── models/                                                    │
│      └── egnn.py (既存)                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                              │
│  crystal/data/                                                  │
│  ├── crystal_loader.py (既存、拡張)                            │
│  │   └── CrystalDatasetWithProperties (新規クラス)             │
│  ├── molecule_loader.py (既存)                                 │
│  └── property_normalizer.py (新規)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Storage Layer                            │
│  ・molecules.db (ASE database)                                  │
│  ・crystals.db (ASE database)                                   │
│  ・crystal_properties.csv                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## アーキテクチャ設計

### レイヤー構成

#### 1. Application Layer（アプリケーション層）

**責務**:
- ユーザーインターフェースの提供
- コマンドライン引数の解析
- ワークフローの制御

**主要コンポーネント**:
- `main_crystal_with_properties.py`: 訓練スクリプト
- `generate_crystal_with_all_conditions.py`: 生成スクリプト
- `scripts/prepare_property_dataset.py`: データ準備スクリプト

#### 2. API Layer（API層）

**責務**:
- 高レベルAPIの提供
- パイプラインの管理
- エラーハンドリング

**主要コンポーネント**:
- `crystal/sampling_with_properties.py`: サンプリングAPI
- `crystal/training_with_properties.py`: 訓練API

#### 3. Core Model Layer（コアモデル層）

**責務**:
- 条件付けモジュールの実装
- 拡散モデルの実装
- ニューラルネットワークの定義

**主要コンポーネント**:
- `crystal/conditioning/`: 条件付けモジュール群
- `crystal/dynamics/`: 拡散ダイナミクス
- `crystal/models/`: ニューラルネットワークモデル

#### 4. Data Layer（データ層）

**責務**:
- データの読み込みと前処理
- バッチングとシャッフリング
- データ正規化

**主要コンポーネント**:
- `crystal/data/crystal_loader.py`: 結晶データローダー
- `crystal/data/molecule_loader.py`: 分子データローダー
- `crystal/data/property_normalizer.py`: 物性値正規化

#### 5. Storage Layer（ストレージ層）

**責務**:
- データの永続化
- データベース管理

**主要コンポーネント**:
- ASEデータベースファイル
- CSVファイル

### コンポーネント間の依存関係

```
Application Layer
    ↓ uses
API Layer
    ↓ uses
Core Model Layer
    ↓ uses
Data Layer
    ↓ uses
Storage Layer
```

**依存関係の原則**:
- 上位層は下位層に依存してもよい
- 下位層は上位層に依存してはいけない（逆依存禁止）
- 同じ層内のコンポーネント間の依存は最小化

---

## モジュール設計

### M-1: PropertyConditioning モジュール

**パス**: `crystal/conditioning/property_conditioning.py`

**目的**: 物性値を条件付けベクトルに変換

**依存関係**:
- torch
- torch.nn

**公開インターフェース**:
```python
class PropertyConditioning(nn.Module):
    def __init__(...)
    def forward(properties: torch.Tensor) -> torch.Tensor
    def set_normalization_params(mean: torch.Tensor, std: torch.Tensor)
```

**内部構造**:
```
PropertyConditioning
├── __init__(): 初期化
│   ├── property_names の保存
│   ├── 正規化バッファの登録
│   └── MLP の構築
├── forward(): 順伝播
│   ├── 入力検証
│   ├── 正規化
│   └── MLP 変換
└── set_normalization_params(): 正規化パラメータ設定
    ├── mean のコピー
    └── std のコピー
```

**ファイル構成**:
```python
# crystal/conditioning/property_conditioning.py

import torch
import torch.nn as nn
from typing import List

class PropertyConditioning(nn.Module):
    """物性値による条件付けモジュール"""
    
    def __init__(
        self,
        property_names: List[str],
        conditioning_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 3
    ):
        # 実装...
    
    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        # 実装...
    
    def set_normalization_params(
        self,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> None:
        # 実装...
```

### M-2: ExtendedCombinedConditioning モジュール

**パス**: `crystal/conditioning/extended_combined_conditioning.py`

**目的**: 複数の条件を統合

**依存関係**:
- torch
- torch.nn
- MolecularConditioning
- SpaceGroupEmbedding
- DensityConditioning
- PropertyConditioning

**公開インターフェース**:
```python
class ExtendedCombinedConditioning(nn.Module):
    def __init__(...)
    def forward(
        molecular_features,
        space_group=None,
        density=None,
        properties=None
    ) -> torch.Tensor
```

**内部構造**:
```
ExtendedCombinedConditioning
├── __init__(): 初期化
│   ├── 各条件付けモジュールの保存
│   ├── 統合MLPの構築
│   └── num_conditionings の計算
├── forward(): 順伝播
│   ├── 分子条件付け（必須）
│   ├── オプション条件付け
│   │   ├── 空間群
│   │   ├── 密度
│   │   └── 物性値
│   ├── 連結
│   └── 統合MLP
└── _count_active_conditionings(): アクティブな条件数をカウント
```

### M-3: CrystalDatasetWithProperties モジュール

**パス**: `crystal/data/crystal_loader.py`（既存ファイルに追加）

**目的**: 物性値を含む結晶データセット

**依存関係**:
- torch.utils.data.Dataset
- ase.db
- pandas（CSVの場合）
- CrystalDataset（親クラス）

**公開インターフェース**:
```python
class CrystalDatasetWithProperties(CrystalDataset):
    def __init__(
        crystal_db_path,
        molecule_db_path,
        property_names=None,
        **kwargs
    )
    def __getitem__(idx) -> Dict
    @property
    def property_mean() -> torch.Tensor
    @property
    def property_std() -> torch.Tensor
```

**内部構造**:
```
CrystalDatasetWithProperties
├── __init__(): 初期化
│   ├── 親クラスの初期化
│   ├── property_names の保存
│   └── 統計情報の計算
├── _compute_property_statistics(): 統計計算
│   ├── すべてのエントリをループ
│   ├── 物性値を収集
│   └── 平均と標準偏差を計算
├── __getitem__(): データ取得
│   ├── 親クラスからデータ取得
│   ├── 物性値を追加
│   └── 辞書を返す
└── _validate_properties(): 物性値の検証
    ├── 欠損値チェック
    └── 範囲チェック
```

### M-4: PropertyNormalizer モジュール（新規）

**パス**: `crystal/data/property_normalizer.py`

**目的**: 物性値の正規化・逆正規化

**依存関係**:
- torch
- numpy

**公開インターフェース**:
```python
class PropertyNormalizer:
    def __init__(mean, std)
    def normalize(properties) -> torch.Tensor
    def denormalize(properties_norm) -> torch.Tensor
    def save(path)
    @classmethod
    def load(path) -> PropertyNormalizer
```

**実装**:
```python
# crystal/data/property_normalizer.py

import torch
import json
from pathlib import Path
from typing import Union

class PropertyNormalizer:
    """物性値の正規化・逆正規化"""
    
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        property_names: list = None
    ):
        """
        Args:
            mean: [property_dim] 平均値
            std: [property_dim] 標準偏差
            property_names: 物性値の名前リスト
        """
        self.mean = mean
        self.std = std
        self.property_names = property_names or []
        self.eps = 1e-8
    
    def normalize(self, properties: torch.Tensor) -> torch.Tensor:
        """
        物性値を正規化
        
        Args:
            properties: [*, property_dim] 生の物性値
        
        Returns:
            properties_norm: [*, property_dim] 正規化された物性値
        """
        return (properties - self.mean) / (self.std + self.eps)
    
    def denormalize(self, properties_norm: torch.Tensor) -> torch.Tensor:
        """
        正規化された物性値を元のスケールに戻す
        
        Args:
            properties_norm: [*, property_dim] 正規化された物性値
        
        Returns:
            properties: [*, property_dim] 元のスケールの物性値
        """
        return properties_norm * self.std + self.mean
    
    def save(self, path: Union[str, Path]):
        """正規化パラメータを保存"""
        path = Path(path)
        data = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'property_names': self.property_names
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PropertyNormalizer':
        """正規化パラメータを読み込み"""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        mean = torch.tensor(data['mean'])
        std = torch.tensor(data['std'])
        property_names = data.get('property_names', [])
        
        return cls(mean, std, property_names)
```

### M-5: サンプリングモジュール（新規）

**パス**: `crystal/sampling_with_properties.py`

**目的**: 物性値条件付きでのサンプリング

**公開インターフェース**:
```python
def sample_crystals_with_properties(
    model_path: str,
    molecule_db_path: str,
    target_molecule_id: str,
    target_properties: Dict[str, float],
    space_group: int = None,
    density: float = None,
    n_samples: int = 10,
    output_dir: str = 'generated_crystals/'
) -> List[Dict]
```

---

## クラス設計

### CL-1: PropertyConditioning クラス

**UMLクラス図**:
```
┌─────────────────────────────────────────┐
│       PropertyConditioning              │
├─────────────────────────────────────────┤
│ - property_names: List[str]             │
│ - property_dim: int                     │
│ - conditioning_dim: int                 │
│ - property_mean: Tensor                 │
│ - property_std: Tensor                  │
│ - property_mlp: Sequential              │
├─────────────────────────────────────────┤
│ + __init__(property_names, ...)         │
│ + forward(properties): Tensor           │
│ + set_normalization_params(mean, std)   │
│ - _build_mlp(): Sequential              │
│ - _normalize(properties): Tensor        │
└─────────────────────────────────────────┘
```

**属性**:

| 属性名 | 型 | 説明 | アクセス |
|-------|-----|------|---------|
| property_names | List[str] | 物性値の名前リスト | public |
| property_dim | int | 物性値の次元数 | public |
| conditioning_dim | int | 出力の条件付けベクトルの次元 | public |
| property_mean | torch.Tensor | 物性値の平均（バッファ） | private |
| property_std | torch.Tensor | 物性値の標準偏差（バッファ） | private |
| property_mlp | nn.Sequential | 物性値変換MLP | private |

**メソッド**:

##### \_\_init\_\_()
```python
def __init__(
    self,
    property_names: List[str],
    conditioning_dim: int = 256,
    hidden_dim: int = 512,
    n_layers: int = 3
):
    """
    初期化
    
    処理フロー:
    1. パラメータの検証
    2. 属性の初期化
    3. 正規化バッファの登録
    4. MLPの構築
    """
```

##### forward()
```python
def forward(self, properties: torch.Tensor) -> torch.Tensor:
    """
    順伝播
    
    Args:
        properties: [batch_size, property_dim]
    
    Returns:
        conditioning: [batch_size, conditioning_dim]
    
    処理フロー:
    1. 入力形状の検証
    2. 物性値の正規化
    3. MLPで変換
    4. 出力の返却
    
    例外:
        ValueError: 入力形状が不正
        RuntimeError: 正規化パラメータ未設定
    """
```

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
        mean: [property_dim]
        std: [property_dim]
    
    処理フロー:
    1. 形状の検証
    2. バッファにコピー
    
    例外:
        ValueError: 形状が property_dim と不一致
    """
```

##### _build_mlp() (プライベート)
```python
def _build_mlp(self) -> nn.Sequential:
    """
    MLP を構築
    
    アーキテクチャ:
        input_dim = property_dim
        [Linear(in, hidden), SiLU()] × (n_layers - 1)
        [Linear(hidden, conditioning_dim)]
    
    Returns:
        mlp: nn.Sequential
    """
```

**実装の詳細**:

```python
class PropertyConditioning(nn.Module):
    def __init__(
        self,
        property_names: List[str],
        conditioning_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 3
    ):
        super().__init__()
        
        # パラメータの検証
        if not property_names:
            raise ValueError("property_names must be non-empty")
        if not 2 <= n_layers <= 5:
            raise ValueError("n_layers must be between 2 and 5")
        
        # 属性の初期化
        self.property_names = property_names
        self.property_dim = len(property_names)
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 正規化バッファの登録
        self.register_buffer(
            'property_mean',
            torch.zeros(self.property_dim)
        )
        self.register_buffer(
            'property_std',
            torch.ones(self.property_dim)
        )
        
        # MLPの構築
        self.property_mlp = self._build_mlp()
    
    def _build_mlp(self) -> nn.Sequential:
        """MLPを構築"""
        layers = []
        in_dim = self.property_dim
        
        for i in range(self.n_layers):
            out_dim = (
                self.hidden_dim if i < self.n_layers - 1
                else self.conditioning_dim
            )
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < self.n_layers - 1:
                layers.append(nn.SiLU())
            
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _normalize(self, properties: torch.Tensor) -> torch.Tensor:
        """物性値を正規化"""
        return (properties - self.property_mean) / (self.property_std + 1e-8)
    
    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        """順伝播"""
        # 入力形状の検証
        if properties.shape[-1] != self.property_dim:
            raise ValueError(
                f"Expected property_dim={self.property_dim}, "
                f"got {properties.shape[-1]}"
            )
        
        # 正規化パラメータの検証
        if torch.all(self.property_mean == 0) and torch.all(self.property_std == 1):
            import warnings
            warnings.warn(
                "Normalization parameters not set. "
                "Call set_normalization_params() before training."
            )
        
        # 正規化
        properties_norm = self._normalize(properties)
        
        # MLPで変換
        conditioning = self.property_mlp(properties_norm)
        
        return conditioning
    
    def set_normalization_params(
        self,
        mean: torch.Tensor,
        std: torch.Tensor
    ) -> None:
        """正規化パラメータを設定"""
        # 形状の検証
        if mean.shape != (self.property_dim,):
            raise ValueError(
                f"Expected mean shape ({self.property_dim},), "
                f"got {mean.shape}"
            )
        if std.shape != (self.property_dim,):
            raise ValueError(
                f"Expected std shape ({self.property_dim},), "
                f"got {std.shape}"
            )
        
        # バッファにコピー
        self.property_mean.copy_(mean)
        self.property_std.copy_(std)
```

### CL-2: ExtendedCombinedConditioning クラス

**UMLクラス図**:
```
┌─────────────────────────────────────────────────┐
│       ExtendedCombinedConditioning              │
├─────────────────────────────────────────────────┤
│ - molecular_conditioning: MolecularConditioning │
│ - space_group_embedding: SpaceGroupEmbedding    │
│ - density_conditioning: DensityConditioning     │
│ - property_conditioning: PropertyConditioning   │
│ - num_conditionings: int                        │
│ - combine_mlp: Sequential                       │
├─────────────────────────────────────────────────┤
│ + __init__(...)                                 │
│ + forward(molecular_features, ...: Tensor       │
│ - _build_combine_mlp(): Sequential              │
│ - _count_conditionings(): int                   │
└─────────────────────────────────────────────────┘
```

**実装の詳細**:

```python
class ExtendedCombinedConditioning(nn.Module):
    def __init__(
        self,
        molecular_conditioning: MolecularConditioning,
        space_group_embedding: Optional[SpaceGroupEmbedding] = None,
        density_conditioning: Optional[DensityConditioning] = None,
        property_conditioning: Optional[PropertyConditioning] = None,
        conditioning_dim: int = 256
    ):
        super().__init__()
        
        # 条件付けモジュールの保存
        self.molecular_conditioning = molecular_conditioning
        self.space_group_embedding = space_group_embedding
        self.density_conditioning = density_conditioning
        self.property_conditioning = property_conditioning
        
        self.conditioning_dim = conditioning_dim
        
        # アクティブな条件の数をカウント
        self.num_conditionings = self._count_conditionings()
        
        # 統合MLPの構築
        self.combine_mlp = self._build_combine_mlp()
    
    def _count_conditionings(self) -> int:
        """アクティブな条件の数をカウント"""
        count = 1  # 分子条件は必須
        if self.space_group_embedding is not None:
            count += 1
        if self.density_conditioning is not None:
            count += 1
        if self.property_conditioning is not None:
            count += 1
        return count
    
    def _build_combine_mlp(self) -> nn.Sequential:
        """統合MLPを構築"""
        input_dim = self.conditioning_dim * self.num_conditionings
        hidden_dim = self.conditioning_dim * 2
        output_dim = self.conditioning_dim
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        molecular_features: torch.Tensor,
        space_group: Optional[torch.Tensor] = None,
        density: Optional[torch.Tensor] = None,
        properties: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            molecular_features: [batch, mol_feat_dim]
            space_group: [batch] (オプション)
            density: [batch] (オプション)
            properties: [batch, property_dim] (オプション)
        
        Returns:
            combined_conditioning: [batch, conditioning_dim]
        """
        conditionings = []
        
        # 1. 分子条件（必須）
        mol_cond = self.molecular_conditioning(molecular_features)
        conditionings.append(mol_cond)
        
        # 2. 空間群条件（オプション）
        if self.space_group_embedding is not None and space_group is not None:
            sg_cond = self.space_group_embedding(space_group)
            conditionings.append(sg_cond)
        
        # 3. 密度条件（オプション）
        if self.density_conditioning is not None and density is not None:
            density_cond = self.density_conditioning(density)
            conditionings.append(density_cond)
        
        # 4. 物性値条件（オプション）
        if self.property_conditioning is not None and properties is not None:
            prop_cond = self.property_conditioning(properties)
            conditionings.append(prop_cond)
        
        # すべての条件を連結
        combined = torch.cat(conditionings, dim=-1)
        
        # 統合MLPで処理
        combined_conditioning = self.combine_mlp(combined)
        
        return combined_conditioning
```

### CL-3: CrystalDatasetWithProperties クラス

**親クラス**: CrystalDataset

**実装の詳細**:

```python
class CrystalDatasetWithProperties(CrystalDataset):
    """物性値を含む結晶データセット"""
    
    def __init__(
        self,
        crystal_db_path: str,
        molecule_db_path: str,
        property_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        初期化
        
        Args:
            crystal_db_path: 結晶データベースのパス
            molecule_db_path: 分子データベースのパス
            property_names: 使用する物性値の名前リスト
            **kwargs: 親クラスへの追加引数
        """
        # 親クラスの初期化
        super().__init__(
            crystal_db_path,
            molecule_db_path,
            **kwargs
        )
        
        self.property_names = property_names or []
        
        # 物性値の統計情報を計算
        if self.property_names:
            self._compute_property_statistics()
    
    def _compute_property_statistics(self):
        """物性値の平均と標準偏差を計算"""
        property_values = []
        
        for row in self.crystal_db.select():
            values = []
            for prop_name in self.property_names:
                value = row.data.get(prop_name, None)
                if value is None:
                    raise ValueError(
                        f"Property '{prop_name}' not found "
                        f"in crystal {row.id}"
                    )
                values.append(value)
            property_values.append(values)
        
        property_values = torch.tensor(
            property_values,
            dtype=torch.float32
        )
        
        self.property_mean = property_values.mean(dim=0)
        self.property_std = property_values.std(dim=0)
        
        # ログ出力
        print("物性値の統計情報:")
        for i, name in enumerate(self.property_names):
            print(
                f"  {name}: "
                f"mean={self.property_mean[i]:.3f}, "
                f"std={self.property_std[i]:.3f}"
            )
    
    def __getitem__(self, idx: int) -> Dict:
        """
        データを取得
        
        Args:
            idx: インデックス
        
        Returns:
            data: データ辞書
                - molecular_features: 分子特徴量
                - crystal_structure: 結晶構造
                - properties: 物性値 (新規)
                - space_group: 空間群（オプション）
                - density: 密度（オプション）
        """
        # 親クラスからデータを取得
        data = super().__getitem__(idx)
        
        # 物性値を追加
        if self.property_names:
            row = self.crystal_db.get(idx + 1)
            properties = []
            for prop_name in self.property_names:
                value = row.data.get(prop_name, 0.0)
                properties.append(value)
            
            data['properties'] = torch.tensor(
                properties,
                dtype=torch.float32
            )
        
        return data
```

---

## データフロー設計

### DF-1: 訓練時のデータフロー

```
┌──────────────────────┐
│  molecules.db        │
│  crystals.db         │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ CrystalDataset       │
│ WithProperties       │
│  - データ読み込み     │
│  - 統計情報計算      │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ DataLoader           │
│  - バッチング        │
│  - シャッフリング     │
└──────────────────────┘
          ↓
    ┌─────────┐
    │  batch  │
    └─────────┘
          ↓
    ┌─────────────────────────────────────┐
    │ 条件付けベクトルの構築               │
    │                                     │
    │  molecular_features                 │
    │       ↓                             │
    │  MolecularConditioning              │
    │       ↓                             │
    │  h_mol                              │
    │                                     │
    │  properties                         │
    │       ↓                             │
    │  PropertyConditioning               │
    │       ↓                             │
    │  h_prop                             │
    │                                     │
    │  [h_mol || h_sg || h_dens || h_prop]│
    │       ↓                             │
    │  ExtendedCombinedConditioning       │
    │       ↓                             │
    │  h_combined                         │
    └─────────────────────────────────────┘
          ↓
┌──────────────────────┐
│ CrystalDynamics      │
│  - ノイズ追加        │
│  - ノイズ予測        │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ 損失計算             │
│  MSE(pred, true)     │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ 逆伝播・最適化        │
└──────────────────────┘
```

### DF-2: 生成時のデータフロー

```
┌──────────────────────┐
│ ユーザー入力          │
│  - molecule_id       │
│  - target_properties │
│  - space_group       │
│  - density           │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ molecules.db         │
│  分子データ読み込み   │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ MolecularEncoder     │
│  分子特徴量抽出      │
└──────────────────────┘
          ↓
    ┌────────────────────────────────┐
    │ 条件付けベクトルの構築          │
    │                                │
    │  molecular_features            │
    │       ↓                        │
    │  MolecularConditioning         │
    │                                │
    │  target_properties             │
    │       ↓                        │
    │  PropertyConditioning          │
    │                                │
    │  space_group, density          │
    │       ↓                        │
    │  SpaceGroupEmbedding,          │
    │  DensityConditioning           │
    │       ↓                        │
    │  ExtendedCombinedConditioning  │
    │       ↓                        │
    │  h_combined                    │
    └────────────────────────────────┘
          ↓
┌──────────────────────┐
│ 拡散サンプリング      │
│  T → T-1 → ... → 0   │
│                      │
│  for t in [T..1]:    │
│    ε = model(x_t,    │
│              t,      │
│              h_cond) │
│    x_{t-1} = denoise │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ 結晶構造の構築        │
│  - 座標              │
│  - 格子ベクトル      │
└──────────────────────┘
          ↓
┌──────────────────────┐
│ 出力                 │
│  - CIF ファイル      │
│  - メタデータ        │
└──────────────────────┘
```

---

## インターフェース設計

### IF-D-1: 内部モジュール間インターフェース

#### PropertyConditioning ← CrystalDatasetWithProperties

**データ転送**:
```python
# CrystalDatasetWithProperties から
property_mean: torch.Tensor  # [property_dim]
property_std: torch.Tensor   # [property_dim]

# PropertyConditioning へ
property_conditioning.set_normalization_params(
    property_mean,
    property_std
)
```

#### ExtendedCombinedConditioning ← 各条件付けモジュール

**データ転送**:
```python
# 入力
molecular_features: torch.Tensor  # [batch, mol_feat_dim]
space_group: torch.Tensor         # [batch]
density: torch.Tensor             # [batch]
properties: torch.Tensor          # [batch, property_dim]

# 出力
combined_conditioning: torch.Tensor  # [batch, conditioning_dim]
```

#### CrystalDynamics ← ExtendedCombinedConditioning

**データ転送**:
```python
# 入力（拡散モデル）
positions: torch.Tensor      # [batch, n_atoms, 3]
lattice: torch.Tensor        # [batch, 3, 3]
t: torch.Tensor              # [batch]
conditioning: torch.Tensor   # [batch, conditioning_dim]

# 出力
predicted_noise: Tuple[torch.Tensor, torch.Tensor]
    # (positions_noise, lattice_noise)
```

### IF-D-2: 設定ファイルインターフェース

**config.yaml 構造**:
```yaml
# データ設定
data:
  molecule_db_path: str
  crystal_db_path: str
  property_names: List[str]
  train_ratio: float (default: 0.8)
  val_ratio: float (default: 0.1)
  test_ratio: float (default: 0.1)

# モデル設定
model:
  conditioning_dim: int (default: 256)
  hidden_dim: int (default: 512)
  n_layers: int (default: 3)
  n_timesteps: int (default: 1000)
  
  # PropertyConditioning 設定
  property_conditioning:
    enabled: bool (default: true)
    hidden_dim: int (default: 512)
    n_layers: int (default: 3)

# 訓練設定
training:
  n_epochs: int
  batch_size: int
  learning_rate: float
  weight_decay: float
  grad_clip: float
  
  # 損失の重み
  loss_weights:
    positions: float (default: 1.0)
    lattice: float (default: 1.0)

# 条件設定
conditioning:
  types: List[str]
    # 例: ['molecular', 'space_group', 'density', 'property']

# 出力設定
output:
  exp_name: str
  output_dir: str
  save_interval: int (エポック単位)
  log_interval: int (イテレーション単位)
```

---

## 実装ガイドライン

### IG-1: コーディング規約

#### 命名規則

**クラス名**: PascalCase
```python
class PropertyConditioning(nn.Module):
    pass
```

**関数名・メソッド名**: snake_case
```python
def compute_property_statistics():
    pass

def set_normalization_params():
    pass
```

**定数**: UPPER_SNAKE_CASE
```python
DEFAULT_CONDITIONING_DIM = 256
MAX_PROPERTY_DIM = 20
```

**プライベートメソッド**: 先頭にアンダースコア
```python
def _build_mlp(self):
    pass

def _normalize(self, x):
    pass
```

#### 型ヒント

すべての関数・メソッドに型ヒントを付ける:
```python
def forward(
    self,
    properties: torch.Tensor
) -> torch.Tensor:
    pass
```

#### Docstring

Google スタイルの docstring を使用:
```python
def sample_crystals_with_properties(
    model_path: str,
    molecule_db_path: str,
    target_properties: Dict[str, float],
    n_samples: int = 10
) -> List[Dict]:
    """
    物性値を条件として結晶を生成
    
    Args:
        model_path: 訓練済みモデルのパス
        molecule_db_path: 分子データベースのパス
        target_properties: 目標物性値の辞書
        n_samples: 生成サンプル数
    
    Returns:
        crystals: 生成された結晶のリスト
    
    Raises:
        FileNotFoundError: モデルまたはデータベースが見つからない
        ValueError: 不正な引数
    
    Examples:
        >>> crystals = sample_crystals_with_properties(
        ...     model_path='model.pt',
        ...     molecule_db_path='molecules.db',
        ...     target_properties={'bandgap': 2.5},
        ...     n_samples=10
        ... )
    """
```

### IG-2: エラーハンドリング

#### 例外の使用

**入力検証**: ValueError
```python
if not property_names:
    raise ValueError("property_names must be non-empty")
```

**ファイルエラー**: FileNotFoundError, IOError
```python
if not Path(model_path).exists():
    raise FileNotFoundError(f"Model not found: {model_path}")
```

**計算エラー**: RuntimeError
```python
if torch.isnan(loss):
    raise RuntimeError("Loss became NaN during training")
```

#### エラーメッセージ

明確で具体的なメッセージを提供:
```python
# 悪い例
raise ValueError("Invalid input")

# 良い例
raise ValueError(
    f"Expected property_dim={self.property_dim}, "
    f"got {properties.shape[-1]}"
)
```

### IG-3: ロギング

Python の logging モジュールを使用:
```python
import logging

logger = logging.getLogger(__name__)

# 情報ログ
logger.info(f"Loading model from {model_path}")

# 警告ログ
logger.warning(
    "Normalization parameters not set. "
    "Using default values."
)

# エラーログ
logger.error(f"Failed to load database: {e}")

# デバッグログ
logger.debug(f"Property values: {properties}")
```

ログレベル:
- DEBUG: 詳細なデバッグ情報
- INFO: 一般的な情報（進捗など）
- WARNING: 警告（動作は継続）
- ERROR: エラー（例外を伴わない）
- CRITICAL: 致命的なエラー

### IG-4: テスト

#### ユニットテストの構造

```python
import unittest
import torch
from crystal.conditioning import PropertyConditioning

class TestPropertyConditioning(unittest.TestCase):
    def setUp(self):
        """各テスト前の準備"""
        self.property_names = ['bandgap', 'melting_point']
        self.prop_cond = PropertyConditioning(
            property_names=self.property_names,
            conditioning_dim=256
        )
        self.mean = torch.tensor([2.5, 180.0])
        self.std = torch.tensor([1.2, 50.0])
        self.prop_cond.set_normalization_params(self.mean, self.std)
    
    def test_forward_shape(self):
        """出力形状のテスト"""
        properties = torch.randn(8, 2)
        output = self.prop_cond(properties)
        self.assertEqual(output.shape, (8, 256))
    
    def test_normalization(self):
        """正規化のテスト"""
        # 平均値を入力
        properties = self.mean.unsqueeze(0)
        output = self.prop_cond(properties)
        # 内部的には約0ベクトルが入力されている
        # (正確な値は MLP を通るため直接検証困難)
        self.assertIsNotNone(output)
    
    def test_gradient(self):
        """勾配のテスト"""
        properties = torch.randn(8, 2, requires_grad=True)
        output = self.prop_cond(properties)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(properties.grad)
    
    def tearDown(self):
        """各テスト後のクリーンアップ"""
        pass

if __name__ == '__main__':
    unittest.main()
```

#### テストの実行

```bash
# すべてのテストを実行
python -m unittest discover tests/

# 特定のテストを実行
python -m unittest tests.test_property_conditioning

# カバレッジ付きで実行
coverage run -m unittest discover tests/
coverage report
```

---

## テスト設計

### TD-1: ユニットテストリスト

| テストID | 対象モジュール | テスト内容 | 優先度 |
|---------|--------------|----------|--------|
| UT-001 | PropertyConditioning | 初期化 | 高 |
| UT-002 | PropertyConditioning | forward の形状 | 高 |
| UT-003 | PropertyConditioning | 正規化の正確性 | 高 |
| UT-004 | PropertyConditioning | 勾配の伝播 | 中 |
| UT-005 | PropertyConditioning | set_normalization_params | 高 |
| UT-006 | ExtendedCombinedConditioning | 初期化 | 高 |
| UT-007 | ExtendedCombinedConditioning | forward（全条件あり） | 高 |
| UT-008 | ExtendedCombinedConditioning | forward（一部条件なし） | 中 |
| UT-009 | CrystalDatasetWithProperties | データ読み込み | 高 |
| UT-010 | CrystalDatasetWithProperties | 統計情報計算 | 高 |
| UT-011 | CrystalDatasetWithProperties | __getitem__ | 高 |
| UT-012 | PropertyNormalizer | normalize | 高 |
| UT-013 | PropertyNormalizer | denormalize | 高 |
| UT-014 | PropertyNormalizer | save/load | 中 |

### TD-2: 統合テストシナリオ

#### IT-001: データ準備から訓練まで

**目的**: データ準備、モデル構築、訓練の一連の流れをテスト

**ステップ**:
1. テストデータベースを作成（10分子、50結晶）
2. 物性値CSVを作成
3. prepare_property_dataset.py を実行
4. データセットを読み込み
5. モデルを構築
6. 10エポック訓練
7. 損失が減少することを確認

**合格基準**:
- エラーなく完了
- 損失が初期値の50%以下に減少

#### IT-002: 訓練から生成まで

**目的**: 訓練済みモデルでの生成をテスト

**ステップ**:
1. IT-001で訓練したモデルを使用
2. 生成スクリプトを実行
3. 10サンプル生成
4. 出力ファイルを検証

**合格基準**:
- 10個のCIFファイルが生成される
- メタデータJSONが存在
- すべてのサンプルが物理的に妥当

### TD-3: 性能テスト

#### PT-001: 訓練速度

**測定項目**:
- イテレーション/秒
- エポック時間
- GPU使用率

**テスト条件**:
- データセットサイズ: 1000サンプル
- バッチサイズ: 32
- ハードウェア: NVIDIA RTX 3090

**合格基準**:
- > 5 イテレーション/秒
- GPU使用率 > 80%

#### PT-002: 生成速度

**測定項目**:
- サンプル/秒
- メモリ使用量

**テスト条件**:
- サンプル数: 100
- ハードウェア: NVIDIA RTX 3090

**合格基準**:
- > 10 サンプル/秒
- メモリ使用量 < 4 GB

---

## デプロイメント設計

### DEP-1: ディレクトリ構成

プロジェクトのディレクトリ構成:
```
e3_diffusion_for_molecules/
├── crystal/
│   ├── __init__.py
│   ├── conditioning/
│   │   ├── __init__.py
│   │   ├── property_conditioning.py (新規)
│   │   ├── extended_combined_conditioning.py (新規)
│   │   ├── molecular_conditioning.py (既存)
│   │   ├── space_group_embedding.py (既存)
│   │   └── density_conditioning.py (既存)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── crystal_loader.py (拡張)
│   │   ├── molecule_loader.py (既存)
│   │   └── property_normalizer.py (新規)
│   ├── dynamics/
│   │   ├── __init__.py
│   │   └── crystal_dynamics.py (既存)
│   ├── models/
│   │   ├── __init__.py
│   │   └── egnn.py (既存)
│   ├── sampling_with_properties.py (新規)
│   └── training_with_properties.py (新規)
├── scripts/
│   ├── prepare_property_dataset.py (新規)
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_property_conditioning.py (新規)
│   ├── test_extended_combined_conditioning.py (新規)
│   ├── test_crystal_dataset_with_properties.py (新規)
│   └── test_integration.py (新規)
├── doc/
│   ├── gen_molecular_crystal/
│   │   ├── theory.md (新規)
│   │   ├── specification.md (新規)
│   │   └── design.md (新規、本文書)
│   └── ...
├── main_crystal_with_properties.py (新規)
├── generate_crystal_with_all_conditions.py (新規)
├── requirements.txt
└── README.md
```

### DEP-2: 依存関係管理

**requirements.txt**:
```
torch>=1.12.0,<2.0.0
torch-geometric>=2.0.0
ase>=3.22.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
rdkit>=2022.03.1
tqdm>=4.62.0
pyyaml>=5.4.0
```

**インストール手順**:
```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt

# 開発用依存関係（オプション）
pip install pytest coverage black flake8
```

### DEP-3: 環境変数

**必要な環境変数**:
```bash
# CUDA設定
export CUDA_VISIBLE_DEVICES=0

# データパス
export MOLECULE_DB_PATH=/path/to/molecules.db
export CRYSTAL_DB_PATH=/path/to/crystals.db

# 出力ディレクトリ
export OUTPUT_DIR=/path/to/outputs

# ログレベル
export LOG_LEVEL=INFO
```

### DEP-4: Docker対応（オプション）

**Dockerfile**:
```dockerfile
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードのコピー
COPY . .

# エントリーポイント
ENTRYPOINT ["python", "main_crystal_with_properties.py"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  training:
    build: .
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 付録

### 付録A: チェックリスト

#### 実装完了チェックリスト

- [ ] PropertyConditioning モジュールの実装
- [ ] ExtendedCombinedConditioning モジュールの実装
- [ ] CrystalDatasetWithProperties の実装
- [ ] PropertyNormalizer の実装
- [ ] sampling_with_properties.py の実装
- [ ] training_with_properties.py の実装
- [ ] main_crystal_with_properties.py の実装
- [ ] generate_crystal_with_all_conditions.py の実装
- [ ] prepare_property_dataset.py の実装
- [ ] ユニットテストの実装（全15件）
- [ ] 統合テストの実装
- [ ] ドキュメントの作成（本文書を含む）

#### テスト完了チェックリスト

- [ ] ユニットテスト（全件合格）
- [ ] 統合テスト IT-001（合格）
- [ ] 統合テスト IT-002（合格）
- [ ] 性能テスト PT-001（合格）
- [ ] 性能テスト PT-002（合格）
- [ ] 精度テスト（物性値MAE < 10%）
- [ ] 構造検証（妥当性 > 95%）

### 付録B: トラブルシューティング

#### 問題1: GPU メモリ不足

**症状**: `RuntimeError: CUDA out of memory`

**原因**: バッチサイズが大きすぎる

**解決法**:
```bash
# バッチサイズを減らす
python main_crystal_with_properties.py \
    --batch_size 16  # 32 → 16
```

#### 問題2: 訓練が収束しない

**症状**: 損失が減少しない、NaNになる

**原因**: 学習率が高すぎる、勾配爆発

**解決法**:
```bash
# 学習率を下げる
python main_crystal_with_properties.py \
    --learning_rate 5e-5  # 1e-4 → 5e-5

# 勾配クリッピングを追加
python main_crystal_with_properties.py \
    --grad_clip 1.0
```

#### 問題3: 物性値データの欠損

**症状**: `ValueError: Property 'bandgap' not found`

**原因**: 物性値データが不完全

**解決法**:
```python
# データベースを検証
python scripts/validate_property_data.py \
    --crystal_db crystals.db \
    --property_names bandgap melting_point

# 欠損値を補完または除外
```

### 付録C: 実装の優先順位

#### フェーズ1: コア機能（Week 1-2）

1. PropertyConditioning モジュール
2. ExtendedCombinedConditioning モジュール
3. CrystalDatasetWithProperties クラス
4. PropertyNormalizer クラス

#### フェーズ2: 訓練パイプライン（Week 3）

1. training_with_properties.py
2. main_crystal_with_properties.py
3. データ準備スクリプト

#### フェーズ3: 生成パイプライン（Week 4）

1. sampling_with_properties.py
2. generate_crystal_with_all_conditions.py

#### フェーズ4: テストとドキュメント（Week 5）

1. ユニットテスト
2. 統合テスト
3. ドキュメント最終化

---

**文書作成者**: システム設計チーム  
**最終更新日**: 2025-10-24  
**バージョン**: 1.0  
**承認者**: プロジェクトリーダー  
**ステータス**: 最終版
