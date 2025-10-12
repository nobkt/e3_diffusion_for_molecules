# 分子性結晶生成のための詳細設計書
# Detailed Design Document for Molecular Crystal Generation

## 概要 (Overview)

本設計書は、E(3)等変拡散モデルを分子性結晶生成に拡張するための詳細な実装設計を提供します。既存のコードベースに最小限の変更で統合できるよう、モジュール構造と実装の詳細を定義します。

This design document provides detailed implementation specifications for extending the E(3) Equivariant Diffusion Model to support molecular crystal generation. The design ensures minimal modifications to the existing codebase through a modular architecture.

---

## 1. システムアーキテクチャ (System Architecture)

### 1.1 ディレクトリ構造 (Directory Structure)

```
e3_diffusion_for_molecules/
├── qm9/                                    # 既存の分子関連コード
│   ├── dataset.py                          # [修正] ASE DBローダーに結晶サポート追加
│   ├── analyze.py                          # [修正] 結晶評価メトリクス追加
│   ├── visualizer.py                       # [修正] 結晶可視化機能追加
│   └── rdkit_functions.py                  # [変更なし] 分子機能は維持
│
├── crystal/                                # [新規] 結晶専用モジュール
│   ├── __init__.py                         # モジュール初期化
│   ├── data/                               # データ処理
│   │   ├── __init__.py
│   │   ├── crystal_loader.py               # 結晶データローダー
│   │   ├── molecule_loader.py              # [新規] 単分子データローダー
│   │   ├── molecular_features.py           # [新規] 分子特徴量抽出
│   │   ├── molecule_crystal_pair.py        # [新規] 分子-結晶ペアリング
│   │   ├── periodic_utils.py               # 周期境界条件ユーティリティ
│   │   ├── coordinate_transform.py         # 座標変換（分数↔デカルト）
│   │   └── symmetry_handler.py             # 空間群・対称性処理
│   │
│   ├── models/                             # モデル拡張
│   │   ├── __init__.py
│   │   ├── periodic_egnn.py                # 周期的EGNN
│   │   ├── molecular_feature_encoder.py    # [新規] 分子特徴量エンコーダー
│   │   ├── lattice_diffusion.py            # 格子パラメータ拡散
│   │   └── crystal_dynamics.py             # 結晶構造拡散統合モデル
│   │
│   ├── conditioning/                        # 条件付けモジュール
│   │   ├── __init__.py
│   │   ├── molecular_conditioning.py        # [新規] 分子特徴量条件付け
│   │   ├── space_group_embedding.py        # 空間群埋め込み
│   │   ├── density_conditioning.py         # 密度条件付け
│   │   └── lattice_conditioning.py         # 格子パラメータ条件付け
│   │
│   ├── evaluation/                         # 評価モジュール
│   │   ├── __init__.py
│   │   ├── crystal_metrics.py              # 結晶評価指標
│   │   ├── polymorph_analyzer.py           # [新規] 結晶多形分析
│   │   ├── structure_validator.py          # 構造妥当性検証
│   │   └── symmetry_analyzer.py            # 対称性分析
│   │
│   └── utils/                              # ユーティリティ
│       ├── __init__.py
│       ├── cell_operations.py              # セル操作
│       ├── neighbor_list.py                # 周期的近傍リスト
│       └── cif_writer.py                   # CIF出力
│
├── configs/                                # 設定ファイル
│   ├── datasets_config.py                  # [修正] 結晶データセット設定追加
│   └── crystal_config.yaml                 # [新規] 結晶専用設定
│
├── equivariant_diffusion/                  # 拡散モデルコア
│   ├── en_diffusion.py                     # [修正] 結晶モード対応
│   └── utils.py                            # [修正] 周期的ユーティリティ追加
│
├── main_crystal.py                         # [新規] 結晶学習用メインスクリプト
├── eval_crystal.py                         # [新規] 結晶評価用スクリプト
├── sample_crystal.py                       # [新規] 結晶サンプリングスクリプト
│
└── tests/                                  # テスト
    ├── test_crystal_loader.py              # [新規] データローダーテスト
    ├── test_molecular_features.py          # [新規] 分子特徴量テスト
    ├── test_periodic_utils.py              # [新規] 周期性テスト
    ├── test_periodic_egnn.py               # [新規] モデルテスト
    └── test_crystal_integration.py         # [新規] 統合テスト
```

---

## 2. データ処理層の設計 (Data Processing Layer Design)

### 2.1 結晶データローダー (Crystal Data Loader)

#### ファイル: `crystal/data/crystal_loader.py`

```python
"""
結晶構造データをASEデータベースから読み込み、PyTorchテンソルに変換
"""

import torch
import numpy as np
from ase.db import connect
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


class CrystalDataset(Dataset):
    """
    分子性結晶データセット
    
    ASE Atomsオブジェクトを内部表現に変換し、バッチ処理可能な形式で提供
    """
    
    def __init__(
        self,
        db_path: str,
        indices: List[int],
        remove_h: bool = False,
        use_fractional_coords: bool = True,
        cutoff_radius: float = 10.0,
        max_atoms: int = 500,
        include_charges: bool = False,
    ):
        """
        Args:
            db_path: ASEデータベースのパス
            indices: 使用するデータのインデックスリスト
            remove_h: 水素原子を除去するか
            use_fractional_coords: 分数座標を使用するか（Falseの場合はデカルト座標）
            cutoff_radius: 近傍計算のカットオフ半径（Å）
            max_atoms: 単位格子あたりの最大原子数
            include_charges: 電荷を含めるか
        """
        self.db_path = db_path
        self.indices = indices
        self.remove_h = remove_h
        self.use_fractional_coords = use_fractional_coords
        self.cutoff_radius = cutoff_radius
        self.max_atoms = max_atoms
        self.include_charges = include_charges
        
        # データベース接続
        self.db = connect(db_path)
        
        # 原子種の辞書を構築
        self._build_atom_encoder()
        
    def _build_atom_encoder(self):
        """データセット全体をスキャンして原子種辞書を構築"""
        all_atomic_numbers = set()
        
        for idx in self.indices:
            row = self.db.get(idx + 1)  # ASE DBは1-indexed
            atoms = row.toatoms()
            all_atomic_numbers.update(atoms.numbers)
        
        # 原子番号でソート
        sorted_atomic_numbers = sorted(all_atomic_numbers)
        
        # エンコーダー・デコーダー作成
        self.atom_encoder = {num: i for i, num in enumerate(sorted_atomic_numbers)}
        self.atom_decoder = sorted_atomic_numbers
        self.num_atom_types = len(self.atom_decoder)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        単一の結晶構造を取得
        
        Returns:
            data: 以下のキーを含む辞書
                - positions: [n_atoms, 3] 原子座標（分数またはデカルト）
                - atom_types: [n_atoms] 原子種インデックス
                - one_hot: [n_atoms, num_atom_types] ワンホット表現
                - charges: [n_atoms] 原子番号（include_chargesがTrueの場合）
                - cell: [3, 3] 単位格子ベクトル
                - cell_params: [6] (a, b, c, α, β, γ)
                - cell_volume: [1] 単位格子体積
                - pbc: [3] 周期境界条件のフラグ
                - num_atoms: [1] 原子数
                - space_group: [1] 空間群番号（利用可能な場合）
                - density: [1] 結晶密度（利用可能な場合）
        """
        # データベースから取得
        db_idx = self.indices[idx]
        row = self.db.get(db_idx + 1)
        atoms = row.toatoms()
        
        # 水素除去
        if self.remove_h:
            mask = atoms.numbers != 1
            atoms = atoms[mask]
        
        # 原子数チェック
        if len(atoms) > self.max_atoms:
            raise ValueError(f"Crystal has {len(atoms)} atoms, exceeding max {self.max_atoms}")
        
        # 座標取得（デカルト座標）
        positions_cart = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # 単位格子情報
        cell_vectors = torch.tensor(atoms.cell.array, dtype=torch.float32)  # [3, 3]
        cell_params = torch.tensor(atoms.cell.cellpar(), dtype=torch.float32)  # [6] (a,b,c,α,β,γ)
        cell_volume = torch.tensor([atoms.cell.volume], dtype=torch.float32)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool)
        
        # 分数座標に変換（オプション）
        if self.use_fractional_coords:
            positions = self._cartesian_to_fractional(positions_cart, cell_vectors)
        else:
            positions = positions_cart
        
        # 原子種情報
        atomic_numbers = atoms.numbers
        atom_types = torch.tensor(
            [self.atom_encoder[num] for num in atomic_numbers],
            dtype=torch.long
        )
        
        # ワンホット表現
        one_hot = torch.zeros(len(atoms), self.num_atom_types, dtype=torch.float32)
        one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
        
        # データ辞書作成
        data = {
            'positions': positions,
            'positions_cart': positions_cart,  # デカルト座標も保持
            'atom_types': atom_types,
            'one_hot': one_hot,
            'cell': cell_vectors,
            'cell_params': cell_params,
            'cell_volume': cell_volume,
            'pbc': pbc,
            'num_atoms': torch.tensor([len(atoms)], dtype=torch.long),
        }
        
        # 電荷（原子番号）
        if self.include_charges:
            data['charges'] = torch.tensor(atomic_numbers, dtype=torch.float32)
        
        # メタデータ（利用可能な場合）
        if hasattr(row, 'space_group'):
            data['space_group'] = torch.tensor([row.space_group], dtype=torch.long)
        elif hasattr(row, 'data') and 'space_group' in row.data:
            data['space_group'] = torch.tensor([row.data['space_group']], dtype=torch.long)
        
        if hasattr(row, 'density'):
            data['density'] = torch.tensor([row.density], dtype=torch.float32)
        elif hasattr(row, 'data') and 'density' in row.data:
            data['density'] = torch.tensor([row.data['density']], dtype=torch.float32)
        
        return data
    
    @staticmethod
    def _cartesian_to_fractional(
        positions_cart: torch.Tensor,
        cell_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        デカルト座標を分数座標に変換
        
        Args:
            positions_cart: [n_atoms, 3] デカルト座標
            cell_vectors: [3, 3] 単位格子ベクトル
            
        Returns:
            positions_frac: [n_atoms, 3] 分数座標
        """
        # cell_vectors^(-1) @ positions_cart^T = positions_frac^T
        positions_frac = torch.linalg.solve(cell_vectors.T, positions_cart.T).T
        return positions_frac


def collate_crystal_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    結晶データのバッチ処理用collate関数
    
    異なるサイズの結晶をパディングして統一的なバッチに変換
    
    Args:
        batch: CrystalDatasetから返されるデータのリスト
        
    Returns:
        batched_data: バッチ化されたデータ
            - positions: [batch_size, max_atoms, 3]
            - atom_types: [batch_size, max_atoms]
            - one_hot: [batch_size, max_atoms, num_atom_types]
            - cell: [batch_size, 3, 3]
            - cell_params: [batch_size, 6]
            - mask: [batch_size, max_atoms] パディング部分をマスク
            - ...
    """
    batch_size = len(batch)
    max_atoms = max(item['num_atoms'].item() for item in batch)
    num_atom_types = batch[0]['one_hot'].shape[1]
    
    # バッチテンソルを初期化
    positions = torch.zeros(batch_size, max_atoms, 3)
    positions_cart = torch.zeros(batch_size, max_atoms, 3)
    atom_types = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    one_hot = torch.zeros(batch_size, max_atoms, num_atom_types)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    
    cell = torch.zeros(batch_size, 3, 3)
    cell_params = torch.zeros(batch_size, 6)
    cell_volume = torch.zeros(batch_size, 1)
    pbc = torch.zeros(batch_size, 3, dtype=torch.bool)
    num_atoms = torch.zeros(batch_size, dtype=torch.long)
    
    # バッチに詰める
    for i, item in enumerate(batch):
        n_atoms = item['num_atoms'].item()
        
        positions[i, :n_atoms] = item['positions']
        positions_cart[i, :n_atoms] = item['positions_cart']
        atom_types[i, :n_atoms] = item['atom_types']
        one_hot[i, :n_atoms] = item['one_hot']
        mask[i, :n_atoms] = True
        
        cell[i] = item['cell']
        cell_params[i] = item['cell_params']
        cell_volume[i] = item['cell_volume']
        pbc[i] = item['pbc']
        num_atoms[i] = n_atoms
    
    batched_data = {
        'positions': positions,
        'positions_cart': positions_cart,
        'atom_types': atom_types,
        'one_hot': one_hot,
        'mask': mask,
        'cell': cell,
        'cell_params': cell_params,
        'cell_volume': cell_volume,
        'pbc': pbc,
        'num_atoms': num_atoms,
    }
    
    # オプションフィールド
    if 'charges' in batch[0]:
        charges = torch.zeros(batch_size, max_atoms)
        for i, item in enumerate(batch):
            n_atoms = item['num_atoms'].item()
            charges[i, :n_atoms] = item['charges']
        batched_data['charges'] = charges
    
    if 'space_group' in batch[0]:
        space_group = torch.stack([item['space_group'] for item in batch])
        batched_data['space_group'] = space_group
    
    if 'density' in batch[0]:
        density = torch.stack([item['density'] for item in batch])
        batched_data['density'] = density
    
    return batched_data
```

### 2.2 周期境界条件ユーティリティ (Periodic Boundary Utilities)

#### ファイル: `crystal/data/periodic_utils.py`

```python
"""
周期境界条件を扱うためのユーティリティ関数
"""

import torch
import numpy as np
from typing import Tuple, Optional


def minimum_image_distance(
    positions1: torch.Tensor,
    positions2: torch.Tensor,
    cell_vectors: torch.Tensor,
    pbc: torch.Tensor,
    use_fractional: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    周期境界条件下での最小イメージ距離を計算
    
    Args:
        positions1: [batch, n1, 3] または [n1, 3]
        positions2: [batch, n2, 3] または [n2, 3]
        cell_vectors: [batch, 3, 3] または [3, 3] 単位格子ベクトル
        pbc: [batch, 3] または [3] 周期境界条件フラグ
        use_fractional: positions1, positions2が分数座標かどうか
        
    Returns:
        distances: [batch, n1, n2] または [n1, n2] 距離
        vectors: [batch, n1, n2, 3] または [n1, n2, 3] ベクトル
    """
    # 分数座標での計算が効率的
    if not use_fractional:
        # デカルト座標から分数座標に変換
        frac1 = cartesian_to_fractional(positions1, cell_vectors)
        frac2 = cartesian_to_fractional(positions2, cell_vectors)
    else:
        frac1 = positions1
        frac2 = positions2
    
    # バッチ対応
    if frac1.dim() == 2:
        frac1 = frac1.unsqueeze(0)
        frac2 = frac2.unsqueeze(0)
        cell_vectors = cell_vectors.unsqueeze(0)
        pbc = pbc.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    batch_size, n1, _ = frac1.shape
    n2 = frac2.shape[1]
    
    # 差ベクトルを計算 [batch, n1, n2, 3]
    frac_diff = frac2.unsqueeze(1) - frac1.unsqueeze(2)
    
    # 周期境界条件を適用（-0.5 < diff <= 0.5 の範囲に正規化）
    pbc_expanded = pbc.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, 3]
    
    # pbcがTrueの次元のみ周期境界を適用
    frac_diff_periodic = torch.where(
        pbc_expanded,
        frac_diff - torch.round(frac_diff),
        frac_diff
    )
    
    # 分数座標での差をデカルト座標に変換
    # vectors = cell_vectors @ frac_diff_periodic
    vectors = torch.einsum('bij,bmnj->bmni', cell_vectors, frac_diff_periodic)
    
    # 距離計算
    distances = torch.norm(vectors, dim=-1)
    
    if squeeze_batch:
        distances = distances.squeeze(0)
        vectors = vectors.squeeze(0)
    
    return distances, vectors


def cartesian_to_fractional(
    positions_cart: torch.Tensor,
    cell_vectors: torch.Tensor
) -> torch.Tensor:
    """
    デカルト座標を分数座標に変換
    
    Args:
        positions_cart: [..., 3] デカルト座標
        cell_vectors: [..., 3, 3] 単位格子ベクトル
        
    Returns:
        positions_frac: [..., 3] 分数座標
    """
    # cell^(-1) @ positions_cart = positions_frac
    # バッチ対応のため、最後の次元で処理
    original_shape = positions_cart.shape
    
    # [..., 3] -> [..., 3, 1] に reshape
    positions_cart_expanded = positions_cart.unsqueeze(-1)
    
    # cell_vectors^(-1) を計算
    cell_inv = torch.linalg.inv(cell_vectors)
    
    # 行列積
    # [..., 3, 3] @ [..., 3, 1] -> [..., 3, 1]
    positions_frac = torch.matmul(cell_inv, positions_cart_expanded)
    
    # [..., 3, 1] -> [..., 3]
    positions_frac = positions_frac.squeeze(-1)
    
    return positions_frac


def fractional_to_cartesian(
    positions_frac: torch.Tensor,
    cell_vectors: torch.Tensor
) -> torch.Tensor:
    """
    分数座標をデカルト座標に変換
    
    Args:
        positions_frac: [..., 3] 分数座標
        cell_vectors: [..., 3, 3] 単位格子ベクトル
        
    Returns:
        positions_cart: [..., 3] デカルト座標
    """
    # cell @ positions_frac = positions_cart
    positions_frac_expanded = positions_frac.unsqueeze(-1)
    positions_cart = torch.matmul(cell_vectors, positions_frac_expanded)
    positions_cart = positions_cart.squeeze(-1)
    
    return positions_cart


def wrap_positions(
    positions_frac: torch.Tensor,
    pbc: torch.Tensor
) -> torch.Tensor:
    """
    分数座標を単位格子内（0〜1）にラップ
    
    Args:
        positions_frac: [..., 3] 分数座標
        pbc: [..., 3] 周期境界条件フラグ
        
    Returns:
        wrapped_positions: [..., 3] ラップされた分数座標
    """
    # pbcがTrueの次元のみラップ
    wrapped = torch.where(
        pbc.unsqueeze(-2) if pbc.dim() < positions_frac.dim() else pbc,
        positions_frac % 1.0,  # 0〜1の範囲にラップ
        positions_frac
    )
    
    return wrapped


def build_neighbor_list(
    positions: torch.Tensor,
    cell_vectors: torch.Tensor,
    pbc: torch.Tensor,
    cutoff_radius: float,
    use_fractional: bool = True,
    max_neighbors: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    周期境界条件を考慮した近傍リストを構築
    
    Args:
        positions: [batch, n_atoms, 3]
        cell_vectors: [batch, 3, 3]
        pbc: [batch, 3]
        cutoff_radius: カットオフ半径（Å）
        use_fractional: positionsが分数座標かどうか
        max_neighbors: 原子あたりの最大近傍数（メモリ効率化のため）
        
    Returns:
        edge_index: [batch, 2, n_edges] エッジのインデックス
        edge_attr: [batch, n_edges, 3] エッジのベクトル
        edge_dist: [batch, n_edges] エッジの距離
    """
    batch_size, n_atoms, _ = positions.shape
    
    # 全ペアの距離を計算
    distances, vectors = minimum_image_distance(
        positions, positions, cell_vectors, pbc, use_fractional
    )
    
    # 自己ループを除外（対角成分を大きな値にする）
    mask_diag = torch.eye(n_atoms, device=distances.device, dtype=torch.bool)
    distances = distances.masked_fill(mask_diag.unsqueeze(0), float('inf'))
    
    # カットオフ半径内のエッジを選択
    edge_mask = distances < cutoff_radius  # [batch, n_atoms, n_atoms]
    
    # バッチごとに処理
    edge_index_list = []
    edge_attr_list = []
    edge_dist_list = []
    
    for b in range(batch_size):
        # エッジのインデックス
        src, dst = torch.where(edge_mask[b])
        
        # 最大近傍数の制限
        if max_neighbors is not None and len(src) > n_atoms * max_neighbors:
            # 距離でソートして近い順に選択
            edge_distances = distances[b, src, dst]
            sorted_indices = torch.argsort(edge_distances)
            
            # 各原子からmax_neighbors個まで選択
            keep_mask = torch.zeros(len(src), dtype=torch.bool, device=src.device)
            for atom_idx in range(n_atoms):
                atom_edges = src == atom_idx
                if atom_edges.sum() > max_neighbors:
                    # この原子からのエッジをmax_neighbors個に制限
                    atom_edge_indices = torch.where(atom_edges)[0]
                    sorted_atom_indices = sorted_indices[
                        torch.isin(sorted_indices, atom_edge_indices)
                    ][:max_neighbors]
                    keep_mask[sorted_atom_indices] = True
                else:
                    keep_mask[atom_edges] = True
            
            src = src[keep_mask]
            dst = dst[keep_mask]
        
        edge_index = torch.stack([src, dst], dim=0)  # [2, n_edges]
        edge_attr = vectors[b, src, dst]  # [n_edges, 3]
        edge_dist = distances[b, src, dst]  # [n_edges]
        
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        edge_dist_list.append(edge_dist)
    
    # パディングしてバッチ化（必要に応じて）
    # ここでは簡略化のため、リストで返す
    # 実際の実装では、統一的な形式にパディングすることを推奨
    
    return edge_index_list, edge_attr_list, edge_dist_list


def compute_cell_volume(cell_vectors: torch.Tensor) -> torch.Tensor:
    """
    単位格子の体積を計算
    
    Args:
        cell_vectors: [..., 3, 3] 単位格子ベクトル
        
    Returns:
        volume: [...] 体積
    """
    # 体積 = |a · (b × c)|
    a = cell_vectors[..., 0, :]
    b = cell_vectors[..., 1, :]
    c = cell_vectors[..., 2, :]
    
    cross_bc = torch.cross(b, c, dim=-1)
    volume = torch.abs(torch.sum(a * cross_bc, dim=-1))
    
    return volume


def cell_params_to_vectors(cell_params: torch.Tensor) -> torch.Tensor:
    """
    格子パラメータ (a, b, c, α, β, γ) から単位格子ベクトルを計算
    
    Args:
        cell_params: [..., 6] (a, b, c, α, β, γ) α,β,γは度単位
        
    Returns:
        cell_vectors: [..., 3, 3]
    """
    a, b, c = cell_params[..., 0], cell_params[..., 1], cell_params[..., 2]
    alpha, beta, gamma = cell_params[..., 3], cell_params[..., 4], cell_params[..., 5]
    
    # 度からラジアンに変換
    alpha_rad = torch.deg2rad(alpha)
    beta_rad = torch.deg2rad(beta)
    gamma_rad = torch.deg2rad(gamma)
    
    # 単位格子ベクトルの計算（標準的な結晶学的配置）
    cos_alpha = torch.cos(alpha_rad)
    cos_beta = torch.cos(beta_rad)
    cos_gamma = torch.cos(gamma_rad)
    sin_gamma = torch.sin(gamma_rad)
    
    # aベクトルはx軸に沿う
    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)
    
    # bベクトルはxy平面上
    bx = b * cos_gamma
    by = b * sin_gamma
    bz = torch.zeros_like(b)
    
    # cベクトル
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = torch.sqrt(c**2 - cx**2 - cy**2 + 1e-10)  # 数値安定性のため小さな値を加算
    
    # ベクトルをスタック
    shape = list(cell_params.shape[:-1]) + [3, 3]
    cell_vectors = torch.zeros(shape, device=cell_params.device, dtype=cell_params.dtype)
    
    cell_vectors[..., 0, 0] = ax
    cell_vectors[..., 0, 1] = ay
    cell_vectors[..., 0, 2] = az
    
    cell_vectors[..., 1, 0] = bx
    cell_vectors[..., 1, 1] = by
    cell_vectors[..., 1, 2] = bz
    
    cell_vectors[..., 2, 0] = cx
    cell_vectors[..., 2, 1] = cy
    cell_vectors[..., 2, 2] = cz
    
    return cell_vectors


def cell_vectors_to_params(cell_vectors: torch.Tensor) -> torch.Tensor:
    """
    単位格子ベクトルから格子パラメータ (a, b, c, α, β, γ) を計算
    
    Args:
        cell_vectors: [..., 3, 3]
        
    Returns:
        cell_params: [..., 6] (a, b, c, α, β, γ) α,β,γは度単位
    """
    a_vec = cell_vectors[..., 0, :]
    b_vec = cell_vectors[..., 1, :]
    c_vec = cell_vectors[..., 2, :]
    
    # 長さ
    a = torch.norm(a_vec, dim=-1)
    b = torch.norm(b_vec, dim=-1)
    c = torch.norm(c_vec, dim=-1)
    
    # 角度（ラジアン）
    cos_alpha = torch.sum(b_vec * c_vec, dim=-1) / (b * c + 1e-10)
    cos_beta = torch.sum(a_vec * c_vec, dim=-1) / (a * c + 1e-10)
    cos_gamma = torch.sum(a_vec * b_vec, dim=-1) / (a * b + 1e-10)
    
    # -1〜1にクリップ（数値誤差対策）
    cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
    cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
    cos_gamma = torch.clamp(cos_gamma, -1.0, 1.0)
    
    alpha = torch.rad2deg(torch.acos(cos_alpha))
    beta = torch.rad2deg(torch.acos(cos_beta))
    gamma = torch.rad2deg(torch.acos(cos_gamma))
    
    # スタック
    cell_params = torch.stack([a, b, c, alpha, beta, gamma], dim=-1)
    
    return cell_params
```

---

## 3. モデル層の設計 (Model Layer Design)

### 3.1 周期的EGNN (Periodic EGNN)

#### ファイル: `crystal/models/periodic_egnn.py`

```python
"""
周期境界条件を考慮したE(3) Equivariant Graph Neural Network
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from crystal.data.periodic_utils import minimum_image_distance


class PeriodicEGNN(nn.Module):
    """
    周期境界条件を考慮したE(3)等変グラフニューラルネットワーク
    
    既存のEGNNを拡張し、周期性を持つ系に対応
    """
    
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        in_edge_nf: int = 0,
        n_layers: int = 4,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
        use_fractional_coords: bool = True,
    ):
        """
        Args:
            in_node_nf: ノード特徴量の入力次元
            hidden_nf: 隠れ層の次元
            out_node_nf: ノード特徴量の出力次元
            in_edge_nf: エッジ特徴量の入力次元
            n_layers: EGNNレイヤー数
            attention: アテンションを使用するか
            normalize: 座標を正規化するか
            tanh: coord_mlpでtanhを使用するか
            use_fractional_coords: 分数座標を使用するか
        """
        super().__init__()
        
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.use_fractional_coords = use_fractional_coords
        
        # ノード埋め込み
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        
        # EGNN層
        self.layers = nn.ModuleList([
            PeriodicEGNNLayer(
                hidden_nf=hidden_nf,
                edge_nf=in_edge_nf,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
            )
            for _ in range(n_layers)
        ])
        
        # 出力層
        self.output_layer = nn.Linear(hidden_nf, out_node_nf)
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            h: [batch, n_atoms, in_node_nf] ノード特徴量
            x: [batch, n_atoms, 3] 座標（分数またはデカルト）
            cell: [batch, 3, 3] 単位格子ベクトル
            pbc: [batch, 3] 周期境界条件
            edge_index: [batch, 2, n_edges] エッジインデックス（オプション）
            edge_attr: [batch, n_edges, edge_nf] エッジ特徴量（オプション）
            node_mask: [batch, n_atoms] ノードマスク（オプション）
            
        Returns:
            h_out: [batch, n_atoms, out_node_nf] 出力ノード特徴量
            x_out: [batch, n_atoms, 3] 出力座標
        """
        # ノード埋め込み
        h = self.embedding(h)
        
        # 各層を通す
        for layer in self.layers:
            h, x = layer(
                h=h,
                x=x,
                cell=cell,
                pbc=pbc,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
            )
        
        # 出力
        h_out = self.output_layer(h)
        x_out = x
        
        return h_out, x_out


class PeriodicEGNNLayer(nn.Module):
    """
    周期的EGNNの単一レイヤー
    """
    
    def __init__(
        self,
        hidden_nf: int,
        edge_nf: int = 0,
        attention: bool = True,
        normalize: bool = False,
        tanh: bool = False,
    ):
        super().__init__()
        
        self.hidden_nf = hidden_nf
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        
        # エッジモデル
        edge_input_nf = hidden_nf * 2 + edge_nf + 1  # h_i, h_j, edge_attr, distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
        )
        
        # 座標更新
        coord_input_nf = hidden_nf
        coord_layers = [
            nn.Linear(coord_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1, bias=False),
        ]
        if tanh:
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)
        
        # ノード更新
        node_input_nf = hidden_nf * 2 + edge_nf
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
        )
        
        # アテンション
        if attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single layer
        """
        batch_size, n_atoms, _ = h.shape
        
        # エッジインデックスが与えられていない場合、全結合
        if edge_index is None:
            # 簡略化のため、バッチ内で全結合とする
            # 実際の実装では、近傍リストを使用すべき
            src = torch.arange(n_atoms, device=h.device).repeat(n_atoms)
            dst = torch.arange(n_atoms, device=h.device).repeat_interleave(n_atoms)
            # 自己ループを除外
            mask = src != dst
            src = src[mask]
            dst = dst[mask]
        else:
            # edge_indexが与えられた場合は使用
            # ここではバッチサイズ1を仮定（実際の実装ではバッチ対応が必要）
            src, dst = edge_index[0]
        
        # エッジごとの距離とベクトルを計算（周期境界条件を考慮）
        distances, rel_coords = minimum_image_distance(
            x, x, cell, pbc, use_fractional=False  # デカルト座標で計算
        )
        
        # バッチサイズ1を仮定した簡略実装
        # 実際の実装では、各バッチを適切に処理する必要がある
        distances_edge = distances[0, src, dst].unsqueeze(-1)  # [n_edges, 1]
        rel_coords_edge = rel_coords[0, src, dst]  # [n_edges, 3]
        
        # エッジ特徴量の計算
        h_src = h[0, src]  # [n_edges, hidden_nf]
        h_dst = h[0, dst]  # [n_edges, hidden_nf]
        
        edge_feat_input = [h_src, h_dst, distances_edge]
        if edge_attr is not None:
            edge_feat_input.append(edge_attr)
        edge_feat_input = torch.cat(edge_feat_input, dim=-1)
        
        edge_feat = self.edge_mlp(edge_feat_input)  # [n_edges, hidden_nf]
        
        # アテンション
        if self.attention:
            att = self.attention_mlp(edge_feat)  # [n_edges, 1]
            edge_feat = edge_feat * att
        
        # 座標更新
        coord_weight = self.coord_mlp(edge_feat)  # [n_edges, 1]
        
        if self.normalize:
            norm = torch.sqrt(distances_edge + 1e-8)
            coord_diff = rel_coords_edge / norm
        else:
            coord_diff = rel_coords_edge
        
        coord_update = coord_weight * coord_diff  # [n_edges, 3]
        
        # 座標をアグリゲーション（scatter_add）
        x_new = x.clone()
        x_new[0].index_add_(0, dst, coord_update)
        
        # ノード特徴量の更新
        # エッジ特徴量をノードにアグリゲーション
        agg_feat = torch.zeros(n_atoms, self.hidden_nf, device=h.device)
        agg_feat.index_add_(0, dst, edge_feat)
        
        node_feat_input = torch.cat([h[0], agg_feat], dim=-1)
        h_update = self.node_mlp(node_feat_input)
        
        h_new = h.clone()
        h_new[0] = h[0] + h_update  # 残差接続
        
        return h_new, x_new
```

### 3.2 格子拡散モデル (Lattice Diffusion Model)

#### ファイル: `crystal/models/lattice_diffusion.py`

```python
"""
格子パラメータの拡散モデル
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class LatticeDiffusion(nn.Module):
    """
    格子パラメータ (a, b, c, α, β, γ) の拡散モデル
    
    格子パラメータは原子座標とは異なるスケールと制約を持つため、
    独立した拡散プロセスとして扱う
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        condition_dim: int = 0,
    ):
        """
        Args:
            hidden_dim: 隠れ層の次元
            num_layers: MLPのレイヤー数
            condition_dim: 条件付けベクトルの次元（0の場合は無条件）
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        
        # 格子パラメータの埋め込み
        # (a, b, c, α, β, γ) + 時間ステップ -> hidden_dim
        lattice_input_dim = 6 + 1  # 6 params + time
        if condition_dim > 0:
            lattice_input_dim += condition_dim
        
        layers = []
        layers.append(nn.Linear(lattice_input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 6))  # 出力: (a, b, c, α, β, γ)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        lattice_params: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        格子パラメータのノイズを予測
        
        Args:
            lattice_params: [batch, 6] (a, b, c, α, β, γ)
            t: [batch, 1] 時間ステップ（0〜1に正規化）
            condition: [batch, condition_dim] 条件付けベクトル（オプション）
            
        Returns:
            noise: [batch, 6] 予測ノイズ
        """
        # 入力を連結
        inputs = [lattice_params, t]
        if condition is not None:
            inputs.append(condition)
        
        x = torch.cat(inputs, dim=-1)
        
        # MLPを通す
        noise = self.mlp(x)
        
        return noise
    
    @staticmethod
    def normalize_lattice_params(lattice_params: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        格子パラメータを正規化
        
        異なるスケールを持つパラメータを学習しやすくするため
        
        Args:
            lattice_params: [batch, 6] (a, b, c, α, β, γ)
            
        Returns:
            normalized: [batch, 6] 正規化されたパラメータ
            stats: 正規化統計（逆変換に使用）
        """
        a, b, c = lattice_params[..., 0], lattice_params[..., 1], lattice_params[..., 2]
        alpha, beta, gamma = lattice_params[..., 3], lattice_params[..., 4], lattice_params[..., 5]
        
        # 長さはlog空間で正規化
        a_log = torch.log(a + 1e-8)
        b_log = torch.log(b + 1e-8)
        c_log = torch.log(c + 1e-8)
        
        # 角度は度からラジアンに変換し、[-π, π]に正規化
        alpha_rad = torch.deg2rad(alpha)
        beta_rad = torch.deg2rad(beta)
        gamma_rad = torch.deg2rad(gamma)
        
        # 統計を計算（バッチ全体）
        lengths_log = torch.stack([a_log, b_log, c_log], dim=-1)
        angles_rad = torch.stack([alpha_rad, beta_rad, gamma_rad], dim=-1)
        
        lengths_mean = lengths_log.mean(dim=0, keepdim=True)
        lengths_std = lengths_log.std(dim=0, keepdim=True) + 1e-8
        
        angles_mean = angles_rad.mean(dim=0, keepdim=True)
        angles_std = angles_rad.std(dim=0, keepdim=True) + 1e-8
        
        # 正規化
        lengths_normalized = (lengths_log - lengths_mean) / lengths_std
        angles_normalized = (angles_rad - angles_mean) / angles_std
        
        # 連結
        normalized = torch.cat([lengths_normalized, angles_normalized], dim=-1)
        
        stats = {
            'lengths_mean': lengths_mean,
            'lengths_std': lengths_std,
            'angles_mean': angles_mean,
            'angles_std': angles_std,
        }
        
        return normalized, stats
    
    @staticmethod
    def denormalize_lattice_params(
        normalized: torch.Tensor,
        stats: dict
    ) -> torch.Tensor:
        """
        正規化された格子パラメータを元に戻す
        
        Args:
            normalized: [batch, 6] 正規化されたパラメータ
            stats: normalize_lattice_paramsから返された統計
            
        Returns:
            lattice_params: [batch, 6] (a, b, c, α, β, γ)
        """
        lengths_normalized = normalized[..., :3]
        angles_normalized = normalized[..., 3:]
        
        # 逆正規化
        lengths_log = lengths_normalized * stats['lengths_std'] + stats['lengths_mean']
        angles_rad = angles_normalized * stats['angles_std'] + stats['angles_mean']
        
        # 指数とラジアン→度変換
        lengths = torch.exp(lengths_log)
        angles = torch.rad2deg(angles_rad)
        
        # 物理的制約を適用
        lengths = torch.clamp(lengths, min=1.0, max=100.0)  # 1〜100 Å
        angles = torch.clamp(angles, min=30.0, max=150.0)  # 30〜150度
        
        lattice_params = torch.cat([lengths, angles], dim=-1)
        
        return lattice_params
```

---

続きは次のセクションで提供します。これにより、設計書が適切に分割され、読みやすくなります。


### 3.3 結晶拡散統合モデル (Crystal Dynamics Model)

#### ファイル: `crystal/models/crystal_dynamics.py`

```python
"""
原子座標と格子パラメータを統合した結晶拡散モデル
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from crystal.models.periodic_egnn import PeriodicEGNN
from crystal.models.lattice_diffusion import LatticeDiffusion


class CrystalDynamics(nn.Module):
    """
    結晶構造の拡散モデル
    
    原子座標と格子パラメータを同時に進化させる
    """
    
    def __init__(
        self,
        in_node_nf: int,
        n_dims: int = 3,
        context_node_nf: int = 0,
        hidden_nf: int = 128,
        n_layers: int = 6,
        attention: bool = True,
        condition_time: bool = True,
        use_fractional_coords: bool = True,
        learn_lattice: bool = True,
    ):
        """
        Args:
            in_node_nf: ノード特徴量の入力次元
            n_dims: 座標の次元（通常は3）
            context_node_nf: コンテキスト（条件付け）特徴量の次元
            hidden_nf: 隠れ層の次元
            n_layers: ネットワークのレイヤー数
            attention: アテンションを使用するか
            condition_time: 時間ステップで条件付けするか
            use_fractional_coords: 分数座標を使用するか
            learn_lattice: 格子パラメータも学習するか
        """
        super().__init__()
        
        self.n_dims = n_dims
        self.condition_time = condition_time
        self.use_fractional_coords = use_fractional_coords
        self.learn_lattice = learn_lattice
        
        # 時間埋め込み
        if condition_time:
            time_embed_dim = hidden_nf
            self.time_embedding = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = 0
        
        # ノード特徴量の合計次元
        total_node_nf = in_node_nf + context_node_nf
        if condition_time:
            total_node_nf += time_embed_dim
        
        # 周期的EGNN（原子座標の処理）
        self.periodic_egnn = PeriodicEGNN(
            in_node_nf=total_node_nf,
            hidden_nf=hidden_nf,
            out_node_nf=n_dims,  # 座標更新を出力
            n_layers=n_layers,
            attention=attention,
            use_fractional_coords=use_fractional_coords,
        )
        
        # 格子拡散（格子パラメータの処理）
        if learn_lattice:
            self.lattice_diffusion = LatticeDiffusion(
                hidden_dim=hidden_nf,
                num_layers=3,
                condition_dim=context_node_nf if context_node_nf > 0 else 0,
            )
    
    def forward(
        self,
        t: torch.Tensor,
        xh: Tuple[torch.Tensor, torch.Tensor],
        cell: torch.Tensor,
        pbc: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            t: [batch] 時間ステップ（0〜1）
            xh: (x, h) タプル
                x: [batch, n_atoms, 3] 座標
                h: [batch, n_atoms, node_nf] ノード特徴量
            cell: [batch, 3, 3] 単位格子ベクトル
            pbc: [batch, 3] 周期境界条件
            node_mask: [batch, n_atoms] ノードマスク
            edge_mask: [batch, n_edges] エッジマスク（オプション）
            context: [batch, context_nf] コンテキストベクトル（オプション）
            
        Returns:
            velocity_x: [batch, n_atoms, 3] 座標の速度
            velocity_h: [batch, n_atoms, node_nf] ノード特徴量の速度
            velocity_cell: [batch, 3, 3] 格子ベクトルの速度（learn_latticeがTrueの場合）
        """
        x, h = xh
        batch_size, n_atoms, _ = x.shape
        
        # 時間埋め込み
        if self.condition_time:
            t_emb = self.time_embedding(t.unsqueeze(-1))  # [batch, time_embed_dim]
            # 各ノードに時間埋め込みを追加
            t_emb_expanded = t_emb.unsqueeze(1).expand(-1, n_atoms, -1)  # [batch, n_atoms, time_embed_dim]
            h = torch.cat([h, t_emb_expanded], dim=-1)
        
        # コンテキスト（条件付け）を追加
        if context is not None:
            # 各ノードにコンテキストを追加
            context_expanded = context.unsqueeze(1).expand(-1, n_atoms, -1)
            h = torch.cat([h, context_expanded], dim=-1)
        
        # 周期的EGNNで原子座標を処理
        velocity_h, velocity_x = self.periodic_egnn(
            h=h,
            x=x,
            cell=cell,
            pbc=pbc,
            node_mask=node_mask,
        )
        
        # マスク適用
        velocity_x = velocity_x * node_mask.unsqueeze(-1)
        velocity_h = velocity_h * node_mask.unsqueeze(-1)
        
        # 格子パラメータの処理
        if self.learn_lattice:
            from crystal.data.periodic_utils import cell_vectors_to_params, cell_params_to_vectors
            
            # 格子ベクトル -> パラメータ
            cell_params = cell_vectors_to_params(cell)  # [batch, 6]
            
            # コンテキストをグローバルプーリング（簡略化）
            if context is not None:
                lattice_context = context
            else:
                # ノード特徴量から集約
                lattice_context = (h * node_mask.unsqueeze(-1)).sum(dim=1) / node_mask.sum(dim=1, keepdim=True)
            
            # 格子拡散で格子パラメータの速度を計算
            velocity_cell_params = self.lattice_diffusion(
                lattice_params=cell_params,
                t=t.unsqueeze(-1),
                condition=lattice_context if context is not None else None,
            )
            
            # パラメータ -> ベクトル
            velocity_cell = cell_params_to_vectors(velocity_cell_params)
        else:
            velocity_cell = torch.zeros_like(cell)
        
        return velocity_x, velocity_h, velocity_cell
    
    def wrap_forward(self, *args, **kwargs):
        """
        既存のコードとの互換性のためのラッパー
        """
        velocity_x, velocity_h, velocity_cell = self.forward(*args, **kwargs)
        
        # 既存のインターフェースに合わせて返す
        return torch.cat([velocity_x, velocity_h], dim=-1), velocity_cell
```

---

## 4. 条件付けモジュールの設計 (Conditioning Module Design)

### 4.1 空間群埋め込み (Space Group Embedding)

#### ファイル: `crystal/conditioning/space_group_embedding.py`

```python
"""
空間群の埋め込み表現
"""

import torch
import torch.nn as nn


class SpaceGroupEmbedding(nn.Module):
    """
    空間群番号を埋め込みベクトルに変換
    
    230種類の空間群それぞれに学習可能な埋め込みを割り当てる
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_space_groups: int = 230,
    ):
        """
        Args:
            embedding_dim: 埋め込みベクトルの次元
            num_space_groups: 空間群の総数（デフォルトは230）
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_space_groups = num_space_groups
        
        # 空間群の埋め込みテーブル
        # space_group番号は1-230なので、インデックスは0-229
        self.embedding = nn.Embedding(
            num_embeddings=num_space_groups + 1,  # +1 for unknown (0)
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        
        # オプション: 空間群の階層構造を考慮したMLP
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, space_group: torch.Tensor) -> torch.Tensor:
        """
        空間群番号を埋め込みベクトルに変換
        
        Args:
            space_group: [batch] 空間群番号（1-230、0は未知）
            
        Returns:
            embedding: [batch, embedding_dim] 埋め込みベクトル
        """
        # 埋め込み取得
        emb = self.embedding(space_group)
        
        # プロジェクション
        emb = self.projection(emb)
        
        return emb
```

### 4.2 密度条件付け (Density Conditioning)

#### ファイル: `crystal/conditioning/density_conditioning.py`

```python
"""
結晶密度を用いた条件付け
"""

import torch
import torch.nn as nn


class DensityConditioning(nn.Module):
    """
    結晶密度を条件付けベクトルに変換
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        density_min: float = 0.5,
        density_max: float = 5.0,
    ):
        """
        Args:
            embedding_dim: 埋め込みベクトルの次元
            density_min: 密度の最小値（g/cm³）
            density_max: 密度の最大値（g/cm³）
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.density_min = density_min
        self.density_max = density_max
        
        # 密度を埋め込みに変換するMLP
        self.mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """
        密度を埋め込みベクトルに変換
        
        Args:
            density: [batch] or [batch, 1] 密度（g/cm³）
            
        Returns:
            embedding: [batch, embedding_dim] 埋め込みベクトル
        """
        # 正規化（0〜1の範囲に）
        density = density.unsqueeze(-1) if density.dim() == 1 else density
        density_normalized = (density - self.density_min) / (self.density_max - self.density_min)
        density_normalized = torch.clamp(density_normalized, 0.0, 1.0)
        
        # MLPを通す
        embedding = self.mlp(density_normalized)
        
        return embedding
```

---

## 5. 評価モジュールの設計 (Evaluation Module Design)

### 5.1 結晶評価メトリクス (Crystal Metrics)

#### ファイル: `crystal/evaluation/crystal_metrics.py`

```python
"""
結晶構造の評価メトリクス
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from crystal.data.periodic_utils import minimum_image_distance, compute_cell_volume


class CrystalMetrics:
    """
    生成された結晶構造の評価メトリクス
    """
    
    def __init__(self, dataset_info: dict):
        """
        Args:
            dataset_info: データセット情報（atom_decoderなど）
        """
        self.dataset_info = dataset_info
        self.atom_decoder = dataset_info.get('atom_decoder', [])
    
    def compute_all_metrics(
        self,
        generated_crystals: List[Dict],
        reference_crystals: List[Dict] = None,
    ) -> Dict[str, float]:
        """
        全ての評価メトリクスを計算
        
        Args:
            generated_crystals: 生成された結晶のリスト
            reference_crystals: 参照結晶のリスト（オプション）
            
        Returns:
            metrics: 評価メトリクスの辞書
        """
        metrics = {}
        
        # 構造的メトリクス
        metrics.update(self.compute_structural_metrics(generated_crystals))
        
        # 妥当性メトリクス
        metrics.update(self.compute_validity_metrics(generated_crystals))
        
        # 参照データとの比較（利用可能な場合）
        if reference_crystals is not None:
            metrics.update(self.compute_distribution_metrics(
                generated_crystals, reference_crystals
            ))
        
        return metrics
    
    def compute_structural_metrics(
        self,
        crystals: List[Dict]
    ) -> Dict[str, float]:
        """
        構造的メトリクスを計算
        """
        metrics = {}
        
        # 格子パラメータの統計
        cell_params_list = [c['cell_params'].numpy() for c in crystals]
        cell_params_array = np.array(cell_params_list)
        
        for i, param_name in enumerate(['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
            metrics[f'{param_name}_mean'] = float(np.mean(cell_params_array[:, i]))
            metrics[f'{param_name}_std'] = float(np.std(cell_params_array[:, i]))
        
        # 体積の統計
        volumes = [c['cell_volume'].item() for c in crystals]
        metrics['volume_mean'] = float(np.mean(volumes))
        metrics['volume_std'] = float(np.std(volumes))
        
        # 密度の統計（利用可能な場合）
        if 'density' in crystals[0]:
            densities = [c['density'].item() for c in crystals]
            metrics['density_mean'] = float(np.mean(densities))
            metrics['density_std'] = float(np.std(densities))
        
        return metrics
    
    def compute_validity_metrics(
        self,
        crystals: List[Dict]
    ) -> Dict[str, float]:
        """
        妥当性メトリクスを計算
        """
        metrics = {}
        
        valid_count = 0
        min_distance_violations = 0
        
        for crystal in crystals:
            is_valid, has_violation = self.check_crystal_validity(crystal)
            
            if is_valid:
                valid_count += 1
            if has_violation:
                min_distance_violations += 1
        
        metrics['validity_ratio'] = valid_count / len(crystals)
        metrics['min_distance_violation_ratio'] = min_distance_violations / len(crystals)
        
        return metrics
    
    def check_crystal_validity(
        self,
        crystal: Dict,
        min_distance_threshold: float = 0.5  # Angstrom
    ) -> Tuple[bool, bool]:
        """
        単一の結晶構造の妥当性をチェック
        
        Returns:
            is_valid: 全体として妥当か
            has_min_distance_violation: 最小距離違反があるか
        """
        positions = crystal['positions_cart']
        cell = crystal['cell']
        pbc = crystal['pbc']
        
        # 周期境界を考慮した距離計算
        distances, _ = minimum_image_distance(
            positions.unsqueeze(0),
            positions.unsqueeze(0),
            cell.unsqueeze(0),
            pbc.unsqueeze(0),
            use_fractional=False,
        )
        
        distances = distances.squeeze(0)
        
        # 対角要素（自己距離）を除外
        n = distances.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=distances.device)
        distances = distances[mask]
        
        # 最小距離チェック
        min_distance = torch.min(distances).item()
        has_min_distance_violation = min_distance < min_distance_threshold
        
        # 格子パラメータの妥当性チェック
        cell_params = crystal['cell_params']
        lengths = cell_params[:3]
        angles = cell_params[3:]
        
        valid_lengths = torch.all(lengths > 1.0) and torch.all(lengths < 100.0)
        valid_angles = torch.all(angles > 30.0) and torch.all(angles < 150.0)
        
        is_valid = valid_lengths and valid_angles and not has_min_distance_violation
        
        return is_valid, has_min_distance_violation
    
    def compute_distribution_metrics(
        self,
        generated_crystals: List[Dict],
        reference_crystals: List[Dict]
    ) -> Dict[str, float]:
        """
        生成分布と参照分布の比較メトリクス
        """
        metrics = {}
        
        # 格子パラメータの分布比較（Wasserstein距離）
        gen_params = np.array([c['cell_params'].numpy() for c in generated_crystals])
        ref_params = np.array([c['cell_params'].numpy() for c in reference_crystals])
        
        from scipy.stats import wasserstein_distance
        
        for i, param_name in enumerate(['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
            wd = wasserstein_distance(gen_params[:, i], ref_params[:, i])
            metrics[f'{param_name}_wasserstein'] = float(wd)
        
        # 体積の分布比較
        gen_volumes = np.array([c['cell_volume'].item() for c in generated_crystals])
        ref_volumes = np.array([c['cell_volume'].item() for c in reference_crystals])
        metrics['volume_wasserstein'] = float(wasserstein_distance(gen_volumes, ref_volumes))
        
        return metrics
```

---

## 6. 統合とインターフェース (Integration and Interfaces)

### 6.1 既存コードとの統合

#### 修正ファイル: `qm9/dataset.py`

既存の`load_ase_database`関数を拡張して結晶モードをサポート:

```python
def load_ase_database(
    db_path,
    split_ratios=(0.8, 0.1, 0.1),
    seed=42,
    include_charges=False,
    remove_h=False,
    remove_duplicates=True,
    duplicate_tolerance=1e-6,
    debug_csv_path=None,
    debug_xyz_path=None,
    crystal_mode=False,  # 新規パラメータ
):
    """
    ASEデータベースから分子または結晶データを読み込む
    
    Parameters
    ----------
    ...
    crystal_mode : bool
        結晶モード（周期境界条件を考慮）
    """
    if crystal_mode:
        # 結晶モードの場合、crystal.data.crystal_loaderを使用
        from crystal.data.crystal_loader import CrystalDataset, collate_crystal_batch
        
        # データベース接続
        from ase.db import connect
        db = connect(db_path)
        
        # インデックス分割
        n_total = len(db)
        indices = np.random.RandomState(seed).permutation(n_total)
        
        n_train = int(split_ratios[0] * n_total)
        n_valid = int(split_ratios[1] * n_total)
        
        train_indices = indices[:n_train].tolist()
        valid_indices = indices[n_train:n_train + n_valid].tolist()
        test_indices = indices[n_train + n_valid:].tolist()
        
        # データセット作成
        train_dataset = CrystalDataset(db_path, train_indices, remove_h=remove_h)
        valid_dataset = CrystalDataset(db_path, valid_indices, remove_h=remove_h)
        test_dataset = CrystalDataset(db_path, test_indices, remove_h=remove_h)
        
        datasets = {
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset,
        }
        
        num_species = train_dataset.num_atom_types
        charge_scale = 1.0  # 結晶では使用しない
        
        return datasets, num_species, charge_scale
    else:
        # 既存の分子モード
        # ... (既存のコード)
```

### 6.2 メインスクリプト

#### 新規ファイル: `main_crystal.py`

```python
"""
結晶構造生成のためのメインスクリプト
"""

import argparse
import torch
from torch.utils.data import DataLoader
import wandb

from configs.datasets_config import get_crystal_dataset_info
from qm9.dataset import load_ase_database
from crystal.models.crystal_dynamics import CrystalDynamics
from crystal.data.crystal_loader import collate_crystal_batch
from equivariant_diffusion import en_diffusion


def main(args):
    # データセット読み込み
    datasets, num_atom_types, _ = load_ase_database(
        db_path=args.ase_db_path,
        split_ratios=args.split_ratios,
        seed=args.seed,
        remove_h=args.remove_h,
        crystal_mode=True,  # 結晶モード
    )
    
    # データローダー作成
    train_loader = DataLoader(
        datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_crystal_batch,
        num_workers=args.num_workers,
    )
    
    valid_loader = DataLoader(
        datasets['valid'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_crystal_batch,
        num_workers=args.num_workers,
    )
    
    # モデル作成
    model = CrystalDynamics(
        in_node_nf=num_atom_types,
        n_dims=3,
        hidden_nf=args.nf,
        n_layers=args.n_layers,
        attention=args.attention,
        use_fractional_coords=args.use_fractional_coords,
        learn_lattice=args.learn_lattice,
    )
    
    # 拡散モデル作成
    diffusion = en_diffusion.EnVariationalDiffusion(
        dynamics=model,
        in_node_nf=num_atom_types,
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
    )
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    diffusion = diffusion.to(device)
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.lr,
        weight_decay=1e-12,
    )
    
    # Wandb初期化
    if not args.no_wandb:
        wandb.init(
            project='crystal_diffusion',
            name=args.exp_name,
            config=vars(args),
        )
    
    # 学習ループ
    for epoch in range(args.n_epochs):
        train_loss = train_epoch(diffusion, train_loader, optimizer, device, epoch)
        valid_loss = validate_epoch(diffusion, valid_loader, device, epoch)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}')
        
        if not args.no_wandb:
            wandb.log({
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'epoch': epoch,
            })
        
        # モデル保存
        if (epoch + 1) % args.save_every == 0:
            save_path = f'outputs/{args.exp_name}/model_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)


def train_epoch(diffusion, train_loader, optimizer, device, epoch):
    # 学習エポックの実装
    # ... (詳細は省略)
    pass


def validate_epoch(diffusion, valid_loader, device, epoch):
    # 検証エポックの実装
    # ... (詳細は省略)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # データ関連
    parser.add_argument('--ase_db_path', type=str, required=True)
    parser.add_argument('--split_ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument('--remove_h', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    
    # モデル関連
    parser.add_argument('--nf', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--attention', type=bool, default=True)
    parser.add_argument('--use_fractional_coords', type=bool, default=True)
    parser.add_argument('--learn_lattice', type=bool, default=True)
    
    # 拡散関連
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
    parser.add_argument('--diffusion_loss_type', type=str, default='l2')
    parser.add_argument('--normalize_factors', nargs=3, type=float, default=[1, 4, 10])
    
    # 学習関連
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=100)
    
    # その他
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    
    args = parser.parse_args()
    main(args)
```

---

## 7. テスト戦略 (Testing Strategy)

### 7.1 ユニットテスト

#### ファイル: `tests/test_periodic_utils.py`

```python
"""
周期境界条件ユーティリティのテスト
"""

import torch
import pytest
from crystal.data.periodic_utils import (
    minimum_image_distance,
    cartesian_to_fractional,
    fractional_to_cartesian,
    cell_params_to_vectors,
    cell_vectors_to_params,
)


def test_coordinate_conversion():
    """座標変換の往復テスト"""
    # 単位格子ベクトル
    cell_vectors = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ]], dtype=torch.float32)
    
    # デカルト座標
    positions_cart = torch.tensor([[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]], dtype=torch.float32)
    
    # デカルト -> 分数 -> デカルト
    positions_frac = cartesian_to_fractional(positions_cart, cell_vectors)
    positions_cart_recovered = fractional_to_cartesian(positions_frac, cell_vectors)
    
    assert torch.allclose(positions_cart, positions_cart_recovered, atol=1e-5)


def test_minimum_image_distance():
    """最小イメージ距離のテスト"""
    # 立方体セル
    cell_vectors = torch.tensor([[
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ]], dtype=torch.float32)
    
    # 2つの原子（周期境界を越える）
    pos1 = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    pos2 = torch.tensor([[[9.0, 0.0, 0.0]]], dtype=torch.float32)
    
    pbc = torch.tensor([[True, True, True]])
    
    distances, vectors = minimum_image_distance(
        pos1, pos2, cell_vectors, pbc, use_fractional=False
    )
    
    # 最小距離は2.0であるべき（9.0 - 1.0 = 8.0 だが、周期的に-2.0が近い）
    expected_distance = 2.0
    assert torch.allclose(distances, torch.tensor([[[expected_distance]]]), atol=1e-4)


def test_cell_param_conversion():
    """格子パラメータとベクトルの変換テスト"""
    # 格子パラメータ (a, b, c, α, β, γ)
    cell_params = torch.tensor([[10.0, 12.0, 15.0, 90.0, 90.0, 90.0]], dtype=torch.float32)
    
    # パラメータ -> ベクトル -> パラメータ
    cell_vectors = cell_params_to_vectors(cell_params)
    cell_params_recovered = cell_vectors_to_params(cell_vectors)
    
    assert torch.allclose(cell_params, cell_params_recovered, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
```

---

## 8. 実装ロードマップ (Implementation Roadmap)

### フェーズ1: データ処理基盤（2週間）

**Week 1:**
- [ ] `crystal/data/crystal_loader.py` 実装
  - CrystalDatasetクラス
  - collate_crystal_batch関数
  - 基本的なテストケース
- [ ] `crystal/data/periodic_utils.py` 実装（基本機能）
  - 座標変換関数
  - 最小イメージ距離計算
  - ユニットテスト作成

**Week 2:**
- [ ] `crystal/data/periodic_utils.py` 実装（応用機能）
  - 周期的近傍リスト構築
  - 格子パラメータ変換
  - 性能最適化
- [ ] データローダーの統合テスト
- [ ] 小規模データセットでの動作確認

### フェーズ2: モデル実装（3週間）

**Week 3:**
- [ ] `crystal/models/periodic_egnn.py` 実装
  - PeriodicEGNNクラスの基本構造
  - メッセージパッシングの実装
  - 単純なテストケース

**Week 4:**
- [ ] PeriodicEGNNの完成と最適化
  - バッチ処理の実装
  - メモリ効率化
  - 勾配チェック

**Week 5:**
- [ ] `crystal/models/lattice_diffusion.py` 実装
  - LatticeDiffusionクラス
  - 格子パラメータの正規化
- [ ] `crystal/models/crystal_dynamics.py` 実装
  - 統合モデルの構築
  - フォワードパスの実装

### フェーズ3: 拡散プロセス統合（2週間）

**Week 6:**
- [ ] 既存の拡散フレームワークとの統合
  - `equivariant_diffusion/en_diffusion.py` の修正
  - 結晶モードのサポート追加
- [ ] 損失関数の実装
  - 原子座標損失
  - 格子パラメータ損失
  - 結合損失

**Week 7:**
- [ ] サンプリング機能の実装
  - 逆拡散プロセス
  - 周期境界条件の適用
- [ ] 統合テストとデバッグ

### フェーズ4: 条件付けと評価（2週間）

**Week 8:**
- [ ] 条件付けモジュールの実装
  - 空間群埋め込み
  - 密度条件付け
  - 格子パラメータ条件付け
- [ ] 条件付き学習の統合

**Week 9:**
- [ ] 評価メトリクスの実装
  - 構造的メトリクス
  - 妥当性チェック
  - 分布比較
- [ ] 評価スクリプトの作成

### フェーズ5: 実験とドキュメント（2週間）

**Week 10:**
- [ ] 小規模データセットでの学習実験
- [ ] ハイパーパラメータチューニング
- [ ] 性能ベンチマーク

**Week 11:**
- [ ] ドキュメントの整備
  - ユーザーガイド
  - APIドキュメント
  - チュートリアル
- [ ] 最終テストとバグ修正

---

## 9. まとめと次のステップ (Summary and Next Steps)

### 実装の優先順位

1. **最優先**: データ処理基盤
   - 周期境界条件の正しい実装が全ての基礎
   
2. **高優先**: 周期的EGNN
   - コアとなるモデルアーキテクチャ
   
3. **中優先**: 格子拡散
   - 結晶生成の完全性に必要
   
4. **低優先**: 空間群対称性
   - Phase 1では単純化、将来の拡張として保留

### 成功基準

- **技術的成功**:
  - 周期境界条件下で正しく動作する
  - 物理的に妥当な結晶構造を生成
  - 既存の分子生成機能との互換性維持
  
- **性能成功**:
  - 1000ステップの拡散を10秒以内で実行（GPU使用時）
  - バッチサイズ32で学習可能
  
- **品質成功**:
  - 生成された構造の80%以上が妥当性チェックをパス
  - 参照データセットとの分布が類似（Wasserstein距離）

### 推奨される開発プラクティス

1. **テスト駆動開発**:
   - 各モジュールに対してユニットテストを先に作成
   
2. **段階的統合**:
   - 小さなコンポーネントから順に統合
   - 各段階で動作確認
   
3. **継続的ベンチマーク**:
   - 小規模データセットで頻繁にテスト
   - 性能劣化を早期に検出

---

## 付録: コード規約とベストプラクティス

### コーディングスタイル

- PEP 8に準拠
- Type hintsを使用
- Docstringを全ての公開関数に追加（Google形式）

### Git ワークフロー

```bash
# 機能ブランチの作成
git checkout -b feature/crystal-data-loader

# 実装とテスト
# ...

# コミット
git add crystal/data/crystal_loader.py tests/test_crystal_loader.py
git commit -m "Implement CrystalDataset for periodic structures"

# プッシュとプルリクエスト
git push origin feature/crystal-data-loader
```

### ドキュメント

- README.mdに使用例を追加
- APIドキュメントをSphinxで生成
- チュートリアルノートブック（Jupyter）を作成

---

**このドキュメントは実装の指針として使用し、実装中に発見された問題に応じて更新してください。**

