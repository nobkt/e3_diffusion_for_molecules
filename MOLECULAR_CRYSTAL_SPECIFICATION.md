# 分子性結晶生成のための拡張仕様書
# Specification for Molecular Crystal Generation Extension

## 概要 (Overview)

本仕様書は、既存のE(3)等変拡散モデル（EDM）を分子性結晶の生成に拡張するための詳細な仕様を定義します。現在のシステムは単一分子の生成に特化していますが、この拡張により周期境界条件を持つ分子性結晶構造の生成が可能になります。

This specification defines the detailed requirements for extending the existing E(3) Equivariant Diffusion Model (EDM) to support molecular crystal generation. The current system specializes in single molecule generation, but this extension will enable generation of molecular crystal structures with periodic boundary conditions.

---

## 1. 要件定義 (Requirements Definition)

### 1.1 機能要件 (Functional Requirements)

#### FR-1: データ入力 (Data Input)
- **FR-1.1**: ASEデータベース形式で分子性結晶構造を読み込む
  - 単位格子パラメータ（a, b, c, α, β, γ）の取得
  - 周期境界条件（pbc）情報の取得
  - 分数座標または絶対座標での原子位置
  - 空間群情報（オプション）
  
- **FR-1.2**: **ホモ結晶のための分子-結晶リンク構造**
  - **単分子データセット**: 各分子のxyz座標情報
  - **分子性結晶データセット**: 各結晶構造と対応する分子IDのリンク
  - **分子-結晶マッピング**: 分子A → 結晶A（1対多、結晶多形を考慮）
  - **分子特徴量の自動抽出**: xyz座標から分子記述子を計算
  
- **FR-1.3**: 複数の結晶構造をバッチ処理できる
  - 異なるサイズの単位格子に対応
  - 異なる空間群の混在に対応
  - 同一分子の異なる結晶多形を区別
  
- **FR-1.4**: 結晶構造の前処理
  - 単位格子の正規化
  - 非対称単位の抽出（オプション）
  - 原子座標の周期境界条件の適用
  - 分子特徴量の正規化

#### FR-2: モデル学習 (Model Training)
- **FR-2.1**: 周期境界条件を考慮した距離計算
  - 最小イメージ規約（Minimum Image Convention）の実装
  - 周期境界を越えた近傍原子の検出
  
- **FR-2.2**: **分子特徴量の統合**
  - 単分子のxyz座標から幾何学的記述子を抽出
  - 分子の形状、サイズ、電荷分布などの特徴量
  - 分子特徴量を結晶生成の条件付けに使用
  - 分子間相互作用の考慮
  
- **FR-2.3**: 格子パラメータの学習
  - 単位格子の形状（a, b, c, α, β, γ）の同時生成
  - 格子パラメータの物理的制約の適用
  - 分子サイズに適した格子パラメータの生成
  
- **FR-2.4**: E(3)等変性の拡張
  - 周期性を保持した等変変換
  - 格子変換に対する共変性
  - 分子回転・並進に対する等変性
  
- **FR-2.5**: **ホモ結晶のための条件付き生成**
  - **分子記述子での条件付け**: 分子構造から自動抽出された特徴量
  - 空間群での条件付け
  - 密度での条件付け
  - 格子定数での条件付け
  - **結晶多形の区別**: 同一分子の異なる結晶形の生成

#### FR-3: サンプル生成 (Sample Generation)
- **FR-3.1**: 結晶構造の生成
  - 単位格子内の原子配置の生成
  - 格子パラメータの生成
  - 対称性の自動適用（オプション）
  
- **FR-3.2**: 生成構造の後処理
  - 周期境界条件の適用
  - 格子の最適化
  - 対称性の検証
  
- **FR-3.3**: 出力形式
  - ASEデータベース形式での保存
  - CIFファイル形式での保存
  - XYZファイル（スーパーセル展開）での保存

#### FR-4: 評価と検証 (Evaluation and Validation)
- **FR-4.1**: 結晶構造の物理的妥当性チェック
  - 原子間距離の検証
  - 格子パラメータの妥当性
  - エネルギー的安定性（オプション）
  
- **FR-4.2**: 統計的評価
  - 空間群分布の比較
  - 格子定数分布の比較
  - 密度分布の比較
  
- **FR-4.3**: 可視化
  - 単位格子の3D可視化
  - スーパーセルの可視化
  - パッキング効率の可視化

### 1.2 非機能要件 (Non-Functional Requirements)

#### NFR-1: パフォーマンス (Performance)
- 単一結晶の生成時間: < 10秒（GPU使用時）
- バッチサイズ: 最大64結晶構造
- メモリ使用量: < 16GB（標準的なGPU）

#### NFR-2: スケーラビリティ (Scalability)
- 単位格子あたり最大500原子まで対応
- データセットサイズ: 10,000〜100,000結晶構造

#### NFR-3: 互換性 (Compatibility)
- 既存の単一分子生成機能との後方互換性を維持
- ASE 3.22以降との互換性
- PyTorch 1.9以降との互換性

#### NFR-4: 拡張性 (Extensibility)
- 新しい空間群の追加が容易
- 新しい条件付け変数の追加が容易
- カスタム結晶メトリクスの追加が容易

---

## 2. データ形式仕様 (Data Format Specification)

### 2.1 入力データ形式

#### 2.1.1 ホモ結晶のためのデータセット構造

**二層構造のデータセット**:

```
molecular_dataset/
├── molecules.db              # 単分子のASEデータベース
│   ├── Molecule 1 (xyz coordinates, no cell, no pbc)
│   ├── Molecule 2
│   └── ...
│
└── crystals.db               # 分子性結晶のASEデータベース
    ├── Crystal 1 (linked to Molecule 1)
    ├── Crystal 2 (linked to Molecule 1, polymorph)
    ├── Crystal 3 (linked to Molecule 2)
    └── ...
```

#### 2.1.2 単分子データ形式
```python
# molecules.db内の単分子
molecule = Atoms(
    symbols=['C', 'H', 'N', ...],      # 原子種
    positions=[[x1,y1,z1], ...],       # 分子のxyz座標（Å）
    # cell, pbcは設定しない（非周期系）
)

# 必須メタデータ（molecule.info内）
molecule.info = {
    'molecule_id': str or int,          # 一意な分子ID
    'smiles': str,                      # SMILES文字列（オプション）
    'molecular_weight': float,          # 分子量
    'molecular_volume': float,          # van der Waals体積
    'inertia_tensor': array,            # 慣性テンソル
    'dipole_moment': float,             # 双極子モーメント（オプション）
    'quadrupole_moment': array,         # 四重極モーメント（オプション）
}
```

#### 2.1.3 分子性結晶データ形式
```python
# crystals.db内の結晶構造
crystal = Atoms(
    symbols=['C', 'H', 'N', ...],      # 原子種
    positions=[[x1,y1,z1], ...],       # 原子座標（Å）
    cell=[[a1,a2,a3], [b1,b2,b3], [c1,c2,c3]],  # 単位格子ベクトル
    pbc=[True, True, True]              # 周期境界条件
)

# 必須フィールド（crystal.info内）
crystal.info = {
    'molecule_id': str or int,          # 対応する分子のID（molecules.dbへのリンク）
    'polymorph_id': str or int,         # 結晶多形の識別子（同一分子の異なる結晶形）
    'space_group': int,                 # 空間群番号（1-230）
    'space_group_symbol': str,          # 空間群記号（例: 'P21/c'）
    'crystal_density': float,           # 密度（g/cm³）
    'crystal_volume': float,            # 単位格子体積（Å³）
    'Z': int,                          # 単位格子あたりの分子数
    'Z_prime': float,                  # 非対称単位あたりの分子数
    'temperature': float,               # 測定温度（K）
    'pressure': float,                  # 測定圧力（GPa）
}
```

#### 2.1.4 分子-結晶リンクの例
```
Molecule A (molecule_id='mol_001')
  ├→ Crystal A-I (polymorph_id='form_I')   # Form I
  ├→ Crystal A-II (polymorph_id='form_II')  # Form II (polymorph)
  └→ Crystal A-III (polymorph_id='form_III') # Form III (another polymorph)

Molecule B (molecule_id='mol_002')
  └→ Crystal B (polymorph_id='form_I')      # Only one known form
```

### 2.2 内部データ表現

#### 2.2.1 分子特徴量の表現
```python
molecular_features = {
    # 幾何学的特徴量
    'molecular_geometry': torch.Tensor,     # [n_mol_atoms, 3] - 分子のxyz座標（重心基準）
    'molecular_volume': torch.Tensor,       # [1] - van der Waals体積
    'molecular_surface_area': torch.Tensor, # [1] - 分子表面積
    'gyration_radius': torch.Tensor,        # [1] - 回転半径
    
    # 形状記述子
    'principal_moments': torch.Tensor,      # [3] - 主慣性モーメント
    'asphericity': torch.Tensor,            # [1] - 非球面度
    'acylindricity': torch.Tensor,          # [1] - 非円筒度
    'shape_anisotropy': torch.Tensor,       # [1] - 形状異方性
    
    # 電子的特徴量（オプション）
    'dipole_moment': torch.Tensor,          # [3] - 双極子モーメントベクトル
    'quadrupole_moment': torch.Tensor,      # [3, 3] - 四重極モーメントテンソル
    'polarizability': torch.Tensor,         # [1] - 分極率（オプション）
    
    # グラフ表現
    'molecular_graph': {
        'atom_types': torch.LongTensor,     # [n_mol_atoms] - 分子内の原子種
        'edge_index': torch.LongTensor,     # [2, n_mol_bonds] - 結合情報
        'edge_attr': torch.Tensor,          # [n_mol_bonds, edge_dim] - 結合特徴
    },
}
```

#### 2.2.2 結晶構造の表現（分子特徴量を含む）
```python
crystal_data = {
    # 結晶の原子情報
    'positions': torch.Tensor,          # [n_atoms, 3] - 分数座標または絶対座標
    'atom_types': torch.LongTensor,     # [n_atoms] - 原子種インデックス
    'charges': torch.Tensor,            # [n_atoms] - 電荷（オプション）
    
    # 格子情報
    'cell_vectors': torch.Tensor,       # [3, 3] - 単位格子ベクトル
    'cell_params': torch.Tensor,        # [6] - (a, b, c, α, β, γ)
    'cell_volume': torch.Tensor,        # [1] - 単位格子体積
    
    # 周期性情報
    'pbc': torch.BoolTensor,            # [3] - 各方向の周期境界条件
    
    # 分子情報（ホモ結晶用）
    'molecule_id': torch.LongTensor,    # [1] - 対応する分子のID
    'polymorph_id': torch.LongTensor,   # [1] - 結晶多形の識別子
    'molecular_features': dict,         # 上記の分子特徴量辞書
    
    # メタデータ
    'space_group': torch.LongTensor,    # [1] - 空間群番号
    'n_atoms': int,                     # 単位格子内の原子数
    'n_molecules': int,                 # 単位格子内の分子数（Z値）
    'Z_prime': float,                   # 非対称単位あたりの分子数
}
```

#### 2.2.3 座標系の定義
- **分数座標 (Fractional Coordinates)**: 単位格子ベクトルを基底とした座標系（0〜1の範囲）
  - 利点: 格子変形に対して不変
  - 使用場面: 格子パラメータの変更時
  
- **絶対座標 (Cartesian Coordinates)**: デカルト座標系（Å単位）
  - 利点: 距離計算が直感的
  - 使用場面: ニューラルネットワークの入力
  
- **分子中心座標 (Molecular Center Coordinates)**: 分子の重心を原点とした座標系
  - 利点: 分子の回転・並進に対して不変
  - 使用場面: 分子特徴量の抽出

変換式:
```
# 結晶座標変換
r_cart = cell_vectors @ r_frac
r_frac = cell_vectors^(-1) @ r_cart

# 分子中心座標
r_mol_center = r_mol - centroid(r_mol)

# 分子の配向（慣性主軸への変換）
r_principal = rotation_matrix(inertia_tensor) @ r_mol_center
```

### 2.3 出力データ形式

#### 2.3.1 ASEデータベース
生成された結晶構造をASE Atomsオブジェクトとして保存

#### 2.3.2 CIFフォーマット
結晶学的情報交換形式（Crystallographic Information File）
- 空間群情報を含む
- 結晶学ソフトウェアとの互換性

#### 2.3.3 XYZフォーマット（拡張）
スーパーセル展開された構造
- 可視化用
- 分子動力学シミュレーション用

---

## 3. システム設計概要 (System Design Overview)

### 3.1 アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                     データ入力層                              │
│  (Data Input Layer)                                         │
├─────────────────────────────────────────────────────────────┤
│  • ASE Database Loader                                      │
│  • Crystal Structure Parser                                 │
│  • Periodic Boundary Handler                                │
│  • Space Group Processor                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   前処理・正規化層                             │
│  (Preprocessing & Normalization Layer)                      │
├─────────────────────────────────────────────────────────────┤
│  • Coordinate Transformer (Frac ↔ Cart)                    │
│  • Cell Normalizer                                          │
│  • Minimum Image Calculator                                 │
│  • Symmetry Analyzer (optional)                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    拡散モデル層                               │
│  (Diffusion Model Layer)                                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │ Position Diffusion│     │ Lattice Diffusion│             │
│  │ (SE(3) Equivariant)│    │ (GL(3) Covariant)│             │
│  └──────────────────┘     └──────────────────┘             │
│           ↓                        ↓                        │
│  ┌────────────────────────────────────────┐                │
│  │   Periodic E(3)-EGNN                   │                │
│  │   - Minimum image convention           │                │
│  │   - Periodic neighbor list             │                │
│  │   - Cell parameter evolution           │                │
│  └────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    条件付けモジュール                           │
│  (Conditioning Module)                                      │
├─────────────────────────────────────────────────────────────┤
│  • Space Group Embedding                                    │
│  • Density Conditioning                                     │
│  • Lattice Parameter Conditioning                           │
│  • Molecular Property Conditioning                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  後処理・検証層                                │
│  (Post-processing & Validation Layer)                       │
├─────────────────────────────────────────────────────────────┤
│  • Symmetry Application (optional)                          │
│  • Structure Optimization                                   │
│  • Physical Validation                                      │
│  • Quality Metrics Calculator                               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    出力・可視化層                              │
│  (Output & Visualization Layer)                             │
├─────────────────────────────────────────────────────────────┤
│  • ASE Database Writer                                      │
│  • CIF File Exporter                                        │
│  • XYZ File Exporter                                        │
│  • 3D Structure Visualizer                                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 主要コンポーネント

#### 3.2.1 Crystal Data Loader
- **役割**: ASEデータベースから結晶構造を読み込み、内部表現に変換
- **入力**: ASEデータベースパス、設定パラメータ
- **出力**: バッチ化された結晶データ

#### 3.2.2 Periodic Boundary Handler
- **役割**: 周期境界条件を考慮した距離・近傍計算
- **機能**:
  - 最小イメージ規約の実装
  - 周期境界を越えた近傍リストの生成
  - 分数座標とデカルト座標の相互変換

#### 3.2.3 Periodic E(3)-EGNN
- **役割**: 周期性を考慮したE(3)等変グラフニューラルネットワーク
- **拡張点**:
  - メッセージパッシングで周期境界条件を考慮
  - 格子パラメータの学習可能な表現
  - 格子変換に対する共変性

#### 3.2.4 Lattice Parameter Model
- **役割**: 単位格子パラメータの生成と進化
- **機能**:
  - 格子パラメータ (a, b, c, α, β, γ) の同時生成
  - 物理的制約の適用（正の長さ、角度範囲）
  - 格子体積の制御

#### 3.2.5 Space Group Conditioner
- **役割**: 空間群情報を用いた条件付き生成
- **機能**:
  - 空間群の埋め込み表現
  - 対称操作の適用（オプション）
  - 空間群制約の強制

---

## 4. 主要な技術課題と解決策 (Key Technical Challenges and Solutions)

### 4.1 周期境界条件の取り扱い

#### 課題
- ユークリッド空間を前提とした既存のE(3)等変性が、周期性のあるトーラス空間では適用できない
- 原子間距離の計算に最小イメージ規約が必要

#### 解決策
- **最小イメージ規約の実装**
  ```python
  def minimum_image_distance(pos1, pos2, cell_vectors):
      """
      周期境界条件下での最小距離を計算
      """
      # デカルト座標を分数座標に変換
      frac1 = cart_to_fractional(pos1, cell_vectors)
      frac2 = cart_to_fractional(pos2, cell_vectors)
      
      # 分数座標での差を-0.5~0.5の範囲に正規化
      frac_diff = frac2 - frac1
      frac_diff = frac_diff - torch.round(frac_diff)
      
      # デカルト座標に戻して距離計算
      cart_diff = fractional_to_cart(frac_diff, cell_vectors)
      distance = torch.norm(cart_diff, dim=-1)
      
      return distance, cart_diff
  ```

- **周期的近傍リストの構築**
  - カットオフ半径内の全ての周期的イメージを考慮
  - セルリスト法による効率的な実装

### 4.2 格子パラメータの学習

#### 課題
- 格子パラメータ (a, b, c, α, β, γ) は原子座標とは異なるスケールと制約を持つ
- 格子の形状と原子配置の協調的な学習が必要

#### 解決策
- **分離された拡散プロセス**
  - 原子座標の拡散: SE(3)等変
  - 格子パラメータの拡散: GL(3)共変
  - 両者を結合したロス関数

- **格子パラメータの正規化**
  ```python
  lattice_params = {
      'lengths': (a, b, c),           # log空間で学習
      'angles': (α, β, γ),            # sin/cos表現で学習
      'volume': V                     # log空間で学習
  }
  ```

- **物理的制約の適用**
  - a, b, c > 0（正の長さ）
  - 0° < α, β, γ < 180°（有効な角度）
  - 体積の妥当性チェック

### 4.3 E(3)等変性の保持

#### 課題
- 周期的な系では、並進対称性が格子ベクトルの整数倍に離散化される
- 格子の回転・変形に対する共変性が必要

#### 解決策
- **相対座標の使用**
  - 絶対座標ではなく、相対的な原子間ベクトルを使用
  - 周期境界条件を考慮した相対ベクトル計算

- **格子共変層の導入**
  ```python
  class LatticeCovariantLayer(nn.Module):
      """
      格子変換に対して共変な層
      """
      def forward(self, positions, cell_matrix):
          # 分数座標での処理（格子に依存しない）
          frac_coords = cart_to_fractional(positions, cell_matrix)
          
          # 等変な特徴抽出
          features = self.process_fractional(frac_coords)
          
          # デカルト座標に戻す（格子変換に共変）
          output = fractional_to_cart(features, cell_matrix)
          
          return output
  ```

### 4.4 空間群対称性の取り扱い

#### 課題
- 230種類の空間群それぞれに固有の対称性がある
- 対称性を厳密に保持すると生成の自由度が制限される

#### 解決策
- **段階的アプローチ**
  1. **Phase 1**: 対称性を無視した自由な生成（P1空間群相当）
  2. **Phase 2**: 事後的な対称性の適用（オプション）
  3. **Phase 3**: 対称性制約下での生成（将来の拡張）

- **対称性の事後適用**
  ```python
  def apply_space_group_symmetry(structure, space_group):
      """
      生成された構造に空間群対称性を適用
      """
      from pymatgen.symmetry import SpaceGroup
      sg = SpaceGroup.from_int_number(space_group)
      
      # 非対称単位の抽出
      asymmetric_unit = extract_asymmetric_unit(structure, sg)
      
      # 対称操作を適用して完全な単位格子を生成
      full_structure = apply_symmetry_operations(asymmetric_unit, sg)
      
      return full_structure
  ```

### 4.5 スケーラビリティ

#### 課題
- 結晶構造は分子よりも多くの原子を含む可能性がある
- 周期的イメージの考慮により計算コストが増大

#### 解決策
- **効率的な近傍リスト**
  - カットオフ半径の導入
  - セルリスト法またはVerletリストの使用

- **バッチサイズの調整**
  - 結晶サイズに応じた動的なバッチサイズ
  - グラディエント累積の活用

- **メモリ最適化**
  - スパースグラフ表現の使用
  - チェックポインティングの活用

---

## 5. 実装フェーズ (Implementation Phases)

### Phase 1: 基盤整備（2-3週間）
- [ ] ASE結晶データローダーの実装
- [ ] 周期境界条件ハンドラの実装
- [ ] 座標変換ユーティリティの実装
- [ ] 最小イメージ規約の実装
- [ ] 基本的なテストケースの作成

### Phase 2: モデル拡張（3-4週間）
- [ ] Periodic E(3)-EGNNの実装
- [ ] 格子パラメータ拡散モデルの実装
- [ ] 結合された損失関数の実装
- [ ] 周期的近傍リストの実装
- [ ] 統合テストの作成

### Phase 3: 条件付け（2-3週間）
- [ ] 空間群埋め込みの実装
- [ ] 密度条件付けの実装
- [ ] 格子パラメータ条件付けの実装
- [ ] 条件付きサンプリングのテスト

### Phase 4: 評価・検証（2週間）
- [ ] 結晶構造評価メトリクスの実装
- [ ] 物理的妥当性チェッカーの実装
- [ ] 統計的評価ツールの実装
- [ ] ベンチマークデータセットでの評価

### Phase 5: 可視化・出力（1-2週間）
- [ ] 結晶構造3D可視化の実装
- [ ] CIFファイル出力の実装
- [ ] スーパーセル展開の実装
- [ ] ドキュメントの整備

---

## 6. 評価指標 (Evaluation Metrics)

### 6.1 構造的指標

#### 6.1.1 格子パラメータの精度
- **Mean Absolute Error (MAE)**: 生成された格子パラメータと参照値の差
- **Distribution Matching**: KLダイバージェンス、Wasserstein距離

#### 6.1.2 原子配置の妥当性
- **Minimum Distance Check**: 原子間の最小距離が妥当な範囲か
- **Coordination Number**: 配位数の分布が妥当か
- **Radial Distribution Function (RDF)**: 動径分布関数の比較

#### 6.1.3 密度の精度
- **Density Error**: 生成された結晶の密度と実験値の比較
- **Volume Error**: 単位格子体積の誤差

### 6.2 対称性指標

#### 6.2.1 空間群の再現性
- **Space Group Accuracy**: 生成された構造の空間群が正しいか
- **Symmetry Score**: 対称性の保持度合い

### 6.3 物理的指標

#### 6.3.1 安定性
- **Molecular Stability**: 各分子が物理的に妥当か
- **Packing Efficiency**: パッキング効率
- **Energy Landscape** (オプション): エネルギー計算による安定性評価

### 6.4 生成品質指標

#### 6.4.1 多様性
- **Diversity Score**: 生成された構造の多様性
- **Uniqueness**: 重複しない構造の割合

#### 6.4.2 新規性
- **Novelty Score**: データセットに存在しない新しい構造の割合

---

## 7. データセット要件 (Dataset Requirements)

### 7.1 推奨データセット

#### 7.1.1 Cambridge Structural Database (CSD)
- 100万以上の有機・金属有機結晶構造
- 高品質な実験データ
- ライセンス: 商用（学術利用可能）

#### 7.1.2 Materials Project
- 無機結晶構造データベース
- DFT計算データ
- ライセンス: オープンソース

#### 7.1.3 カスタムデータセット
- 特定の分子種に特化したデータセット
- ASE DB形式で準備

### 7.2 データセットの前処理

#### 7.2.1 品質フィルタリング
- 不完全な構造の除外
- 異常な格子パラメータの除外
- 重複構造の除外

#### 7.2.2 正規化
- 単位格子の標準化
- 原子座標の正規化
- 空間群の標準化

---

## 8. 使用例 (Usage Examples)

### 8.1 基本的な使用方法

```python
# 結晶データの読み込み
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path crystals.db \
    --crystal_mode True \
    --n_epochs 3000 \
    --batch_size 32 \
    --exp_name crystal_generation

# 条件付き生成（空間群指定）
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path crystals.db \
    --crystal_mode True \
    --conditioning space_group \
    --exp_name crystal_conditional

# サンプル生成
python eval_sample.py \
    --model_path outputs/crystal_generation \
    --n_samples 1000 \
    --output_format cif \
    --crystal_mode True
```

### 8.2 高度な使用例

```python
# 密度と格子定数での条件付け
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path crystals.db \
    --crystal_mode True \
    --conditioning density lattice_params \
    --exp_name crystal_multi_conditional

# カスタム評価
python eval_analyze.py \
    --model_path outputs/crystal_generation \
    --n_samples 10000 \
    --crystal_mode True \
    --evaluate_symmetry True \
    --evaluate_packing True
```

---

## 9. リスクと制約 (Risks and Limitations)

### 9.1 技術的リスク

#### 9.1.1 計算コスト
- **リスク**: 周期境界条件により計算量が大幅に増加
- **対策**: 効率的なアルゴリズムの実装、カットオフ半径の最適化

#### 9.1.2 収束性
- **リスク**: 格子パラメータと原子座標の同時学習が不安定
- **対策**: 段階的学習、適切な正規化、学習率スケジューリング

#### 9.1.3 対称性の保持
- **リスク**: 生成された構造が対称性を完全に保持できない
- **対策**: 緩和された対称性チェック、事後的な対称性適用

### 9.2 データ制約

#### 9.2.1 データの入手性
- **制約**: 高品質な結晶構造データはライセンスが必要な場合がある
- **対策**: オープンソースデータセットの活用、カスタムデータセットの作成

#### 9.2.2 データの多様性
- **制約**: 特定の空間群や分子種に偏りがある可能性
- **対策**: データ拡張、バランシング手法の適用

### 9.3 物理的制約

#### 9.3.1 生成構造の安定性
- **制約**: 生成された構造が必ずしも安定とは限らない
- **対策**: エネルギー評価の追加、構造最適化の組み込み

---

## 10. 将来の拡張 (Future Extensions)

### 10.1 短期的拡張（6ヶ月以内）
- ポリモルフ生成（同一分子の異なる結晶形）
- 共結晶の生成（2種類以上の分子を含む結晶）
- 温度・圧力依存性の考慮

### 10.2 中期的拡張（6-12ヶ月）
- 対称性制約下での直接生成
- エネルギー誘導型生成
- 逆設計（目的特性からの結晶設計）

### 10.3 長期的拡張（12ヶ月以上）
- 無機結晶への拡張
- 表面・界面の考慮
- 動的プロセス（結晶成長、相転移）のシミュレーション

---

## 11. 参考文献 (References)

### 11.1 理論的背景
1. Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D" (2022)
2. Jiao et al. "Crystal Diffusion Variational Autoencoder for Periodic Material Generation" (2023)
3. Xie et al. "Crystal Diffusion Generative Models" (2021)

### 11.2 周期系の取り扱い
4. Schütt et al. "SchNet - A deep learning architecture for molecules and materials" (2018)
5. Choudhary et al. "Atomistic Line Graph Neural Network for improved materials property predictions" (2021)

### 11.3 空間群と対称性
6. Hahn, T. "International Tables for Crystallography" (2005)
7. Aroyo et al. "Crystallography online: Bilbao Crystallographic Server" (2011)

---

## 付録 A: 用語集 (Glossary)

- **分子性結晶 (Molecular Crystal)**: 分子が結晶格子を形成した固体
- **単位格子 (Unit Cell)**: 結晶構造の最小繰り返し単位
- **空間群 (Space Group)**: 結晶の対称性を記述する数学的群
- **分数座標 (Fractional Coordinates)**: 単位格子ベクトルを基底とした座標
- **最小イメージ規約 (Minimum Image Convention)**: 周期境界条件下で最も近い距離を計算する規約
- **非対称単位 (Asymmetric Unit)**: 対称操作により単位格子全体を生成できる最小単位

---

## 付録 B: 設定ファイル例 (Configuration File Example)

```yaml
# crystal_config.yaml

dataset:
  name: "crystal_ase_db"
  path: "data/molecular_crystals.db"
  split_ratios: [0.8, 0.1, 0.1]
  remove_duplicates: true
  
crystal_settings:
  mode: "crystal"  # "molecule" or "crystal"
  coordinate_system: "fractional"  # "fractional" or "cartesian"
  use_minimum_image: true
  cutoff_radius: 10.0  # Angstroms
  max_neighbors: 50
  
  lattice:
    learn_lattice: true
    lattice_noise_schedule: "polynomial_2"
    normalize_lattice: true
    constraints:
      min_length: 3.0  # Angstroms
      max_length: 50.0  # Angstroms
      min_angle: 30.0  # degrees
      max_angle: 150.0  # degrees
  
  symmetry:
    apply_space_group: false  # Phase 1: false
    space_group_constraint: null  # null or specific space group number
    check_symmetry: true  # validate after generation

model:
  type: "periodic_egnn_dynamics"
  n_layers: 9
  nf: 256
  attention: true
  periodic_message_passing: true
  
conditioning:
  properties: ["space_group", "density", "lattice_params"]
  space_group_embedding_dim: 64
  
training:
  n_epochs: 3000
  batch_size: 32
  lr: 1e-4
  diffusion_steps: 1000
  diffusion_noise_schedule: "polynomial_2"
  
evaluation:
  metrics: ["lattice_mae", "density_error", "rdf_similarity", "space_group_accuracy"]
  validate_physics: true
  check_minimum_distances: true
```

---

## 改訂履歴 (Revision History)

| バージョン | 日付 | 変更内容 | 著者 |
|----------|------|---------|------|
| 1.0 | 2025-01-XX | 初版作成 | - |

