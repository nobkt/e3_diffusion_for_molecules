# APIリファレンス（日本語）

**作成日**: 2025年10月25日  
**バージョン**: 1.0  
**対象**: E3 Diffusion for Molecules

---

## 目次

1. [qm9モジュール](#qm9モジュール)
   - [models](#qmmodels)
   - [dataset](#qmdataset)
   - [utils](#qmutils)
   - [analyze](#qmanalyze)
   - [sampling](#qmsampling)
2. [crystalモジュール](#crystalモジュール)
   - [models](#crystalmodels)
   - [conditioning](#crystalconditioning)
   - [utils](#crystalutils)
3. [egnnモジュール](#egnnモジュール)
   - [egnn_new](#egnn_new)
   - [models](#egnnmodels)
4. [equivariant_diffusionモジュール](#equivariant_diffusionモジュール)
   - [en_diffusion](#en_diffusion)
   - [distributions](#distributions)

---

## qm9モジュール

分子生成のための主要モジュール。QM9データセットと汎用分子生成機能を提供します。

### qm9.models

分子生成モデルの構築と最適化を担当するモジュール。

#### `get_model(args, device, dataset_info, dataloader_train)`

E(3)同変拡散モデルを構築します。

**パラメータ**:
- `args` (argparse.Namespace): モデル設定パラメータ
  - `nf` (int): 隠れ層の次元数（デフォルト: 256）
  - `n_layers` (int): EGNNレイヤー数（デフォルト: 9）
  - `diffusion_steps` (int): 拡散ステップ数（デフォルト: 1000）
  - `diffusion_noise_schedule` (str): ノイズスケジュール（'polynomial'または'cosine'）
  - `diffusion_noise_precision` (float): ノイズの精度（デフォルト: 1e-5）
  - `diffusion_loss_type` (str): 損失関数のタイプ（デフォルト: 'l2'）
  - `conditioning` (list): 条件付けに使用する性質のリスト
  - `context_node_nf` (int): コンテキスト特徴の次元数
  - `include_charges` (bool): 電荷を含めるかどうか
  - `attention` (bool): アテンション機構を使用するかどうか
  - `tanh` (bool): tanh活性化関数を使用するかどうか
  - `model` (str): モデルのタイプ（'egnn_dynamics'など）
  - `norm_constant` (float): 正規化定数
  - `inv_sublayers` (int): 不変サブレイヤーの数
  - `sin_embedding` (bool): sin埋め込みを使用するかどうか
  - `normalization_factor` (float): 正規化係数
  - `aggregation_method` (str): 集約方法（'sum'または'mean'）
  - `condition_time` (bool): 時間条件付けを使用するかどうか
  - `normalize_factors` (list): 正規化係数のリスト
  - `probabilistic_model` (str): 確率モデルのタイプ（'diffusion'）
- `device` (torch.device): 計算デバイス（cuda/cpu）
- `dataset_info` (dict): データセット情報
  - `atom_decoder` (list): 原子タイプのデコーダー
  - `n_nodes` (dict): ノード数の分布
- `dataloader_train` (DataLoader): 訓練データローダー（性質分布推定用）

**戻り値**:
- `model` (EnVariationalDiffusion): 拡散モデル
- `nodes_dist` (DistributionNodes): ノード数分布
- `prop_dist` (DistributionProperty or None): 性質分布

**使用例**:
```python
from qm9.models import get_model
from qm9 import utils as qm9_utils
from qm9.dataset import retrieve_dataloaders

# データセット情報の取得
dataset_info = qm9_utils.get_dataset_info('qm9', remove_h=False)

# データローダーの取得
dataloaders, charge_scale = retrieve_dataloaders(args)

# モデルの構築
model, nodes_dist, prop_dist = get_model(
    args, device, dataset_info, dataloaders['train']
)
```

**注意事項**:
- 条件付き生成の場合、`args.conditioning`にリスト形式で性質名を設定する必要があります
- GPU使用時は十分なメモリが必要です（推奨: 8GB以上）
- `args.context_node_nf`は条件付けする特徴の総次元数に設定します

---

#### `get_optim(args, generative_model)`

モデルのオプティマイザを構築します。

**パラメータ**:
- `args` (argparse.Namespace): 設定パラメータ
  - `lr` (float): 学習率（デフォルト: 1e-4）
- `generative_model` (torch.nn.Module): 最適化するモデル

**戻り値**:
- `optimizer` (torch.optim.AdamW): AdamWオプティマイザ

**使用例**:
```python
from qm9.models import get_optim

optimizer = get_optim(args, model)
```

**注意事項**:
- デフォルトでAdamWを使用します（weight_decay=1e-12, amsgrad=True）
- 学習率スケジューラは別途設定する必要があります

---

#### クラス: `DistributionNodes`

分子のノード数（原子数）の分布を表現するクラス。

**初期化パラメータ**:
- `histogram` (dict): ノード数の頻度分布

**メソッド**:

##### `sample(n_samples=1)`
ノード数分布からサンプリングします。

**パラメータ**:
- `n_samples` (int): サンプル数

**戻り値**:
- `n_nodes` (torch.Tensor): サンプリングされたノード数（shape: [n_samples]）

##### `log_prob(batch_n_nodes)`
ノード数の対数確率を計算します。

**パラメータ**:
- `batch_n_nodes` (torch.Tensor): ノード数のバッチ（shape: [batch_size]）

**戻り値**:
- `log_prob` (torch.Tensor): 対数確率（shape: [batch_size]）

**使用例**:
```python
# モデル構築時に自動的に作成されます
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloader)

# ノード数のサンプリング
n_nodes = nodes_dist.sample(n_samples=10)  # 10個の分子のノード数をサンプリング
```

---

#### クラス: `DistributionProperty`

分子性質の分布を表現するクラス。条件付き生成に使用します。

**初期化パラメータ**:
- `dataloader` (DataLoader): データローダー
- `properties` (list): 性質名のリスト

**メソッド**:

##### `sample(n_nodes)`
性質分布からサンプリングします。

**パラメータ**:
- `n_nodes` (int): ノード数

**戻り値**:
- `properties` (torch.Tensor): サンプリングされた性質（shape: [n_properties]）

##### `sample_batch(nodesxsample)`
バッチサンプリングを行います。

**パラメータ**:
- `nodesxsample` (torch.Tensor): 各サンプルのノード数

**戻り値**:
- `properties` (torch.Tensor): サンプリングされた性質（shape: [batch_size, n_properties]）

**使用例**:
```python
# 条件付き生成時にコンテキストを作成
if prop_dist is not None:
    sampled_props = prop_dist.sample(n_nodes)
```

---

### qm9.dataset

データセットの読み込みと前処理を担当するモジュール。

#### `retrieve_dataloaders(args)`

QM9またはGEOMデータセットのデータローダーを取得します。

**パラメータ**:
- `args` (argparse.Namespace): データセット設定
  - `dataset` (str): データセット名（'qm9', 'geom', 'ase_db'など）
  - `batch_size` (int): バッチサイズ
  - `num_workers` (int): ワーカー数
  - `filter_n_atoms` (int or None): 原子数によるフィルタリング
  - `remove_h` (bool): 水素原子を除去するかどうか
  - `include_charges` (bool): 電荷を含めるかどうか

**戻り値**:
- `dataloaders` (dict): 'train', 'valid', 'test'キーを持つデータローダーの辞書
- `charge_scale` (float): 電荷のスケール係数

**使用例**:
```python
from qm9.dataset import retrieve_dataloaders

dataloaders, charge_scale = retrieve_dataloaders(args)
train_loader = dataloaders['train']
valid_loader = dataloaders['valid']
test_loader = dataloaders['test']

# バッチの取得
for batch in train_loader:
    positions = batch['positions']  # shape: [batch_size, n_nodes, 3]
    one_hot = batch['one_hot']      # shape: [batch_size, n_nodes, n_atom_types]
    charges = batch['charges']      # shape: [batch_size, n_nodes]
    atom_mask = batch['atom_mask']  # shape: [batch_size, n_nodes]
    edge_mask = batch['edge_mask']  # shape: [batch_size * n_nodes * n_nodes]
    break
```

**注意事項**:
- データセットが存在しない場合は自動的にダウンロードされます
- `remove_h=True`の場合、水素原子が除去されます
- バッチサイズは利用可能なメモリに応じて調整してください

---

#### `load_ase_database(db_path, split_ratios=(0.8, 0.1, 0.1), seed=42, ...)`

ASEデータベースからデータセットを読み込みます。

**パラメータ**:
- `db_path` (str): ASEデータベースファイルのパス
- `split_ratios` (tuple): 訓練、検証、テストの分割比率（合計1.0）
- `seed` (int): 再現性のためのランダムシード
- `include_charges` (bool): 原子電荷を含めるかどうか（デフォルト: False）
- `remove_h` (bool): 水素原子を除去するかどうか
- `remove_duplicates` (bool): 重複分子を除去するかどうか
- `duplicate_tolerance` (float): 重複判定の許容誤差（Angstrom）
- `debug_csv_path` (str, optional): デバッグ用CSVファイルのパス
- `debug_xyz_path` (str, optional): デバッグ用XYZファイルのディレクトリパス

**戻り値**:
- `datasets` (dict): 'train', 'valid', 'test'キーを持つデータセットの辞書
- `num_species` (int): ユニークな原子種の数
- `charge_scale` (float): 電荷のスケール係数

**使用例**:
```python
from qm9.dataset import load_ase_database

# ASEデータベースの読み込み
datasets, num_species, charge_scale = load_ase_database(
    db_path='molecules.db',
    split_ratios=(0.8, 0.1, 0.1),
    seed=42,
    remove_h=False,
    remove_duplicates=True,
    duplicate_tolerance=1e-6
)

train_dataset = datasets['train']
valid_dataset = datasets['valid']
test_dataset = datasets['test']

print(f"訓練データ: {len(train_dataset)} 分子")
print(f"検証データ: {len(valid_dataset)} 分子")
print(f"テストデータ: {len(test_dataset)} 分子")
```

**注意事項**:
- ASEパッケージが必要です（`pip install ase`）
- データベースファイルが存在しない場合はエラーが発生します
- 重複除去はデフォルトで有効です
- カスタム性質（分子記述子など）はASEデータベースのkey_value_pairsから読み込まれます

---

### qm9.utils

ユーティリティ関数を提供するモジュール。

#### `compute_mean_mad(dataloaders, properties, dataset_name)`

性質の平均とMAD（Mean Absolute Deviation）を計算します。

**パラメータ**:
- `dataloaders` (dict): データローダーの辞書
- `properties` (list): 性質名のリスト
- `dataset_name` (str): データセット名（'qm9', 'ase_db'など）

**戻り値**:
- `property_norms` (dict): 各性質の平均とMADを含む辞書

**使用例**:
```python
from qm9.utils import compute_mean_mad

# 性質の正規化パラメータを計算
property_norms = compute_mean_mad(
    dataloaders, 
    ['alpha', 'homo', 'lumo'], 
    'qm9'
)

# 平均とMADの取得
alpha_mean = property_norms['alpha']['mean']
alpha_mad = property_norms['alpha']['mad']

print(f"Alpha - Mean: {alpha_mean:.4f}, MAD: {alpha_mad:.4f}")
```

**注意事項**:
- 正規化は `(value - mean) / mad` で行います
- 多次元特徴の場合は次元ごとに平均とMADが計算されます
- MADがゼロの場合は1e-8にクリップされます

---

#### `prepare_context(conditioning, minibatch, property_norms)`

条件付き生成のためのコンテキストテンソルを準備します。

**パラメータ**:
- `conditioning` (list): 条件付けに使用する性質のリスト
- `minibatch` (dict): ミニバッチデータ
- `property_norms` (dict): 性質の正規化パラメータ

**戻り値**:
- `context` (torch.Tensor): コンテキストテンソル（shape: [batch_size, n_nodes, context_dim]）
- `context_node_nf` (int): コンテキストの次元数

**使用例**:
```python
from qm9.utils import prepare_context, compute_mean_mad

# 正規化パラメータの計算
property_norms = compute_mean_mad(dataloaders, ['alpha', 'homo'], 'qm9')

# コンテキストの準備
for batch in train_loader:
    context, context_dim = prepare_context(
        conditioning=['alpha', 'homo'],
        minibatch=batch,
        property_norms=property_norms
    )
    break

print(f"Context shape: {context.shape}")
print(f"Context dimension: {context_dim}")
```

**注意事項**:
- スカラー性質は全ノードにブロードキャストされます
- 多次元特徴（官能基エンコーディングなど）も正しく処理されます
- `global_features`として定義された特徴は常にブロードキャストされます

---

#### `get_adj_matrix(n_nodes, batch_size, device)`

隣接行列のエッジインデックスを取得します（全結合グラフ）。

**パラメータ**:
- `n_nodes` (int): ノード数
- `batch_size` (int): バッチサイズ
- `device` (torch.device): デバイス

**戻り値**:
- `edges` (list): [row_indices, col_indices]のリスト

**使用例**:
```python
from qm9.utils import get_adj_matrix

edges = get_adj_matrix(n_nodes=19, batch_size=32, device='cuda')
```

**注意事項**:
- 結果はキャッシュされるため、同じパラメータでの2回目以降の呼び出しは高速です
- 全結合グラフを生成します（自己ループを含む）

---

### qm9.analyze

生成された分子の品質を評価するモジュール。

#### `check_stability(positions, atom_type, dataset_info, debug=False)`

分子の化学的安定性をチェックします。

**パラメータ**:
- `positions` (np.ndarray): 原子座標（shape: [n_atoms, 3]）
- `atom_type` (np.ndarray): 原子タイプ（shape: [n_atoms]）
- `dataset_info` (dict): データセット情報
  - `atom_encoder` (dict): 原子タイプのエンコーダー
  - `atom_decoder` (list): 原子タイプのデコーダー
- `debug` (bool): デバッグ情報を出力するかどうか

**戻り値**:
- `stability_results` (dict): 安定性に関する情報
  - `mol_stable` (bool): 分子が安定かどうか
  - `atm_stable` (int): 安定な原子の数
  - `n_atoms` (int): 総原子数
  - `num_bonds` (int): 結合数

**使用例**:
```python
from qm9.analyze import check_stability
from qm9 import utils as qm9_utils

dataset_info = qm9_utils.get_dataset_info('qm9', remove_h=False)

# 生成された分子の評価
stability = check_stability(
    positions=x.cpu().numpy(),
    atom_type=atom_types.cpu().numpy(),
    dataset_info=dataset_info,
    debug=False
)

if stability['mol_stable']:
    print("分子は化学的に安定です")
else:
    print(f"不安定な原子: {stability['n_atoms'] - stability['atm_stable']}")
```

**注意事項**:
- 結合の許容数は原子タイプに基づいて決定されます
- 結合長は原子間距離から推定されます
- RDKitまたはOpenBabelが利用可能な場合、より詳細な分析が可能です

---

#### `analyze_stability_for_molecules(one_hot, positions, atom_decoder, verbose=False)`

バッチ内の複数分子の安定性を一括解析します。

**パラメータ**:
- `one_hot` (torch.Tensor): One-hot原子タイプ（shape: [batch_size, n_nodes, n_atom_types]）
- `positions` (torch.Tensor): 原子座標（shape: [batch_size, n_nodes, 3]）
- `atom_decoder` (list): 原子タイプのデコーダー
- `verbose` (bool): 詳細情報を出力するかどうか

**戻り値**:
- `molecule_stable` (float): 安定な分子の割合
- `atom_stable` (float): 安定な原子の割合
- `num_bonds_per_mol` (list): 各分子の結合数

**使用例**:
```python
from qm9.analyze import analyze_stability_for_molecules

# バッチ評価
mol_stable, atom_stable, bonds = analyze_stability_for_molecules(
    one_hot=one_hot_batch,
    positions=positions_batch,
    atom_decoder=dataset_info['atom_decoder'],
    verbose=True
)

print(f"分子安定性: {mol_stable:.2%}")
print(f"原子安定性: {atom_stable:.2%}")
print(f"平均結合数: {np.mean(bonds):.2f}")
```

---

### qm9.sampling

サンプリングとチェーン生成を担当するモジュール。

#### `sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None)`

拡散モデルからサンプリングチェーンを生成します。

**パラメータ**:
- `args` (argparse.Namespace): 設定パラメータ
- `device` (torch.device): デバイス
- `flow` (EnVariationalDiffusion): 拡散モデル
- `n_tries` (int): 試行回数
- `dataset_info` (dict): データセット情報
- `prop_dist` (DistributionProperty, optional): 性質分布

**戻り値**:
- `chain` (torch.Tensor): サンプリングチェーン
- `x` (torch.Tensor): 最終座標
- `one_hot` (torch.Tensor): 最終原子タイプ

**使用例**:
```python
from qm9.sampling import sample_chain

chain, x, one_hot = sample_chain(
    args=args,
    device=device,
    flow=model,
    n_tries=1,
    dataset_info=dataset_info,
    prop_dist=prop_dist
)

print(f"Chain shape: {chain.shape}")
print(f"Final positions: {x.shape}")
```

**注意事項**:
- チェーンは100フレームで保存されます
- 最後のフレームは10回繰り返されます（可視化用）
- 不安定な分子の場合は再試行されます

---

## crystalモジュール

結晶構造生成のための専用モジュール。

### crystal.models

結晶生成モデルを提供します。

#### クラス: `CrystalDynamics`

結晶構造のための動力学モデル。

**初期化パラメータ**:
- `dynamics` (EGNN): ベースとなるEGNNモデル
- `in_node_nf` (int): 入力ノード特徴数
- `context_node_nf` (int): コンテキスト特徴数
- `lattice_dim` (int): 格子パラメータの次元数（デフォルト: 6）

**メソッド**:

##### `forward(t, xh, node_mask, edge_mask, context)`

順伝播を実行します。

**パラメータ**:
- `t` (torch.Tensor): 時間ステップ
- `xh` (torch.Tensor): 座標と特徴の結合テンソル
- `node_mask` (torch.Tensor): ノードマスク
- `edge_mask` (torch.Tensor): エッジマスク
- `context` (torch.Tensor, optional): コンテキスト

**戻り値**:
- `output` (torch.Tensor): 動力学の出力

**使用例**:
```python
from crystal.models import CrystalDynamics

crystal_dynamics = CrystalDynamics(
    dynamics=base_egnn,
    in_node_nf=11,
    context_node_nf=64,
    lattice_dim=6
)
```

---

### crystal.conditioning

結晶条件付けのためのモジュール。

#### クラス: `SpaceGroupEmbedding`

空間群埋め込みを生成するクラス。

**初期化パラメータ**:
- `num_space_groups` (int): 空間群の数（デフォルト: 230）
- `embedding_dim` (int): 埋め込み次元数

**メソッド**:

##### `forward(space_group_ids)`

空間群IDから埋め込みベクトルを生成します。

**パラメータ**:
- `space_group_ids` (torch.Tensor): 空間群ID（shape: [batch_size]）

**戻り値**:
- `embeddings` (torch.Tensor): 埋め込みベクトル（shape: [batch_size, embedding_dim]）

**使用例**:
```python
from crystal.conditioning import SpaceGroupEmbedding

sg_embedding = SpaceGroupEmbedding(
    num_space_groups=230,
    embedding_dim=64
)

# 空間群P1 (1), P-1 (2), P2 (3)の埋め込み
sg_ids = torch.tensor([1, 2, 3])
embeddings = sg_embedding(sg_ids)
print(f"Embeddings shape: {embeddings.shape}")  # [3, 64]
```

**注意事項**:
- 空間群番号は1-230の範囲である必要があります
- 7つの結晶系すべてがサポートされています

---

#### クラス: `DensityConditioning`

密度条件付けを実装するクラス。

**初期化パラメータ**:
- `hidden_dim` (int): 隠れ層の次元数

**メソッド**:

##### `forward(density_values)`

密度値から条件付けベクトルを生成します。

**パラメータ**:
- `density_values` (torch.Tensor): 密度値（g/cm³、shape: [batch_size]）

**戻り値**:
- `conditioning` (torch.Tensor): 条件付けベクトル

**使用例**:
```python
from crystal.conditioning import DensityConditioning

density_cond = DensityConditioning(hidden_dim=64)

# 密度1.0, 1.5, 2.0 g/cm³の条件付け
densities = torch.tensor([1.0, 1.5, 2.0])
cond_vectors = density_cond(densities)
```

---

### crystal.utils

結晶関連のユーティリティ関数。

#### `get_crystal_systems()`

結晶系の情報を取得します。

**戻り値**:
- `crystal_systems` (dict): 結晶系の情報
  - 各結晶系の名前、空間群番号、格子パラメータ制約

**使用例**:
```python
from crystal.utils import get_crystal_systems

systems = get_crystal_systems()
print(f"立方晶系: 空間群 {systems['cubic']['space_groups']}")
```

---

#### `generate_wyckoff_positions(space_group, n_atoms)`

Wyckoff位置を生成します。

**パラメータ**:
- `space_group` (int): 空間群番号（1-230）
- `n_atoms` (int): 原子数

**戻り値**:
- `positions` (np.ndarray): Wyckoff位置（shape: [n_atoms, 3]）

**使用例**:
```python
from crystal.utils import generate_wyckoff_positions

# 空間群P1での4原子のWyckoff位置
positions = generate_wyckoff_positions(space_group=1, n_atoms=4)
```

**注意事項**:
- spglibパッケージが必要です
- 対称性を考慮した位置が生成されます

---

## egnnモジュール

E(3)同変グラフニューラルネットワークの実装。

### egnn_new

#### クラス: `EGNN`

E(3)同変グラフニューラルネットワーク。

**初期化パラメータ**:
- `in_node_nf` (int): 入力ノード特徴数
- `hidden_nf` (int): 隠れ特徴数
- `out_node_nf` (int): 出力ノード特徴数
- `in_edge_nf` (int): エッジ特徴数（デフォルト: 0）
- `n_layers` (int): レイヤー数
- `attention` (bool): アテンションを使用するかどうか
- `normalize` (bool): 座標を正規化するかどうか
- `tanh` (bool): tanh活性化を使用するかどうか

**メソッド**:

##### `forward(h, x, edge_index, node_mask=None, edge_mask=None)`

順伝播を実行します。

**パラメータ**:
- `h` (torch.Tensor): ノード特徴（shape: [batch_size, n_nodes, in_node_nf]）
- `x` (torch.Tensor): ノード座標（shape: [batch_size, n_nodes, 3]）
- `edge_index` (torch.Tensor): エッジインデックス
- `node_mask` (torch.Tensor, optional): ノードマスク
- `edge_mask` (torch.Tensor, optional): エッジマスク

**戻り値**:
- `h` (torch.Tensor): 更新されたノード特徴
- `x` (torch.Tensor): 更新された座標

**使用例**:
```python
from egnn.egnn_new import EGNN

egnn = EGNN(
    in_node_nf=15,
    hidden_nf=128,
    out_node_nf=15,
    n_layers=4,
    attention=True
)

h_out, x_out = egnn(h, x, edge_index, node_mask, edge_mask)
```

**注意事項**:
- E(3)同変性が厳密に保たれます
- 座標更新は相対位置に基づいて行われます

---

### egnn.models

#### クラス: `EGNN_dynamics_QM9`

QM9用の動力学EGNNモデル。

**初期化パラメータ**:
- `in_node_nf` (int): 入力ノード特徴数
- `context_node_nf` (int): コンテキスト特徴数
- `n_dims` (int): 空間次元数（通常は3）
- `device` (torch.device): デバイス
- `hidden_nf` (int): 隠れ特徴数
- `act_fn` (nn.Module): 活性化関数
- `n_layers` (int): レイヤー数
- `attention` (bool): アテンションを使用するかどうか
- `tanh` (bool): tanh活性化を使用するかどうか
- `mode` (str): モデルモード
- `norm_constant` (float): 正規化定数
- `inv_sublayers` (int): 不変サブレイヤー数
- `sin_embedding` (bool): sin埋め込みを使用するかどうか
- `normalization_factor` (float): 正規化係数
- `aggregation_method` (str): 集約方法

**メソッド**:

##### `forward(t, xh, node_mask, edge_mask, context)`

時間tでの動力学を計算します。

**パラメータ**:
- `t` (torch.Tensor): 時間ステップ
- `xh` (torch.Tensor): 座標と特徴
- `node_mask` (torch.Tensor): ノードマスク
- `edge_mask` (torch.Tensor): エッジマスク
- `context` (torch.Tensor, optional): コンテキスト

**戻り値**:
- `output` (torch.Tensor): 動力学の出力

**使用例**:
```python
from egnn.models import EGNN_dynamics_QM9
import torch.nn as nn

dynamics = EGNN_dynamics_QM9(
    in_node_nf=15,
    context_node_nf=3,
    n_dims=3,
    device=device,
    hidden_nf=256,
    act_fn=nn.SiLU(),
    n_layers=9,
    attention=True,
    tanh=True,
    mode='egnn_dynamics',
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=False,
    normalization_factor=1.0,
    aggregation_method='sum'
)
```

---

## equivariant_diffusionモジュール

同変拡散モデルの実装。

### en_diffusion

#### クラス: `EnVariationalDiffusion`

E(n)同変変分拡散モデル。

**初期化パラメータ**:
- `dynamics` (nn.Module): 動力学モデル
- `in_node_nf` (int): 入力ノード特徴数
- `n_dims` (int): 空間次元数
- `timesteps` (int): 拡散ステップ数（デフォルト: 1000）
- `noise_schedule` (str): ノイズスケジュール（'polynomial'または'cosine'）
- `noise_precision` (float): ノイズ精度
- `loss_type` (str): 損失関数タイプ
- `norm_values` (list): 正規化値
- `include_charges` (bool): 電荷を含めるかどうか

**メソッド**:

##### `forward(x, h, node_mask, edge_mask, context)`

訓練時の順伝播を実行します。

**パラメータ**:
- `x` (torch.Tensor): 座標
- `h` (torch.Tensor): ノード特徴
- `node_mask` (torch.Tensor): ノードマスク
- `edge_mask` (torch.Tensor): エッジマスク
- `context` (torch.Tensor, optional): コンテキスト

**戻り値**:
- `loss` (torch.Tensor): 損失値
- `log_dict` (dict): ログ情報

##### `sample(n_samples, n_nodes, node_mask, edge_mask, context=None, fix_noise=False)`

新しい分子をサンプリングします。

**パラメータ**:
- `n_samples` (int): サンプル数
- `n_nodes` (int): ノード数
- `node_mask` (torch.Tensor): ノードマスク
- `edge_mask` (torch.Tensor): エッジマスク
- `context` (torch.Tensor, optional): コンテキスト
- `fix_noise` (bool): ノイズを固定するかどうか

**戻り値**:
- `x` (torch.Tensor): サンプリングされた座標
- `h` (torch.Tensor): サンプリングされた特徴

##### `sample_chain(n_samples, n_nodes, node_mask, edge_mask, context=None, keep_frames=None)`

サンプリングチェーン全体を生成します。

**パラメータ**:
- `n_samples` (int): サンプル数
- `n_nodes` (int): ノード数
- `node_mask` (torch.Tensor): ノードマスク
- `edge_mask` (torch.Tensor): エッジマスク
- `context` (torch.Tensor, optional): コンテキスト
- `keep_frames` (int, optional): 保存するフレーム数

**戻り値**:
- `chain` (torch.Tensor): サンプリングチェーン

**使用例**:
```python
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion

# モデルの初期化
model = EnVariationalDiffusion(
    dynamics=dynamics_net,
    in_node_nf=15,
    n_dims=3,
    timesteps=1000,
    noise_schedule='polynomial',
    noise_precision=1e-5,
    loss_type='l2',
    norm_values=[1.0, 1.0, 1.0],
    include_charges=True
)

# サンプリング
x, h = model.sample(
    n_samples=10,
    n_nodes=19,
    node_mask=node_mask,
    edge_mask=edge_mask,
    context=context
)

print(f"Sampled positions: {x.shape}")  # [10, 19, 3]
print(f"Sampled features: {h.shape}")    # [10, 19, n_features]
```

**注意事項**:
- 訓練時は`forward()`、サンプリング時は`sample()`を使用します
- コンテキストは条件付き生成に使用されます
- ノイズスケジュールは'polynomial'（推奨）または'cosine'を選択できます

---

### distributions

#### 関数: `polynomial_schedule(timesteps, s=1e-4, power=3.0)`

多項式ノイズスケジュールを生成します。

**パラメータ**:
- `timesteps` (int): タイムステップ数
- `s` (float): 安定性パラメータ
- `power` (float): 多項式の次数

**戻り値**:
- `alphas2` (np.ndarray): αスケジュール

**使用例**:
```python
from equivariant_diffusion.en_diffusion import polynomial_schedule

alphas = polynomial_schedule(timesteps=1000, s=1e-4, power=3.0)
```

---

#### 関数: `cosine_beta_schedule(timesteps, s=0.008, raise_to_power=1)`

コサインベータスケジュールを生成します。

**パラメータ**:
- `timesteps` (int): タイムステップ数
- `s` (float): オフセットパラメータ
- `raise_to_power` (float): べき乗パラメータ

**戻り値**:
- `alphas_cumprod` (np.ndarray): 累積α

**使用例**:
```python
from equivariant_diffusion.en_diffusion import cosine_beta_schedule

alphas = cosine_beta_schedule(timesteps=1000, s=0.008)
```

---

## 共通パターンと使用例

### 基本的な分子生成ワークフロー

```python
import torch
from qm9.dataset import retrieve_dataloaders
from qm9.models import get_model, get_optim
from qm9 import utils as qm9_utils

# 1. データセットの準備
dataloaders, charge_scale = retrieve_dataloaders(args)
dataset_info = qm9_utils.get_dataset_info(args.dataset, args.remove_h)

# 2. モデルの構築
model, nodes_dist, prop_dist = get_model(
    args, device, dataset_info, dataloaders['train']
)

# 3. オプティマイザの準備
optimizer = get_optim(args, model)

# 4. 訓練ループ
for epoch in range(args.n_epochs):
    for batch in dataloaders['train']:
        # バッチデータの取得
        x = batch['positions'].to(device)
        h = batch['one_hot'].to(device)
        node_mask = batch['atom_mask'].to(device)
        edge_mask = batch['edge_mask'].to(device)
        
        # 順伝播
        loss, log = model(x, h, node_mask, edge_mask)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. サンプリング
with torch.no_grad():
    n_nodes = nodes_dist.sample(n_samples=10)
    node_mask = torch.ones(10, n_nodes.max(), 1).to(device)
    edge_mask = (1 - torch.eye(n_nodes.max())).unsqueeze(0)
    edge_mask = edge_mask.repeat(10, 1, 1).view(-1, 1).to(device)
    
    x, h = model.sample(10, n_nodes.max(), node_mask, edge_mask)
```

---

### 条件付き生成ワークフロー

```python
from qm9.utils import compute_mean_mad, prepare_context

# 1. 性質の正規化パラメータを計算
property_norms = compute_mean_mad(
    dataloaders, 
    ['alpha', 'homo', 'lumo'], 
    args.dataset
)

# 2. 訓練時のコンテキスト準備
for batch in dataloaders['train']:
    context, context_dim = prepare_context(
        conditioning=['alpha', 'homo', 'lumo'],
        minibatch=batch,
        property_norms=property_norms
    )
    
    # モデルの順伝播（コンテキスト付き）
    loss, log = model(
        batch['positions'].to(device),
        batch['one_hot'].to(device),
        batch['atom_mask'].to(device),
        batch['edge_mask'].to(device),
        context=context
    )

# 3. サンプリング時のターゲット性質指定
target_alpha = 75.0
target_homo = -0.25
target_lumo = 0.1

# 正規化
target_context = torch.zeros(1, n_nodes, context_dim).to(device)
target_context[0, :, 0] = (target_alpha - property_norms['alpha']['mean']) / property_norms['alpha']['mad']
target_context[0, :, 1] = (target_homo - property_norms['homo']['mean']) / property_norms['homo']['mad']
target_context[0, :, 2] = (target_lumo - property_norms['lumo']['mean']) / property_norms['lumo']['mad']

# サンプリング
x, h = model.sample(1, n_nodes, node_mask, edge_mask, context=target_context)
```

---

### ASEデータベースからのカスタムデータセット

```python
from qm9.dataset import load_ase_database
from torch.utils.data import DataLoader

# 1. ASEデータベースの読み込み
datasets, num_species, charge_scale = load_ase_database(
    db_path='custom_molecules.db',
    split_ratios=(0.8, 0.1, 0.1),
    seed=42,
    include_charges=False,
    remove_h=False,
    remove_duplicates=True,
    duplicate_tolerance=1e-6
)

# 2. DataLoaderの作成
train_loader = DataLoader(
    datasets['train'],
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 3. カスタム性質の利用
for batch in train_loader:
    # ASEデータベースから読み込まれた性質
    if 'molecular_weight' in batch:
        mol_weight = batch['molecular_weight']
    if 'functional_groups_encoding' in batch:
        func_groups = batch['functional_groups_encoding']
```

---

### 結晶生成ワークフロー

```python
from crystal.models import CrystalDynamics
from crystal.conditioning import SpaceGroupEmbedding, DensityConditioning

# 1. 空間群埋め込みの準備
sg_embedding = SpaceGroupEmbedding(
    num_space_groups=230,
    embedding_dim=64
)

# 2. 密度条件付けの準備
density_cond = DensityConditioning(hidden_dim=64)

# 3. 結晶動力学モデルの構築
crystal_model = CrystalDynamics(
    dynamics=base_egnn,
    in_node_nf=11,
    context_node_nf=128,
    lattice_dim=6
)

# 4. 条件付き結晶生成
space_group_id = torch.tensor([14])  # P21/c (monoclinic)
target_density = torch.tensor([1.5])   # g/cm³

sg_embed = sg_embedding(space_group_id)
density_embed = density_cond(target_density)

# コンテキストの結合
context = torch.cat([sg_embed, density_embed], dim=-1)

# サンプリング
x, h = model.sample(1, n_nodes, node_mask, edge_mask, context=context)
```

---

## パフォーマンス最適化のヒント

### メモリ使用量の削減

```python
# 1. 勾配チェックポインティング
import torch.utils.checkpoint as checkpoint

# 2. 混合精度訓練
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        loss, log = model(x, h, node_mask, edge_mask)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 3. バッチサイズの調整
# GPU: 32-64, CPU: 8-16を推奨
```

---

### 訓練の高速化

```python
# 1. DataLoaderのワーカー数を増やす
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # CPUコア数に応じて調整
    pin_memory=True  # GPUの場合
)

# 2. 学習率スケジューラの使用
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

# 3. Early Stopping
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(args.n_epochs):
    # 訓練
    train_loss = train_epoch(model, train_loader, optimizer)
    
    # 検証
    val_loss = validate(model, valid_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    scheduler.step()
```

---

## エラーハンドリング

### 一般的なエラーと対処法

```python
# 1. CUDA out of memory
try:
    loss, log = model(x, h, node_mask, edge_mask)
except RuntimeError as e:
    if 'out of memory' in str(e):
        # バッチサイズを減らす
        print("OOM Error: バッチサイズを減らしてください")
        torch.cuda.empty_cache()

# 2. NaN損失
if torch.isnan(loss):
    print("Warning: NaN loss detected")
    # 学習率を下げるか、勾配クリッピングを使用
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 不安定なサンプリング
from qm9.analyze import check_stability

stability = check_stability(x, atom_types, dataset_info)
if not stability['mol_stable']:
    print("Warning: 不安定な分子が生成されました")
    # 拡散ステップ数を増やすか、ノイズスケジュールを調整
```

---

## デバッグとロギング

```python
import wandb

# 1. Weights & Biases統合
wandb.init(project='e3_diffusion', config=args)

for epoch in range(args.n_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, valid_loader)
    
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    })

# 2. 詳細なデバッグ情報
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Batch shapes - x: {x.shape}, h: {h.shape}")
logger.info(f"Training loss: {loss.item():.4f}")
```

---

## まとめ

このAPIリファレンスは、E3 Diffusion for Moleculesの主要なクラスと関数を網羅しています。

### 重要なポイント

1. **E(3)同変性**: 全てのモデルは回転・並進・反転に対して同変
2. **条件付き生成**: `prepare_context()`を使用して柔軟な条件付けが可能
3. **カスタムデータセット**: ASEデータベースフォーマットで簡単に独自データを使用可能
4. **結晶生成**: 230個全ての空間群をサポート
5. **品質評価**: `check_stability()`で生成分子の化学的妥当性を評価

### さらなる情報

- **チュートリアル**: `tutorials/`ディレクトリの日本語チュートリアルを参照
- **理論**: `doc/complete_theory_ja.md`で数学的背景を学習
- **ユーザーガイド**: `doc/complete_user_guide_ja.md`で実践的な使用方法を確認
- **ユースケース**: `doc/use_cases_ja.md`で実際の応用例を参照

---

**このAPIリファレンスは継続的に更新されます。**

**最終更新**: 2025年10月25日  
**バージョン**: 1.0  
**ライセンス**: MITライセンス
