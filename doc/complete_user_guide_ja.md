# E3同変拡散モデル 完全詳細ユーザーガイド

**Version 1.0 | 最終更新: 2025年10月24日**

---

## 目次

1. [はじめに](#はじめに)
2. [システム概要](#システム概要)
3. [インストールと環境構築](#インストールと環境構築)
4. [基本的な分子生成](#基本的な分子生成)
5. [条件付き分子生成](#条件付き分子生成)
6. [結晶構造生成](#結晶構造生成)
7. [ASEデータベース統合](#aseデータベース統合)
8. [分子記述子による条件付け](#分子記述子による条件付け)
9. [評価と解析](#評価と解析)
10. [高度な使用方法](#高度な使用方法)
11. [コマンドライン完全リファレンス](#コマンドライン完全リファレンス)
12. [トラブルシューティング](#トラブルシューティング)
13. [性能最適化](#性能最適化)

---

## はじめに

### このガイドについて

本ガイドは、E3同変拡散モデル（E3 Equivariant Diffusion Model）を用いた3D分子および結晶構造生成システムの完全な使用方法を解説します。本システムは以下の特徴を持ちます：

- **E(3)同変性の保証**: 回転、平行移動、反射に対する厳密な不変性
- **フォールバックなし**: 全ての操作が理論的に健全であり、近似的な回避策を使用しません
- **物理的妥当性**: 生成される構造は化学的・物理的制約を満たします
- **拡張可能性**: QM9、GEOM-Drugs、ASEデータベースなど多様なデータソースに対応

### 対象読者

- 分子生成AIの研究者
- 計算化学・材料科学の研究者
- 機械学習エンジニア
- PyTorchの基礎知識を持つ方

### 必要な前提知識

- Pythonプログラミング（中級レベル）
- PyTorchの基礎
- 分子化学の基本概念
- コマンドラインの使用経験

---

## システム概要

### アーキテクチャ

本システムは以下のコアコンポーネントで構成されています：

```
e3_diffusion_for_molecules/
├── equivariant_diffusion/    # コア拡散モデル実装
│   ├── en_diffusion.py       # 主要な拡散アルゴリズム
│   ├── distributions.py      # 確率分布
│   ├── crystal_distributions.py  # 結晶用分布
│   └── utils.py              # ユーティリティ関数
├── egnn/                     # E(n)同変グラフニューラルネットワーク
│   ├── egnn.py               # EGNN実装
│   ├── egnn_new.py           # 改良版EGNN
│   └── models.py             # モデル定義
├── qm9/                      # QM9データセットと分子処理
│   ├── dataset.py            # データセット読み込み
│   ├── models.py             # 分子生成モデル
│   ├── analyze.py            # 分子解析
│   ├── sampling.py           # サンプリング
│   └── property_prediction/  # 性質予測器
└── crystal/                  # 結晶生成拡張
    ├── models/               # 結晶用モデル
    ├── conditioning/         # 条件付けモジュール
    ├── data/                 # 結晶データ処理
    ├── evaluation/           # 評価指標
    └── utils/                # 結晶ユーティリティ
```

### 主要機能

#### 1. 分子生成機能

- **無条件生成**: 学習分布からの自由なサンプリング
- **条件付き生成**: 特定の物性値を持つ分子の生成
- **複数性質条件付け**: 複数の物性を同時に制御
- **厳密な条件付け**: 目標値との正確な一致を保証

#### 2. 結晶生成機能

- **ホモ結晶生成**: 単一分子種からなる結晶構造
- **空間群条件付け**: 230種の空間群から選択可能
- **密度制御**: 目標密度での結晶生成
- **周期境界条件**: 物理的に正確な結晶シミュレーション
- **CIFファイル出力**: 標準形式での構造保存

#### 3. データセット対応

- **QM9**: 約134,000の小分子（最大9個の重原子）
- **GEOM-Drugs**: 大規模な医薬品様分子データセット
- **ASEデータベース**: カスタムデータセットの統合

#### 4. 評価・解析機能

- **安定性解析**: 分子の化学的安定性評価
- **RDKit検証**: 有効な分子構造の確認
- **一意性・新規性**: 生成された分子の多様性評価
- **結晶メトリクス**: 結晶構造の品質評価

---

## インストールと環境構築

### 基本インストール

#### 1. リポジトリのクローン

```bash
git clone https://github.com/nobkt/e3_diffusion_for_molecules.git
cd e3_diffusion_for_molecules
```

#### 2. 依存パッケージのインストール

**基本的な依存関係**:
```bash
pip install -r requirements.txt
```

`requirements.txt`の主要な内容：
- `torch>=1.9.0`
- `torch-geometric>=2.0.0`
- `torch-scatter`
- `torch-sparse`
- `torch-cluster`
- `numpy>=1.19.0`
- `matplotlib>=3.3.0`
- `tqdm`

#### 3. オプションの依存関係

**RDKit（分子可視化と検証用）**:
```bash
conda install -c conda-forge rdkit
```

**ASEとOpenBabel（結晶処理用）**:
```bash
pip install ase spglib openbabel-wheel
```

**Jupyter Notebook（チュートリアル用）**:
```bash
pip install jupyter ipykernel
```

### 環境の確認

インストールが正しく完了したことを確認：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric: OK')"
python -c "import rdkit; print('RDKit: OK')"  # オプション
```

### GPUサポートの設定

**CUDA対応PyTorchのインストール**:
```bash
# CUDA 11.3の場合
pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# PyTorch Geometricも対応バージョンをインストール
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

**GPU動作確認**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
```

### ディレクトリ構造の準備

必要なディレクトリを作成：

```bash
mkdir -p outputs          # モデルチェックポイント保存用
mkdir -p data/qm9        # QM9データセット用
mkdir -p data/geom       # GEOM-Drugsデータセット用
mkdir -p generated_samples  # 生成サンプル保存用
```

---

## 基本的な分子生成

### QM9データセットでの学習

#### 最小限の設定での学習

```bash
python main_qm9.py \
    --exp_name my_first_model \
    --n_epochs 100 \
    --batch_size 64 \
    --nf 128 \
    --n_layers 6 \
    --diffusion_steps 500
```

**パラメータの説明**:
- `--exp_name`: 実験名（出力ディレクトリ名）
- `--n_epochs`: 学習エポック数
- `--batch_size`: バッチサイズ
- `--nf`: モデルの隠れ層次元数
- `--n_layers`: EGNNレイヤー数
- `--diffusion_steps`: 拡散ステップ数

#### 推奨される本格的な学習設定

```bash
python main_qm9.py \
    --exp_name production_model \
    --n_epochs 3000 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 9 \
    --lr 1e-4 \
    --diffusion_steps 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_loss_type l2 \
    --normalize_factors 1,4,10 \
    --test_epochs 20 \
    --ema_decay 0.9999 \
    --n_stability_samples 1000 \
    --save_model True
```

**高度なパラメータの説明**:
- `--diffusion_noise_schedule`: ノイズスケジュール（`polynomial_2`, `cosine`など）
- `--diffusion_noise_precision`: 数値精度（推奨: 1e-5）
- `--normalize_factors`: 座標・特徴・電荷の正規化係数
- `--ema_decay`: Exponential Moving Average の減衰率
- `--n_stability_samples`: 評価時に生成するサンプル数

#### 学習の監視

学習中は以下の情報が出力されます：

```
Epoch 10/3000:
  Train Loss: 15.234
  Val Loss: 15.891
  Atom Stability: 94.3%
  Molecule Stability: 87.6%
  Time: 125.3s
```

**重要な指標**:
- **Train/Val Loss**: 拡散損失（低いほど良い）
- **Atom Stability**: 原子レベルの安定性（高いほど良い）
- **Molecule Stability**: 分子レベルの安定性（高いほど良い）

#### 学習の再開

学習を中断した場合、チェックポイントから再開可能：

```bash
python main_qm9.py \
    --exp_name production_model \
    --resume outputs/production_model \
    --start_epoch 1500
```

### 分子の生成

学習済みモデルから分子を生成：

```bash
python eval_sample.py \
    --model_path outputs/production_model \
    --n_samples 1000 \
    --batch_size 100
```

**出力**:
- `outputs/production_model/samples/`: 生成された分子構造
- XYZファイル形式で保存
- RDKitがあれば2D/3D画像も生成

#### カスタムサンプリング設定

```bash
python eval_sample.py \
    --model_path outputs/production_model \
    --n_samples 5000 \
    --batch_size 100 \
    --save_xyz True \
    --save_mol True \
    --visualize True
```

### 生成品質の評価

```bash
python eval_analyze.py \
    --model_path outputs/production_model \
    --n_samples 10000
```

**評価指標**:

1. **Validity（有効性）**: 化学的に有効な分子の割合
2. **Uniqueness（一意性）**: 重複しない分子の割合
3. **Novelty（新規性）**: 学習データに無い分子の割合
4. **Atom Stability**: 原子の安定性
5. **Molecule Stability**: 分子全体の安定性

**出力例**:
```
Evaluation Results:
==================
Validity: 96.3%
Uniqueness: 99.1%
Novelty: 89.7%
Atom Stability: 97.8%
Molecule Stability: 93.2%
```

---

## 条件付き分子生成

### 単一性質での条件付け

特定の物性値を持つ分子を生成します。

#### 学習

```bash
python main_qm9.py \
    --exp_name cond_alpha \
    --model egnn_dynamics \
    --lr 1e-4 \
    --nf 192 \
    --n_layers 9 \
    --save_model True \
    --diffusion_steps 1000 \
    --n_epochs 3000 \
    --n_stability_samples 500 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --normalize_factors 1,8,1 \
    --conditioning alpha \
    --dataset qm9_second_half
```

**条件付け可能な性質**:
- `alpha`: 分極率（polarizability）
- `gap`: HOMO-LUMOギャップ
- `homo`: HOMOエネルギー
- `lumo`: LUMOエネルギー
- `mu`: 双極子モーメント
- `Cv`: 熱容量

#### 性質予測器の学習

条件付き生成の評価のため、性質予測器を学習：

```bash
cd qm9/property_prediction
python main_qm9_prop.py \
    --num_workers 2 \
    --lr 5e-4 \
    --property alpha \
    --exp_name class_alpha \
    --model_name egnn \
    --n_epochs 500
```

#### 条件付き生成の実行

特定の性質値で分子を生成：

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/cond_alpha \
    --property alpha \
    --n_sweeps 10 \
    --task qualitative
```

**n_sweeps**: 性質値の範囲を分割するステップ数

#### 定量的評価

生成された分子が目標性質を満たしているか評価：

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/cond_alpha \
    --classifiers_path qm9/property_prediction/outputs/class_alpha \
    --property alpha \
    --iterations 100 \
    --batch_size 100 \
    --task edm
```

### 複数性質での条件付け

複数の性質を同時に制御して分子を生成：

```bash
python main_qm9.py \
    --exp_name multi_cond \
    --conditioning alpha gap homo \
    --nf 256 \
    --n_layers 9 \
    --n_epochs 3000 \
    --batch_size 64
```

### 厳密な条件付け生成

目標値との厳密な一致を保証する生成方法：

```python
# exact_conditioning_demo.py を使用
python exact_conditioning_demo.py \
    --model_path outputs/cond_alpha \
    --property alpha \
    --target_value 60.5 \
    --tolerance 0.1 \
    --n_samples 100
```

**パラメータ**:
- `--target_value`: 目標となる性質値
- `--tolerance`: 許容誤差（絶対値）
- `--n_samples`: 生成するサンプル数

---

## 結晶構造生成

### ホモ結晶の学習

単一分子種からなる結晶構造を生成するモデルを学習：

```bash
python main_crystal.py \
    --exp_name crystal_model \
    --batch_size 32 \
    --nf 128 \
    --n_layers 6 \
    --n_epochs 1000 \
    --lr 1e-4 \
    --crystal_type homocrystal \
    --space_groups all \
    --include_pbc True
```

**結晶特有のパラメータ**:
- `--crystal_type`: 結晶タイプ（`homocrystal`、`heterocrystal`など）
- `--space_groups`: 学習する空間群（`all` または特定の番号）
- `--include_pbc`: 周期境界条件の使用

### 条件付き結晶生成

#### 空間群条件付け

特定の空間群で結晶を生成：

```bash
python main_crystal_with_properties.py \
    --exp_name crystal_sg_cond \
    --conditioning space_group \
    --space_groups 1,2,14,19,61,62 \
    --n_epochs 1000
```

#### 密度条件付け

目標密度で結晶を生成：

```bash
python main_crystal_with_properties.py \
    --exp_name crystal_density_cond \
    --conditioning density \
    --density_range 0.8,2.5 \
    --n_epochs 1000
```

#### 複合条件付け

空間群と密度を同時に制御：

```bash
python main_crystal_with_properties.py \
    --exp_name crystal_multi_cond \
    --conditioning space_group density \
    --space_groups 14,19,61 \
    --density_range 1.0,2.0 \
    --n_epochs 1000
```

### 結晶の生成とエクスポート

```bash
python generate_crystal_with_all_conditions.py \
    --model_path outputs/crystal_multi_cond \
    --space_group 14 \
    --density 1.5 \
    --n_samples 10 \
    --output_dir generated_crystals
```

**出力**:
- CIFファイル: 標準的な結晶構造フォーマット
- XYZファイル: 座標情報
- PNG画像: 構造の可視化（オプション）

### 結晶品質の評価

```python
from crystal.evaluation import CrystalMetrics, StructureValidator

# 結晶メトリクスの計算
metrics = CrystalMetrics()
results = metrics.evaluate_crystal(crystal_structure)

print(f"Space Group Match: {results['space_group_accuracy']}")
print(f"Density Error: {results['density_error']}")
print(f"Symmetry Score: {results['symmetry_score']}")

# 構造検証
validator = StructureValidator()
is_valid = validator.validate(crystal_structure)
print(f"Structure Valid: {is_valid}")
```

---

## ASEデータベース統合

ASE（Atomic Simulation Environment）データベースを使用してカスタムデータセットで学習可能です。

### ASEデータベースの作成

```python
from ase import Atoms
from ase.db import connect

# データベース接続
db = connect('my_molecules.db')

# 分子の追加
atoms = Atoms(
    'C6H6',  # ベンゼン
    positions=[
        [0.0, 1.4, 0.0],
        [1.2, 0.7, 0.0],
        # ... 他の原子座標
    ]
)

# 性質データの追加
db.write(
    atoms,
    data={
        'molecular_weight': 78.11,
        'energy': -234.567,
        'dipole_moment': 0.0
    }
)
```

### ASEデータベースでの学習

```bash
python main_qm9.py \
    --exp_name ase_model \
    --dataset ase_db \
    --ase_db_path my_molecules.db \
    --batch_size 64 \
    --nf 128 \
    --n_layers 6 \
    --n_epochs 1000
```

### 重複除去機能

ASEデータベースは自動的に重複分子を検出・除去します：

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path my_molecules.db \
    --remove_duplicates True \
    --duplicate_tolerance 1e-6
```

**パラメータ**:
- `--remove_duplicates`: 重複除去の有効化
- `--duplicate_tolerance`: 重複判定の閾値（Å単位）

---

## 分子記述子による条件付け

ASEデータベースと併用して、分子記述子による高度な条件付けが可能です。

### 利用可能な分子記述子

1. **molecular_weight**: 分子量
2. **pi_conjugation_ratio**: π共役系の比率
3. **atom_types_encoding**: 原子タイプのエンコーディング
4. **functional_groups_encoding**: 官能基のエンコーディング

### 分子記述子の計算と保存

```python
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import Descriptors

db = connect('molecules_with_descriptors.db')

for mol in molecule_list:
    # 分子記述子の計算
    mol_weight = Descriptors.MolWt(mol)
    pi_ratio = calculate_pi_conjugation_ratio(mol)
    atom_encoding = encode_atom_types(mol)
    func_groups = encode_functional_groups(mol)
    
    # データベースに保存
    db.write(
        mol_to_atoms(mol),
        data={
            'molecular_weight': mol_weight,
            'pi_conjugation_ratio': pi_ratio,
            'atom_types_encoding': atom_encoding,
            'functional_groups_encoding': func_groups
        }
    )
```

### 分子記述子での条件付け学習

```bash
python main_qm9.py \
    --exp_name descriptor_cond \
    --dataset ase_db \
    --ase_db_path molecules_with_descriptors.db \
    --conditioning molecular_weight pi_conjugation_ratio \
    --nf 192 \
    --n_layers 9 \
    --n_epochs 3000 \
    --batch_size 64
```

### 複数記述子での生成

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/descriptor_cond \
    --property molecular_weight \
    --target_values 100,150,200,250,300 \
    --n_samples 100
```

---

## 評価と解析

### 包括的な評価

生成された分子の全ての品質指標を評価：

```bash
python eval_analyze.py \
    --model_path outputs/production_model \
    --n_samples 10000 \
    --save_results True \
    --output_file evaluation_results.json
```

**出力される指標**:

1. **基本指標**
   - Validity: 有効な分子の割合
   - Uniqueness: 一意な分子の割合
   - Novelty: 新規分子の割合

2. **安定性指標**
   - Atom Stability: 原子レベルの安定性
   - Molecule Stability: 分子レベルの安定性

3. **構造指標**
   - Bond Length Distribution: 結合長分布
   - Bond Angle Distribution: 結合角分布
   - 原子価の正しさ

4. **化学的指標**
   - Aromaticity: 芳香族性
   - Ring Statistics: 環構造統計
   - Functional Group Distribution: 官能基分布

### RDKitによる検証

RDKitを使用した詳細な分子検証：

```python
from qm9.rdkit_functions import mol_from_xyz
from qm9.analyze import check_stability

# XYZファイルから分子を読み込み
mol = mol_from_xyz('generated_molecule.xyz')

# 安定性チェック
atom_stable, mol_stable, validity = check_stability(
    positions, atom_types, charges, dataset_info
)

print(f"Atom Stable: {atom_stable}")
print(f"Molecule Stable: {mol_stable}")
print(f"Valid: {validity}")
```

### 結晶構造の評価

```python
from crystal.evaluation import (
    CrystalMetrics,
    StructureValidator,
    SymmetryAnalyzer
)

# メトリクス計算
metrics = CrystalMetrics()
results = metrics.evaluate_batch(crystal_structures)

# 対称性解析
symmetry_analyzer = SymmetryAnalyzer()
space_group = symmetry_analyzer.detect_space_group(crystal)
print(f"Detected Space Group: {space_group}")

# 構造検証
validator = StructureValidator()
validation_results = validator.validate_comprehensive(crystal)
print(validation_results)
```

### カスタム評価スクリプト

独自の評価指標を実装：

```python
from qm9.dataset import retrieve_dataloaders
from qm9.analyze import analyze_stability_for_molecules
import torch

# モデルとデータの読み込み
model = load_model('outputs/production_model')
dataloaders = retrieve_dataloaders(args)

# サンプリング
samples = []
for i in range(n_samples):
    x, h, node_mask = model.sample(n_nodes, device)
    samples.append((x, h, node_mask))

# カスタム評価
def custom_evaluation(samples):
    # 独自の評価ロジック
    pass

results = custom_evaluation(samples)
```

---

## 高度な使用方法

### EMAモデルの使用

Exponential Moving Average モデルは学習の安定性を向上させます：

```bash
python main_qm9.py \
    --exp_name ema_model \
    --ema_decay 0.9999 \
    --save_model True \
    --test_epochs 10
```

生成時にEMAモデルを使用：

```bash
python eval_sample.py \
    --model_path outputs/ema_model \
    --use_ema True \
    --n_samples 1000
```

### カスタムノイズスケジュール

異なるノイズスケジュールを試す：

```bash
# Polynomial schedule
python main_qm9.py \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_steps 1000

# Cosine schedule
python main_qm9.py \
    --diffusion_noise_schedule cosine \
    --diffusion_steps 1000
```

### 学習データの分割

データセットを分割して学習：

```bash
# 後半のデータのみで学習
python main_qm9.py \
    --dataset qm9_second_half \
    --exp_name qm9_half_model

# カスタム分割
python main_qm9.py \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15
```

### バッチ生成と並列処理

大量のサンプルを効率的に生成：

```python
import torch
from torch.multiprocessing import Pool

def generate_batch(model, batch_size, n_nodes, device):
    with torch.no_grad():
        x, h = model.sample(batch_size, n_nodes, device)
    return x, h

# 並列生成
n_processes = 4
n_samples_per_process = 1000

with Pool(n_processes) as pool:
    results = pool.starmap(
        generate_batch,
        [(model, n_samples_per_process, n_nodes, device)] * n_processes
    )
```

### CSV出力とデータエクスポート

生成結果をCSV形式でエクスポート：

```bash
python demo_csv_export.py \
    --model_path outputs/production_model \
    --n_samples 1000 \
    --output_csv generated_molecules.csv
```

CSV出力の例：

```python
from crystal.utils import CIFWriter
import pandas as pd

# 分子情報の収集
molecule_data = []
for i, (x, h) in enumerate(generated_molecules):
    mol_info = {
        'id': i,
        'n_atoms': len(x),
        'molecular_weight': calculate_weight(h),
        'coordinates': x.tolist(),
        'atom_types': h.argmax(dim=-1).tolist()
    }
    molecule_data.append(mol_info)

# DataFrameに変換して保存
df = pd.DataFrame(molecule_data)
df.to_csv('molecules.csv', index=False)
```

---

## コマンドライン完全リファレンス

### main_qm9.py

QM9データセットでの分子生成モデル学習。

#### 基本パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--exp_name` | str | 必須 | 実験名 |
| `--n_epochs` | int | 3000 | 学習エポック数 |
| `--batch_size` | int | 64 | バッチサイズ |
| `--lr` | float | 1e-4 | 学習率 |

#### モデルパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--model` | str | egnn_dynamics | モデルタイプ |
| `--nf` | int | 256 | 隠れ層の次元数 |
| `--n_layers` | int | 9 | EGNNレイヤー数 |
| `--attention` | bool | True | アテンション機構の使用 |
| `--tanh` | bool | True | tanh活性化関数の使用 |

#### 拡散パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--diffusion_steps` | int | 1000 | 拡散ステップ数 |
| `--diffusion_noise_schedule` | str | polynomial_2 | ノイズスケジュール |
| `--diffusion_noise_precision` | float | 1e-5 | 数値精度 |
| `--diffusion_loss_type` | str | l2 | 損失関数タイプ |

#### データセットパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--dataset` | str | qm9 | データセット名 |
| `--remove_h` | bool | False | 水素原子の除去 |
| `--include_charges` | bool | True | 電荷情報の使用 |

#### ASEデータベースパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--ase_db_path` | str | None | ASEデータベースパス |
| `--remove_duplicates` | bool | True | 重複除去 |
| `--duplicate_tolerance` | float | 1e-6 | 重複判定閾値 |

#### 条件付けパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--conditioning` | list | None | 条件付ける性質 |
| `--property` | str | None | 単一性質の指定 |

#### 学習制御パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--test_epochs` | int | 20 | 評価の実行間隔 |
| `--save_model` | bool | True | モデルの保存 |
| `--ema_decay` | float | 0.9999 | EMA減衰率 |
| `--n_stability_samples` | int | 1000 | 安定性評価サンプル数 |
| `--normalize_factors` | list | [1,4,10] | 正規化係数 |

### main_crystal.py

結晶構造生成モデルの学習。

#### 結晶固有パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--crystal_type` | str | homocrystal | 結晶タイプ |
| `--space_groups` | str | all | 空間群の指定 |
| `--include_pbc` | bool | True | 周期境界条件 |
| `--lattice_dim` | int | 64 | 格子パラメータ次元 |

### eval_sample.py

学習済みモデルからのサンプリング。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--model_path` | str | 必須 | モデルパス |
| `--n_samples` | int | 100 | サンプル数 |
| `--batch_size` | int | 100 | バッチサイズ |
| `--save_xyz` | bool | True | XYZ保存 |
| `--visualize` | bool | False | 可視化 |

### eval_analyze.py

生成品質の評価。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--model_path` | str | 必須 | モデルパス |
| `--n_samples` | int | 10000 | 評価サンプル数 |
| `--save_results` | bool | True | 結果保存 |

### eval_conditional_qm9.py

条件付き生成と評価。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--generators_path` | str | 必須 | 生成器パス |
| `--classifiers_path` | str | None | 分類器パス |
| `--property` | str | 必須 | 性質名 |
| `--task` | str | qualitative | タスクタイプ |
| `--n_sweeps` | int | 10 | 性質値の分割数 |

---

## トラブルシューティング

### よくある問題と解決方法

#### 1. CUDA Out of Memory

**症状**: 学習中やサンプリング中にCUDAメモリ不足エラー

**解決方法**:
```bash
# バッチサイズを減らす
python main_qm9.py --batch_size 16  # デフォルト: 64

# モデルサイズを減らす
python main_qm9.py --nf 64 --n_layers 4  # デフォルト: 256, 9

# 評価サンプル数を減らす
python main_qm9.py --n_stability_samples 100  # デフォルト: 1000
```

#### 2. RDKit Not Found

**症状**: `ImportError: No module named 'rdkit'`

**解決方法**:
```bash
# Condaでインストール（推奨）
conda install -c conda-forge rdkit

# または、RDKitなしで続行
# 多くの機能はRDKitなしでも動作します
```

#### 3. 学習が遅い

**症状**: エポックごとの処理時間が長すぎる

**解決方法**:
```bash
# 評価頻度を下げる
python main_qm9.py --test_epochs 50  # デフォルト: 20

# 安定性評価サンプルを減らす
python main_qm9.py --n_stability_samples 100

# データローダーのワーカー数を増やす
python main_qm9.py --num_workers 4
```

#### 4. NaN Loss

**症状**: 学習中に損失がNaNになる

**解決方法**:
```bash
# 学習率を下げる
python main_qm9.py --lr 5e-5  # デフォルト: 1e-4

# ノイズ精度を上げる
python main_qm9.py --diffusion_noise_precision 1e-6  # デフォルト: 1e-5

# 勾配クリッピングを有効化
python main_qm9.py --clip_grad 1.0
```

#### 5. ASEデータベース読み込みエラー

**症状**: ASEデータベースの読み込みに失敗

**解決方法**:
```python
# データベースの整合性をチェック
from ase.db import connect
db = connect('your_database.db')
print(f"Total molecules: {len(db)}")

# データベースを再構築
python build_geom_dataset.py --rebuild
```

#### 6. 結晶生成でのPBCエラー

**症状**: 周期境界条件の処理でエラー

**解決方法**:
```bash
# PBCを無効化してテスト
python main_crystal.py --include_pbc False

# カットオフ距離を調整
python main_crystal.py --cutoff_distance 5.0
```

### デバッグモード

詳細なデバッグ情報を出力：

```bash
# Verbose モード
python main_qm9.py --verbose True --debug True

# ログレベルの設定
python main_qm9.py --log_level DEBUG
```

### パフォーマンスプロファイリング

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    # 学習ループ
    for batch in dataloader:
        loss = model(batch)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 性能最適化

### GPU最適化

#### 混合精度学習

```bash
python main_qm9.py \
    --use_amp True \
    --amp_opt_level O1
```

**利点**:
- メモリ使用量の削減
- 学習速度の向上（約1.5-2倍）
- 最小限の精度損失

#### マルチGPU学習

```bash
# DataParallel
python main_qm9.py \
    --multi_gpu True \
    --gpu_ids 0,1,2,3

# DistributedDataParallel（推奨）
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main_qm9.py \
    --distributed True
```

### メモリ最適化

#### 勾配累積

バッチサイズを増やさずに大きな実効バッチサイズを実現：

```bash
python main_qm9.py \
    --batch_size 16 \
    --gradient_accumulation_steps 4
    # 実効バッチサイズ: 16 * 4 = 64
```

#### チェックポイント機能

メモリ消費を削減（学習速度とのトレードオフ）：

```bash
python main_qm9.py \
    --use_checkpoint True
```

### データローディング最適化

```bash
python main_qm9.py \
    --num_workers 8 \
    --prefetch_factor 2 \
    --pin_memory True
```

### サンプリング最適化

```python
# バッチサイズを調整
python eval_sample.py \
    --batch_size 200 \
    --n_samples 10000

# 並列サンプリング
python eval_sample.py \
    --num_workers 4 \
    --parallel True
```

### 結晶生成の最適化

```python
from crystal.evaluation import PerformanceOptimization

optimizer = PerformanceOptimization()

# 近傍リスト計算の最適化
optimizer.optimize_neighbor_list(cutoff=5.0, cell_size=2.5)

# 対称性演算のキャッシング
optimizer.enable_symmetry_cache()

# バッチ処理
crystals = optimizer.batch_generate(
    model, n_samples=1000, batch_size=50
)
```

---

## 付録

### A. 性質一覧（QM9）

| 性質 | 記号 | 単位 | 説明 |
|-----|------|------|------|
| Polarizability | alpha | Bohr³ | 分極率 |
| HOMO Energy | homo | Hartree | 最高被占軌道エネルギー |
| LUMO Energy | lumo | Hartree | 最低空軌道エネルギー |
| HOMO-LUMO Gap | gap | Hartree | エネルギーギャップ |
| Dipole Moment | mu | Debye | 双極子モーメント |
| Heat Capacity | Cv | cal/mol/K | 熱容量 |

### B. 空間群一覧（結晶）

結晶系と代表的な空間群：

| 結晶系 | 空間群番号 | ヘルマン・モーガン記号 |
|--------|-----------|---------------------|
| 三斜晶系 | 1, 2 | P1, P-1 |
| 単斜晶系 | 3-15 | P2, P21, C2, Pm, ... |
| 斜方晶系 | 16-74 | P222, P2221, ... |
| 正方晶系 | 75-142 | P4, P41, P42, ... |
| 三方晶系 | 143-167 | P3, P31, P32, ... |
| 六方晶系 | 168-194 | P6, P61, P62, ... |
| 立方晶系 | 195-230 | P23, Pm-3, F23, ... |

### C. データ形式仕様

#### XYZ形式

```
6
Benzene molecule
C  0.000  1.400  0.000
C  1.212  0.700  0.000
C  1.212 -0.700  0.000
C  0.000 -1.400  0.000
C -1.212 -0.700  0.000
C -1.212  0.700  0.000
```

#### CIF形式

```
data_crystal
_cell_length_a    5.0
_cell_length_b    5.0
_cell_length_c    5.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.0
```

### D. 参考文献

1. Hoogeboom, E., et al. "Equivariant Diffusion for Molecule Generation in 3D." ICML 2022.
2. Satorras, V. G., et al. "E(n) Equivariant Graph Neural Networks." ICML 2021.
3. Ho, J., et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.

### E. ライセンス

本ソフトウェアはMITライセンスの下で公開されています。

---

**このガイドの作成日**: 2025年10月24日  
**バージョン**: 1.0  
**著者**: E3 Diffusion Development Team

本ガイドは、システムの全機能を網羅的に解説し、フォールバック処理を一切使用しない厳密な実装を前提としています。不明な点や追加の質問がある場合は、GitHubリポジトリのIssuesセクションでお問い合わせください。
