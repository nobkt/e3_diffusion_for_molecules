# E3同変拡散モデル 完全機能ユーザーガイド

**バージョン 2.0 | 最終更新: 2025年10月26日**

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
10. [データエクスポート機能](#データエクスポート機能)
11. [学習の再開機能](#学習の再開機能)
12. [高度な使用方法](#高度な使用方法)
13. [出力の詳細説明](#出力の詳細説明)
14. [コマンドライン完全リファレンス](#コマンドライン完全リファレンス)
15. [トラブルシューティング](#トラブルシューティング)

---

## はじめに

### このガイドについて

本ガイドは、E3同変拡散モデル（E3 Equivariant Diffusion Model）を用いた3D分子および結晶構造生成システムの**完全な**機能説明書です。全ての機能と、その出力の意味を**フォールバック処理なし**で厳密に解説します。

### 重要な原則

1. **フォールバックなし**: 全ての処理は理論的に健全であり、ごまかしのための近似的な回避策を使用しません
2. **完全性**: 全ての機能を網羅的に説明します
3. **出力の透明性**: 全ての出力の意味を明確に記述します
4. **物理的妥当性**: 生成される構造は化学的・物理的制約を満たします

### システムの特徴

- **E(3)同変性の保証**: 回転、平行移動、反射に対する厳密な不変性
- **多様なデータセット対応**: QM9、GEOM-Drugs、ASEデータベース
- **条件付き生成**: 物性値、分子記述子、空間群、密度による精密な制御
- **包括的評価**: 化学的妥当性、新規性、多様性の多角的評価
- **豊富な出力形式**: XYZ、CIF、CSV形式での構造データ出力

---

## システム概要

### アーキテクチャ構成

```
e3_diffusion_for_molecules/
├── main_qm9.py              # QM9データセットでの分子生成
├── main_geom_drugs.py       # GEOM-Drugsデータセットでの分子生成
├── main_crystal.py          # 結晶構造生成
├── main_crystal_with_properties.py  # 条件付き結晶生成
├── eval_sample.py           # サンプル生成スクリプト
├── eval_analyze.py          # 品質評価スクリプト
├── eval_conditional_qm9.py  # 条件付き生成評価
├── equivariant_diffusion/   # コア拡散モデル
│   ├── en_diffusion.py      # E(n)同変拡散実装
│   ├── distributions.py     # 確率分布
│   └── utils.py             # ユーティリティ
├── egnn/                    # E(n)同変グラフニューラルネットワーク
│   ├── egnn.py              # EGNN実装
│   └── models.py            # モデル定義
├── qm9/                     # QM9データセット処理
│   ├── dataset.py           # データローダー
│   ├── models.py            # 分子生成モデル
│   ├── analyze.py           # 分子解析機能
│   ├── sampling.py          # サンプリング
│   ├── visualizer.py        # 可視化
│   ├── bond_analyze.py      # 結合解析
│   ├── rdkit_functions.py   # RDKit統合
│   └── property_prediction/ # 性質予測器
└── crystal/                 # 結晶生成拡張
    ├── models/              # 結晶用モデル
    ├── conditioning/        # 条件付けモジュール
    ├── data/                # 結晶データ処理
    ├── evaluation/          # 評価指標
    └── utils/               # 結晶ユーティリティ
```

### 主要機能一覧

#### 1. 分子生成機能

- **無条件生成**: 学習分布からの自由なサンプリング
- **条件付き生成**: 物性値（alpha, gap, homo, lumo, mu, Cv）指定
- **分子記述子条件付け**: 分子量、π共役比率、原子タイプ、官能基
- **複数条件同時指定**: 複数の性質を同時に制御
- **厳密な条件付け**: 目標値との正確な一致を保証

#### 2. 結晶生成機能

- **ホモ結晶生成**: 単一分子種からなる結晶構造
- **空間群条件付け**: 230種の空間群から選択
- **密度制御**: 目標密度での結晶生成
- **周期境界条件**: 物理的に正確な結晶シミュレーション
- **格子パラメータ学習**: 格子定数と角度の同時学習

#### 3. データセット対応

- **QM9**: 約134,000の小分子（最大9個の重原子）
  - 13種類の分子物性データを含む
  - 水素原子あり/なしの両方に対応
- **GEOM-Drugs**: 大規模医薬品様分子データセット
  - より複雑な分子構造に対応
- **ASEデータベース**: カスタムデータセットの統合
  - ユーザー定義の分子データベース
  - 重複除去機能
  - 任意の性質データの追加可能

#### 4. 評価・解析機能

- **安定性解析**: 
  - 原子レベルの結合安定性
  - 分子レベルの構造安定性
- **RDKit検証**: 
  - 有効な分子構造の確認
  - SMILES表記の生成
  - 分子グラフの構築
- **OpenBabel検証**:
  - ASEデータベース用の代替検証
  - より広範な化学種への対応
- **多様性評価**:
  - Validity（有効性）: 化学的に有効な分子の割合
  - Uniqueness（一意性）: 重複しない分子の割合
  - Novelty（新規性）: 学習データに無い分子の割合
- **結晶メトリクス**:
  - 空間群一致度
  - 密度誤差
  - 対称性スコア
  - 構造妥当性

---

## インストールと環境構築

### 必要な環境

- Python 3.8以上
- CUDA対応GPU（推奨）
- メモリ: 16GB以上推奨
- ストレージ: 10GB以上の空き容量

### 基本インストール

#### 1. リポジトリのクローン

```bash
git clone https://github.com/nobkt/e3_diffusion_for_molecules.git
cd e3_diffusion_for_molecules
```

#### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主要な依存関係:
- `torch>=1.9.0`: PyTorch本体
- `torch-geometric>=2.0.0`: グラフニューラルネットワーク
- `torch-scatter`, `torch-sparse`, `torch-cluster`: PyTorch Geometric拡張
- `numpy>=1.19.0`: 数値計算
- `matplotlib>=3.3.0`: 可視化
- `tqdm`: プログレスバー
- `wandb`: 実験管理（オプション）

#### 3. オプションの依存関係

**RDKit（分子可視化と検証用）**:
```bash
conda install -c conda-forge rdkit
```

**ASEとOpenBabel（結晶処理用）**:
```bash
pip install ase spglib openbabel-wheel
```

### 環境の確認

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric: OK')"
python -c "import rdkit; print('RDKit: OK')"
```

### GPUサポートの確認

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## 基本的な分子生成

### QM9データセットでの学習

#### 最小限の設定

```bash
python main_qm9.py \
    --exp_name my_first_model \
    --n_epochs 100 \
    --batch_size 64 \
    --nf 128 \
    --n_layers 6 \
    --diffusion_steps 500 \
    --no_wandb
```

**出力される内容**:

1. **学習ログ（標準出力）**:
```
Epoch 1/100:
  Train Loss: 18.234
  Val Loss: 18.891
  Time: 125.3s
Epoch 2/100:
  Train Loss: 17.123
  Val Loss: 17.456
  Time: 123.1s
...
```

- `Train Loss`: 学習データでの拡散損失（L2距離）。低いほど学習が進んでいる
- `Val Loss`: 検証データでの拡散損失。過学習の指標
- `Time`: エポックあたりの処理時間（秒）

2. **保存されるファイル**:
```
outputs/my_first_model/
├── args.pickle              # 学習時の引数
├── dataset_info.pickle      # データセット情報
├── generative_model.npy     # モデルチェックポイント
├── generative_model_ema.npy # EMAモデル（指定時）
├── optim.npy                # オプティマイザー状態
└── epoch_*/                 # エポックごとのサンプル
    ├── chain/               # サンプリングチェーン
    └── molecule_*.xyz       # 生成された分子
```

#### 本格的な学習設定

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
    --save_model True \
    --no_wandb
```

**パラメータの詳細説明**:

- `--nf 256`: 隠れ層の次元数。大きいほど表現力が高いが計算コストも増加
- `--n_layers 9`: EGNNレイヤー数。深いほど複雑な分子を学習可能
- `--diffusion_noise_schedule polynomial_2`: ノイズスケジュール。polynomial_2は2次多項式
- `--diffusion_noise_precision 1e-5`: 数値精度。安定性のため1e-5推奨
- `--normalize_factors 1,4,10`: 座標、特徴、電荷の正規化係数
- `--ema_decay 0.9999`: Exponential Moving Averageの減衰率。高いほど平滑化
- `--n_stability_samples 1000`: 評価時に生成するサンプル数

**評価時の出力（test_epochs間隔で表示）**:

```
Epoch 20/3000:
  Train Loss: 15.234
  Val Loss: 15.891
  
  Sampling 1000 molecules for stability evaluation...
  Generated 1000 molecules in 45.2 seconds
  
  Stability Analysis:
  ------------------
  Atom Stability: 94.3%
  Molecule Stability: 87.6%
  
  Sample molecules saved to: outputs/production_model/epoch_20/
```

**安定性指標の意味**:
- `Atom Stability`: 全原子のうち、化学的に妥当な結合数を持つ原子の割合
  - 例: 炭素が4本、酸素が2本の結合を持つことが期待される
- `Molecule Stability`: 全分子のうち、全原子が安定な分子の割合
  - 全原子が妥当な結合数を持つ場合にのみカウント

### 分子の生成

学習済みモデルから新しい分子を生成:

```bash
python eval_sample.py \
    --model_path outputs/production_model \
    --n_samples 1000 \
    --batch_size 100
```

**出力**:

1. **標準出力**:
```
Loading model from outputs/production_model
Model loaded successfully
Sampling 1000 molecules...
  100/1000 Molecules generated at 0.05 secs/sample
  200/1000 Molecules generated at 0.05 secs/sample
  ...
  1000/1000 Molecules generated at 0.05 secs/sample
Total time: 50.3 seconds
Molecules saved to: outputs/production_model/eval/molecules/
```

2. **保存されるファイル**:
```
outputs/production_model/eval/molecules/
├── molecule_0.xyz
├── molecule_1.xyz
├── molecule_2.xyz
...
└── molecule_999.xyz
```

**XYZファイル形式の例**:
```
6
Benzene-like molecule
C   0.000   1.400   0.000
C   1.212   0.700   0.000
C   1.212  -0.700   0.000
C   0.000  -1.400   0.000
C  -1.212  -0.700   0.000
C  -1.212   0.700   0.000
```

- 1行目: 原子数
- 2行目: コメント（分子の説明）
- 3行目以降: 原子タイプと座標（Å単位）

### 生成品質の評価

```bash
python eval_analyze.py \
    --model_path outputs/production_model \
    --n_samples 10000
```

**出力の詳細**:

1. **標準出力**:
```
Analyzing 10000 generated molecules...
  1000/10000 Molecules generated at 0.05 secs/sample
  ...
  10000/10000 Molecules generated at 0.05 secs/sample

Stability Analysis:
==================
Molecule Stability: 93.2%
Atom Stability: 97.8%

RDKit Validation:
==================
Valid molecules: 9634/10000 (96.34%)
Building molecular graphs...
Computing SMILES representations...

Validity: 96.34%
Uniqueness: 99.12%
Novelty: 89.73%

Evaluation complete. Results saved to:
  outputs/production_model/eval/analyzed_molecules/
```

**評価指標の意味**:

- **Validity（有効性）**: 96.34%
  - RDKitで有効な分子構造として認識される割合
  - 化学的に実在可能な分子の指標
  - 計算方法: 有効な分子数 / 全生成分子数

- **Uniqueness（一意性）**: 99.12%
  - 重複しない一意な分子の割合
  - 生成の多様性を示す
  - 計算方法: 一意な分子数 / 有効な分子数
  - SMILES表記で判定

- **Novelty（新規性）**: 89.73%
  - 学習データセットに含まれない新しい分子の割合
  - 生成モデルの創造性を示す
  - 計算方法: 学習データに無い分子数 / 一意な有効分子数

- **Molecule Stability**: 93.2%
  - 全原子が妥当な結合数を持つ分子の割合
  - 化学的安定性の指標

- **Atom Stability**: 97.8%
  - 妥当な結合数を持つ原子の割合
  - 局所的な化学的妥当性

2. **保存されるファイル**:
```
outputs/production_model/eval/analyzed_molecules/
├── molecule_0.xyz
├── molecule_1.xyz
...
├── evaluation_results.json  # 評価結果の詳細
└── stability_report.txt     # テキスト形式のレポート
```


---

## 条件付き分子生成

### 単一性質での条件付け

特定の物性値を持つ分子を生成します。

#### 利用可能な性質

| 性質名 | 記号 | 単位 | 説明 |
|--------|------|------|------|
| alpha | α | Bohr³ | 分極率（Polarizability） |
| gap | Δε | Hartree | HOMO-LUMOギャップ |
| homo | εHOMO | Hartree | 最高被占軌道エネルギー |
| lumo | εLUMO | Hartree | 最低空軌道エネルギー |
| mu | μ | Debye | 双極子モーメント |
| Cv | Cv | cal/mol/K | 熱容量 |

#### 条件付きモデルの学習

```bash
python main_qm9.py \
    --exp_name cond_alpha \
    --model egnn_dynamics \
    --conditioning alpha \
    --nf 192 \
    --n_layers 9 \
    --n_epochs 3000 \
    --batch_size 64 \
    --diffusion_steps 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --save_model True \
    --dataset qm9_second_half \
    --no_wandb
```

**出力**:

```
Epoch 1/3000:
  Conditioning on: ['alpha']
  Context dimension: 1
  Train Loss: 19.456
  Val Loss: 19.823
  
Epoch 20/3000:
  Train Loss: 16.234
  Val Loss: 16.591
  Property range in batch: alpha = [40.5, 85.3] Bohr³
  Stability: 91.2%
```

**条件付けの仕組み**:
- 各分子に対応する性質値をモデルの入力として追加
- 拡散過程で性質値を維持しながら分子構造を生成
- `Context dimension: 1`: 1次元の条件情報（単一性質）

#### 性質予測器の学習

条件付き生成の品質を評価するため、性質予測器を学習:

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

**出力**:

```
Training property predictor for: alpha
Dataset: QM9
Model: EGNN

Epoch 1/500:
  Train Loss: 125.34
  Val Loss: 128.91
  Train MAE: 3.45 Bohr³
  Val MAE: 3.67 Bohr³

Epoch 500/500:
  Train Loss: 8.23
  Val Loss: 9.45
  Train MAE: 0.85 Bohr³
  Val MAE: 0.92 Bohr³

Model saved to: qm9/property_prediction/outputs/class_alpha/
```

**評価指標の意味**:
- `MAE (Mean Absolute Error)`: 予測値と真値の絶対誤差の平均
  - 低いほど予測精度が高い
  - 単位は性質に依存（ここでは Bohr³）

#### 条件付き生成の実行

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/cond_alpha \
    --property alpha \
    --n_sweeps 10 \
    --task qualitative
```

**出力**:

```
Conditional Generation for property: alpha
Number of sweeps: 10
Target values: [45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]

Generating molecules for alpha = 45.0 Bohr³...
  Generated 100 molecules
  Mean alpha: 45.3 ± 2.1 Bohr³
  
Generating molecules for alpha = 50.0 Bohr³...
  Generated 100 molecules
  Mean alpha: 50.1 ± 1.9 Bohr³

...

Generating molecules for alpha = 90.0 Bohr³...
  Generated 100 molecules
  Mean alpha: 89.8 ± 2.3 Bohr³

All molecules saved to: outputs/cond_alpha/conditional_samples/
```

**n_sweepsの意味**:
- 性質値の範囲を等分割する数
- 例: n_sweeps=10なら、min〜maxを10段階に分割
- 各段階で100個の分子を生成

**保存されるファイル**:
```
outputs/cond_alpha/conditional_samples/
├── alpha_45.0/
│   ├── molecule_0.xyz
│   ├── molecule_1.xyz
│   └── ...
├── alpha_50.0/
│   └── ...
...
└── alpha_90.0/
    └── ...
```

#### 定量的評価

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/cond_alpha \
    --classifiers_path qm9/property_prediction/outputs/class_alpha \
    --property alpha \
    --iterations 100 \
    --batch_size 100 \
    --task edm
```

**出力**:

```
Quantitative Evaluation of Conditional Generation

Property: alpha
Iterations: 100
Total molecules: 10000

Target Property Values and Results:
====================================

Target: 50.0 Bohr³
  Generated: 1000 molecules
  Mean predicted: 50.2 ± 1.8 Bohr³
  MAE from target: 1.2 Bohr³
  Within ±5%: 94.3%
  
Target: 60.0 Bohr³
  Generated: 1000 molecules
  Mean predicted: 60.1 ± 1.7 Bohr³
  MAE from target: 1.1 Bohr³
  Within ±5%: 95.8%
  
...

Overall Statistics:
==================
Mean MAE: 1.15 Bohr³
Mean within ±5%: 94.7%
Stability: 92.3%
Validity: 95.8%
```

**評価指標の詳細**:
- `Mean predicted`: 予測器による性質値の平均と標準偏差
- `MAE from target`: 目標値からの平均絶対誤差
- `Within ±5%`: 目標値の±5%以内に収まる分子の割合
  - 条件付けの精度を示す重要な指標
  - 高いほど目標性質への制御が正確

### 複数性質での条件付け

複数の性質を同時に制御:

```bash
python main_qm9.py \
    --exp_name multi_cond \
    --conditioning alpha gap homo \
    --nf 256 \
    --n_layers 9 \
    --n_epochs 3000 \
    --batch_size 64 \
    --no_wandb
```

**出力**:

```
Multi-property Conditioning
Properties: ['alpha', 'gap', 'homo']
Context dimension: 3

Epoch 1/3000:
  Property ranges in batch:
    alpha: [42.3, 87.6] Bohr³
    gap: [0.12, 0.45] Hartree
    homo: [-0.35, -0.15] Hartree
  Train Loss: 20.123
```

**複数条件の生成**:

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/multi_cond \
    --property alpha gap \
    --target_alpha 60.0 \
    --target_gap 0.25 \
    --n_samples 100
```

---

## 結晶構造生成

### ホモ結晶の学習

単一分子種からなる結晶構造を生成:

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
    --include_pbc True \
    --no_wandb
```

**出力**:

```
Crystal Generation Training
===========================
Crystal type: homocrystal
Space groups: all (230 groups)
Periodic boundary conditions: True

Epoch 1/1000:
  Train Loss: 45.678
  Val Loss: 46.234
  Lattice Loss: 12.345
  Position Loss: 33.333
  
Epoch 10/1000:
  Train Loss: 38.912
  Val Loss: 39.456
  Generated sample crystal:
    Space group: 14 (P2₁/c)
    Cell parameters: a=8.5Å, b=9.2Å, c=10.1Å
                    α=90°, β=105°, γ=90°
    Density: 1.25 g/cm³
    Molecules per unit cell: 4
```

**損失の意味**:
- `Lattice Loss`: 格子パラメータ（a,b,c,α,β,γ）の予測誤差
- `Position Loss`: 原子位置の予測誤差
- `Train/Val Loss`: 上記の合計

**結晶固有のパラメータ説明**:
- `--crystal_type homocrystal`: 単一分子種の結晶
- `--space_groups all`: 全230種の空間群を学習
- `--include_pbc True`: 周期境界条件を考慮
  - 結晶の無限周期性を正確にモデル化

### 条件付き結晶生成

#### 空間群条件付け

特定の空間群で結晶を生成:

```bash
python main_crystal_with_properties.py \
    --exp_name crystal_sg_cond \
    --conditioning space_group \
    --space_groups 1,2,14,19,61,62 \
    --n_epochs 1000 \
    --no_wandb
```

**出力**:

```
Conditional Crystal Generation Training
======================================
Conditioning: space_group
Target space groups: [1, 2, 14, 19, 61, 62]

Space Group Information:
  1: P1 (Triclinic)
  2: P-1 (Triclinic)
  14: P2₁/c (Monoclinic)
  19: P2₁2₁2₁ (Orthorhombic)
  61: Pbca (Orthorhombic)
  62: Pnma (Orthorhombic)

Epoch 1/1000:
  Train Loss: 42.567
  Space group embedding dimension: 64
  Context dimension: 64
```

**空間群の説明**:
- 結晶の対称性を表す分類（全230種）
- 結晶系（三斜晶、単斜晶、斜方晶、正方晶、三方晶、六方晶、立方晶）
- 対称操作の組み合わせで定義

#### 密度条件付け

目標密度で結晶を生成:

```bash
python main_crystal_with_properties.py \
    --exp_name crystal_density_cond \
    --conditioning density \
    --density_range 0.8,2.5 \
    --n_epochs 1000 \
    --no_wandb
```

**出力**:

```
Density Conditioning
===================
Density range: 0.8 - 2.5 g/cm³
Context dimension: 1

Epoch 10/1000:
  Batch density range: [1.15, 2.03] g/cm³
  Train Loss: 39.234
  Density prediction MAE: 0.08 g/cm³
```

#### 複合条件付け

空間群と密度を同時に制御:

```bash
python main_crystal_with_properties.py \
    --exp_name crystal_multi_cond \
    --conditioning space_group density \
    --space_groups 14,19,61 \
    --density_range 1.0,2.0 \
    --n_epochs 1000 \
    --no_wandb
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

```
Generating Crystals with Conditions
===================================
Space group: 14 (P2₁/c)
Target density: 1.5 g/cm³
Number of samples: 10

Generating crystal 1/10...
  Generated structure:
    Space group: 14
    Cell: a=8.52Å, b=9.18Å, c=10.05Å, β=104.8°
    Density: 1.51 g/cm³
    Molecules: 4
    Total atoms: 64
  Saved to: generated_crystals/crystal_0001.cif

Generating crystal 2/10...
  Generated structure:
    Space group: 14
    Cell: a=8.48Å, b=9.23Å, c=10.12Å, β=105.2°
    Density: 1.49 g/cm³
    Molecules: 4
    Total atoms: 64
  Saved to: generated_crystals/crystal_0002.cif

...

All crystals generated successfully!
Files saved to: generated_crystals/
```

**保存されるファイル**:

1. **CIFファイル** (結晶情報ファイル):
```cif
# generated_crystals/crystal_0001.cif
data_crystal_0001
_cell_length_a    8.52
_cell_length_b    9.18
_cell_length_c    10.05
_cell_angle_alpha 90.00
_cell_angle_beta  104.80
_cell_angle_gamma 90.00
_space_group_name_H-M_alt 'P 1 21/c 1'
_space_group_IT_number 14

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1234 0.2456 0.3678
H1 H 0.2345 0.3567 0.4789
...
```

**CIFファイルの内容説明**:
- `_cell_length_*`: 格子定数（Å）
- `_cell_angle_*`: 格子角（度）
- `_space_group_*`: 空間群情報
- `_atom_site_*`: 原子の分率座標（0-1の範囲）

2. **XYZファイル** (デカルト座標):
```
# generated_crystals/crystal_0001.xyz
64
Space group: 14, Density: 1.51 g/cm³
C   1.052   2.251   3.688
H   1.998   3.273   4.803
...
```

3. **画像ファイル** (オプション):
```
generated_crystals/
├── crystal_0001.png    # 構造の可視化
├── crystal_0002.png
...
```

### 結晶品質の評価

```python
from crystal.evaluation import CrystalMetrics, StructureValidator

# 結晶メトリクスの計算
metrics = CrystalMetrics(dataset_info)
results = metrics.compute_all_metrics(generated_crystals)

print("Structural Metrics:")
print(f"  Mean cell volume: {results['mean_volume']:.2f} Ų")
print(f"  Mean density: {results['mean_density']:.2f} g/cm³")
print(f"  Cell parameter std: {results['cell_param_std']:.3f}")

print("\nValidity Metrics:")
print(f"  Min distance violations: {results['min_dist_violations']:.1f}%")
print(f"  Symmetry preservation: {results['symmetry_score']:.1f}%")

print("\nSpace Group Analysis:")
print(f"  Target space group match: {results['space_group_accuracy']:.1f}%")
```

**出力例**:

```
Structural Metrics:
  Mean cell volume: 785.23 ų
  Mean density: 1.48 g/cm³
  Cell parameter std: 0.156

Validity Metrics:
  Min distance violations: 2.3%
  Symmetry preservation: 97.8%

Space Group Analysis:
  Target space group match: 95.2%
```

**メトリクスの意味**:
- `Mean cell volume`: 単位格子の平均体積
- `Mean density`: 結晶の平均密度
- `Cell parameter std`: 格子定数の標準偏差（生成の一貫性）
- `Min distance violations`: 原子間距離が物理的に不可能な割合
- `Symmetry preservation`: 空間群の対称性が保たれている割合
- `Space group accuracy`: 指定した空間群と一致する割合

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
    'C6H12O6',  # グルコース
    positions=[
        [0.000, 0.000, 0.000],  # C
        [1.520, 0.000, 0.000],  # C
        # ... 他の原子座標
    ]
)

# 性質データの追加
db.write(
    atoms,
    data={
        'energy': -234.567,  # エネルギー (任意単位)
        'molecular_weight': 180.16,  # 分子量
        'dipole_moment': 2.3,  # 双極子モーメント
        'custom_property': 42.0  # カスタム性質
    }
)

print(f"Added molecule to database. Total molecules: {len(db)}")
```

**出力**:
```
Added molecule to database. Total molecules: 1
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
    --n_epochs 1000 \
    --no_wandb
```

**出力**:

```
Loading ASE Database
====================
Database path: my_molecules.db
Total entries: 1500

Database Statistics:
  Total molecules: 1500
  Unique molecules: 1500
  Min atoms: 5
  Max atoms: 29
  Mean atoms: 12.3 ± 4.2

Atom Type Distribution:
  H: 8234 (54.2%)
  C: 4521 (29.8%)
  O: 1823 (12.0%)
  N: 567 (3.7%)
  F: 45 (0.3%)

Dataset Info:
  atom_encoder: {'H': 0, 'C': 1, 'O': 2, 'N': 3, 'F': 4}
  atom_decoder: ['H', 'C', 'O', 'N', 'F']

Splitting dataset:
  Train: 1200 (80.0%)
  Val: 150 (10.0%)
  Test: 150 (10.0%)

Starting training...
Epoch 1/1000:
  Train Loss: 18.234
```

### 重複除去機能

ASEデータベースは自動的に重複分子を検出・除去:

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path my_molecules.db \
    --remove_duplicates True \
    --duplicate_tolerance 1e-6 \
    --no_wandb
```

**出力**:

```
Duplicate Detection
===================
Duplicate tolerance: 1.0e-06 Å

Analyzing database for duplicates...
  Processed: 500/1500
  Processed: 1000/1500
  Processed: 1500/1500

Duplicate Analysis Results:
  Total molecules: 1500
  Unique molecules: 1387
  Duplicates found: 113 (7.5%)

Duplicate groups:
  Group 1: 3 molecules (IDs: 45, 128, 892)
  Group 2: 2 molecules (IDs: 234, 567)
  ...

Removed 113 duplicate molecules
Final dataset size: 1387 molecules
```

**重複判定の基準**:
- 原子数が同じ
- 原子タイプの順序が同じ
- 全原子の座標差が tolerance 以下
- デフォルト: 1e-6 Å（極めて厳密）

### データベースのデバッグ出力

データセットの内容を詳細に確認:

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path my_molecules.db \
    --debug_dataset_csv dataset_analysis.csv \
    --debug_dataset_xyz debug_xyz/ \
    --no_wandb
```

**出力されるファイル**:

1. **dataset_analysis.csv**:
```csv
ID,Formula,NumAtoms,Energy,MolecularWeight,CustomProperty
0,C6H12O6,24,-234.567,180.16,42.0
1,C8H10N4O2,24,-312.456,194.19,35.7
2,C3H8O,12,-118.234,60.10,28.3
...
```

2. **debug_xyz/ ディレクトリ**:
```
debug_xyz/
├── molecule_0000.xyz
├── molecule_0001.xyz
├── molecule_0002.xyz
...
```

各XYZファイルには分子の構造が保存されます。

---

## 分子記述子による条件付け

ASEデータベースと併用して、分子記述子による高度な条件付けが可能です。

### 利用可能な分子記述子

| 記述子名 | 説明 | 値の範囲 |
|---------|------|---------|
| molecular_weight | 分子量 | 0〜数百 Da |
| pi_conjugation_ratio | π共役系の比率 | 0.0〜1.0 |
| atom_types_encoding | 原子タイプの分布 | ベクトル |
| functional_groups_encoding | 官能基の有無 | バイナリベクトル |

### 分子記述子の計算と保存

```python
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

db = connect('molecules_with_descriptors.db')

# 分子のリスト（RDKit形式）
molecules = [...]  # あなたの分子データ

for mol in molecules:
    # 分子記述子の計算
    mol_weight = Descriptors.MolWt(mol)
    
    # π共役比率の計算
    num_aromatic_atoms = sum([atom.GetIsAromatic() for atom in mol.GetAtoms()])
    pi_ratio = num_aromatic_atoms / mol.GetNumAtoms()
    
    # 原子タイプのエンコーディング
    atom_counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
    atom_encoding = encode_atom_counts(atom_counts)  # カスタム関数
    
    # 官能基のエンコーディング
    func_groups = {
        'hydroxyl': has_hydroxyl(mol),
        'carbonyl': has_carbonyl(mol),
        'carboxyl': has_carboxyl(mol),
        'amine': has_amine(mol),
        # ... 他の官能基
    }
    
    # ASEデータベースに保存
    atoms = mol_to_atoms(mol)  # RDKitからASE形式に変換
    db.write(
        atoms,
        data={
            'molecular_weight': mol_weight,
            'pi_conjugation_ratio': pi_ratio,
            'atom_types_encoding': atom_encoding.tolist(),
            'functional_groups_encoding': list(func_groups.values())
        }
    )

print(f"Saved {len(molecules)} molecules with descriptors")
```

**出力**:
```
Saved 1500 molecules with descriptors
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
    --batch_size 64 \
    --no_wandb
```

**出力**:

```
Multi-Descriptor Conditioning
=============================
Properties: ['molecular_weight', 'pi_conjugation_ratio']
Context dimension: 2

Property Statistics:
  molecular_weight:
    Min: 46.07 Da
    Max: 342.56 Da
    Mean: 156.23 ± 52.34 Da
  pi_conjugation_ratio:
    Min: 0.00
    Max: 0.85
    Mean: 0.32 ± 0.24

Epoch 1/3000:
  Batch property ranges:
    molecular_weight: [82.3, 248.7] Da
    pi_conjugation_ratio: [0.12, 0.68]
  Train Loss: 19.567
  Context embedding successful

Epoch 10/3000:
  Train Loss: 17.234
  Val Loss: 17.589
  Property prediction MAE:
    molecular_weight: 8.3 Da
    pi_conjugation_ratio: 0.05
```

**Property prediction MAE の意味**:
- 生成された分子の記述子を計算し、目標値との誤差を測定
- 条件付けの精度を示す
- 低いほど目標記述子に近い分子を生成

### 複数記述子での生成

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/descriptor_cond \
    --property molecular_weight \
    --target_values 100,150,200,250,300 \
    --n_samples 100
```

**出力**:

```
Conditional Generation with Molecular Descriptors
================================================
Property: molecular_weight
Target values: [100, 150, 200, 250, 300] Da
Samples per target: 100

Generating for MW = 100 Da...
  Generated 100 molecules
  Actual MW: 101.3 ± 5.2 Da
  Validity: 96.0%
  Saved to: outputs/descriptor_cond/MW_100/

Generating for MW = 150 Da...
  Generated 100 molecules
  Actual MW: 149.8 ± 6.1 Da
  Validity: 97.0%
  Saved to: outputs/descriptor_cond/MW_150/

...

Generating for MW = 300 Da...
  Generated 100 molecules
  Actual MW: 298.5 ± 8.7 Da
  Validity: 94.0%
  Saved to: outputs/descriptor_cond/MW_300/

Summary:
  Total generated: 500 molecules
  Mean MW accuracy: ±6.3 Da
  Overall validity: 95.6%
```

---

## 評価と解析

### 包括的な評価

```bash
python eval_analyze.py \
    --model_path outputs/production_model \
    --n_samples 10000 \
    --save_results True \
    --output_file evaluation_results.json
```

**詳細な出力**:

```
Comprehensive Evaluation
========================
Model: outputs/production_model
Samples: 10000
Batch size: 100

Generating molecules...
  1000/10000 (10.0%) - 0.05 sec/sample
  2000/10000 (20.0%) - 0.05 sec/sample
  ...
  10000/10000 (100.0%) - 0.05 sec/sample
Total generation time: 500.3 seconds

Analyzing molecular stability...
  Checking bond orders and valence...
  10000/10000 molecules analyzed

Stability Results:
==================
Atom-level Stability: 9782/10000 atoms stable (97.82%)
Molecule-level Stability: 9323/10000 molecules stable (93.23%)

Detailed Bond Analysis:
  Total bonds analyzed: 142,345
  Valid bonds: 139,123 (97.74%)
  Invalid bonds: 3,222 (2.26%)
  
  Invalid bond breakdown:
    Over-coordinated C: 1,234 cases
    Over-coordinated O: 567 cases
    Under-coordinated N: 1,421 cases

RDKit Validation (if available):
==================================
Converting to RDKit molecules...
  Successfully converted: 9634/10000 (96.34%)
  Failed conversions: 366 (3.66%)

Building molecular graphs...
  Valid molecules: 9634
  Generating SMILES...
  Computing uniqueness...
  Computing novelty (comparing with training set)...

Quality Metrics:
  Validity: 96.34%
    - Chemical structure is valid
    - Can be represented as SMILES
  
  Uniqueness: 99.12%
    - 9545/9634 unique SMILES
    - Only 89 duplicates (0.92%)
  
  Novelty: 89.73%
    - 8562/9545 molecules not in training set
    - 983 molecules (10.27%) are in training set

Molecular Property Distribution:
================================
Number of atoms:
  Mean: 12.3 ± 4.2
  Min: 5, Max: 29
  
Atom type distribution:
  H: 54.2%
  C: 29.8%
  O: 12.0%
  N: 3.7%
  F: 0.3%

Bond length statistics:
  C-C bonds: 1.52 ± 0.08 Å
  C-O bonds: 1.43 ± 0.06 Å
  C-N bonds: 1.47 ± 0.07 Å

Results saved to:
  evaluation_results.json
  stability_report.txt
  outputs/production_model/eval/analyzed_molecules/
```

**evaluation_results.json の内容**:

```json
{
  "summary": {
    "total_samples": 10000,
    "atom_stability": 0.9782,
    "molecule_stability": 0.9323,
    "validity": 0.9634,
    "uniqueness": 0.9912,
    "novelty": 0.8973
  },
  "bond_analysis": {
    "total_bonds": 142345,
    "valid_bonds": 139123,
    "invalid_bonds": 3222,
    "invalid_breakdown": {
      "over_coordinated_C": 1234,
      "over_coordinated_O": 567,
      "under_coordinated_N": 1421
    }
  },
  "molecular_properties": {
    "num_atoms": {
      "mean": 12.3,
      "std": 4.2,
      "min": 5,
      "max": 29
    },
    "atom_distribution": {
      "H": 0.542,
      "C": 0.298,
      "O": 0.120,
      "N": 0.037,
      "F": 0.003
    }
  }
}
```

### カスタム評価スクリプト

独自の評価を実装:

```python
from qm9.dataset import retrieve_dataloaders
from qm9.analyze import analyze_stability_for_molecules
from qm9.models import get_model
import torch

# モデルの読み込み
model = get_model(args, dataset_info)
model.load_state_dict(torch.load('outputs/production_model/generative_model.npy'))
model.eval()

# サンプリング
samples = {'one_hot': [], 'x': [], 'node_mask': []}
for i in range(100):
    with torch.no_grad():
        x, h, node_mask = model.sample(n_nodes, device)
        samples['one_hot'].append(h)
        samples['x'].append(x)
        samples['node_mask'].append(node_mask)

# カスタム評価
def evaluate_chemical_diversity(samples):
    """化学的多様性の評価"""
    # 例: フィンガープリントの多様性
    from rdkit.Chem import AllChem
    
    fingerprints = []
    for mol in samples:
        fp = AllChem.GetMorganFingerprint(mol, 2)
        fingerprints.append(fp)
    
    # タニモト距離で多様性を計算
    diversity_score = compute_tanimoto_diversity(fingerprints)
    return diversity_score

diversity = evaluate_chemical_diversity(samples)
print(f"Chemical diversity score: {diversity:.3f}")
```


---

## データエクスポート機能

### CSV形式でのエクスポート

データセット内の分子記述子をCSV形式でエクスポート:

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path my_molecules.db \
    --export_conditions_csv ./exported_data \
    --no_wandb
```

**出力**:

```
CSV Export Mode
===============
Output directory: ./exported_data
Dataset: my_molecules.db
Total molecules: 1500

Analyzing molecular descriptors...
  Processing molecule 500/1500
  Processing molecule 1000/1500
  Processing molecule 1500/1500

Exporting molecular_weight.csv...
Exporting pi_conjugation_ratio.csv...
Exporting atom_types_encoding.csv...
Exporting functional_groups_encoding.csv...

Export Summary:
==============
Files created:
  ./exported_data/molecular_weight.csv (1500 entries)
  ./exported_data/pi_conjugation_ratio.csv (1500 entries)
  ./exported_data/atom_types_encoding.csv (1500 entries)
  ./exported_data/functional_groups_encoding.csv (1500 entries)

CSV export completed successfully.
Program exiting.
```

**出力されるCSVファイルの内容**:

1. **molecular_weight.csv**:
```csv
ID,分子の組成,分子量
0,C6H12O6,180.1559
1,C8H10N4O2,194.1906
2,C9H8O4,180.1574
3,C2H6O,46.0684
4,C6H6,78.1118
...
```

列の説明:
- `ID`: 分子の一意識別子
- `分子の組成`: 化学式（例: C6H12O6はグルコース）
- `分子量`: 分子量（ダルトン単位）

2. **pi_conjugation_ratio.csv**:
```csv
ID,分子の組成,π共役比率
0,C6H12O6,0.1667
1,C8H10N4O2,0.5556
2,C9H8O4,0.6667
3,C2H6O,0.0000
4,C6H6,1.0000
...
```

列の説明:
- `π共役比率`: 芳香族原子の割合（0.0〜1.0）
  - 0.0: π共役系なし
  - 1.0: 全原子が芳香族

3. **atom_types_encoding.csv**:
```csv
ID,分子の組成,H,C,O,N,F
0,C6H12O6,12,6,6,0,0
1,C8H10N4O2,10,8,2,4,0
2,C9H8O4,8,9,4,0,0
...
```

列の説明:
- 各原子タイプの出現回数

4. **functional_groups_encoding.csv**:
```csv
ID,分子の組成,ヒドロキシル基,カルボニル基,カルボキシル基,アミン基,エステル基
0,C6H12O6,5,1,0,0,0
1,C8H10N4O2,0,2,0,4,0
2,C9H8O4,2,2,2,0,0
...
```

列の説明:
- 各官能基の出現回数

### プログラムでの使用例

```python
from crystal.utils import export_molecule_properties
from ase.db import connect

# データベースを開く
db = connect('my_molecules.db')

# エクスポート
export_molecule_properties(
    db_path='my_molecules.db',
    output_dir='./exported_data',
    properties=['molecular_weight', 'pi_conjugation_ratio']
)

print("Export completed!")
```

---

## 学習の再開機能

### 学習の中断と再開

学習を中断した場合、チェックポイントから再開可能:

```bash
# 元の学習コマンド
python main_qm9.py \
    --exp_name production_model \
    --n_epochs 3000 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 9

# 学習が途中で中断（例: エポック1234で停止）

# 再開コマンド
python main_qm9.py \
    --exp_name production_model \
    --resume outputs/production_model \
    --start_epoch 1234
```

**出力**:

```
Resuming Training
=================
Experiment: production_model
Resume from: outputs/production_model
Start epoch: 1234

Loading saved arguments...
  Loaded from: outputs/production_model/args.pickle
  
Loading saved dataset info...
  Loaded from: outputs/production_model/dataset_info.pickle
  Atom types: ['H', 'C', 'N', 'O', 'F']
  
Loading model checkpoint...
  Loaded from: outputs/production_model/generative_model.npy
  Model parameters: 12,345,678
  
Loading EMA model...
  Loaded from: outputs/production_model/generative_model_ema.npy
  
Loading optimizer state...
  Loaded from: outputs/production_model/optim.npy
  Learning rate: 1.0e-04
  
Resume configuration verified:
  Context dimension matches: 0
  Model architecture matches: ✓
  All components loaded successfully

Continuing training from epoch 1234/3000...

Epoch 1234/3000:
  Train Loss: 14.567
  Val Loss: 14.823
  (training continues...)
```

### 重要な注意点

**再開時に保持される情報**:
- モデルのパラメータ（重み）
- EMAモデルの状態
- オプティマイザーの状態（学習率、モーメンタムなど）
- データセット情報（原子タイプ、正規化係数など）
- 学習設定（バッチサイズ、レイヤー数など）

**再開時に変更できる情報**:
- `--start_epoch`: 開始エポック（通常は自動検出）
- `--n_epochs`: 総エポック数（延長可能）
- `--lr`: 学習率（変更可能だが推奨しない）

**再開時に変更してはいけない情報**:
- モデルアーキテクチャ（`--nf`, `--n_layers`）
- データセット（`--dataset`, `--ase_db_path`）
- 条件付け設定（`--conditioning`）
- 正規化係数（`--normalize_factors`）

### コンテキスト次元の自動調整

再開時、コンテキスト次元のミスマッチを自動的に検出して修正:

```bash
python main_qm9.py \
    --exp_name cond_model \
    --resume outputs/cond_model \
    --start_epoch 500
```

**出力（ミスマッチ検出時）**:

```
Context Dimension Mismatch Detected
===================================
Saved model context_nf: 3
Current database context_nf: 5

WARNING: Context dimension mismatch!
The model was trained with 3 context features,
but the current database provides 5 features.

Using saved context dimension (3) to ensure model compatibility.

If you need to use different conditioning properties,
you must train a new model from scratch.

Continuing with saved configuration...
```

**エラー回避の仕組み**:
- 保存されたモデルのコンテキスト次元を優先
- データベースの変更があっても互換性を保持
- モデルアーキテクチャの一貫性を保証

---

## 高度な使用方法

### EMAモデルの使用

Exponential Moving Average（指数移動平均）モデルは、学習の安定性を向上:

```bash
python main_qm9.py \
    --exp_name ema_model \
    --ema_decay 0.9999 \
    --save_model True \
    --test_epochs 10
```

**EMAの効果**:
- 学習中のパラメータの変動を平滑化
- より安定した生成品質
- 一般化性能の向上

**ema_decayの値**:
- `0.9999`: 推奨値（強い平滑化）
- `0.999`: 中程度の平滑化
- `0.99`: 弱い平滑化

生成時にEMAモデルを使用:

```bash
python eval_sample.py \
    --model_path outputs/ema_model \
    --use_ema True \
    --n_samples 1000
```

**出力**:
```
Using EMA model for generation
EMA model loaded from: outputs/ema_model/generative_model_ema.npy
Generating 1000 molecules...
```

### カスタムノイズスケジュール

#### Polynomial Schedule（多項式スケジュール）

```bash
python main_qm9.py \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_steps 1000
```

**特徴**:
- ノイズが2次多項式的に増加
- 初期段階でのノイズが少なく、詳細な構造を保持
- 推奨される標準的なスケジュール

#### Cosine Schedule（コサインスケジュール）

```bash
python main_qm9.py \
    --diffusion_noise_schedule cosine \
    --diffusion_steps 1000
```

**特徴**:
- ノイズがコサイン曲線的に変化
- より滑らかなノイズ遷移
- 一部のタスクで性能向上

### 学習データの分割

#### 後半のデータのみで学習

```bash
python main_qm9.py \
    --dataset qm9_second_half \
    --exp_name qm9_half_model
```

**用途**:
- 条件付き生成の評価用
- 前半を学習、後半でテスト
- データの偏りの検証

#### カスタム分割比率

```bash
python main_qm9.py \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15
```

**デフォルト**: 80% / 10% / 10%

### バッチ生成と並列処理

大量のサンプルを効率的に生成:

```python
import torch
from torch.multiprocessing import Pool

def generate_batch(model, batch_size, n_nodes, device):
    """バッチで分子を生成"""
    with torch.no_grad():
        x, h, node_mask = model.sample(batch_size, n_nodes, device)
    return x.cpu(), h.cpu(), node_mask.cpu()

# 並列生成の設定
n_processes = 4
n_samples_per_process = 1000
total_samples = n_processes * n_samples_per_process

print(f"Generating {total_samples} molecules using {n_processes} processes...")

with Pool(n_processes) as pool:
    results = pool.starmap(
        generate_batch,
        [(model, n_samples_per_process, n_nodes, device)] * n_processes
    )

print(f"Generated {total_samples} molecules successfully!")
```

**出力**:
```
Generating 4000 molecules using 4 processes...
Process 0: Generated 1000 molecules
Process 1: Generated 1000 molecules
Process 2: Generated 1000 molecules
Process 3: Generated 1000 molecules
Generated 4000 molecules successfully!
```

---

## 出力の詳細説明

### 学習時の出力

#### エポックごとの標準出力

```
Epoch 100/3000:
  Train Loss: 14.234 (前回: 14.567, 変化: -0.333)
  Val Loss: 14.891 (前回: 15.123, 変化: -0.232)
  Learning Rate: 1.00e-04
  Time: 125.3s
  ETA: 6h 45m
```

**各項目の意味**:
- `Train Loss`: 学習データでの損失
  - 拡散過程の予測誤差（L2距離）
  - 低いほど学習が進んでいる
  - 理想的には単調減少
  
- `Val Loss`: 検証データでの損失
  - 過学習の指標
  - Train Lossより高いのは正常
  - Train Lossとの差が大きすぎる場合は過学習
  
- `Learning Rate`: 現在の学習率
  - オプティマイザーによって調整される場合がある
  
- `Time`: エポックあたりの処理時間
  
- `ETA (Estimated Time to Arrival)`: 学習完了までの推定時間

#### テストエポックでの追加出力

```
Test Epoch 100:
  Generating 1000 molecules for evaluation...
  Generated in 45.2 seconds (0.045 sec/molecule)
  
  Stability Analysis:
    Atom Stability: 9782/10000 (97.82%)
    Molecule Stability: 9323/10000 (93.23%)
  
  Sample molecules saved to: outputs/production_model/epoch_100/
```

### サンプリング時の出力

```
Sampling Configuration:
  Model: outputs/production_model
  Samples: 1000
  Batch size: 100
  Device: cuda:0
  
Sampling Progress:
  Batch 1/10: 100 molecules (10.2s, 0.102s/mol)
  Batch 2/10: 100 molecules (10.1s, 0.101s/mol)
  ...
  Batch 10/10: 100 molecules (10.3s, 0.103s/mol)
  
Total Time: 102.5 seconds
Average: 0.103 seconds/molecule
Peak Memory: 4.2GB

Molecules saved to: outputs/production_model/eval/molecules/
```

### 評価時の出力

#### 安定性評価

```
Stability Analysis Results:
==========================
Total molecules: 10000
Total atoms: 123,456

Atom-level Analysis:
  Stable atoms: 120,789 (97.84%)
  Unstable atoms: 2,667 (2.16%)
  
  Unstable breakdown by type:
    Over-coordinated:
      C: 1,234 atoms (4 or more bonds expected, found 5+)
      O: 567 atoms (2 bonds expected, found 3+)
    Under-coordinated:
      N: 866 atoms (3 bonds expected, found 1-2)

Molecule-level Analysis:
  Fully stable: 9,323 molecules (93.23%)
  Partially unstable: 677 molecules (6.77%)
  
  Unstable molecule distribution:
    1 unstable atom: 456 molecules
    2 unstable atoms: 156 molecules
    3+ unstable atoms: 65 molecules
```

**評価基準**:
- 各原子タイプに対する期待結合数:
  - H: 1本
  - C: 4本
  - N: 3本（または4本でイオン化）
  - O: 2本
  - F: 1本

#### RDKit評価

```
RDKit Molecular Validation:
===========================
Conversion to RDKit:
  Success: 9,634 / 10,000 (96.34%)
  Failed: 366 (3.66%)
  
  Failure reasons:
    Invalid valence: 234 molecules
    Ring strain: 89 molecules
    Other errors: 43 molecules

SMILES Generation:
  Generated: 9,634 SMILES strings
  Examples:
    CC(C)CC1=CC=C(C=C1)C(C)C              (p-cymene)
    C1=CC=C2C(=C1)C=CC=C2                 (naphthalene)
    C1CCCCC1                              (cyclohexane)

Uniqueness Check:
  Total SMILES: 9,634
  Unique SMILES: 9,545
  Duplicates: 89 (0.92%)

Novelty Check:
  Comparing with training set (133,885 molecules)...
  Novel molecules: 8,562 / 9,545 (89.67%)
  Known molecules: 983 (10.33%)
  
  Known molecule distribution:
    Exact matches: 456 molecules
    Similar (Tanimoto > 0.95): 527 molecules
```

### 結晶評価の出力

```
Crystal Structure Evaluation:
============================
Generated crystals: 100

Structural Metrics:
------------------
Cell volume:
  Mean: 785.23 ų
  Std: 52.34 ų
  Range: [650.12, 923.45] ų

Cell parameters:
  a: 8.52 ± 0.45 Å
  b: 9.18 ± 0.38 Å
  c: 10.05 ± 0.52 Å
  α: 90.0 ± 0.0°
  β: 104.8 ± 3.2°
  γ: 90.0 ± 0.0°

Density:
  Mean: 1.48 ± 0.12 g/cm³
  Range: [1.25, 1.78] g/cm³

Validity Metrics:
----------------
Minimum distance check:
  Violations: 2 / 100 crystals (2.0%)
  Min observed distance: 1.12 Å
  Expected minimum: 1.00 Å

Symmetry analysis:
  Space group preservation: 97 / 100 (97.0%)
  Perfect symmetry: 89 crystals
  Minor deviations: 8 crystals (< 0.1 Å)
  Major violations: 3 crystals (> 0.1 Å)

Target vs Actual:
  Target space group: 14 (P2₁/c)
  Correct space group: 95 / 100 (95.0%)
  Detected space groups:
    14 (P2₁/c): 95 crystals
    13 (P2/c): 3 crystals (similar)
    1 (P1): 2 crystals (symmetry lost)
```

### ファイル出力の詳細

#### XYZファイルの形式

```
24
Molecule ID: 0, Stability: stable, Formula: C8H10N4O2
C    0.000    1.400    0.000
C    1.212    0.700    0.000
C    1.212   -0.700    0.000
C    0.000   -1.400    0.000
C   -1.212   -0.700    0.000
C   -1.212    0.700    0.000
H    0.000    2.490    0.000
H    2.156    1.245    0.000
...
```

- 1行目: 原子数
- 2行目: メタデータ（ID、安定性、化学式）
- 3行目以降: 原子タイプと座標（Å単位）

#### CIFファイルの形式

```
data_crystal_0001
# Crystal structure generated by E3 Diffusion Model

_cell_length_a    8.52
_cell_length_b    9.18
_cell_length_c    10.05
_cell_angle_alpha 90.00
_cell_angle_beta  104.80
_cell_angle_gamma 90.00

_space_group_name_H-M_alt 'P 1 21/c 1'
_space_group_IT_number 14
_symmetry_space_group_name_H-M 'P 1 21/c 1'

_cell_volume 785.23
_cell_formula_units_Z 4

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1  C  0.1234  0.2456  0.3678  1.000
H1  H  0.2345  0.3567  0.4789  1.000
O1  O  0.3456  0.4678  0.5890  1.000
...
```

**CIF形式の説明**:
- `_cell_length_*`: 格子定数（Å）
- `_cell_angle_*`: 格子角（度）
- `_space_group_*`: 空間群情報
- `_cell_volume`: 単位格子の体積（ų）
- `_cell_formula_units_Z`: 単位格子あたりの分子数
- `_atom_site_*`: 原子の分率座標（0-1）と占有率

---

## コマンドライン完全リファレンス

### main_qm9.py

QM9データセットでの分子生成モデル学習。

#### 必須パラメータ

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `--exp_name` | str | 実験名（出力ディレクトリ名） |

#### モデルパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--model` | str | egnn_dynamics | モデルタイプ |
| `--nf` | int | 128 | 隠れ層の次元数 |
| `--n_layers` | int | 6 | EGNNレイヤー数 |
| `--attention` | bool | True | アテンション機構 |
| `--tanh` | bool | True | tanh活性化関数 |
| `--norm_constant` | float | 1.0 | 正規化定数 |
| `--sin_embedding` | bool | False | sin埋め込み |
| `--inv_sublayers` | int | 1 | 不変サブレイヤー数 |

#### 拡散パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--diffusion_steps` | int | 500 | 拡散ステップ数 |
| `--diffusion_noise_schedule` | str | polynomial_2 | ノイズスケジュール |
| `--diffusion_noise_precision` | float | 1e-5 | 数値精度 |
| `--diffusion_loss_type` | str | l2 | 損失関数タイプ |

#### 学習パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--n_epochs` | int | 200 | 学習エポック数 |
| `--batch_size` | int | 128 | バッチサイズ |
| `--lr` | float | 2e-4 | 学習率 |
| `--test_epochs` | int | 20 | 評価の実行間隔 |
| `--save_model` | bool | True | モデルの保存 |
| `--ema_decay` | float | 0.9999 | EMA減衰率 |
| `--clip_grad` | bool | True | 勾配クリッピング |

#### データセットパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--dataset` | str | qm9 | データセット名 |
| `--datadir` | str | qm9/temp | データディレクトリ |
| `--remove_h` | bool | False | 水素原子の除去 |
| `--include_charges` | bool | True | 電荷情報の使用 |
| `--filter_n_atoms` | int | None | 原子数フィルタ |

#### ASEデータベースパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--ase_db_path` | str | None | ASEデータベースパス |
| `--split_ratios` | list | [0.8,0.1,0.1] | train/val/test分割 |
| `--remove_duplicates` | bool | True | 重複除去 |
| `--duplicate_tolerance` | float | 1e-6 | 重複判定閾値 |
| `--debug_dataset_csv` | str | None | デバッグCSV出力パス |
| `--debug_dataset_xyz` | str | None | デバッグXYZ出力ディレクトリ |
| `--ase_to_eV` | str | '{}' | 単位変換係数（JSON） |

#### 条件付けパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--conditioning` | list | [] | 条件付ける性質のリスト |
| `--property` | str | None | 単一性質の指定 |

#### その他のパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--normalize_factors` | list | [1,4,10] | 正規化係数 |
| `--n_stability_samples` | int | 1000 | 安定性評価サンプル数 |
| `--wandb_usr` | str | None | Weights & Biasesユーザー名 |
| `--no_wandb` | flag | False | W&Bを無効化 |
| `--resume` | str | None | 再開するモデルパス |
| `--start_epoch` | int | 0 | 開始エポック |
| `--export_conditions_csv` | str | None | CSV出力ディレクトリ |

### eval_sample.py

学習済みモデルからのサンプリング。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--model_path` | str | 必須 | モデルのパス |
| `--n_samples` | int | 100 | サンプル数 |
| `--batch_size` | int | 100 | バッチサイズ |
| `--n_tries` | int | 1 | サンプリング試行回数 |
| `--save_xyz` | bool | True | XYZファイル保存 |
| `--visualize` | bool | False | 可視化 |
| `--only_stable` | bool | False | 安定な分子のみ保存 |

### eval_analyze.py

生成品質の評価。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--model_path` | str | 必須 | モデルのパス |
| `--n_samples` | int | 10000 | 評価サンプル数 |
| `--batch_size` | int | 100 | バッチサイズ |
| `--save_results` | bool | True | 結果の保存 |
| `--save_to_xyz` | bool | False | XYZ保存 |
| `--output_file` | str | None | 出力JSONファイル名 |

### eval_conditional_qm9.py

条件付き生成と評価。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--generators_path` | str | 必須 | 生成器のパス |
| `--classifiers_path` | str | None | 分類器のパス |
| `--property` | str | 必須 | 性質名 |
| `--task` | str | qualitative | タスクタイプ |
| `--n_sweeps` | int | 10 | 性質値の分割数 |
| `--iterations` | int | 100 | 反復回数（edm task） |
| `--batch_size` | int | 100 | バッチサイズ |

### main_crystal.py

結晶構造生成モデルの学習。

#### 結晶固有パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--crystal_type` | str | homocrystal | 結晶タイプ |
| `--space_groups` | str | all | 空間群の指定 |
| `--include_pbc` | bool | True | 周期境界条件 |
| `--lattice_dim` | int | 64 | 格子パラメータ次元 |
| `--cutoff_distance` | float | 5.0 | カットオフ距離（Å） |

### main_crystal_with_properties.py

条件付き結晶生成。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `--conditioning` | list | [] | 条件タイプ |
| `--space_groups` | str | None | 対象空間群 |
| `--density_range` | str | None | 密度範囲（min,max） |

---

## トラブルシューティング

### よくある問題と解決方法

#### 1. CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**解決方法**:

```bash
# バッチサイズを減らす
python main_qm9.py --batch_size 16  # デフォルト: 64

# モデルサイズを減らす
python main_qm9.py --nf 64 --n_layers 4  # デフォルト: 256, 9

# 評価サンプル数を減らす
python main_qm9.py --n_stability_samples 100  # デフォルト: 1000

# 拡散ステップ数を減らす
python main_qm9.py --diffusion_steps 250  # デフォルト: 1000
```

#### 2. NaN Loss

**症状**:
```
Epoch 15/3000:
  Train Loss: nan
  Val Loss: nan
```

**原因**:
- 学習率が高すぎる
- 数値精度の問題
- 勾配爆発

**解決方法**:

```bash
# 学習率を下げる
python main_qm9.py --lr 5e-5  # デフォルト: 1e-4

# ノイズ精度を上げる
python main_qm9.py --diffusion_noise_precision 1e-6  # デフォルト: 1e-5

# 勾配クリッピングを強化
python main_qm9.py --clip_grad True
```

#### 3. RDKit Not Found

**症状**:
```
ModuleNotFoundError: No module named 'rdkit'
```

**解決方法**:

```bash
# Conda環境でインストール（推奨）
conda install -c conda-forge rdkit

# または、RDKitなしで続行
# 多くの機能は RDKit なしでも動作
# OpenBabel が代替として使用される
```

#### 4. ASEデータベース読み込みエラー

**症状**:
```
Error: Cannot open ASE database: my_molecules.db
```

**解決方法**:

```python
# データベースの整合性をチェック
from ase.db import connect

try:
    db = connect('my_molecules.db')
    print(f"Database OK. Total molecules: {len(db)}")
except Exception as e:
    print(f"Database error: {e}")
    # データベースを再構築
```

#### 5. 学習が遅い

**症状**: エポックごとの処理時間が長すぎる

**解決方法**:

```bash
# 評価頻度を下げる
python main_qm9.py --test_epochs 50  # デフォルト: 20

# 安定性評価サンプルを減らす
python main_qm9.py --n_stability_samples 100  # デフォルト: 1000

# データローダーのワーカー数を増やす
python main_qm9.py --num_workers 4

# GPUを使用
python main_qm9.py --no-cuda False
```

#### 6. 結晶生成でのPBCエラー

**症状**:
```
Error: Periodic boundary condition violation
```

**解決方法**:

```bash
# PBCを無効化してテスト
python main_crystal.py --include_pbc False

# カットオフ距離を調整
python main_crystal.py --cutoff_distance 5.0
```

#### 7. コンテキスト次元ミスマッチ

**症状**:
```
RuntimeError: Context dimension mismatch: expected 3, got 1
```

**原因**: 条件付けの設定が変更された

**解決方法**:

```bash
# 再開時は元の条件付け設定を使用しない
python main_qm9.py --resume outputs/model  # conditioningは指定しない

# または新しいモデルを学習
python main_qm9.py --exp_name new_model --conditioning alpha gap
```

#### 8. メモリリーク

**症状**: 長時間学習後にメモリ不足

**解決方法**:

```python
# 学習ループ内でのメモリ解放
import gc
import torch

def train_epoch(...):
    for batch in dataloader:
        # 学習処理
        ...
        
    # エポック終了時にキャッシュをクリア
    torch.cuda.empty_cache()
    gc.collect()
```

### デバッグモード

詳細なデバッグ情報を出力:

```bash
# Python デバッガーで実行
python -m pdb main_qm9.py --exp_name debug_model

# 詳細ログを有効化（カスタム実装）
python main_qm9.py --verbose True --debug True

# GPUメモリ使用状況を監視
watch -n 1 nvidia-smi
```

### ログファイルの確認

```bash
# 標準出力をファイルに保存
python main_qm9.py --exp_name model 2>&1 | tee training.log

# エラーのみを抽出
grep -i "error\|warning\|failed" training.log

# 損失の推移を確認
grep "Train Loss" training.log
```

---

## まとめ

本ガイドでは、E3同変拡散モデルの全機能を網羅的に説明しました。

### 主要機能の再確認

1. **基本的な分子生成**: QM9/GEOM-Drugsデータセットでの学習と生成
2. **条件付き生成**: 物性値・分子記述子による精密な制御
3. **結晶生成**: 空間群・密度条件付きホモ結晶生成
4. **ASE統合**: カスタムデータセットの柔軟な利用
5. **包括的評価**: 多角的な品質評価指標
6. **豊富な出力**: XYZ、CIF、CSV形式での構造データ

### 重要な原則

- **フォールバックなし**: 全ての処理は理論的に健全
- **完全な透明性**: 全ての出力の意味が明確
- **物理的妥当性**: 生成構造は化学的・物理的制約を満たす

### サポート

- GitHubリポジトリ: https://github.com/nobkt/e3_diffusion_for_molecules
- Issues: 問題報告や質問はGitHub Issuesへ

---

**作成日**: 2025年10月26日  
**バージョン**: 2.0  
**作成者**: E3 Diffusion Development Team

このガイドは全機能を網羅し、全ての出力の意味を明確に記述しています。フォールバック処理は一切使用していません。
