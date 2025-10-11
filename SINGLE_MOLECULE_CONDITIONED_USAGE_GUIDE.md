# 単分子条件付き分子性結晶生成 - 使用ガイド
# Single-Molecule Conditioned Molecular Crystal Generation - Usage Guide

## 📖 概要 (Overview)

本ガイドでは、単分子の構造情報を条件として分子性結晶を生成するシステムの使用方法を説明します。

**This guide explains how to use the system for generating molecular crystals conditioned on single molecule structures.**

---

## 🚀 クイックスタート (Quick Start)

### 1. インストール (Installation)

```bash
# リポジトリのクローン
git clone https://github.com/nobkt/e3_diffusion_for_molecules.git
cd e3_diffusion_for_molecules

# 依存パッケージのインストール
pip install -r requirements.txt

# 追加の依存関係
pip install rdkit scipy
```

### 2. データセットの準備 (Dataset Preparation)

#### 2.1 分子データベースの作成

```python
from ase import Atoms
from ase.db import connect

# 分子データベースを作成
mol_db = connect('data/molecules.db')

# 例: ベンゼン分子
benzene = Atoms(
    symbols=['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[
        [0.000, 1.400, 0.000],   # C1
        [1.212, 0.700, 0.000],   # C2
        [1.212, -0.700, 0.000],  # C3
        [0.000, -1.400, 0.000],  # C4
        [-1.212, -0.700, 0.000], # C5
        [-1.212, 0.700, 0.000],  # C6
        [0.000, 2.480, 0.000],   # H1
        [2.148, 1.240, 0.000],   # H2
        [2.148, -1.240, 0.000],  # H3
        [0.000, -2.480, 0.000],  # H4
        [-2.148, -1.240, 0.000], # H5
        [-2.148, 1.240, 0.000],  # H6
    ]
)

# データベースに保存
# 注意: smiles と formula はオプション（人間の可読性のため）
# モデルの訓練・生成には原子座標と原子番号のみが使用されます
mol_id = mol_db.write(benzene, smiles='c1ccccc1', formula='C6H6')
print(f"Molecule ID: {mol_id}")

# または、smilesとformulaなしでも保存可能（完全に動作します）
# mol_id = mol_db.write(benzene)
```

> **📝 重要な注意事項 / Important Note:**
> 
> **日本語:** `smiles` と `formula` はオプションのメタデータです。モデルの訓練と生成に使用されるのは、原子の3D座標と原子番号のみです。RDKitを使用せず、OpenBabelなどの代替ツールで分子を扱う場合でも、SMILESなしでデータベースを作成できます。詳細は [DATASET_REQUIREMENTS_FAQ.md](./DATASET_REQUIREMENTS_FAQ.md) を参照してください。
> 
> **English:** `smiles` and `formula` are optional metadata. Only 3D atomic coordinates and atomic numbers are used for model training and generation. You can create databases without SMILES even when using alternative tools like OpenBabel instead of RDKit. See [DATASET_REQUIREMENTS_FAQ.md](./DATASET_REQUIREMENTS_FAQ.md) for details.

#### 2.2 結晶データベースの作成

```python
# 結晶データベースを作成
crystal_db = connect('data/crystals.db')

# 例: ベンゼン結晶 (4分子/単位格子)
benzene_crystal = Atoms(
    symbols=['C']*24 + ['H']*24,  # 4 molecules × 12 atoms
    positions=[...],  # 単位格子内の全原子座標
    cell=[
        [7.39, 0.0, 0.0],
        [0.0, 9.42, 0.0],
        [0.0, 0.0, 6.81]
    ],
    pbc=[True, True, True]
)

# molecule_idを指定して保存
crystal_db.write(
    benzene_crystal,
    molecule_id=mol_id,  # 対応する分子のID
    Z=4,                 # 単位格子あたりの分子数
    space_group=14,      # P21/c
    density=1.15,        # g/cm³
)
```

### 3. モデルの訓練 (Training)

#### 3.1 コマンドライン

```bash
python main_crystal.py \
    --exp_name benzene_crystal_gen \
    --dataset molecular_crystal \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --molecule_conditioning True \
    --conditioning_method film \
    --molecule_encoder_layers 6 \
    --molecule_encoder_hidden 256 \
    --molecule_encoding_dim 128 \
    --n_layers 9 \
    --nf 256 \
    --n_epochs 3000 \
    --batch_size 32 \
    --lr 1e-4 \
    --guidance_scale 2.0 \
    --diffusion_steps 1000 \
    --save_model outputs/models
```

#### 3.2 Python API

```python
from crystal_generation import train_conditional_model, load_paired_datasets

# データセット読み込み
datasets, dataset_info = load_paired_datasets(
    molecule_db_path='data/molecules.db',
    crystal_db_path='data/crystals.db',
    split_ratios=[0.8, 0.1, 0.1],
    remove_h=False,
)

# モデル訓練
model, mol_encoder = train_conditional_model(
    datasets=datasets,
    dataset_info=dataset_info,
    config={
        'exp_name': 'my_crystal_model',
        'molecule_encoding_dim': 128,
        'hidden_nf': 256,
        'n_layers': 9,
        'n_epochs': 3000,
        'batch_size': 32,
        'lr': 1e-4,
        'guidance_scale': 2.0,
    }
)

# モデル保存
torch.save({
    'model_state_dict': model.state_dict(),
    'mol_encoder_state_dict': mol_encoder.state_dict(),
    'dataset_info': dataset_info,
    'config': config,
}, 'outputs/models/crystal_model.pt')
```

### 4. 結晶の生成 (Generation)

#### 4.1 コマンドライン

```bash
python sample_crystal.py \
    --model_path outputs/models/crystal_model.pt \
    --molecule_db_path data/molecules.db \
    --molecule_ids 1,2,3 \
    --n_samples_per_molecule 10 \
    --guidance_scale 2.0 \
    --output_dir outputs/generated \
    --output_format cif \
    --save_visualization True
```

#### 4.2 Python API

```python
from crystal_generation import generate_crystals_from_molecules

# チェックポイント読み込み
checkpoint = torch.load('outputs/models/crystal_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
mol_encoder.load_state_dict(checkpoint['mol_encoder_state_dict'])

# 新しい分子から結晶を生成
from ase.db import connect
mol_db = connect('data/molecules.db')
molecule = mol_db.get(1).toatoms()  # 分子ID=1

# 結晶生成
generated_crystals = generate_crystals_from_molecules(
    molecules=[molecule],
    model=model,
    molecule_encoder=mol_encoder,
    n_samples_per_molecule=10,
    guidance_scale=2.0,
    Z=4,  # molecules per unit cell
)

# 保存
for i, crystal in enumerate(generated_crystals):
    crystal.write(f'outputs/generated/crystal_{i}.cif')
    crystal.write(f'outputs/generated/crystal_{i}.xyz')  # supercell展開版
```

---

## ❓ よくある質問 (Frequently Asked Questions)

### Q1: SMILESとformulaの情報は必須ですか？

**A: いいえ、必須ではありません。**

- モデルの訓練と生成には、原子の3D座標と原子番号のみが使用されます
- `smiles` と `formula` はオプションのメタデータで、人間がデータを理解しやすくするための補助情報です
- RDKitを使わずOpenBabelなどで分子を扱う場合でも、SMILESなしでデータベースを作成できます
- 詳細は [DATASET_REQUIREMENTS_FAQ.md](./DATASET_REQUIREMENTS_FAQ.md) を参照してください

### Q2: ASEデータベースに最低限必要な情報は？

**A: 以下の情報のみが必須です：**

**分子データベース:**
- 原子のシンボルまたは原子番号
- 原子の3D座標

**結晶データベース:**
- 原子のシンボルまたは原子番号
- 原子の3D座標
- セル情報（cell vectors, PBC）
- `molecule_id`（対応する分子のID）

**推奨メタデータ:** `Z`, `space_group`, `density`

### Q3: OpenBabelで分子を準備できますか？

**A: はい、完全に可能です。**

システムは3D座標と原子番号のみから動作するため、OpenBabelで分子を準備し、ASE Atomsオブジェクトに変換してデータベースに保存できます。RDKitやSMILESは不要です。

### Q4: データベースにカスタムメタデータを追加できますか？

**A: はい、ASEデータベースの任意のkey-valueペアを保存できます。**

```python
mol_id = mol_db.write(
    molecule,
    my_custom_field='value',
    energy=-123.45,
    source='my_calculation'
)
```

ただし、モデルが使用するのは3D構造情報のみで、カスタムメタデータは訓練には使用されません。

---

## 📊 詳細な使用例 (Detailed Examples)

### 例1: 密度を指定した生成

```python
# 密度条件付き生成
from crystal.conditioning import DensityConditioning

crystals = generate_crystals_with_density(
    molecule=toluene,
    target_density=1.2,  # g/cm³
    model=model,
    mol_encoder=mol_encoder,
    n_samples=10,
)
```

### 例2: 多形生成（異なる結晶構造）

```python
# 異なる初期化で複数の結晶多形を生成
polymorphs = []

for seed in range(10):
    torch.manual_seed(seed)
    
    crystal = generate_crystals_from_molecules(
        molecules=[molecule],
        model=model,
        molecule_encoder=mol_encoder,
        n_samples_per_molecule=1,
        guidance_scale=2.0,
        initial_cell=None,  # ランダム初期化
    )[0]
    
    polymorphs.append(crystal)

# 重複除去
unique_polymorphs = remove_duplicate_structures(polymorphs, threshold=0.1)
print(f"Found {len(unique_polymorphs)} unique polymorphs")
```

### 例3: バッチ処理

```python
# 複数分子から並列に結晶生成
molecules = [mol_db.get(i).toatoms() for i in range(1, 101)]  # 100分子

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def generate_batch(mol):
    return generate_crystals_from_molecules(
        molecules=[mol],
        model=model,
        molecule_encoder=mol_encoder,
        n_samples_per_molecule=5,
    )

with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    results = list(executor.map(generate_batch, molecules))

# 全結果を平坦化
all_crystals = [crystal for batch in results for crystal in batch]
print(f"Generated {len(all_crystals)} crystals total")
```

---

## 🔧 高度な設定 (Advanced Configuration)

### カスタム条件付け方法

```python
from crystal.models import ConditionalCrystalDynamics

# FiLM (Feature-wise Linear Modulation)
model = ConditionalCrystalDynamics(
    conditioning_method='film',  # デフォルト
    ...
)

# Additive conditioning
model = ConditionalCrystalDynamics(
    conditioning_method='add',
    ...
)

# Cross-attention conditioning
model = ConditionalCrystalDynamics(
    conditioning_method='cross_attention',
    ...
)
```

### ガイダンススケールの調整

```python
# 低いガイダンス: より多様だが入力分子からの逸脱が大きい
crystals_diverse = generate_crystals(guidance_scale=1.0)

# 標準ガイダンス: バランスが良い
crystals_balanced = generate_crystals(guidance_scale=2.0)

# 高いガイダンス: 入力分子に忠実だが多様性が低い
crystals_faithful = generate_crystals(guidance_scale=5.0)
```

---

## 📈 評価とメトリクス (Evaluation and Metrics)

### 生成結果の評価

```python
from crystal.evaluation import evaluate_generated_crystals

metrics = evaluate_generated_crystals(
    generated_crystals=generated_crystals,
    reference_crystals=test_crystals,
    input_molecules=test_molecules,
)

print("Evaluation Metrics:")
print(f"  Molecular Consistency Rate: {metrics['consistency_rate']:.2%}")
print(f"  Average RMSD: {metrics['avg_rmsd']:.3f} Å")
print(f"  Validity Rate: {metrics['validity_rate']:.2%}")
print(f"  Lattice MAE: {metrics['lattice_mae']:.3f}")
print(f"  Density MAE: {metrics['density_mae']:.3f} g/cm³")
```

### 個別メトリクスの計算

```python
from crystal.evaluation import (
    compute_molecular_consistency,
    compute_crystal_quality,
    compute_packing_efficiency,
)

# 分子一致性
consistency = compute_molecular_consistency(
    generated_crystal,
    input_molecule,
    tolerance=0.5,  # Å
)

# 結晶品質
quality = compute_crystal_quality(
    generated_crystal,
    min_distance_threshold=0.7,  # Å
)

# パッキング効率
packing_eff = compute_packing_efficiency(generated_crystal)
```

---

## 🐛 トラブルシューティング (Troubleshooting)

### 問題1: メモリ不足

**症状:** `RuntimeError: CUDA out of memory`

**解決策:**
```python
# バッチサイズを減らす
config['batch_size'] = 16  # デフォルト: 32

# より小さいモデルを使用
config['hidden_nf'] = 128  # デフォルト: 256
config['n_layers'] = 6     # デフォルト: 9

# Gradient accumulationを使用
config['gradient_accumulation_steps'] = 2
```

### 問題2: 訓練が収束しない

**症状:** 損失が減少しない

**解決策:**
```python
# 学習率を調整
config['lr'] = 5e-5  # より小さい学習率

# ウォームアップを追加
config['lr_warmup_steps'] = 1000

# 損失重みを調整
config['loss_weights'] = {
    'coord': 1.0,
    'feature': 1.0,
    'lattice': 0.1,
    'consistency': 0.5,
}
```

### 問題3: 生成される結晶が物理的に無効

**症状:** 原子間距離が小さすぎる

**解決策:**
```python
# より多くの拡散ステップを使用
config['diffusion_steps'] = 2000  # デフォルト: 1000

# 異なるノイズスケジュールを試す
config['diffusion_noise_schedule'] = 'cosine'  # or 'linear'

# 後処理で構造最適化
from ase.optimize import BFGS
optimized_crystal = optimize_structure(generated_crystal)
```

---

## 📚 参考資料 (References)

### ドキュメント
- [データセット要件FAQ](./DATASET_REQUIREMENTS_FAQ.md) - SMILESとformulaは必須？
- [理論説明書](./SINGLE_MOLECULE_CONDITIONED_CRYSTAL_THEORY.md) - 数式付き詳細理論
- [仕様書](./SINGLE_MOLECULE_CONDITIONED_CRYSTAL_SPEC.md) - システム仕様
- [既存の結晶生成ドキュメント](./MOLECULAR_CRYSTAL_SPECIFICATION.md)

### 論文
1. Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D" (ICML 2022)
2. Jiao et al. "Crystal Diffusion Variational Autoencoder" (2023)
3. Satorras et al. "E(n) Equivariant Graph Neural Networks" (ICML 2021)

### コード例
- [データローダー](./crystal/data/molecular_crystal_loader.py)
- [分子エンコーダー](./crystal/models/molecule_encoder.py)
- [条件付きダイナミクス](./crystal/models/conditional_crystal_dynamics.py)

---

## 💡 ベストプラクティス (Best Practices)

### データセット準備
1. **品質チェック**: 全データの原子座標と格子パラメータを検証
2. **対応確認**: 各結晶の `molecule_id` が正しく設定されているか確認
3. **化学式チェック**: 結晶の化学式が分子の Z 倍になっているか確認

### モデル訓練
1. **小規模実験**: まず100サンプルで10エポック訓練して動作確認
2. **検証頻度**: 10エポックごとにモデルを保存し、検証データで評価
3. **早期停止**: 検証損失が改善しない場合は訓練を停止

### サンプリング
1. **温度調整**: `guidance_scale` を 1.0〜5.0 で試す
2. **多様性確保**: 異なるランダムシードで複数生成
3. **後処理**: 生成後に妥当性チェックと軽い構造最適化

---

## 🎯 パフォーマンスチューニング (Performance Tuning)

### GPU使用の最適化

```python
# Mixed precision training
config['use_amp'] = True  # Automatic Mixed Precision

# Data loading
config['num_workers'] = 4
config['pin_memory'] = True

# Model parallelism (複数GPU)
config['dp'] = True  # DataParallel
```

### メモリ効率化

```python
# Gradient checkpointing
config['gradient_checkpointing'] = True

# Smaller batches with gradient accumulation
config['batch_size'] = 8
config['gradient_accumulation_steps'] = 4  # 実効バッチサイズ = 32
```

---

## 📞 サポート (Support)

### 問題報告
- GitHub Issues: https://github.com/nobkt/e3_diffusion_for_molecules/issues

### コミュニティ
- Discussions: https://github.com/nobkt/e3_diffusion_for_molecules/discussions

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-11  
**Status:** Ready for use
