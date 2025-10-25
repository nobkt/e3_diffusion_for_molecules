# E3 Diffusion for Molecules - 日本語チュートリアル

このディレクトリには、E3同変拡散モデルを使用した分子・結晶生成のための包括的な日本語チュートリアルが含まれています。

## チュートリアル概要

| チュートリアル | トピック | 所要時間 | 前提知識 | セル数 |
|--------------|---------|---------|---------|--------|
| 01 | 基本的な分子生成 | 15-20分 | なし | 20 |
| 02 | 条件付き生成 | 20-30分 | チュートリアル01 | 23 |
| 03 | 結晶生成 | 30分 | チュートリアル01 | 20 |
| 04 | 分子記述子とASE | 20分 | チュートリアル01 | 24 |
| 05 | 評価と解析 | 25分 | チュートリアル01 | 30 |
| 06 | 高度な結晶条件付け | 30分 | チュートリアル03 | 24 |

**合計**: 141セル、約2-3時間の学習内容

## クイックスタート

### 環境セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/nobkt/e3_diffusion_for_molecules.git
cd e3_diffusion_for_molecules

# 依存関係のインストール
pip install -r requirements.txt

# オプション: RDKitの可視化用
conda install -c conda-forge rdkit

# Jupyter Notebookの起動
jupyter notebook tutorials/
```

### チュートリアルの実行順序

1. **01から順番に実行**: 段階的な学習のため、番号順に進めてください
2. **全セルを順次実行**: 上から下へ順番にコードセルを実行してください
3. **パラメータの調整**: 実験のために自由にパラメータを変更できます

## チュートリアル詳細

### チュートリアル01: 基本的な分子生成

**学習内容**:
- QM9データセットの読み込み
- E(3)同変拡散モデルの構築
- モデルの学習
- 新しい分子のサンプリング
- 生成品質の評価

**主要コード例**:
```python
from qm9 import dataset
from qm9.models import get_model

dataloaders = dataset.retrieve_dataloaders(args)
model, nodes_dist, _ = get_model(args, device, dataset_info)

# 生成
x, h = model.sample(n_samples, n_nodes, node_mask, edge_mask)
```

**出力**: 安定性メトリクスを持つ生成された3D分子

---

### チュートリアル02: 条件付き生成

**学習内容**:
- 性質の正規化
- 条件付きモデルの学習
- ターゲット性質での生成
- 性質スイープ
- 複数性質での条件付け
- Classifier-Free Guidance

**主要コード例**:
```python
from qm9.utils import prepare_context, compute_mean_mad

# 性質の正規化
property_norms = compute_mean_mad(dataloaders, ['alpha'], 'qm9')

# 条件付き生成
context = prepare_context(['alpha'], data, property_norms)
x, h = model.sample(..., context=context)
```

**出力**: 制御された性質を持つ分子

---

### チュートリアル03: 結晶生成

**学習内容**:
- ASEデータベースを使用した結晶データ
- 周期境界条件の実装
- 分子特徴抽出
- ユニットセルの学習
- CIFファイルのエクスポート

**主要コード例**:
```python
from crystal.models import CrystalDynamics
from crystal.conditioning import MolecularConditioning
from ase.io import write

# 結晶生成
crystal = crystal_model.sample(..., context=context, pbc=[True,True,True])

# CIFエクスポート
write('output.cif', atoms)
```

**出力**: CIF形式の結晶構造

---

### チュートリアル04: 分子記述子とASE

**学習内容**:
- ASEデータベースの完全な作成手順（厳密な原子座標）
- 分子記述子の計算（分子量、官能基、π共役比）
- 14種類の官能基のSMARTSパターンマッチング
- OpenBabel統合による高度な記述子抽出
- カスタムデータセットでの学習例
- 厳密な条件付き生成の検証

**特徴**:
- **フォールバックなし**: 全ての処理は厳密
- **SMARTS パターン**: 14種類の官能基を正確に検出
- **検証**: 生成された分子の記述子を厳密にチェック

**主要コード例**:
```python
from ase import Atoms
from ase.db import connect
from rdkit import Chem

# データベース作成
db = connect('molecules.db')
atoms = Atoms('C6H6', positions=...)
db.write(atoms, data={'molecular_weight': 78.0})

# 官能基検出
pattern = Chem.MolFromSmarts('[OX2H]')  # ヒドロキシル基
matches = mol.GetSubstructMatches(pattern)
```

**出力**: 記述子による条件付き生成

---

### チュートリアル05: 評価と解析

**学習内容**:
- 安定性メトリクスの詳細計算（原子・分子レベル）
- 許容結合数の厳密な適用
- RDKit統合による化学的妥当性検証
- SMILES重複除去による一意性評価
- 新規性アルゴリズム
- プロパティ分布解析（分子量、logP、原子数、結合数）
- 包括的評価パイプライン
- 品質ベンチマーク（優秀/良好/要改善）

**評価指標**:
- **原子安定性**: 結合数の妥当性チェック
- **分子安定性**: 化学的妥当性
- **一意性**: 重複分子の検出
- **新規性**: 訓練データとの比較

**主要コード例**:
```python
from qm9.analyze import check_stability, analyze_stability_for_molecules

# 安定性チェック
atom_stable, mol_stable, validity = check_stability(
    positions, atom_types, charges, dataset_info
)

# 包括的評価
python eval_analyze.py --model_path outputs/model --n_samples 10000
```

**出力**: 品質メトリクスと可視化

---

### チュートリアル06: 高度な結晶条件付け

**学習内容**:
- 複合条件付けの完全実装（分子+空間群+密度）
- 7つの結晶系と230個全空間群のサポート
- 空間群埋め込みの厳密な検証
- 物理的妥当性を考慮した密度ターゲティング
- 多形生成アルゴリズムの詳細実装
- Wyckoff位置の理論と取り扱い
- 対称性解析の詳細（5項目の検証）
- 高度なCIF操作（標準フォーマット準拠）
- 結晶構造の可視化手法
- 段階的学習戦略（3ステージ）

**230個の空間群**:
- **三斜晶系** (1-2): 最も低い対称性
- **単斜晶系** (3-15): β≠90°の制約
- **斜方晶系** (16-74): 直交軸
- **正方晶系** (75-142): a=b≠c
- **三方晶系** (143-167): α=β=γ≠90°
- **六方晶系** (168-194): γ=120°
- **立方晶系** (195-230): 最高対称性

**主要コード例**:
```python
# 複合条件付け
mol_context = mol_conditioning(mol_features)
sg_context = sg_embedding(space_group)
dens_context = density_conditioning(density)
context = torch.cat([mol_context, sg_context, dens_context], dim=-1)

# 多形生成
crystal = model.sample(..., context=context)

# 対称性検証
validation_results = validate_symmetry(atoms, space_group_number)
```

**出力**: 制御された結晶構造（空間群と密度を満たす）

---

## カバーされている機能

### 分子生成 (チュートリアル01-02, 04-05)
- ✅ 基本的な無条件生成
- ✅ 性質による条件付き生成
- ✅ 複数性質での条件付け
- ✅ 厳密な条件付き生成
- ✅ 分子記述子による条件付け
- ✅ QM9、GEOM-Drugs、ASEデータセット

### 結晶生成 (チュートリアル03, 06)
- ✅ ホモ結晶生成
- ✅ 分子特徴抽出
- ✅ 空間群条件付け（230個全て）
- ✅ 密度条件付け
- ✅ ユニットセルパラメータ学習
- ✅ 周期境界条件
- ✅ CIFファイルエクスポート

### 評価 (チュートリアル05)
- ✅ 安定性メトリクス
- ✅ RDKit妥当性
- ✅ 一意性
- ✅ 新規性
- ✅ 構造検証
- ✅ 結晶メトリクス

## 設計原則

全てのチュートリアルは、以下の核心原則に従っています:

### 1. フォールバックなし（ごまかしのためのfallbackは絶対にしない）
- 全てのエラーケースを明示的に処理
- 明確なエラーメッセージ
- 無言の修正なし
- 厳密な入力検証

### 2. 理論的健全性
- E(3)同変性の維持
- 適切なPBC処理
- 物理的制約の強制
- 数学的厳密性

### 3. 実用的使いやすさ
- 実行可能なコードスニペット
- 明確な説明
- 実例
- パフォーマンスのヒント

## よくある問題

### CUDA Out of Memory
```python
# バッチサイズを減らす
args.batch_size = 16  # またはそれ以下

# モデルサイズを減らす
args.nf = 64
args.n_layers = 4
```

### RDKit Not Found
```bash
# condaでインストール
conda install -c conda-forge rdkit

# またはRDKitなしで継続（一部機能が無効）
```

### 学習が遅い
```python
# テストサンプル数を減らす
args.n_stability_samples = 100

# 評価頻度を下げる
args.test_epochs = 10
```

## 学習のヒント

1. **シンプルから始める**: チュートリアル01から始め、基礎を理解
2. **コードを実行**: 各セルを実行し、出力を観察
3. **実験**: パラメータを変更し、効果を確認
4. **コメントを読む**: コードコメントは重要な概念を説明
5. **エラーを確認**: エラーメッセージは有益、フォールバックではない
6. **スケールアップ**: 速度のため小規模なモデル/データセットから始める

## 本番使用

チュートリアルでは、迅速な学習のため小規模なデータセットとモデルを使用します。本番環境では:

```bash
# 完全なQM9学習
python main_qm9.py \
    --exp_name production \
    --n_epochs 1000 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 9 \
    --diffusion_steps 1000 \
    --lr 1e-4 \
    --ema_decay 0.9999
```

**学習時間**: V100 GPU で 2-3日

## 追加リソース

- **ドキュメント**: `../doc/`
  - `complete_theory_ja.md`: 数学的基礎（完全版）
  - `complete_user_guide_ja.md`: 完全使用ガイド
  - `continuation_plan_ja.md`: 継続作成計画

- **サンプルスクリプト**:
  - `../main_qm9.py`: QM9学習
  - `../main_crystal.py`: 結晶学習
  - `../eval_analyze.py`: 評価
  - `../eval_conditional_qm9.py`: 条件付き生成

- **テスト**: `../tests/`
  - 全モジュールの単体テスト
  - 統合テスト
  - 使用例パターン

## サポート

質問や問題がある場合:
1. チュートリアルのコメントとドキュメントを確認
2. エラーメッセージを確認（有用に設計されています）
3. `../doc/complete_user_guide_ja.md`のユーザーマニュアルを参照
4. 詳細を含むGitHub Issueを開く

## 引用

研究でこのコードを使用する場合は、以下を引用してください:

```bibtex
@article{hoogeboom2022equivariant,
  title={Equivariant Diffusion for Molecule Generation in 3D},
  author={Hoogeboom, Emiel and Satorras, V{\'\i}ctor Garcia and Vignac, Cl{\'e}ment and Welling, Max},
  journal={ICML},
  year={2022}
}
```

---

**バージョン**: 2.0  
**作成日**: 2025年10月25日  
**ステータス**: 全6チュートリアル完成（フェーズ1完了）

**Happy Learning! 🚀 頑張ってください！**
