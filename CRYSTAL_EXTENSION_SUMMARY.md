# 分子性結晶生成拡張 - プロジェクト要約
# Molecular Crystal Generation Extension - Project Summary

## 🎯 プロジェクト概要 (Project Overview)

このプロジェクトは、既存のE(3)等変拡散モデル(EDM)を**ホモ結晶（同一分子からなる分子性結晶）の生成**に拡張するための包括的な仕様と設計を提供します。重要な特徴として、**単分子のEGNN特徴量を結晶生成モデルに統合**することで、分子の構造情報を活用した理論的に正しい結晶生成を実現します。

This project provides comprehensive specifications and design for extending the existing E(3) Equivariant Diffusion Model (EDM) to support **homocrystal generation (molecular crystals composed of identical molecules)**. A key feature is the **integration of single-molecule EGNN features into the crystal generation model**, enabling theoretically sound crystal generation that leverages molecular structural information.

---

## 📚 ドキュメント一覧 (Documentation Index)

### 1. 📖 [CRYSTAL_EXTENSION_README.md](./CRYSTAL_EXTENSION_README.md) - **START HERE**
**概要とナビゲーションガイド**

全てのドキュメントの概要を提供し、読者を適切なセクションに案内します。

- ドキュメント構成の説明
- 主要な技術的特徴のサマリー
- クイックスタートガイド
- 実装フェーズのタイムライン
- コード例の参照方法

**推奨**: まずこのドキュメントから読み始めてください。

---

### 2. 📋 [MOLECULAR_CRYSTAL_SPECIFICATION.md](./MOLECULAR_CRYSTAL_SPECIFICATION.md)
**要件仕様書 - Requirements Specification**

プロジェクトの要件と仕様を詳細に定義します。

#### 主要セクション:
- **Section 1**: 要件定義（機能要件・非機能要件）
- **Section 2**: データ形式仕様
- **Section 3**: システム設計概要
- **Section 4**: 主要な技術課題と解決策
- **Section 5**: 実装フェーズ
- **Section 6**: 評価指標
- **Section 7**: データセット要件
- **Section 8**: 使用例
- **Section 9**: リスクと制約
- **Section 10**: 将来の拡張

**対象読者**: プロジェクトマネージャー、アーキテクト、要件定義担当者

**文書サイズ**: 約750行、29KB

---

### 3. 🏗️ [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md)
**詳細設計書 - Detailed Design Document**

実装に必要な具体的な設計とコード例を提供します。

#### 主要セクション:
- **Section 1**: ディレクトリ構造
- **Section 2**: データ処理層の設計（完全なコード例付き）
  - CrystalDataset クラス
  - 周期境界条件ユーティリティ
  - 座標変換関数
- **Section 3**: モデル層の設計（完全なコード例付き）
  - Periodic EGNN
  - Lattice Diffusion
  - Crystal Dynamics
- **Section 4**: 条件付けモジュール
- **Section 5**: 評価モジュール
- **Section 6**: 統合とインターフェース
- **Section 7**: テスト戦略
- **Section 8**: 実装ロードマップ（11週間の詳細計画）
- **Section 9**: まとめと次のステップ

**対象読者**: 開発者、実装担当者

**文書サイズ**: 約2,200行、72KB

**特徴**: すぐに使える完全なPythonコード例を多数含む

---

### 4. 🏛️ [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md)
**アーキテクチャ図 - Architecture Diagrams**

システム全体の構造を視覚的に表現します。

#### 主要内容:
- システム全体のアーキテクチャ図
- データフロー（学習フェーズ・サンプリングフェーズ）
- モジュール間の依存関係
- 座標系の変換フロー
- 周期境界条件の処理方法
- 格子パラメータの表現方法
- 実装の流れ（週ごと）

**対象読者**: 全てのステークホルダー（視覚的理解が必要な場合）

**文書サイズ**: 約500行、22KB

**特徴**: ASCIIアートによる詳細な図解

---

## 🎯 主要な技術的貢献 (Key Technical Contributions)

### 1. ホモ結晶生成のための分子-結晶データセット統合 (Molecule-Crystal Dataset Integration for Homocrystals)

**課題**: 既存のEDMは単一分子のみを扱うため、分子性結晶の生成には不十分。

**解決策**:
- **分離データセット**: molecules.db（単分子）+ crystals.db（結晶）
- **molecule_idによるリンク**: 理論的に正しい対応関係の管理
- **ポリモルフ対応**: 1分子:N結晶の関係をネイティブサポート
- **ヒューリスティック不使用**: fallbackなしの厳格な実装

**実装場所**: `crystal/data/molecule_loader.py`, `crystal/data/molecule_crystal_mapper.py`

---

### 2. 単分子EGNN特徴量の統合 (Single-Molecule EGNN Feature Integration)

**課題**: 結晶生成時に構成分子の構造情報を活用する必要がある。

**解決策**:
- **MolecularEncoder**: 単分子のxyz座標からEGNN特徴量を抽出
- **幾何学的特徴**: 分子サイズ、体積、主軸方向を計算
- **MolecularConditioning**: 分子特徴量を結晶生成の主要条件として使用
- **事前学習済みモデル対応**: 既存の単分子EGNNモデルを再利用可能

**実装場所**: `crystal/models/molecular_encoder.py`, `crystal/conditioning/molecular_conditioning.py`

---

### 3. 周期境界条件のサポート (Periodic Boundary Conditions)

**課題**: 既存のEDMはユークリッド空間を前提としており、周期性のあるトーラス空間には対応していない。

**解決策**:
- 最小イメージ規約（Minimum Image Convention）の実装
- 周期的近傍リストの構築
- 分数座標系の導入

**実装場所**: `crystal/data/periodic_utils.py`

---

### 4. 格子パラメータの学習 (Lattice Parameter Learning)

**課題**: 格子パラメータ (a, b, c, α, β, γ) は原子座標とは異なるスケールと制約を持つ。

**解決策**:
- 独立した拡散プロセス（原子座標とは分離）
- log空間での長さ、sin/cos表現での角度
- 物理的制約の適用（正の長さ、妥当な角度範囲）

**実装場所**: `crystal/models/lattice_diffusion.py`

---

### 3. E(3)等変性の拡張 (E(3) Equivariance Extension)

**課題**: 周期的な系では、並進対称性が格子ベクトルの整数倍に離散化される。

**解決策**:
- 相対座標の使用（絶対座標ではなく）
- 格子共変層の導入
- 周期境界を考慮したメッセージパッシング

**実装場所**: `crystal/models/periodic_egnn.py`

---

### 5. 条件付き生成 (Conditional Generation)

**新機能**:
- **★ 分子EGNN特徴量（PRIMARY）**: 単分子の構造情報を結晶生成に反映
- **空間群**: 230種類の空間群による条件付け
- **密度**: 結晶密度による条件付け
- **格子パラメータ**: 特定の格子定数での生成
- **分子特性**: 既存の分子記述子も利用可能

**実装場所**: `crystal/conditioning/`

---

## 📊 実装スケジュール (Implementation Schedule)

### 総期間: 10-14週間 (Full Timeline: 10-14 weeks)

```
Week 1-2:  Data Foundation (データ基盤)
  ├─ crystal/data/periodic_utils.py
  └─ crystal/data/crystal_loader.py

Week 3-5:  Model Core (モデルコア)
  ├─ crystal/models/periodic_egnn.py
  ├─ crystal/models/lattice_diffusion.py
  └─ crystal/models/crystal_dynamics.py

Week 6-7:  Diffusion Integration (拡散統合)
  └─ equivariant_diffusion/en_diffusion.py (modify)

Week 8-9:  Conditioning & Evaluation (条件付けと評価)
  ├─ crystal/conditioning/*.py
  └─ crystal/evaluation/*.py

Week 10-11: Integration & Testing (統合とテスト)
  ├─ main_crystal.py
  ├─ eval_crystal.py
  └─ tests/*.py
```

---

## 💡 使用方法 (How to Use This Documentation)

### シナリオ 1: プロジェクトの概要を理解したい

1. [CRYSTAL_EXTENSION_README.md](./CRYSTAL_EXTENSION_README.md) を読む
2. [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md) で視覚的に確認
3. [MOLECULAR_CRYSTAL_SPECIFICATION.md](./MOLECULAR_CRYSTAL_SPECIFICATION.md) のSection 1-3 を読む

**所要時間**: 1-2時間

---

### シナリオ 2: 実装を開始したい

1. [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md) のSection 1 でディレクトリ構造を確認
2. Section 8 の実装ロードマップを確認
3. Week 1のタスク（`crystal/data/periodic_utils.py`）から開始
4. 各関数の実装例を参照しながらコーディング
5. Section 7 のテスト例を参考にユニットテストを作成

**推奨**: 週ごとに進捗を確認し、各フェーズ終了時に統合テストを実行

---

### シナリオ 3: 特定の技術的問題を解決したい

#### 周期境界条件について:
- [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md) Section 2.2
- [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md) の周期境界条件セクション

#### 格子パラメータの学習について:
- [MOLECULAR_CRYSTAL_SPECIFICATION.md](./MOLECULAR_CRYSTAL_SPECIFICATION.md) Section 4.2
- [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md) Section 3.2

#### E(3)等変性の保持について:
- [MOLECULAR_CRYSTAL_SPECIFICATION.md](./MOLECULAR_CRYSTAL_SPECIFICATION.md) Section 4.3
- [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md) Section 3.1

---

## 🔍 コード例のクイックリファレンス (Code Example Quick Reference)

### データローダー:
```python
# 詳細: MOLECULAR_CRYSTAL_DESIGN.md Section 2.1-2.3
from crystal.data.molecule_loader import MoleculeDataset
from crystal.data.crystal_loader import CrystalDataset
from crystal.data.molecule_crystal_mapper import MoleculeCrystalMapper

# 分子データセット
molecule_dataset = MoleculeDataset(
    db_path='molecules.db',
    remove_h=False,
)

# マッパー
mapper = MoleculeCrystalMapper()
mapper.build_from_databases('molecules.db', 'crystals.db')

# 結晶データセット（分子データと連携）
crystal_dataset = CrystalDataset(
    db_path='crystals.db',
    indices=[0, 1, 2, ...],
    molecule_dataset=molecule_dataset,
    molecule_crystal_mapper=mapper,
    use_fractional_coords=True,
)
```

### 分子EGNN特徴量抽出:
```python
# 詳細: MOLECULAR_CRYSTAL_DESIGN.md Section 3.0
from crystal.models.molecular_encoder import MolecularEncoder

encoder = MolecularEncoder(
    in_node_nf=num_atom_types,
    hidden_nf=128,
    global_feature_dim=128,
    pretrained_path='pretrained_molecule_egnn.pt'  # オプション
)

# 分子特徴量を抽出
molecular_features = encoder(
    h=molecule_one_hot,  # [batch, n_atoms, num_types]
    x=molecule_positions,  # [batch, n_atoms, 3]
    node_mask=molecule_mask
)
# Returns: global_features, mol_size, mol_volume, principal_axes
```

### 分子特徴量条件付け:
```python
# 詳細: MOLECULAR_CRYSTAL_DESIGN.md Section 4.0
from crystal.conditioning.molecular_conditioning import MolecularConditioning

mol_cond = MolecularConditioning(
    molecular_feature_dim=128,
    conditioning_dim=256,
    use_geometry=True
)

# 分子特徴量を条件付けベクトルに変換
conditioning_vector = mol_cond(molecular_features)
# [batch, 256]
```

### 最小イメージ距離:
```python
# 詳細: MOLECULAR_CRYSTAL_DESIGN.md Section 2.2
from crystal.data.periodic_utils import minimum_image_distance

distances, vectors = minimum_image_distance(
    positions1, positions2, cell_vectors, pbc
)
```

### 周期的EGNN:
```python
# 詳細: MOLECULAR_CRYSTAL_DESIGN.md Section 3.1
from crystal.models.periodic_egnn import PeriodicEGNN

model = PeriodicEGNN(
    in_node_nf=num_atom_types,
    hidden_nf=128,
    n_layers=6,
)
```

### 学習スクリプト:
```python
# 詳細: MOLECULAR_CRYSTAL_DESIGN.md Section 6.2
python main_crystal.py \
    --ase_db_path crystals.db \
    --nf 128 --n_layers 6 \
    --learn_lattice True \
    --exp_name my_crystal_exp
```

---

## 📈 成功基準 (Success Criteria)

### Phase 1 完了基準:
- ✅ ASEデータベースから結晶データを読み込める
- ✅ 周期境界条件下で正しい距離計算ができる
- ✅ 座標変換（分数↔デカルト）が正確に動作する
- ✅ 全ての関数がユニットテストをパスする

### Phase 2 完了基準:
- ✅ 周期的EGNNが動作し、勾配が正しく計算される
- ✅ 格子拡散モデルが動作する
- ✅ 統合モデルで学習が開始できる
- ✅ 小規模データセット（10結晶）で収束する

### Phase 3 完了基準:
- ✅ 空間群で条件付けした学習ができる
- ✅ 密度で条件付けした学習ができる
- ✅ 条件付きサンプリングが可能

### Phase 4 完了基準:
- ✅ 生成構造の妥当性を評価できる（最小距離、密度など）
- ✅ 参照データセットと統計的に比較できる
- ✅ 評価レポートを自動生成できる

### Phase 5 完了基準:
- ✅ 結晶構造を3D可視化できる
- ✅ CIFファイルとして出力できる
- ✅ エンドツーエンドのパイプラインが動作する

### 最終成功基準:
- 🎉 **生成された結晶の80%以上が物理的に妥当**
- 🎉 **参照データセットとの分布がWasserstein距離で近い**
- 🎉 **1000ステップの拡散を10秒以内で実行（GPU使用時）**
- 🎉 **既存の分子生成機能が全て正常に動作**

---

## 🚀 次のステップ (Next Steps)

### 1. 環境準備
```bash
# 必要なパッケージのインストール
pip install ase torch numpy scipy

# リポジトリのクローン（既に完了している場合はスキップ）
cd /path/to/e3_diffusion_for_molecules
```

### 2. データセットの準備
- ASEデータベース形式で結晶構造を準備
- 推奨: 小規模データセット（10-100結晶）から開始
- データベースの内容を確認:
```python
from ase.db import connect
db = connect('crystals.db')
print(f'Total structures: {len(db)}')
```

### 3. 実装開始
- [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md) Section 8 の Week 1 タスクから開始
- `crystal/data/periodic_utils.py` の実装
- ユニットテストを並行して作成

### 4. 継続的な検証
- 各関数を実装後、すぐにテスト
- 小規模データセットで動作確認
- 問題があれば該当ドキュメントセクションを再確認

---

## 📞 サポート (Support)

### ドキュメントの構成が分からない場合:
→ [CRYSTAL_EXTENSION_README.md](./CRYSTAL_EXTENSION_README.md) を参照

### 技術的な詳細が必要な場合:
→ [MOLECULAR_CRYSTAL_SPECIFICATION.md](./MOLECULAR_CRYSTAL_SPECIFICATION.md) または [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md)

### 視覚的な理解が必要な場合:
→ [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md)

### コード例が必要な場合:
→ [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md) の各セクション

---

## 📊 ドキュメント統計 (Documentation Statistics)

| ドキュメント | 行数 | サイズ | 主な対象読者 |
|------------|-----|-------|------------|
| CRYSTAL_EXTENSION_README.md | 383 | 13KB | 全員（導入） |
| MOLECULAR_CRYSTAL_SPECIFICATION.md | 756 | 29KB | PM、アーキテクト |
| MOLECULAR_CRYSTAL_DESIGN.md | 2,172 | 72KB | 開発者 |
| ARCHITECTURE_DIAGRAM.md | 497 | 22KB | 全員（視覚的理解） |
| **合計** | **3,808** | **136KB** | |

**完全なコード例**: 10以上の主要コンポーネント
**実装期間**: 10-14週間
**主要技術**: E(3) Equivariance, Periodic Boundaries, Diffusion Models, Crystal Structure Generation

---

## 🌟 期待される成果 (Expected Outcomes)

このプロジェクトの完了により、以下が達成されます:

1. **技術的成果**:
   - 周期境界条件を正しく扱える拡散モデル
   - 物理的に妥当な分子性結晶を生成できるシステム
   - 条件付き生成により、望ましい特性を持つ結晶を設計可能

2. **学術的貢献**:
   - E(3)等変拡散モデルの周期系への拡張
   - 格子パラメータと原子座標の同時学習手法
   - 結晶生成のための新しい評価メトリクス

3. **実用的価値**:
   - 新材料設計の加速
   - 結晶多形（ポリモルフ）の予測
   - 創薬における結晶形探索の効率化

---

## 📝 更新履歴 (Revision History)

| 日付 | バージョン | 変更内容 |
|------|----------|---------|
| 2025-01-XX | 1.0 | 初版作成 - 全ドキュメントを統合したサマリー |

---

**Good luck with your implementation! 🚀**

このドキュメントパッケージが、分子性結晶生成システムの実装に役立つことを願っています。
