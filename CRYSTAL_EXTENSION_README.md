# 分子性結晶生成への拡張 - ドキュメント概要
# Molecular Crystal Generation Extension - Documentation Overview

このドキュメントパッケージは、E(3)等変拡散モデル(EDM)を**ホモ結晶（同一分子からなる分子性結晶）**の生成に拡張するための包括的な仕様と設計を提供します。**単分子のEGNN特徴量を結晶生成に統合**することで、理論的に正しい結晶生成を実現します。

This documentation package provides comprehensive specifications and design for extending the E(3) Equivariant Diffusion Model (EDM) to support **homocrystal generation (molecular crystals composed of identical molecules)**. By **integrating single-molecule EGNN features**, we achieve theoretically sound crystal generation.

---

## 📚 ドキュメント構成 (Document Structure)

### 1. [MOLECULAR_CRYSTAL_SPECIFICATION.md](./MOLECULAR_CRYSTAL_SPECIFICATION.md)
**分子性結晶生成のための拡張仕様書**

このドキュメントは、プロジェクトの**要件定義**と**仕様**を詳細に記述しています。

#### 主な内容:
- **要件定義** (Section 1)
  - 機能要件: データ入力、モデル学習、サンプル生成、評価・検証
  - 非機能要件: パフォーマンス、スケーラビリティ、互換性、拡張性

- **データ形式仕様** (Section 2)
  - ASEデータベース形式の詳細
  - 内部データ表現（結晶構造、座標系）
  - 出力データ形式（ASE DB、CIF、XYZ）

- **システム設計概要** (Section 3)
  - 全体アーキテクチャ図
  - 各レイヤーの責務
  - 主要コンポーネントの説明

- **技術課題と解決策** (Section 4)
  - 周期境界条件の取り扱い
  - 格子パラメータの学習
  - E(3)等変性の保持
  - 空間群対称性の取り扱い
  - スケーラビリティ

- **実装フェーズ** (Section 5)
  - Phase 1: 基盤整備（2-3週間）
  - Phase 2: モデル拡張（3-4週間）
  - Phase 3: 条件付け（2-3週間）
  - Phase 4: 評価・検証（2週間）
  - Phase 5: 可視化・出力（1-2週間）

- **評価指標** (Section 6)
  - 構造的指標（格子パラメータ、原子配置、密度）
  - 対称性指標（空間群の再現性）
  - 物理的指標（安定性、パッキング効率）
  - 生成品質指標（多様性、新規性）

- **データセット要件** (Section 7)
  - 推奨データセット（CSD、Materials Project）
  - データセットの前処理

- **使用例** (Section 8)
  - 基本的な使用方法
  - 高度な使用例

- **リスクと制約** (Section 9)
  - 技術的リスク
  - データ制約
  - 物理的制約

- **将来の拡張** (Section 10)
  - 短期的拡張（ポリモルフ、共結晶）
  - 中期的拡張（対称性制約、エネルギー誘導）
  - 長期的拡張（無機結晶、表面・界面）

### 2. [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md)
**分子性結晶生成のための詳細設計書**

このドキュメントは、実装に必要な**具体的な設計**と**コード例**を提供しています。

#### 主な内容:
- **ディレクトリ構造** (Section 1)
  - 新規モジュール `crystal/` の詳細構成
  - 既存ファイルへの修正箇所
  - テストファイルの配置

- **データ処理層の設計** (Section 2)
  - **`crystal/data/crystal_loader.py`**
    - `CrystalDataset` クラスの実装
    - `collate_crystal_batch` 関数
    - 完全なコード例付き
  - **`crystal/data/periodic_utils.py`**
    - 最小イメージ距離計算
    - 座標変換（分数↔デカルト）
    - 周期的近傍リスト構築
    - 格子パラメータ変換
    - 全関数の完全実装例

- **モデル層の設計** (Section 3)
  - **`crystal/models/periodic_egnn.py`**
    - `PeriodicEGNN` クラス
    - `PeriodicEGNNLayer` クラス
    - 周期境界条件を考慮したメッセージパッシング
  - **`crystal/models/lattice_diffusion.py`**
    - `LatticeDiffusion` クラス
    - 格子パラメータの正規化/逆正規化
  - **`crystal/models/crystal_dynamics.py`**
    - 原子座標と格子パラメータを統合したモデル

- **条件付けモジュールの設計** (Section 4)
  - 空間群埋め込み (`space_group_embedding.py`)
  - 密度条件付け (`density_conditioning.py`)
  - 格子パラメータ条件付け

- **評価モジュールの設計** (Section 5)
  - `CrystalMetrics` クラス
  - 構造的メトリクス、妥当性チェック、分布比較

- **統合とインターフェース** (Section 6)
  - 既存の `qm9/dataset.py` への修正
  - 新規メインスクリプト `main_crystal.py`
  - 完全な使用例

- **テスト戦略** (Section 7)
  - ユニットテストの例
  - `test_periodic_utils.py` の実装

- **実装ロードマップ** (Section 8)
  - 11週間の詳細な実装計画
  - 各週のタスクとマイルストーン
  - チェックリスト形式

- **まとめと次のステップ** (Section 9)
  - 実装の優先順位
  - 成功基準
  - 開発プラクティス

---

## 🎯 主要な技術的特徴 (Key Technical Features)

### 1. ホモ結晶生成のための分子-結晶統合 ★NEW★
- **分離データセット**: molecules.db（単分子）+ crystals.db（結晶）の明確な分離
- **molecule_idリンク**: 理論的に正しい分子-結晶対応関係
- **ポリモルフ対応**: 1分子:N結晶の関係をネイティブサポート
- **ヒューリスティック不使用**: fallbackなしの厳格な実装

### 2. 単分子EGNN特徴量の統合 ★PRIMARY FEATURE★
- **MolecularEncoder**: 単分子からEGNN特徴量を自動抽出
- **幾何学的特徴**: 分子サイズ、体積、主軸方向の計算
- **MolecularConditioning**: 分子特徴量を結晶生成の主要条件として使用
- **事前学習済みモデル**: 既存の単分子EGNNを再利用可能

### 3. 周期境界条件のサポート
- **最小イメージ規約**: 周期境界を越えた原子間距離を正確に計算
- **周期的近傍リスト**: カットオフ半径内の全ての周期イメージを考慮
- **分数座標系**: 格子変形に対して不変な座標表現

### 4. 格子パラメータの学習
- **独立した拡散プロセス**: 原子座標とは異なるスケールと制約を持つ格子パラメータを別途処理
- **物理的制約**: 正の長さ、妥当な角度範囲を保証
- **正規化戦略**: log空間での長さ、sin/cos表現での角度
- **分子サイズとの整合性**: 分子特徴量から格子サイズを推定

### 5. E(3)等変性の拡張
- **周期的EGNN**: 既存のEGNNを周期系に拡張
- **格子共変性**: 格子変換に対して共変な層の導入
- **相対座標の使用**: 並進不変性の保持
- **分子内・分子間相互作用の分離**: 理論的に正しい相互作用表現

### 6. 条件付き生成
- **★ 分子EGNN特徴量（PRIMARY）**: 単分子の構造情報による条件付け
- **空間群**: 230種類の空間群による条件付け
- **密度**: 結晶密度による条件付け
- **格子パラメータ**: 特定の格子定数での生成

---

## 📊 実装フェーズと期間 (Implementation Phases and Timeline)

| フェーズ | 期間 | 主要タスク | 成果物 |
|---------|------|-----------|--------|
| **Phase 1: 基盤整備** | 2-3週間 | データローダー、周期境界ユーティリティ | `crystal/data/` モジュール |
| **Phase 2: モデル拡張** | 3-4週間 | Periodic EGNN、格子拡散 | `crystal/models/` モジュール |
| **Phase 3: 条件付け** | 2-3週間 | 空間群埋め込み、密度条件付け | `crystal/conditioning/` モジュール |
| **Phase 4: 評価・検証** | 2週間 | 評価メトリクス、妥当性チェック | `crystal/evaluation/` モジュール |
| **Phase 5: 可視化・出力** | 1-2週間 | 3D可視化、CIF出力 | `crystal/utils/` モジュール |
| **総計** | **10-14週間** | | **完全な結晶生成システム** |

---

## 🚀 クイックスタート (Quick Start)

### 1. ドキュメントの読み方

**初めての方:**
1. まず [MOLECULAR_CRYSTAL_SPECIFICATION.md](./MOLECULAR_CRYSTAL_SPECIFICATION.md) のSection 1-3を読んで全体像を把握
2. Section 4で技術的な課題と解決策を理解
3. Section 8の使用例を確認

**実装者の方:**
1. [MOLECULAR_CRYSTAL_DESIGN.md](./MOLECULAR_CRYSTAL_DESIGN.md) のSection 1でディレクトリ構造を確認
2. Section 2-6で各モジュールの詳細設計とコード例を参照
3. Section 8の実装ロードマップに従って開発

### 2. 推奨される実装順序

```
1. crystal/data/periodic_utils.py
   ↓ (座標変換、最小イメージ距離)
2. crystal/data/crystal_loader.py
   ↓ (データローダー)
3. crystal/models/periodic_egnn.py
   ↓ (周期的EGNN)
4. crystal/models/lattice_diffusion.py
   ↓ (格子拡散)
5. crystal/models/crystal_dynamics.py
   ↓ (統合モデル)
6. crystal/conditioning/*.py
   ↓ (条件付けモジュール)
7. crystal/evaluation/*.py
   ↓ (評価モジュール)
8. main_crystal.py
   ↓ (メインスクリプト)
9. テストとドキュメント
```

---

## 📖 コード例の参照方法 (How to Reference Code Examples)

各設計ドキュメントには、**そのまま使える完全なコード例**が含まれています:

### データ処理層の例:
```python
# MOLECULAR_CRYSTAL_DESIGN.md の Section 2.1 参照
from crystal.data.crystal_loader import CrystalDataset, collate_crystal_batch

dataset = CrystalDataset(
    db_path='crystals.db',
    indices=[0, 1, 2, ...],
    use_fractional_coords=True,
)
```

### 周期境界条件の例:
```python
# MOLECULAR_CRYSTAL_DESIGN.md の Section 2.2 参照
from crystal.data.periodic_utils import minimum_image_distance

distances, vectors = minimum_image_distance(
    positions1, positions2, cell_vectors, pbc,
    use_fractional=True
)
```

### モデルの例:
```python
# MOLECULAR_CRYSTAL_DESIGN.md の Section 3 参照
from crystal.models.crystal_dynamics import CrystalDynamics

model = CrystalDynamics(
    in_node_nf=num_atom_types,
    hidden_nf=128,
    n_layers=6,
    learn_lattice=True,
)
```

---

## 🧪 テスト戦略 (Testing Strategy)

### ユニットテストの例:
```python
# tests/test_periodic_utils.py
def test_minimum_image_distance():
    """最小イメージ距離のテスト"""
    # 詳細は MOLECULAR_CRYSTAL_DESIGN.md Section 7.1 参照
    ...
```

### 統合テスト:
- 小規模データセット（10-100結晶）で動作確認
- 各フェーズ完了時に統合テスト実行
- CI/CDパイプラインへの統合

---

## 📈 評価指標 (Evaluation Metrics)

### 構造的指標:
- 格子パラメータの精度（MAE、分布マッチング）
- 原子配置の妥当性（最小距離、配位数、RDF）
- 密度の精度

### 対称性指標:
- 空間群の再現性
- 対称性スコア

### 物理的指標:
- 分子の安定性
- パッキング効率
- エネルギーランドスケープ（オプション）

### 生成品質指標:
- 多様性スコア
- 新規性スコア

詳細は **MOLECULAR_CRYSTAL_SPECIFICATION.md Section 6** を参照。

---

## 🔧 実装時の注意点 (Implementation Notes)

### 1. 既存コードとの互換性
- 既存の分子生成機能を破壊しない
- `crystal_mode` フラグで動作を切り替え
- 既存のテストが全てパスすることを確認

### 2. 性能最適化
- カットオフ半径の適切な設定（10 Å推奨）
- バッチサイズの調整（結晶サイズに応じて）
- スパースグラフ表現の使用

### 3. 数値安定性
- 格子パラメータの正規化
- 角度計算でのクリッピング
- ゼロ除算の回避

### 4. デバッグ戦略
- 小規模データセット（10結晶）で開始
- 各コンポーネントを個別にテスト
- 可視化による確認（単位格子、原子配置）

---

## 🌟 期待される成果 (Expected Outcomes)

### Phase 1 完了時:
- ✅ ASEデータベースから結晶データを読み込める
- ✅ 周期境界条件下で正しい距離計算ができる
- ✅ 座標変換が正確に動作する

### Phase 2 完了時:
- ✅ 周期的EGNNが動作する
- ✅ 格子パラメータの拡散モデルが動作する
- ✅ 統合モデルで学習が開始できる

### Phase 3 完了時:
- ✅ 空間群で条件付けできる
- ✅ 密度で条件付けできる
- ✅ 条件付きサンプリングが可能

### Phase 4 完了時:
- ✅ 生成構造の妥当性を評価できる
- ✅ 参照データセットと比較できる
- ✅ 評価メトリクスが計算できる

### Phase 5 完了時:
- ✅ 結晶構造を3D可視化できる
- ✅ CIFファイルとして出力できる
- ✅ 完全な結晶生成パイプラインが動作する

### 最終成果:
- 🎉 **物理的に妥当な分子性結晶を生成できるシステム**
- 🎉 **条件付き生成により、望ましい特性を持つ結晶を設計できる**
- 🎉 **既存の分子生成機能と共存する統合システム**

---

## 📚 関連文献 (References)

詳細な参考文献リストは **MOLECULAR_CRYSTAL_SPECIFICATION.md Section 11** を参照。

主要論文:
1. Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D" (2022)
2. Jiao et al. "Crystal Diffusion Variational Autoencoder for Periodic Material Generation" (2023)
3. Xie et al. "Crystal Diffusion Generative Models" (2021)

---

## 💡 サポートとフィードバック (Support and Feedback)

実装中に問題が発生した場合:
1. まず該当するドキュメントセクションを再確認
2. ユニットテストを実行して問題を特定
3. 小規模データセットでデバッグ

このドキュメントは**実装の指針**として使用し、実装中に発見された問題に応じて更新してください。

---

## 📝 ドキュメントの更新履歴 (Document Revision History)

| 日付 | バージョン | 変更内容 |
|------|----------|---------|
| 2025-01-XX | 1.0 | 初版作成 - 仕様書と設計書を作成 |

---

**Good luck with your implementation! 🚀**

この拡張により、E(3)等変拡散モデルが分子だけでなく、結晶構造の生成にも対応できるようになります。
