# 物性値を条件とした分子性結晶生成 - 詳細ドキュメント

## 概要

このディレクトリには、物性値を条件として分子性結晶（ホモ結晶）を生成するシステムの詳細ドキュメントが含まれています。

これらのドキュメントは、PR#143および`doc/molecular_crystal_generation_spec.md`のQ4における物性値を条件とした分子性結晶生成に関する議論を基に作成されています。

## ドキュメント構成

### 1. theory.md - 詳細理論説明書

**対象読者**: 研究者、開発者、データサイエンティスト

**内容**:
- 拡散モデルの数学的基礎
- E(3)等変性の理論
- 条件付き拡散モデルの定式化
- PropertyConditioningの理論的根拠
- 物性値条件付けの数学的定式化
- 多重条件統合の理論

**ページ数**: 約650行

### 2. specification.md - 詳細仕様書

**対象読者**: システム設計者、実装者、テスター

**内容**:
- システム要件（ハードウェア、ソフトウェア）
- 機能仕様
  - データ読み込み機能
  - PropertyConditioning モジュール仕様
  - ExtendedCombinedConditioning モジュール仕様
  - 訓練・生成機能仕様
- データ仕様（データベース構造、CSV形式）
- インターフェース仕様（CLI、Python API）
- アルゴリズム仕様
- 性能要件
- 制約事項
- 検証要件

**ページ数**: 約1470行

### 3. design.md - 詳細設計書

**対象読者**: 実装者、レビュアー、保守担当者

**内容**:
- アーキテクチャ設計（5層構成）
- モジュール設計
  - PropertyConditioning
  - ExtendedCombinedConditioning
  - CrystalDatasetWithProperties
  - PropertyNormalizer
- クラス設計（UMLダイアグラム、実装詳細）
- データフロー設計（訓練時、生成時）
- インターフェース設計
- 実装ガイドライン
  - コーディング規約
  - エラーハンドリング
  - ロギング
  - テスト
- テスト設計
- デプロイメント設計

**ページ数**: 約1700行

## 主要な特徴

### 1. 理論的正当性

すべてのドキュメントにおいて、**ヒューリスティックなfallbackを絶対に使用しない**という原則を貫いています。すべての処理は理論的に定義され、数学的根拠を持っています。

### 2. E(3)等変性の保証

結晶構造生成において、E(3)（ユークリッド群）等変性を厳密に保つ設計となっています。座標系の選択に依存しない一貫した予測を保証します。

### 3. 物性値の直接的条件付け

PropertyConditioningモジュールにより、物性値（バンドギャップ、融点、誘電率など）を直接的な生成条件として使用できます。これにより、目標物性値を満たす結晶を効率的に生成可能です。

### 4. 多重条件の統合

ExtendedCombinedConditioningにより、以下の条件を統一的に扱います：
- **分子条件**: 分子の3D構造から抽出されるEGNN特徴量
- **結晶条件**: 空間群、密度、格子定数
- **物性値条件**: バンドギャップ、融点など（新規）

## 使用方法

### ドキュメントの読み方

1. **研究者・理論的背景を理解したい方**
   - `theory.md` を最初に読む
   - 数学的定式化と理論的根拠を理解

2. **システム設計者・全体像を把握したい方**
   - `specification.md` を最初に読む
   - システム要件と機能仕様を理解

3. **実装者・コードを書く方**
   - `design.md` を最初に読む
   - クラス設計と実装ガイドラインに従う

4. **段階的な理解**
   1. `theory.md` で理論を理解
   2. `specification.md` で要件を確認
   3. `design.md` で実装を開始

## 実装の優先順位

ドキュメントでは、実装を4つのフェーズに分けています：

### フェーズ1: コア機能（Week 1-2）
- PropertyConditioning モジュール
- ExtendedCombinedConditioning モジュール
- CrystalDatasetWithProperties クラス
- PropertyNormalizer クラス

### フェーズ2: 訓練パイプライン（Week 3）
- training_with_properties.py
- main_crystal_with_properties.py
- データ準備スクリプト

### フェーズ3: 生成パイプライン（Week 4）
- sampling_with_properties.py
- generate_crystal_with_all_conditions.py

### フェーズ4: テストとドキュメント（Week 5）
- ユニットテスト
- 統合テスト
- ドキュメント最終化

## 主要コンポーネント

### PropertyConditioning

物性値ベクトルを条件付けベクトルに変換するモジュール。

**入力**: 物性値ベクトル（例: [bandgap, melting_point]）  
**出力**: 条件付けベクトル（次元: conditioning_dim）

**特徴**:
- 物性値の正規化（平均0、標準偏差1）
- 多層パーセプトロン（MLP）による変換
- 訓練時に計算された統計情報を使用

### ExtendedCombinedConditioning

複数の条件を統合するモジュール。

**入力**:
- 分子特徴量（必須）
- 空間群（オプション）
- 密度（オプション）
- 物性値（オプション）

**出力**: 統合条件付けベクトル

**特徴**:
- 柔軟な条件の組み合わせ
- アテンション機構または線形結合による統合
- モジュラー設計

### CrystalDatasetWithProperties

物性値を含む結晶データセット。

**機能**:
- ASEデータベースからの結晶データ読み込み
- 物性値の自動抽出
- 統計情報の計算（平均、標準偏差）
- バッチ処理対応

## データ形式

### 分子データベース (molecules.db)

ASE SQLite3形式。必須フィールド：
- molecule_id
- positions（原子座標）
- numbers（原子番号）
- molecular_weight
- pi_conjugation_ratio

### 結晶データベース (crystals.db)

ASE SQLite3形式。必須フィールド：
- crystal_id
- molecule_id（リンク用）
- positions（原子座標）
- cell（格子ベクトル）
- space_group
- density

物性値フィールド（新規追加）：
- bandgap（バンドギャップ）
- melting_point（融点）
- dielectric_constant（誘電率）
- その他

### 物性値CSV (crystal_properties.csv)

```csv
crystal_id,property_name,property_value
crystal_001,bandgap,2.5
crystal_001,melting_point,180.0
```

## 使用例

### データ準備

```bash
python scripts/prepare_property_dataset.py \
    --input_molecules_db data/molecules.db \
    --input_crystals_db data/crystals.db \
    --property_file data/crystal_properties.csv \
    --output_molecules_db data/molecules_with_props.db \
    --output_crystals_db data/crystals_with_props.db
```

### モデル訓練

```bash
python main_crystal_with_properties.py \
    --molecule_db_path data/molecules_with_props.db \
    --crystal_db_path data/crystals_with_props.db \
    --property_names bandgap melting_point dielectric_constant \
    --conditioning space_group density \
    --n_epochs 500 \
    --batch_size 32 \
    --exp_name crystal_with_properties
```

### 結晶生成

```bash
python generate_crystal_with_all_conditions.py \
    --model_path outputs/crystal_with_properties/model.pt \
    --molecule_db_path data/molecules_with_props.db \
    --molecular_weight_min 100 \
    --molecular_weight_max 200 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --space_group 14 \
    --density 1.2 \
    --n_samples 100 \
    --output_dir generated_crystals/
```

## 参照

- **PR#143**: 物性値条件付け機能の提案と議論
- **doc/molecular_crystal_generation_spec.md Q4**: 物性値を条件とした分子性結晶生成の実現方法
- **doc/theory.md**: 分子性結晶生成の基本理論
- **doc/design.md**: 既存システムの設計
- **doc/user_manual.md**: ユーザーマニュアル

## バージョン履歴

- **v1.0** (2025-10-24): 初版作成
  - theory.md: 理論説明書
  - specification.md: 詳細仕様書
  - design.md: 詳細設計書

## ライセンス

このドキュメントは、プロジェクトのライセンス（LICENSEファイル参照）に従います。

## お問い合わせ

ドキュメントに関する質問や提案は、GitHubのIssueまたはPull Requestでお願いします。

---

**最終更新日**: 2025-10-24  
**バージョン**: 1.0  
**文書管理者**: E(3)等変拡散モデル研究チーム
