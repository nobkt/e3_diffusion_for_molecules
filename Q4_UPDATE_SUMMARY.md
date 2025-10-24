# Q4更新サマリー - 物性値データが利用可能な場合の対応

## 更新内容

`doc/molecular_crystal_generation_spec.md` のQ4セクションを更新し、物性値データが利用可能な場合の直接的な実装方法を追加しました。

## 主な変更点

### 1. 新しいセクションの追加

Q4の冒頭に **「【新規】物性値データが利用可能な場合の推奨アプローチ」** を追加しました。このセクションでは：

- 単分子構造 + 分子性結晶構造 + 物性値の三つ組データを活用
- PropertyConditioning モジュールによる直接的な物性値条件付け
- ExtendedCombinedConditioning での統合
- データローダーの拡張

### 2. 完全な実装ガイド

以下のコンポーネントの詳細な実装コードを提供：

#### PropertyConditioning モジュール
```
crystal/conditioning/property_conditioning.py（新規）
```
- 物性値（バンドギャップ、融点など）を条件付けベクトルに変換
- 正規化機能を含む
- 複数の物性値を同時に扱える

#### ExtendedCombinedConditioning
```
crystal/conditioning/__init__.py（拡張）
```
- 分子条件 + 空間群条件 + 密度条件 + 物性値条件を統合
- 既存のアーキテクチャと互換性を保つ

#### CrystalDatasetWithProperties
```
crystal/data/crystal_loader.py（拡張）
```
- 物性値をロードするデータローダー
- 物性値の統計情報（平均、標準偏差）を自動計算

### 3. データ準備スクリプト

**prepare_property_dataset.py**（新規作成推奨）
- 物性値CSVファイルから結晶データベースに物性値を追加
- データ形式: `crystal_id, property_name, property_value`

### 4. 訓練・生成スクリプト

**main_crystal_with_properties.py**（新規作成推奨）
- 物性値条件付けを含む訓練スクリプト
- 既存の訓練ループに物性値条件を統合

**generate_crystal_with_all_conditions.py**（新規作成推奨）
- 分子条件 + 物性値条件 + 結晶条件をすべて指定して生成
- 2段階のワークフロー：
  1. 分子条件で分子を選択
  2. 選択した分子と物性値条件で結晶を生成

## あなたの要求への回答

### 要求の再確認

```
・ある物性値を満たす分子性結晶を生成させたい
・その分子性結晶は同じ分子から構成されるホモ結晶である
・その分子性結晶の構成分子に対して、molecular_weight、pi_conjugation_ratio、
  atom_types_encoding、functional_groups_encodingの条件を課したい
```

### 実現方法

**完全に実現可能です**。以下の手順で実装してください：

#### ステップ1: データセットの準備

あなたが持っているデータを以下の形式で整理：

```
molecules.db:
  - molecule_id
  - 原子座標
  - molecular_weight
  - pi_conjugation_ratio
  - atom_types
  - functional_groups

crystals.db:
  - crystal_id
  - molecule_id
  - 原子座標、格子定数
  - space_group
  - density
  
crystal_properties.csv:
  - crystal_id
  - property_name
  - property_value
```

#### ステップ2: PropertyConditioning の実装

ドキュメントに記載されたコードをそのまま実装。

#### ステップ3: 訓練

```bash
python main_crystal_with_properties.py \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --property_names bandgap melting_point \
    --conditioning space_group density \
    --n_epochs 500
```

#### ステップ4: 生成

```bash
python generate_crystal_with_all_conditions.py \
    --model_path outputs/*/generative_model.npy \
    --molecule_db_path data/molecules.db \
    --molecular_weight_min 100 \
    --molecular_weight_max 200 \
    --pi_conjugation_ratio_min 0.3 \
    --pi_conjugation_ratio_max 0.7 \
    --target_bandgap 2.5 \
    --target_melting_point 180.0 \
    --space_group 14 \
    --density 1.2 \
    --n_samples 100
```

## 実装の優先度

### 推奨度: ⭐⭐⭐⭐⭐

物性値データが利用可能な場合、このアプローチは最も効果的です：

**利点**:
- ✅ エンドツーエンドで物性値を直接制御可能
- ✅ 分子条件と物性値条件を同時に指定可能
- ✅ 理論的に最もエレガントで効果的
- ✅ 訓練後の生成が高速
- ✅ 既存のシステムとの互換性を保つ

**実装の難易度**: 中程度

- データ準備: 簡単（CSVファイルの作成）
- モジュール実装: 中程度（PyTorchの基本的な知識が必要）
- 統合: 既存システムへの最小限の変更で可能

## 実装スケジュール（推奨）

| フェーズ | 期間 | タスク |
|---------|------|--------|
| **フェーズ1: データ準備** | 1週間 | 物性値CSVファイルの作成、データベース更新 |
| **フェーズ2: 実装** | 2-3週間 | PropertyConditioningモジュール、データローダー、訓練・生成スクリプト |
| **フェーズ3: 訓練と検証** | 1-2週間 | パイロット訓練、生成結果の検証、調整 |
| **フェーズ4: 本格運用** | 継続 | 全データセットでの訓練、様々な条件でのテスト |

## 次のアクション

1. **ドキュメントの確認**: `doc/molecular_crystal_generation_spec.md` のQ4セクションを読む
2. **データの準備**: 物性値データを crystal_properties.csv 形式で準備
3. **実装開始**: PropertyConditioning モジュールから実装開始
4. **パイロットテスト**: 小規模データセットで動作確認

## 参考資料

- **更新されたドキュメント**: `doc/molecular_crystal_generation_spec.md` Q4セクション
- **実装コード例**: ドキュメント内に完全なコードスニペットを記載
- **データ形式**: ドキュメント内にCSV形式の例を記載

## サポート

実装中に不明点があれば、ドキュメントの該当セクションを参照してください。特に：

- **PropertyConditioning モジュール**: 行580-654
- **ExtendedCombinedConditioning**: 行659-761
- **CrystalDatasetWithProperties**: 行766-835
- **データ準備スクリプト**: 行842-922
- **訓練スクリプト**: 行927-997
- **生成スクリプト**: 行1002-1166
- **使用例**: 行1173-1270
- **あなたの要求への具体的回答**: 行1405-1511

---

**更新日**: 2025-10-24  
**ドキュメントバージョン**: 1.2  
**更新者**: GitHub Copilot
