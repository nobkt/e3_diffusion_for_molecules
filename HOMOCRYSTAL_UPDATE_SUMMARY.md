# ホモ結晶生成対応 - ドキュメント更新サマリー
# Homocrystal Generation Support - Documentation Update Summary

## 📋 更新概要 (Update Overview)

全6つのドキュメントファイルを更新し、以下の要件を追加しました:

1. **ホモ結晶の生成**: 同一分子からなる分子性結晶の生成
2. **単分子EGNN特徴量の統合**: 構成分子のEGNN特徴量を結晶生成に活用
3. **分子-結晶データセット構造**: 分子データセットと結晶データセットの分離管理
4. **ポリモルフ対応**: 1分子:N結晶の関係をサポート

## 🎯 重要な設計原則

### 1. ヒューリスティック・fallbackの完全排除
- molecule_idの厳格な必須化
- 理論的に正しい特徴量統合のみ実装
- エラー時のfallbackは一切なし

### 2. 分離データセット構造
```
molecules.db (単分子データ)
  ├─ Molecule A: xyz座標、原子種
  ├─ Molecule B: xyz座標、原子種
  └─ ...

crystals.db (結晶データ)
  ├─ Crystal A1: 結晶構造 + molecule_id="A"
  ├─ Crystal A2: 結晶構造 + molecule_id="A" (polymorph)
  ├─ Crystal B1: 結晶構造 + molecule_id="B"
  └─ ...

molecule_crystal_map.json (マッピング情報)
  {
    "A": {"crystal_ids": ["A1", "A2"], "num_polymorphs": 2},
    "B": {"crystal_ids": ["B1"], "num_polymorphs": 1},
    ...
  }
```

## 📝 更新されたファイル

### 1. MOLECULAR_CRYSTAL_SPECIFICATION.md
**更新内容:**
- 概要にホモ結晶と分子EGNN特徴量について記載
- FR-1.2: 分子-結晶データセットペアリング要件
- FR-1.3: 単分子EGNN特徴量抽出要件
- FR-2.1: 分子EGNN特徴量の統合要件
- Section 2.1.2: 詳細なデータセット構造（79行追加）
- Section 2.2.2: 分子EGNN特徴量表現（52行追加）
- アーキテクチャ図に分子特徴量抽出層を追加

### 2. ARCHITECTURE_DIAGRAM.md
**更新内容:**
- システム全体図を更新（molecule DB + crystal DB）
- データ処理層に「分子EGNN特徴量抽出」セクション追加
- 条件付け層で分子特徴量をPRIMARYとして明示
- 学習フェーズのデータフローを更新（分子特徴量の流れを追加）

### 3. MOLECULAR_CRYSTAL_DESIGN.md
**更新内容（最も大きな変更）:**
- 概要を更新
- ディレクトリ構造に4つの新規モジュール追加
  - `molecule_loader.py`
  - `molecule_crystal_mapper.py`
  - `molecular_encoder.py`
  - `molecular_conditioning.py`
- Section 2.1: MoleculeDataset実装（105行）
- Section 2.2: MoleculeCrystalMapper実装（83行）
- Section 2.3: CrystalDatasetをmolecule_id対応に更新
- Section 3.0: MolecularEncoder実装（218行、完全なコード）
- Section 4.0: MolecularConditioning + CombinedConditioning実装（233行、完全なコード）

### 4. CRYSTAL_EXTENSION_SUMMARY.md
**更新内容:**
- プロジェクト概要を更新
- 主要な技術的貢献に2つの新セクション追加：
  1. ホモ結晶生成のための分子-結晶データセット統合
  2. 単分子EGNN特徴量の統合
- コード例を更新（分子特徴量の使用方法を追加）

### 5. CRYSTAL_EXTENSION_README.md
**更新内容:**
- 概要を更新
- 主要な技術的特徴を6セクションに拡張
  - ホモ結晶統合をセクション1に追加
  - 単分子EGNN特徴量をセクション2（PRIMARY FEATURE）に追加
- 各セクションに詳細な説明を追加

### 6. README_CRYSTAL_EXTENSION.md
**更新内容:**
- 概要文をホモ結晶と分子EGNN特徴量に言及するよう更新

## 🔑 新規追加された主要コンポーネント

### 1. MoleculeDataset
```python
# crystal/data/molecule_loader.py
class MoleculeDataset(Dataset):
    """
    単分子データセット
    既存の単分子生成モデルと同じ形式
    """
```
**機能:**
- molecules.dbから単分子のxyz座標を読み込み
- molecule_idの取得と管理
- 原子種のエンコーディング

### 2. MoleculeCrystalMapper
```python
# crystal/data/molecule_crystal_mapper.py
class MoleculeCrystalMapper:
    """
    分子-結晶の対応関係を管理
    """
```
**機能:**
- molecule_id → crystal_idsのマッピング
- ポリモルフ情報の管理
- JSONファイルへの保存・読み込み

### 3. MolecularEncoder
```python
# crystal/models/molecular_encoder.py
class MolecularEncoder(nn.Module):
    """
    単分子からEGNN特徴量を抽出
    """
```
**機能:**
- 既存のEGNNモデルを再利用
- グローバル特徴量の抽出（分子レベル）
- 幾何学的性質の計算（サイズ、体積、主軸）
- 事前学習済みモデルのロード対応

### 4. MolecularConditioning
```python
# crystal/conditioning/molecular_conditioning.py
class MolecularConditioning(nn.Module):
    """
    分子特徴量を結晶生成の条件付けに使用（PRIMARY）
    """
```
**機能:**
- 分子グローバル特徴量を条件付けベクトルに変換
- 幾何学的性質の統合
- MLPによる特徴変換

### 5. CombinedConditioning
```python
# crystal/conditioning/molecular_conditioning.py
class CombinedConditioning(nn.Module):
    """
    複数の条件付けを統合（分子特徴量が主軸）
    """
```
**機能:**
- 分子特徴量（必須）
- 空間群（オプション）
- 密度（オプション）
- 学習可能な重み付け統合

## 📊 統計情報

| ドキュメント | 追加行数 | 主な変更 |
|------------|---------|---------|
| MOLECULAR_CRYSTAL_SPECIFICATION.md | ~200 | データセット構造、要件定義 |
| ARCHITECTURE_DIAGRAM.md | ~80 | アーキテクチャ図、データフロー |
| MOLECULAR_CRYSTAL_DESIGN.md | ~700 | 4つの新規クラス実装 |
| CRYSTAL_EXTENSION_SUMMARY.md | ~60 | 技術的貢献、コード例 |
| CRYSTAL_EXTENSION_README.md | ~40 | 技術的特徴の拡張 |
| README_CRYSTAL_EXTENSION.md | ~10 | 概要の更新 |
| **合計** | **~1,090行** | **6ファイル更新** |

## 🚀 次のステップ

### 実装時の優先順位

1. **Phase 1: データ基盤（Week 1-2）**
   - `crystal/data/molecule_loader.py`
   - `crystal/data/molecule_crystal_mapper.py`
   - `crystal/data/crystal_loader.py`（molecule_id対応）

2. **Phase 2: 分子特徴量抽出（Week 3-4）**
   - `crystal/models/molecular_encoder.py`
   - 既存EGNNとの統合
   - テストケースの作成

3. **Phase 3: 条件付けモジュール（Week 5-6）**
   - `crystal/conditioning/molecular_conditioning.py`
   - 他の条件付けとの統合

4. **Phase 4: 統合テスト（Week 7-8）**
   - エンドツーエンドの動作確認
   - 小規模データセットでの検証

## ⚠️ 重要な注意事項

### 必須要件
1. **molecule_idは必須**: crystals.dbの全エントリに必要
2. **molecules.dbの存在**: 結晶データセットには対応する分子データセットが必須
3. **ヒューリスティック禁止**: molecule_idがない場合は明示的にエラー
4. **fallback禁止**: データ不整合時の暗黙的な補完は行わない

### データセット準備
```python
# 正しいデータセット準備の例
from ase.db import connect

# 分子データベース
mol_db = connect('molecules.db')
mol_db.write(molecule_atoms, molecule_id='mol_001')

# 結晶データベース
crys_db = connect('crystals.db')
crys_db.write(
    crystal_atoms,
    molecule_id='mol_001',  # 必須！
    crystal_id='crys_001',
    polymorph_id='A'  # オプション
)
```

## 📚 参考セクション

各ドキュメントの主要セクション:

- **MOLECULAR_CRYSTAL_SPECIFICATION.md**
  - Section 2.1.2: ホモ結晶のための分子-結晶データセット構造
  - Section 2.2.2: 単分子EGNN特徴量の表現

- **MOLECULAR_CRYSTAL_DESIGN.md**
  - Section 2.1: MoleculeDataset実装
  - Section 2.2: MoleculeCrystalMapper実装
  - Section 3.0: MolecularEncoder実装
  - Section 4.0: MolecularConditioning実装

- **ARCHITECTURE_DIAGRAM.md**
  - データ処理層: 分子EGNN特徴量抽出
  - 条件付け層: 分子特徴量の優先順位

---

**作成日**: 2025-01-XX  
**更新者**: GitHub Copilot  
**レビュー**: 必要

このドキュメントは、ホモ結晶生成対応のための全ドキュメント更新の要約です。
実装時は各ドキュメントの詳細セクションを参照してください。
