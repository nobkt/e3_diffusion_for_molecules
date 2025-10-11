# クイックリファレンス: データセット要件 / Quick Reference: Dataset Requirements

## 🎯 最も重要なポイント / Most Important Point

### 日本語
**SMILESとformulaはオプションです。必須ではありません。**

モデルは原子の3D座標と原子番号のみを使用します。

### English
**SMILES and formula are optional. They are NOT required.**

The model only uses 3D atomic coordinates and atomic numbers.

---

## 📋 必須データ / Required Data

### 分子データベース / Molecule Database

```python
from ase import Atoms
from ase.db import connect

mol_db = connect('molecules.db')

# 最小限の例 / Minimal example
molecule = Atoms(
    symbols=['C', 'H', 'H', 'H', 'H'],  # ✅ 必須 / Required
    positions=[...]                      # ✅ 必須 / Required
)

mol_id = mol_db.write(molecule)  # これだけで動作 / This works!
```

### 結晶データベース / Crystal Database

```python
crystal_db = connect('crystals.db')

crystal = Atoms(
    symbols=[...],              # ✅ 必須 / Required
    positions=[...],            # ✅ 必須 / Required
    cell=[...],                 # ✅ 必須 / Required
    pbc=[True, True, True]      # ✅ 必須 / Required
)

crystal_db.write(
    crystal,
    molecule_id=mol_id,  # ✅ 必須 / Required
    Z=4,                 # 🟡 推奨 / Recommended
    space_group=14,      # 🟡 推奨 / Recommended
    density=1.15         # 🟡 推奨 / Recommended
)
```

---

## ❌ オプションデータ / Optional Data

以下は**不要**です / The following are **NOT required**:

```python
# ❌ 以下は任意 / Optional
mol_db.write(
    molecule,
    smiles='C1CCC1',        # ❌ オプション / Optional
    formula='C4H8',         # ❌ オプション / Optional
    any_custom_field='...'  # ❌ オプション / Optional
)
```

---

## 🔧 使用可能なツール / Available Tools

### ✅ OpenBabel - 完全対応 / Fully Supported

```python
# OpenBabelで分子を準備 / Prepare molecule with OpenBabel
from openbabel import openbabel
# ... OpenBabel processing ...

# ASE Atomsに変換 / Convert to ASE Atoms
from ase import Atoms
molecule = Atoms(symbols=symbols, positions=positions)

# SMILESなしで保存 / Save without SMILES
mol_id = mol_db.write(molecule)  # ✅ 動作します / Works!
```

### ✅ RDKit - 対応（オプション） / Supported (Optional)

```python
# RDKitを使う場合 / If using RDKit
from rdkit import Chem
# ... RDKit processing ...

# SMILESを保存したい場合のみ / Only if you want to save SMILES
mol_id = mol_db.write(molecule, smiles=smiles)  # オプション / Optional
```

---

## 📖 詳細ドキュメント / Detailed Documentation

より詳しい情報は以下を参照：

### 日本語
- **[データセット要件FAQ](./DATASET_REQUIREMENTS_FAQ.md)** - 包括的な説明
- **[対応まとめ](./DATASET_CLARIFICATION_SUMMARY.md)** - 調査結果の詳細
- **[使用ガイド](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md)** - 完全なチュートリアル

### English
- **[Dataset Requirements FAQ](./DATASET_REQUIREMENTS_FAQ.md)** - Comprehensive explanation
- **[Clarification Summary](./DATASET_CLARIFICATION_SUMMARY.md)** - Detailed findings
- **[Usage Guide](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md)** - Complete tutorial

---

## 🚀 クイックスタート / Quick Start

### 1. 分子データベース作成 / Create Molecule Database

```python
from ase import Atoms
from ase.db import connect

mol_db = connect('molecules.db')

# 簡単な分子 / Simple molecule
water = Atoms('H2O', positions=[[0,0,0], [0.757,0.586,0], [-0.757,0.586,0]])
mol_id = mol_db.write(water)  # SMILESなし / No SMILES needed!
```

### 2. 結晶データベース作成 / Create Crystal Database

```python
crystal_db = connect('crystals.db')

# 結晶構造 / Crystal structure
crystal = Atoms(
    symbols=['H']*8 + ['O']*4,  # 4 water molecules
    positions=[...],
    cell=[[10,0,0], [0,10,0], [0,0,10]],
    pbc=True
)

crystal_db.write(crystal, molecule_id=mol_id, Z=4)
```

### 3. モデル訓練 / Train Model

```python
from crystal.data.molecular_crystal_loader import load_paired_datasets

datasets, info = load_paired_datasets(
    molecule_db_path='molecules.db',
    crystal_db_path='crystals.db'
)
# SMILESなしで動作！ / Works without SMILES!
```

---

## ✅ チェックリスト / Checklist

データベース作成時の確認事項：

- [ ] 原子座標が正しい / Atomic positions are correct
- [ ] 原子シンボルまたは番号が正しい / Atomic symbols/numbers are correct
- [ ] （結晶）セル情報がある / (Crystal) Cell information exists
- [ ] （結晶）molecule_idが設定されている / (Crystal) molecule_id is set
- [ ] ❌ SMILESは**不要** / SMILES is **NOT needed**
- [ ] ❌ formulaは**不要** / formula is **NOT needed**

---

## 💡 重要なメモ / Important Notes

### 日本語
1. **3D座標と原子番号のみ使用** - SMILESは内部で使われません
2. **OpenBabel完全対応** - RDKitは不要です
3. **メタデータは任意** - 何でも保存できますが、訓練には影響しません
4. **後方互換性あり** - 既存のSMILES付きデータベースも動作します

### English
1. **Only 3D coordinates and atomic numbers used** - SMILES not used internally
2. **Full OpenBabel support** - RDKit not required
3. **Metadata is optional** - Can store anything, but won't affect training
4. **Backward compatible** - Existing databases with SMILES still work

---

**Version:** 1.0  
**Date:** 2025-10-11  
**Status:** ✅ Ready to use
