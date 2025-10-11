# データセット要件に関するFAQ / Dataset Requirements FAQ

## 質問 / Question

**Q: 単分子条件付き分子性結晶生成のデータセットにSMILESとformulaの情報は必須ですか？**

**Q: Are SMILES and formula information mandatory for the single-molecule conditioned molecular crystal generation dataset?**

---

## 回答 / Answer

### 日本語

**いいえ、SMILESとformulaの情報は必須ではありません。**

#### 詳細説明

1. **実際のデータローダーでの使用状況**
   - `crystal/data/molecular_crystal_loader.py`の`MolecularCrystalDataset`クラスを確認したところ、SMILESやformulaの情報は一切使用されていません
   - データローダーが使用するのは以下の情報のみです：
     - 原子座標（positions）
     - 原子番号（atomic numbers / atom types）
     - セル情報（cell vectors, cell parameters - 結晶の場合）
     - メタデータ（`molecule_id`, `space_group`, `density`, `Z` など）

2. **USAGE GUIDEのサンプルコードについて**
   - `SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md`のサンプルコード（59行目）：
     ```python
     mol_id = mol_db.write(benzene, smiles='c1ccccc1', formula='C6H6')
     ```
   - このコードは**例示目的**であり、SMILESやformulaは任意の付加情報（key-value data）として保存されているだけです
   - ASE databaseのメタデータとして保存されますが、モデルの訓練や生成には使用されません

3. **crystal/ディレクトリ内のコード調査結果**
   - `crystal/`ディレクトリ配下のすべてのPythonファイルを調査した結果、`smiles`や`formula`を参照するコードは一切存在しませんでした
   - つまり、結晶生成システムの内部ではこれらの情報を全く使用していません

4. **必要な情報**
   以下の情報のみが必須です：
   
   **分子データベース（molecules.db）:**
   - 原子のシンボルまたは原子番号
   - 原子の3D座標
   
   **結晶データベース（crystals.db）:**
   - 原子のシンボルまたは原子番号
   - 原子の3D座標
   - セル情報（cell vectors, PBC）
   - `molecule_id`（対応する分子のID）
   - （推奨）`Z`, `space_group`, `density`などのメタデータ

5. **OpenBabelとRDKitの使い分けについて**
   - ご質問で言及されている通り、SMILESを経由したRDKit処理を避け、OpenBabelで代替する方針は完全に問題ありません
   - 単分子条件付き分子性結晶生成システムは、3D座標と原子番号のみから動作するように設計されています
   - SMILESやformulaはあくまで人間がデータを理解しやすくするための補助情報であり、システムの動作には不要です

#### 実装上の確認

`crystal/data/molecular_crystal_loader.py`の`__getitem__`メソッド（104-181行）を参照すると：

```python
# Get molecule data
molecule_row = self.molecule_db.get(molecule_id)
molecule_atoms = molecule_row.toatoms()
```

このように、データベースから`.toatoms()`でASE Atomsオブジェクトを取得しているだけで、SMILESやformulaにアクセスしていません。

#### 結論

**SMILESとformulaはデータベースに保存する必要はありません。** 以下のような最小限のコードでデータベースを作成できます：

```python
from ase import Atoms
from ase.db import connect

# 分子データベース
mol_db = connect('data/molecules.db')

benzene = Atoms(
    symbols=['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[...]  # 3D coordinates
)

# SMILESやformulaなしで保存可能
mol_id = mol_db.write(benzene)

# または、任意のメタデータとして保存（オプション）
# mol_id = mol_db.write(benzene, my_custom_field='any_value')
```

---

### English

**No, SMILES and formula information are NOT mandatory.**

#### Detailed Explanation

1. **Usage in Actual Data Loader**
   - After examining the `MolecularCrystalDataset` class in `crystal/data/molecular_crystal_loader.py`, we confirmed that SMILES and formula information are not used at all
   - The data loader only uses the following information:
     - Atomic positions
     - Atomic numbers (atom types)
     - Cell information (cell vectors, cell parameters - for crystals)
     - Metadata (`molecule_id`, `space_group`, `density`, `Z`, etc.)

2. **About the Sample Code in USAGE GUIDE**
   - The sample code in `SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md` (line 59):
     ```python
     mol_id = mol_db.write(benzene, smiles='c1ccccc1', formula='C6H6')
     ```
   - This code is for **demonstration purposes**, and SMILES and formula are stored as optional additional information (key-value data)
   - They are saved as ASE database metadata but are not used for model training or generation

3. **Code Investigation Results in crystal/ Directory**
   - After investigating all Python files under the `crystal/` directory, we found no code that references `smiles` or `formula`
   - This means the crystal generation system does not use this information internally at all

4. **Required Information**
   Only the following information is mandatory:
   
   **Molecule Database (molecules.db):**
   - Atomic symbols or atomic numbers
   - 3D atomic coordinates
   
   **Crystal Database (crystals.db):**
   - Atomic symbols or atomic numbers
   - 3D atomic coordinates
   - Cell information (cell vectors, PBC)
   - `molecule_id` (ID of the corresponding molecule)
   - (Recommended) Metadata such as `Z`, `space_group`, `density`

5. **Regarding OpenBabel vs RDKit**
   - As mentioned in your question, the approach of avoiding RDKit processing via SMILES and using OpenBabel instead is completely acceptable
   - The single-molecule conditioned molecular crystal generation system is designed to work solely from 3D coordinates and atomic numbers
   - SMILES and formula are merely auxiliary information to help humans understand the data and are not required for system operation

#### Implementation Verification

Referring to the `__getitem__` method (lines 104-181) in `crystal/data/molecular_crystal_loader.py`:

```python
# Get molecule data
molecule_row = self.molecule_db.get(molecule_id)
molecule_atoms = molecule_row.toatoms()
```

As shown, it simply retrieves an ASE Atoms object from the database using `.toatoms()`, without accessing SMILES or formula.

#### Conclusion

**You do NOT need to store SMILES and formula in the database.** You can create a database with minimal code like this:

```python
from ase import Atoms
from ase.db import connect

# Molecule database
mol_db = connect('data/molecules.db')

benzene = Atoms(
    symbols=['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
    positions=[...]  # 3D coordinates
)

# Can save without SMILES or formula
mol_id = mol_db.write(benzene)

# Or save with arbitrary metadata (optional)
# mol_id = mol_db.write(benzene, my_custom_field='any_value')
```

---

## 補足情報 / Additional Information

### データベースに保存される情報の優先順位 / Priority of Information Stored in Database

#### 必須 / Mandatory
1. 原子座標（3D positions） / Atomic 3D positions
2. 原子番号またはシンボル / Atomic numbers or symbols
3. （結晶の場合）セル情報 / (For crystals) Cell information
4. （結晶の場合）`molecule_id` / (For crystals) `molecule_id`

#### 推奨 / Recommended
1. `Z` - 単位格子あたりの分子数 / Number of molecules per unit cell
2. `space_group` - 空間群 / Space group
3. `density` - 密度 (g/cm³) / Density (g/cm³)

#### オプション / Optional
1. `smiles` - SMILES文字列（人間の可読性のため） / SMILES string (for human readability)
2. `formula` - 化学式（人間の可読性のため） / Chemical formula (for human readability)
3. その他任意のメタデータ / Any other custom metadata

---

## 関連ファイル / Related Files

- **データローダー実装**: `crystal/data/molecular_crystal_loader.py`
- **使用ガイド**: `SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md`
- **サンプルスクリプト**: `create_test_paired_dataset.py`

---

## 更新履歴 / Change History

- **2025-10-11**: 初版作成 / Initial version created

