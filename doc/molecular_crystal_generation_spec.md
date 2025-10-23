# 分子性結晶生成モデル機能仕様書

## 目次
1. [概要](#概要)
2. [システムアーキテクチャ](#システムアーキテクチャ)
3. [生成条件の詳細](#生成条件の詳細)
4. [質問への回答](#質問への回答)
5. [使用例](#使用例)
6. [まとめ](#まとめ)

---

## 概要

本仕様書は、E(3)等変拡散モデルを用いた分子性結晶（ホモ結晶）生成機能について、その仕様と生成条件を整理したものです。

### 分子性結晶生成とは

分子性結晶生成とは、**単一の分子が周期的に配列して形成される結晶構造**を生成する機能です。本システムでは、以下の特徴を持ちます：

- **ホモ結晶（Homocrystal）**: 同一の分子で構成される結晶のみをサポート
- **周期境界条件（PBC）**: 結晶の周期性を正確に扱う
- **分子EGNN特徴量の統合**: 単分子の構造情報を結晶生成に活用
- **理論的正当性**: ヒューリスティックなfallbackを使用せず、理論的に正しい生成を実現

### システムの構成要素

```
┌──────────────────────────────────────────────────────┐
│           単分子生成モデル                             │
│  ・molecular_weight                                  │
│  ・pi_conjugation_ratio                              │
│  ・atom_types_encoding                               │
│  ・functional_groups_encoding                        │
└──────────────────────────────────────────────────────┘
                    ↓ （分子の3D構造）
┌──────────────────────────────────────────────────────┐
│        分子EGNN特徴量エンコーダ（新規）                │
│  ・MolecularEncoder                                  │
│  ・EGNN特徴量抽出                                     │
│  ・幾何学的特徴の計算                                 │
└──────────────────────────────────────────────────────┘
                    ↓ （分子特徴量）
┌──────────────────────────────────────────────────────┐
│        分子性結晶生成モデル（拡張）                    │
│  【分子条件（必須）】                                  │
│  ・分子EGNN特徴量                                     │
│  ・分子の幾何学的性質                                 │
│  【結晶条件（オプション）】                            │
│  ・space_group（空間群）                              │
│  ・density（密度）                                    │
│  ・lattice_params（格子定数）                         │
└──────────────────────────────────────────────────────┘
                    ↓
              生成された結晶構造
```

---

## システムアーキテクチャ

### データフロー

分子性結晶生成システムは、以下のデータフローで動作します：

```
1. 分子データ読み込み
   molecules.db → MoleculeDataset
   
2. 結晶データ読み込み
   crystals.db → CrystalDataset
   
3. 分子-結晶マッピング
   MoleculeCrystalMapper が molecule_id でリンク
   
4. 分子特徴量抽出
   MolecularEncoder が分子の3D構造から特徴量を抽出
   
5. 結晶生成条件の構築
   MolecularConditioning が特徴量を条件付けベクトルに変換
   （オプション）SpaceGroupEmbedding, DensityConditioning を追加
   
6. 結晶構造生成
   CrystalDynamics が条件付きで結晶を生成
```

### 主要コンポーネント

#### 1. MoleculeDataset（分子データローダー）
- **役割**: 単分子の3D構造を読み込み
- **入力**: molecules.db（ASEデータベース）
- **出力**: 分子の原子座標、原子種、molecule_id

#### 2. CrystalDataset（結晶データローダー）
- **役割**: 結晶構造を読み込み、周期境界条件を処理
- **入力**: crystals.db（ASEデータベース）
- **出力**: 結晶の原子座標、格子ベクトル、molecule_id

#### 3. MoleculeCrystalMapper（分子-結晶マッパー）
- **役割**: 分子IDと結晶IDの対応を管理
- **機能**: 
  - 1分子：N結晶の関係をサポート（ポリモルフ対応）
  - molecule_idによるリンク（必須）

#### 4. MolecularEncoder（分子特徴量エンコーダ）
- **役割**: 単分子からEGNN特徴量を抽出
- **入力**: 分子の3D構造（positions, atom_types）
- **出力**: 
  - global_features: 分子レベルの特徴ベクトル
  - mol_size: 分子のサイズ
  - mol_volume: 分子の体積
  - principal_axes: 主軸方向

#### 5. MolecularConditioning（分子条件付けモジュール）
- **役割**: 分子特徴量を結晶生成の条件付けベクトルに変換
- **重要性**: 結晶生成における**最も重要な条件付け**
- **入力**: MolecularEncoderからの特徴量
- **出力**: conditioning_vector（結晶生成用）

#### 6. CrystalDynamics（結晶拡散モデル）
- **役割**: 条件付きで結晶構造を生成
- **機能**:
  - 原子座標の拡散（周期境界条件対応）
  - 格子パラメータの拡散
  - 分子条件による条件付け

---

## 生成条件の詳細

### 単分子生成の条件

単分子生成モデルで使用できる生成条件は以下の通りです：

| 条件名 | 型 | 説明 | 用途 |
|--------|-----|------|------|
| **molecular_weight** | float | 分子量（原子質量単位） | 分子のサイズ制御 |
| **pi_conjugation_ratio** | float [0, 1] | π共役比率 = (二重結合数 + 芳香族結合数) / 全結合数 | 芳香族性・共役度の制御 |
| **atom_types_encoding** | バイナリベクトル | 各元素の有無を示すエンコーディング | 元素組成の制御 |
| **functional_groups_encoding** | バイナリベクトル | 各官能基の有無を示すエンコーディング | 官能基の制御 |

**使用例**:
```bash
# 単分子生成
python main_qm9.py \
    --conditioning molecular_weight pi_conjugation_ratio \
    --dataset ase_db \
    --ase_db_path molecules.db
```

### 分子性結晶生成の条件

分子性結晶生成では、以下の2種類の条件を使用できます：

#### A. 分子の持つ条件（分子条件）

分子条件は、**結晶を構成する分子の特性**を指定するものです。これらは、MolecularEncoderを通じて自動的に抽出されます。

| 条件の種類 | 抽出方法 | 説明 |
|-----------|----------|------|
| **分子EGNN特徴量** | MolecularEncoder | 分子の構造的特徴を深層学習で抽出 |
| **分子の幾何学的性質** | MolecularEncoder | 分子サイズ、体積、主軸方向 |

**重要**: 単分子生成で使用する条件（molecular_weight, pi_conjugation_ratioなど）は、**結晶生成では直接使用しません**。代わりに、これらの性質を持つ分子の3D構造をMolecularEncoderに入力し、より高次の特徴量を抽出して使用します。

#### B. 結晶固有の条件（結晶条件）

結晶条件は、**結晶そのものの性質**を指定するものです。これらはすべてオプションです。

| 条件名 | 型 | 説明 | 用途 |
|--------|-----|------|------|
| **space_group** | int (1-230) | 空間群番号 | 結晶の対称性を制御 |
| **density** | float | 結晶密度（g/cm³） | パッキング効率を制御 |
| **lattice_params** | float[6] | 格子定数 (a, b, c, α, β, γ) | 単位格子の形状・サイズを制御 |

**使用例**:
```bash
# 分子性結晶生成（空間群と密度を指定）
python main_crystal.py \
    --molecule_db_path molecules.db \
    --crystal_db_path crystals.db \
    --conditioning space_group density \
    --exp_name crystal_with_sg_density
```

### 条件の階層構造

```
分子性結晶の生成条件
│
├─ 【必須】分子条件
│   ├─ 分子EGNN特徴量（自動抽出）
│   │   └─ 単分子の3D構造から深層学習で抽出
│   └─ 分子の幾何学的性質（自動計算）
│       ├─ mol_size（分子サイズ）
│       ├─ mol_volume（分子体積）
│       └─ principal_axes（主軸方向）
│
└─ 【オプション】結晶条件
    ├─ space_group（空間群番号）
    ├─ density（密度）
    └─ lattice_params（格子定数）
```

---

## 質問への回答

### Q1: 単分子の生成条件を分子性結晶の生成条件に使うことは可能か

**回答**: **可能ですが、直接的ではなく、間接的に使用されます。**

#### 詳細説明

単分子生成の条件（molecular_weight, pi_conjugation_ratio, atom_types_encoding, functional_groups_encoding）は、結晶生成において以下のように扱われます：

1. **直接使用はしない**: これらの条件を結晶生成モデルに直接入力することはありません。

2. **間接的な使用**: 以下のプロセスで間接的に活用されます：
   ```
   単分子の条件 → 分子の3D構造を選択/生成
                     ↓
            MolecularEncoder で特徴量抽出
                     ↓
              分子EGNN特徴量（高次特徴）
                     ↓
            結晶生成の条件付けに使用
   ```

3. **なぜ直接使用しないのか**:
   - 単分子の条件（molecular_weightなど）は、**スカラー値や単純なエンコーディング**
   - 結晶生成には、**分子の3D構造的な情報**が必要
   - MolecularEncoderが分子の構造を深層学習で解析し、より豊かな特徴量を抽出

#### 具体例

**ケース1**: 特定の分子量を持つ分子の結晶を生成したい場合

```bash
# ステップ1: 分子量50の分子を選択
# molecules.db から molecular_weight ≈ 50 の分子を選択

# ステップ2: その分子IDを使って結晶を生成
python main_crystal.py \
    --molecule_db_path molecules.db \
    --crystal_db_path crystals.db \
    --target_molecule_id "mol_12345"
```

**ケース2**: π共役性の高い分子の結晶を生成したい場合

```bash
# ステップ1: π共役比率が高い分子を選択
# molecules.db から pi_conjugation_ratio > 0.8 の分子を選択

# ステップ2: その分子の結晶を生成
# （MolecularEncoderが自動的にπ共役の構造的特徴を抽出）
```

#### 結論

| 単分子条件 | 結晶生成での使用 | 使用方法 |
|-----------|-----------------|---------|
| molecular_weight | ○（間接的） | 該当する分子量の分子を選択し、その3D構造から特徴量を抽出 |
| pi_conjugation_ratio | ○（間接的） | 該当するπ共役比率の分子を選択し、その3D構造から特徴量を抽出 |
| atom_types_encoding | ○（間接的） | 該当する元素組成の分子を選択し、その3D構造から特徴量を抽出 |
| functional_groups_encoding | ○（間接的） | 該当する官能基を持つ分子を選択し、その3D構造から特徴量を抽出 |

**重要**: これらの条件は結晶生成モデルに直接渡されるのではなく、**分子の選択基準として使用**され、選択された分子の**3D構造的特徴**が結晶生成に活用されます。

---

### Q2: 構成分子やその他の生成条件を何も指定しなくても生成可能か

**回答**: **理論的には可能ですが、実用的には推奨されません。**

#### 詳細説明

##### 無条件生成のメカニズム

1. **技術的可能性**: 
   - 拡散モデルは無条件生成（unconditional generation）をサポート
   - 条件付けベクトルをゼロベクトルまたはランダムベクトルとして生成可能

2. **実際の動作**:
   ```python
   # 無条件生成の例（疑似コード）
   conditioning_vector = torch.zeros(batch_size, conditioning_dim)
   # または
   conditioning_vector = torch.randn(batch_size, conditioning_dim)
   
   # 結晶構造を生成
   crystal = crystal_diffusion_model.sample(
       conditioning=conditioning_vector,
       n_samples=n_samples
   )
   ```

##### 無条件生成の問題点

1. **生成結果の制御不能**:
   - どのような分子の結晶が生成されるか予測不可能
   - 物理的に不安定な構造が生成される可能性が高い

2. **品質の低下**:
   - 分子の構造情報がないため、分子間のパッキングが不適切
   - 格子定数が現実的でない値になる可能性

3. **理論的正当性の欠如**:
   - 本システムは「分子の構造情報を活用して結晶を生成する」設計
   - 分子情報なしでは、設計思想に反する

##### 推奨される最小限の条件

**最低限必要な条件**: **構成分子の指定**（molecule_id）

```bash
# 推奨される最小限の使用方法
python main_crystal.py \
    --molecule_db_path molecules.db \
    --crystal_db_path crystals.db \
    --target_molecule_id "mol_12345"
# 結晶条件（space_group, densityなど）は指定しない
```

この場合：
- ✅ 分子の構造情報（EGNN特徴量）が使用される
- ✅ 分子間パッキングが適切に行われる
- ✅ 物理的に妥当な結晶構造が生成される
- ⚠️ 空間群や密度は学習データの分布に基づいてランダムに決定される

#### 生成条件の推奨レベル

| 条件レベル | 指定する条件 | 生成品質 | 推奨度 |
|-----------|-------------|---------|--------|
| **レベル0** | なし（無条件） | ❌ 低い | ⛔ 非推奨 |
| **レベル1** | 構成分子のみ | ⭐⭐⭐ 良い | ✅ 推奨（最小限） |
| **レベル2** | 構成分子 + 空間群 | ⭐⭐⭐⭐ とても良い | ✅✅ 推奨 |
| **レベル3** | 構成分子 + 空間群 + 密度 | ⭐⭐⭐⭐⭐ 最良 | ✅✅✅ 最推奨 |
| **レベル4** | 構成分子 + 空間群 + 密度 + 格子定数 | ⭐⭐⭐⭐⭐ 最良（厳密制御） | ✅✅✅ 最推奨（詳細制御が必要な場合） |

#### 結論

```
質問: 何も指定しなくても生成可能か？
回答: 技術的には可能だが、実用的には非推奨

推奨: 最低限、構成分子（molecule_id）を指定すること
理由: 分子の構造情報が結晶生成の品質を大きく左右するため
```

---

### Q3: どの生成条件で生成可能か

**回答**: **構成分子の条件（分子EGNN特徴量）のみで生成可能です。結晶固有の条件（空間群、密度、格子定数）はすべてオプションです。**

#### 詳細説明

##### 必須条件

| 条件 | 必要性 | 理由 |
|-----|--------|------|
| **分子EGNN特徴量** | **必須** | 結晶を構成する分子の構造情報が必要 |
| **分子の幾何学的性質** | **必須** | 分子サイズ、体積などが結晶のパッキングに影響 |

これらは、**構成分子を指定することで自動的に取得**されます。

##### オプション条件

| 条件 | 必要性 | 効果 |
|-----|--------|------|
| **space_group** | オプション | 指定しない場合、学習データの分布から自動決定 |
| **density** | オプション | 指定しない場合、学習データの分布から自動決定 |
| **lattice_params** | オプション | 指定しない場合、学習データの分布から自動決定 |

#### 生成可能な条件の組み合わせ

##### パターン1: 分子条件のみ（最小限）

```bash
python main_crystal.py \
    --molecule_db_path molecules.db \
    --crystal_db_path crystals.db \
    --target_molecule_id "mol_12345"
```

**生成される情報**:
- ✅ 原子座標（周期境界条件対応）
- ✅ 格子ベクトル（自動決定）
- ✅ 格子定数（自動決定）
- ⚠️ 空間群（学習データの分布から）
- ⚠️ 密度（学習データの分布から）

##### パターン2: 分子条件 + 空間群

```bash
python main_crystal.py \
    --molecule_db_path molecules.db \
    --crystal_db_path crystals.db \
    --target_molecule_id "mol_12345" \
    --conditioning space_group \
    --space_group 14  # P21/c
```

**生成される情報**:
- ✅ 原子座標（周期境界条件対応）
- ✅ 格子ベクトル（空間群の制約を満たす）
- ✅ 格子定数（空間群の制約を満たす）
- ✅ 空間群（指定された値）
- ⚠️ 密度（学習データの分布から）

##### パターン3: 分子条件 + 空間群 + 密度（推奨）

```bash
python main_crystal.py \
    --molecule_db_path molecules.db \
    --crystal_db_path crystals.db \
    --target_molecule_id "mol_12345" \
    --conditioning space_group density \
    --space_group 14 \
    --density 1.5
```

**生成される情報**:
- ✅ 原子座標（周期境界条件対応）
- ✅ 格子ベクトル（空間群と密度の制約を満たす）
- ✅ 格子定数（空間群と密度の制約を満たす）
- ✅ 空間群（指定された値）
- ✅ 密度（指定された値）

##### パターン4: 分子条件 + 空間群 + 密度 + 格子定数（完全制御）

```bash
python main_crystal.py \
    --molecule_db_path molecules.db \
    --crystal_db_path crystals.db \
    --target_molecule_id "mol_12345" \
    --conditioning space_group density lattice_params \
    --space_group 14 \
    --density 1.5 \
    --lattice_params 10.0 12.0 15.0 90.0 100.0 90.0
```

**生成される情報**:
- ✅ 原子座標（周期境界条件対応）
- ✅ 格子ベクトル（指定された格子定数から計算）
- ✅ 格子定数（指定された値）
- ✅ 空間群（指定された値）
- ✅ 密度（指定された値）

#### 条件の依存関係

```
分子EGNN特徴量（必須）
    ↓
原子座標の生成
    ↓
格子パラメータの生成
    ├─→ space_group 指定あり → 制約を満たす格子を生成
    ├─→ density 指定あり → 密度を満たす格子を生成
    └─→ lattice_params 指定あり → 指定された格子を使用
```

#### 各条件の相互作用

| 指定条件の組み合わせ | 結果 | 注意点 |
|---------------------|------|--------|
| 分子のみ | すべて自動決定 | 多様な結果が得られる |
| 分子 + 空間群 | 対称性が保証される | 特定の対称性を持つ結晶を生成 |
| 分子 + 密度 | 密度が制御される | パッキング効率が指定される |
| 分子 + 空間群 + 密度 | 対称性と密度が両方制御される | **最も推奨される組み合わせ** |
| 分子 + すべての結晶条件 | 完全に制御される | 詳細な制御が可能だが、条件が矛盾しないよう注意 |

#### 結論

```
質問: 単分子の条件と結晶の条件だけで生成可能か？
回答: YES

必須条件: 
  - 分子EGNN特徴量（構成分子を指定することで自動取得）
  - 分子の幾何学的性質（構成分子を指定することで自動計算）

オプション条件:
  - space_group（空間群）
  - density（密度）
  - lattice_params（格子定数）

推奨される最小限の組み合わせ:
  - 分子条件（必須） + space_group + density
```

---

## 使用例

### 例1: 最小限の条件で生成（分子のみ指定）

```bash
# molecules.db と crystals.db を準備
# crystals.db には molecule_id が記録されている必要がある

python main_crystal.py \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --target_molecule_id "benzene_001" \
    --n_epochs 200 \
    --batch_size 32 \
    --exp_name benzene_crystal_minimal
```

**生成される結晶**:
- ベンゼン分子で構成される結晶
- 空間群、密度、格子定数は学習データの分布から決定

### 例2: 空間群を指定して生成

```bash
python main_crystal.py \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --target_molecule_id "benzene_001" \
    --conditioning space_group \
    --space_group 14 \
    --n_epochs 200 \
    --batch_size 32 \
    --exp_name benzene_crystal_sg14
```

**生成される結晶**:
- ベンゼン分子で構成される結晶
- 空間群 P21/c（14番）
- 密度と格子定数は学習データの分布から決定

### 例3: 空間群と密度を指定して生成（推奨）

```bash
python main_crystal.py \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --target_molecule_id "benzene_001" \
    --conditioning space_group density \
    --space_group 14 \
    --density 1.2 \
    --n_epochs 200 \
    --batch_size 32 \
    --exp_name benzene_crystal_sg14_d12
```

**生成される結晶**:
- ベンゼン分子で構成される結晶
- 空間群 P21/c（14番）
- 密度 1.2 g/cm³
- 格子定数は空間群と密度の制約を満たすように決定

### 例4: すべての条件を指定して生成

```bash
python main_crystal.py \
    --molecule_db_path data/molecules.db \
    --crystal_db_path data/crystals.db \
    --target_molecule_id "benzene_001" \
    --conditioning space_group density lattice_params \
    --space_group 14 \
    --density 1.2 \
    --lattice_params 10.0 12.0 15.0 90.0 95.0 90.0 \
    --n_epochs 200 \
    --batch_size 32 \
    --exp_name benzene_crystal_full_control
```

**生成される結晶**:
- ベンゼン分子で構成される結晶
- 空間群 P21/c（14番）
- 密度 1.2 g/cm³
- 格子定数 a=10.0Å, b=12.0Å, c=15.0Å, α=90°, β=95°, γ=90°

### 例5: サンプリング（学習済みモデルから結晶を生成）

```bash
# 学習済みモデルを使って新しい結晶をサンプリング
python crystal/sampling.py \
    --model_path outputs/benzene_crystal_sg14_d12/generative_model.npy \
    --molecule_db_path data/molecules.db \
    --target_molecule_id "benzene_001" \
    --n_samples 100 \
    --conditioning space_group density \
    --space_group 14 \
    --density 1.2 \
    --output_dir samples/benzene_crystals \
    --output_format cif
```

**出力**:
- 100個の結晶構造（CIFファイル）
- すべてベンゼン分子、空間群14、密度1.2 g/cm³

### 例6: 複数の分子で結晶を生成

```bash
# ベンゼンとナフタレンの結晶を生成
for mol_id in "benzene_001" "naphthalene_001"; do
    python main_crystal.py \
        --molecule_db_path data/molecules.db \
        --crystal_db_path data/crystals.db \
        --target_molecule_id "$mol_id" \
        --conditioning space_group density \
        --space_group 14 \
        --density 1.2 \
        --n_epochs 200 \
        --batch_size 32 \
        --exp_name "${mol_id}_crystal"
done
```

---

## まとめ

### 主要なポイント

#### 1. 分子条件の取り扱い

- 単分子生成の条件（molecular_weight, pi_conjugation_ratioなど）は、結晶生成に**間接的に**使用される
- 直接使用されるのではなく、これらの条件を満たす分子の**3D構造的特徴（EGNN特徴量）**が使用される
- プロセス: `単分子条件 → 分子選択 → 3D構造 → EGNN特徴量 → 結晶生成条件`

#### 2. 無条件生成の可否

- **技術的には可能**だが、**実用的には非推奨**
- 最低限、構成分子（molecule_id）を指定することを強く推奨
- 理由: 分子の構造情報がないと、物理的に不安定な結晶が生成される可能性が高い

#### 3. 必要十分な生成条件

**必須条件**:
- ✅ 分子EGNN特徴量（構成分子を指定すれば自動取得）
- ✅ 分子の幾何学的性質（構成分子を指定すれば自動計算）

**オプション条件**（すべて省略可能）:
- ⭕ space_group（空間群）
- ⭕ density（密度）
- ⭕ lattice_params（格子定数）

**推奨される組み合わせ**:
```
分子条件（必須） + space_group + density
```

### 生成条件の比較表

| 生成モード | 必要な条件 | 生成品質 | 推奨度 | 用途 |
|-----------|-----------|---------|--------|------|
| **無条件** | なし | ❌ 低い | ⛔ 非推奨 | - |
| **分子のみ** | 構成分子 | ⭐⭐⭐ 良い | ✅ 推奨 | 探索的な結晶生成 |
| **分子+空間群** | 構成分子 + space_group | ⭐⭐⭐⭐ とても良い | ✅✅ 推奨 | 特定の対称性を持つ結晶 |
| **分子+密度** | 構成分子 + density | ⭐⭐⭐⭐ とても良い | ✅✅ 推奨 | 特定の密度を持つ結晶 |
| **分子+空間群+密度** | 構成分子 + space_group + density | ⭐⭐⭐⭐⭐ 最良 | ✅✅✅ 最推奨 | **標準的な使用** |
| **完全制御** | すべての条件 | ⭐⭐⭐⭐⭐ 最良 | ✅✅✅ 最推奨 | 詳細な制御が必要な場合 |

### 設計思想

本システムは以下の設計思想に基づいています：

1. **理論的正当性**: ヒューリスティックなfallbackを使用せず、理論的に正しい生成を実現
2. **分子構造の活用**: 単分子の3D構造情報を深層学習で抽出し、結晶生成に活用
3. **柔軟な条件付け**: 必須条件を最小限にしつつ、オプション条件で詳細な制御を可能に
4. **周期境界条件の厳密な取り扱い**: 最小イメージ規約などを正確に実装

### 参考資料

- **理論的背景**: `doc/theory.md`
- **詳細設計**: `doc/design.md`
- **使用方法**: `doc/user_manual.md`
- **チュートリアル**: `tutorials/`

---

## 付録: データ準備

### 分子データベースの準備

```python
from ase import Atoms
from ase.db import connect

# データベース作成
db = connect('molecules.db', append=False)

# ベンゼン分子を追加
benzene = Atoms(
    'C6H6',
    positions=[
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [2.1, 1.2, 0.0],
        [1.4, 2.4, 0.0],
        [0.0, 2.4, 0.0],
        [-0.7, 1.2, 0.0],
        # ... 水素原子の座標
    ]
)

db.write(
    benzene,
    data={
        'molecule_id': 'benzene_001',
        'molecular_weight': 78.11,
        'pi_conjugation_ratio': 1.0,
    }
)
```

### 結晶データベースの準備

```python
from ase import Atoms
from ase.db import connect
import numpy as np

# データベース作成
db = connect('crystals.db', append=False)

# ベンゼン結晶を追加
benzene_crystal = Atoms(
    'C24H24',  # 単位格子内に4分子
    positions=[...],  # 原子座標
    cell=[10.0, 12.0, 15.0, 90.0, 95.0, 90.0],  # 格子定数
    pbc=[True, True, True]  # 周期境界条件
)

db.write(
    benzene_crystal,
    data={
        'crystal_id': 'benzene_crystal_001',
        'molecule_id': 'benzene_001',  # ← 重要: 構成分子を指定
        'space_group': 14,
        'density': 1.2,
    }
)
```

### molecule_crystal_map.json の準備（オプション）

```json
{
  "benzene_001": {
    "molecule_id": "benzene_001",
    "crystal_ids": ["benzene_crystal_001", "benzene_crystal_002"],
    "polymorphs": {
      "alpha": "benzene_crystal_001",
      "beta": "benzene_crystal_002"
    },
    "num_polymorphs": 2
  }
}
```

---

**文書作成日**: 2025-01-XX  
**バージョン**: 1.0  
**ステータス**: 完成
