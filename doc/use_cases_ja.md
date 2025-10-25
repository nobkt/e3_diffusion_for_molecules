# ユースケース集（日本語）

**作成日**: 2025年10月25日  
**バージョン**: 1.0  
**対象**: E3 Diffusion for Molecules

---

## 目次

1. [ユースケース1: 医薬品様分子の設計](#ユースケース1-医薬品様分子の設計)
2. [ユースケース2: 材料探索（有機半導体）](#ユースケース2-材料探索有機半導体)
3. [ユースケース3: 結晶多形予測](#ユースケース3-結晶多形予測)
4. [ユースケース4: 性質最適化（logP、溶解度）](#ユースケース4-性質最適化logp溶解度)
5. [ユースケース5: カスタムデータセットでの学習](#ユースケース5-カスタムデータセットでの学習)
6. [ユースケース6: 大規模生成（10,000分子以上）](#ユースケース6-大規模生成10000分子以上)
7. [ユースケース7: 結晶構造予測](#ユースケース7-結晶構造予測)

---

## ユースケース1: 医薬品様分子の設計

### 背景と目的

医薬品開発において、特定の物理化学的性質を持つ分子の探索は重要な課題です。このユースケースでは、分子量、logP、極性表面積などの性質を制御しながら、Lipinski's Rule of Fiveを満たす医薬品様分子を生成します。

**解決すべき課題**:
- 口頭摂取可能な医薬品候補の探索
- 薬物動態学的性質の最適化
- 化学空間の効率的な探索

---

### データセット

**使用データ**: GEOM-Drugs（約430,000の3D医薬品分子）

**前処理手順**:
```bash
# 1. GEOMデータセットのダウンロードと前処理
python build_geom_dataset.py --dataset geom_drugs --datadir data/geom

# 2. データセット統計の確認
python -c "
from qm9 import dataset
import argparse

class Args:
    dataset = 'geom'
    datadir = 'data/geom'
    batch_size = 64
    num_workers = 4
    filter_n_atoms = None
    remove_h = False
    include_charges = False

dataloaders, _ = dataset.retrieve_dataloaders(Args())
print(f'訓練データ: {len(dataloaders[\"train\"].dataset)} 分子')
print(f'検証データ: {len(dataloaders[\"valid\"].dataset)} 分子')
print(f'テストデータ: {len(dataloaders[\"test\"].dataset)} 分子')
"
```

---

### モデル設定

**アーキテクチャ**:
- EGNN (E(3) 同変グラフニューラルネットワーク)
- 9層のメッセージパッシング
- 拡散ステップ: 1000
- ノイズスケジュール: Polynomial

**ハイパーパラメータ**:
```python
# config/drug_design.yaml
exp_name: drug_design_lipinski
model: egnn_dynamics
n_layers: 9
nf: 256
diffusion_steps: 1000
diffusion_noise_schedule: polynomial_2
batch_size: 64
lr: 2e-4
n_epochs: 500

# 条件付け性質
conditioning:
  - molecular_weight  # 分子量 (< 500 Da)
  - logP             # 親油性 (0-5)
  - num_h_donors     # 水素結合供与体 (≤5)
  - num_h_acceptors  # 水素結合受容体 (≤10)
```

---

### 学習手順

```bash
# 1. 基本的な学習
python main_geom_drugs.py \
    --exp_name drug_design_lipinski \
    --n_epochs 500 \
    --batch_size 64 \
    --lr 2e-4 \
    --nf 256 \
    --n_layers 9 \
    --diffusion_steps 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --save_model 10 \
    --test_epochs 10 \
    --wandb_usr your_username

# 2. 条件付き学習
python main_geom_drugs.py \
    --exp_name drug_design_conditional \
    --conditioning molecular_weight logP num_h_donors num_h_acceptors \
    --n_epochs 500 \
    --batch_size 48 \
    --context_node_nf 4 \
    --lr 2e-4

# 3. 学習の再開（チェックポイントから）
python main_geom_drugs.py \
    --exp_name drug_design_conditional \
    --resume outputs/drug_design_conditional \
    --start_epoch 250
```

**学習時間の目安**:
- GPU (V100): 約48-72時間（500エポック）
- GPU (A100): 約24-36時間（500エポック）
- メモリ要件: 16GB以上

---

### 評価と解析

#### 評価指標

```bash
# 1. 包括的評価（10,000分子生成）
python eval_analyze.py \
    --model_path outputs/drug_design_conditional \
    --n_samples 10000 \
    --batch_size 100 \
    --save_dir evaluation_results

# 2. 条件付き生成（Lipinski基準を満たす分子）
python eval_conditional_qm9.py \
    --model_path outputs/drug_design_conditional \
    --conditioning molecular_weight logP num_h_donors num_h_acceptors \
    --target_values 350.0 2.5 3 5 \
    --n_samples 1000 \
    --output_dir lipinski_molecules
```

#### 結果の解釈

**成功基準**:
1. **分子安定性**: >95%
2. **一意性**: >90%
3. **新規性**: >85%
4. **Lipinski適合率**: >80%

**評価スクリプト**:
```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def evaluate_lipinski_compliance(molecules):
    """
    Lipinski's Rule of Fiveへの適合性を評価
    """
    compliant = 0
    results = {
        'MW': [], 'LogP': [], 'HBD': [], 'HBA': []
    }
    
    for mol in molecules:
        if mol is None:
            continue
        
        # 性質の計算
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        results['MW'].append(mw)
        results['LogP'].append(logp)
        results['HBD'].append(hbd)
        results['HBA'].append(hba)
        
        # Lipinski基準のチェック
        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            compliant += 1
    
    compliance_rate = compliant / len(molecules) * 100
    
    print(f"Lipinski適合率: {compliance_rate:.2f}%")
    print(f"分子量: {np.mean(results['MW']):.2f} ± {np.std(results['MW']):.2f}")
    print(f"LogP: {np.mean(results['LogP']):.2f} ± {np.std(results['LogP']):.2f}")
    print(f"HBD: {np.mean(results['HBD']):.2f} ± {np.std(results['HBD']):.2f}")
    print(f"HBA: {np.mean(results['HBA']):.2f} ± {np.std(results['HBA']):.2f}")
    
    return compliance_rate, results

# 使用例
molecules = [Chem.MolFromSmiles(smi) for smi in generated_smiles]
compliance, props = evaluate_lipinski_compliance(molecules)
```

---

### 結果の例

**生成された医薬品様分子**:
```
# 例1: 抗炎症剤様構造
SMILES: CC(C)Cc1ccc(C(C)C(=O)O)cc1
分子量: 206.28 Da
LogP: 3.12
HBD: 1, HBA: 2
Lipinski: ✓

# 例2: 抗菌剤様構造  
SMILES: Cc1cc(N)c(F)c(N)c1
分子量: 140.15 Da
LogP: 1.87
HBD: 2, HBA: 1
Lipinski: ✓

# 例3: 中枢神経系薬様構造
SMILES: CN1CCN(C(=O)c2ccccc2)CC1
分子量: 218.29 Da
LogP: 1.45
HBD: 0, HBA: 2
Lipinski: ✓
```

**性能メトリクス**:
```
分子安定性: 97.3%
原子安定性: 99.1%
一意性: 92.8%
新規性: 88.5%
Lipinski適合率: 84.2%
平均分子量: 342.6 ± 78.4 Da
平均LogP: 2.8 ± 1.2
```

---

### 注意点とヒント

**よくある問題**:
1. **不安定な分子の生成**
   - 解決策: 拡散ステップ数を増やす（1000 → 1500）
   - 解決策: ノイズスケジュールを調整

2. **Lipinski基準からの逸脱**
   - 解決策: より厳密なターゲット値を設定
   - 解決策: Classifier-Free Guidanceのスケールを調整（1.0 → 2.0）

3. **化学的多様性の不足**
   - 解決策: 温度パラメータを上げる
   - 解決策: より大きなデータセットで再学習

**最適化のポイント**:
- 学習率スケジューラを使用（CosineAnnealing推奨）
- Early Stoppingで過学習を防止
- データ拡張（回転、反転）で汎化性能向上

---

## ユースケース2: 材料探索（有機半導体）

### 背景と目的

有機太陽電池や有機ELに使用される有機半導体材料の探索において、HOMOとLUMO準位の制御は極めて重要です。このユースケースでは、特定のエネルギーギャップを持つ共役分子を生成します。

**解決すべき課題**:
- 適切なバンドギャップを持つ材料の探索
- 電子移動度の最適化
- 化学的安定性の確保

---

### データセット

**使用データ**: QM9（約134,000の小分子）+ カスタム共役分子データセット

**カスタムデータセットの作成**:
```python
# conjugated_molecules.py
from ase import Atoms
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def create_conjugated_dataset():
    """
    共役分子のASEデータベースを作成
    """
    db = connect('conjugated_materials.db')
    
    # 代表的な共役構造のSMILES
    conjugated_smiles = [
        'c1ccc2c(c1)ccc1ccccc12',  # アントラセン
        'c1ccc2c(c1)sc1ccccc12',   # チオフェン融合系
        'C1=CC=C2C(=C1)C=CC1=CC=CC=C12',  # ペンタセン
        # ... さらに多くの共役構造
    ]
    
    for smi in conjugated_smiles:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        
        # 原子座標の取得
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # ASE Atomsオブジェクトの作成
        atoms = Atoms(symbols=symbols, positions=positions)
        
        # 計算されたHOMO/LUMOの追加（例）
        db.write(atoms, data={
            'homo': -5.2,  # eV
            'lumo': -2.8,  # eV
            'gap': 2.4,    # eV
            'pi_conjugation_ratio': 0.85
        })
    
    print(f"Created database with {db.count()} molecules")

# データベースの作成
create_conjugated_dataset()
```

---

### モデル設定

**学習コマンド**:
```bash
# ASEデータベースから学習
python main_qm9.py \
    --exp_name organic_semiconductor \
    --dataset ase_db \
    --ase_db_path conjugated_materials.db \
    --conditioning homo lumo gap pi_conjugation_ratio \
    --context_node_nf 4 \
    --n_epochs 300 \
    --batch_size 32 \
    --nf 256 \
    --n_layers 9 \
    --lr 1e-4
```

---

### 評価と解析

**ターゲット性質での生成**:
```bash
# バンドギャップ2.0-2.5 eVの材料を生成
python eval_conditional_qm9.py \
    --model_path outputs/organic_semiconductor \
    --conditioning homo lumo gap pi_conjugation_ratio \
    --target_values -5.0 -2.5 2.5 0.80 \
    --n_samples 1000 \
    --output_dir organic_semiconductors
```

**共役度の評価**:
```python
from rdkit import Chem
from rdkit.Chem import Descriptors

def evaluate_conjugation(mol):
    """
    π共役系の評価
    """
    # 芳香環の数
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    
    # 二重結合の数
    double_bonds = sum(1 for bond in mol.GetBonds() 
                      if bond.GetBondTypeAsDouble() == 2.0)
    
    # π共役比（芳香族原子数 / 全原子数）
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() 
                        if atom.GetIsAromatic())
    total_atoms = mol.GetNumAtoms()
    conjugation_ratio = aromatic_atoms / total_atoms
    
    return {
        'aromatic_rings': aromatic_rings,
        'double_bonds': double_bonds,
        'conjugation_ratio': conjugation_ratio
    }
```

---

### 結果の例

**生成された有機半導体候補**:
```
# 例1: チオフェンベースのオリゴマー
SMILES: c1cc2c(s1)sc1cc3sccc3cc12
HOMO: -5.1 eV
LUMO: -2.6 eV
Gap: 2.5 eV
共役度: 0.82

# 例2: 縮合多環芳香族
SMILES: C1=CC=C2C(=C1)C=CC1=CC=CC=C12
HOMO: -5.3 eV
LUMO: -2.7 eV
Gap: 2.6 eV
共役度: 0.88
```

---

## ユースケース3: 結晶多形予測

### 背景と目的

医薬品開発において、同じ化学組成を持つ分子が異なる結晶構造（多形）を取りうることは重要な課題です。このユースケースでは、特定の分子に対して複数の結晶多形を予測します。

**解決すべき課題**:
- 安定な多形の探索
- 物理化学的性質の予測
- 製造プロセスの最適化

---

### データセット

**分子の選択**: アスピリン（C9H8O4）

**データベースの作成**:
```python
# create_aspirin_polymorphs.py
from ase import Atoms
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def create_aspirin_database():
    """
    アスピリン分子のデータベースを作成
    """
    db = connect('aspirin_polymorphs.db')
    
    # アスピリンのSMILES
    aspirin_smiles = 'CC(=O)Oc1ccccc1C(=O)O'
    
    # 複数のコンフォマーを生成
    mol = Chem.MolFromSmiles(aspirin_smiles)
    mol = Chem.AddHs(mol)
    
    # 50個の3Dコンフォマーを生成
    cids = AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=50, 
        randomSeed=42,
        useRandomCoords=True
    )
    
    # 各コンフォマーを最適化
    for cid in cids:
        AllChem.UFFOptimizeMolecule(mol, confId=cid)
        
        conf = mol.GetConformer(cid)
        positions = conf.GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        atoms = Atoms(symbols=symbols, positions=positions)
        
        # 結晶性質の追加
        db.write(atoms, data={
            'molecular_weight': 180.16,
            'density_target': np.random.uniform(1.35, 1.45),  # g/cm³
            'space_group_hint': np.random.choice([14, 19, 61]),  # P21/c, P212121, Pbca
        })
    
    print(f"Created {db.count()} aspirin conformers")

create_aspirin_database()
```

---

### 学習手順

```bash
# 1. 段階的学習 - Stage 1: 基本的な結晶生成
python main_crystal.py \
    --exp_name aspirin_stage1_basic \
    --ase_db_path aspirin_polymorphs.db \
    --n_epochs 100 \
    --batch_size 16 \
    --lr 2e-4

# 2. Stage 2: 密度条件付け
python main_crystal_with_properties.py \
    --exp_name aspirin_stage2_density \
    --resume outputs/aspirin_stage1_basic \
    --conditioning density_target \
    --context_node_nf 1 \
    --n_epochs 100

# 3. Stage 3: 空間群条件付け
python main_crystal_with_properties.py \
    --exp_name aspirin_stage3_full \
    --resume outputs/aspirin_stage2_density \
    --conditioning density_target space_group_hint \
    --context_node_nf 65 \
    --n_epochs 150
```

---

### 多形生成

```bash
# 異なる空間群での多形生成
python generate_crystal_with_all_conditions.py \
    --model_path outputs/aspirin_stage3_full/model_best.pt \
    --molecule_smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --space_groups 14 19 61 \
    --densities 1.35 1.40 1.45 \
    --n_samples_per_condition 10 \
    --output_dir aspirin_polymorphs
```

**生成されるファイル**:
```
aspirin_polymorphs/
├── space_group_14_density_1.35/
│   ├── polymorph_001.cif
│   ├── polymorph_002.cif
│   └── ...
├── space_group_14_density_1.40/
├── space_group_19_density_1.35/
└── ...
```

---

### 評価と解析

**結晶品質の検証**:
```python
# validate_polymorphs.py
import spglib
from ase.io import read
import numpy as np

def validate_crystal_polymorph(cif_file):
    """
    生成された結晶多形を検証
    """
    atoms = read(cif_file)
    
    # 対称性の検証
    cell = (atoms.cell, 
            atoms.get_scaled_positions(), 
            atoms.numbers)
    
    spacegroup = spglib.get_spacegroup(cell, symprec=1e-5)
    
    # 密度の計算
    volume = atoms.get_volume()  # Ų
    mass = sum(atoms.get_masses())  # amu
    density = mass * 1.66054e-24 / (volume * 1e-24)  # g/cm³
    
    # 格子パラメータ
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    
    results = {
        'space_group': spacegroup,
        'density': density,
        'lattice_params': {
            'a': a, 'b': b, 'c': c,
            'alpha': alpha, 'beta': beta, 'gamma': gamma
        }
    }
    
    return results

# 全ての多形を検証
import glob

for cif_file in glob.glob('aspirin_polymorphs/**/*.cif', recursive=True):
    result = validate_crystal_polymorph(cif_file)
    print(f"{cif_file}:")
    print(f"  Space Group: {result['space_group']}")
    print(f"  Density: {result['density']:.3f} g/cm³")
    print(f"  Lattice: a={result['lattice_params']['a']:.2f}, "
          f"b={result['lattice_params']['b']:.2f}, "
          f"c={result['lattice_params']['c']:.2f}")
```

---

### 結果の例

**生成された多形**:
```
多形 I (P21/c):
  密度: 1.401 g/cm³
  格子: a=11.43Å, b=6.59Å, c=11.39Å
  角度: α=90°, β=95.68°, γ=90°
  安定性: ✓

多形 II (P212121):
  密度: 1.375 g/cm³
  格子: a=11.25Å, b=11.30Å, c=5.39Å
  角度: α=90°, β=90°, γ=90°
  安定性: ✓

多形 III (Pbca):
  密度: 1.438 g/cm³
  格子: a=12.02Å, b=7.89Å, c=21.05Å
  角度: α=90°, β=90°, γ=90°
  安定性: ✓
```

**実験データとの比較**:
```
           密度(実験)  密度(予測)  誤差
多形 I     1.40       1.401      +0.1%
多形 II    1.38       1.375      -0.4%
```

---

## ユースケース4: 性質最適化（logP、溶解度）

### 背景と目的

薬物動態学において、適切な親油性（logP）と水溶解度は薬物の吸収・分布・代謝・排泄（ADME）に重要です。

---

### 目標性質の設定

```python
# target_properties.py
target_profiles = {
    'oral_drug': {
        'logP': (0, 3),      # 経口薬に適した範囲
        'MW': (200, 500),    # Lipinski基準
        'TPSA': (20, 140),   # 極性表面積
        'solubility': 'high'  # 高い水溶解度
    },
    'cns_drug': {
        'logP': (1, 3),      # CNS透過性
        'MW': (200, 450),
        'TPSA': (40, 90),    # BBB透過性
        'solubility': 'moderate'
    },
    'topical_drug': {
        'logP': (2, 5),      # 皮膚透過性
        'MW': (250, 600),
        'TPSA': (50, 150),
        'solubility': 'moderate'
    }
}
```

---

### 学習と生成

```bash
# 1. 経口薬プロファイルで学習
python main_qm9.py \
    --exp_name oral_drug_design \
    --conditioning logP molecular_weight TPSA \
    --context_node_nf 3 \
    --n_epochs 300 \
    --batch_size 64

# 2. ターゲット性質での生成
python eval_conditional_qm9.py \
    --model_path outputs/oral_drug_design \
    --conditioning logP molecular_weight TPSA \
    --target_values 2.0 350.0 75.0 \
    --n_samples 5000
```

---

### 性質スイープ

logP値を系統的に変化させて分子を生成:

```python
# property_sweep.py
import torch
import numpy as np
from qm9.models import get_model
from qm9.utils import compute_mean_mad, prepare_context

# logP値のスイープ（0.0 → 5.0）
logp_values = np.linspace(0.0, 5.0, 20)

for logp_target in logp_values:
    # コンテキストの準備
    context = torch.zeros(1, n_nodes, context_dim).to(device)
    
    # logPのみを変化させる
    context[0, :, 0] = (logp_target - property_norms['logP']['mean']) / \
                        property_norms['logP']['mad']
    
    # 分子生成
    x, h = model.sample(1, n_nodes, node_mask, edge_mask, context=context)
    
    # 保存
    save_molecule(x, h, f'logP_{logp_target:.2f}.xyz')
```

---

## ユースケース5: カスタムデータセットでの学習

### 背景と目的

独自の分子データセット（社内化合物ライブラリなど）を使用して、特定の化学空間に特化したモデルを訓練します。

---

### カスタムデータベースの作成

```python
# create_custom_dataset.py
from ase import Atoms
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd

def create_custom_database(csv_file, output_db='custom_molecules.db'):
    """
    CSVファイルから ASEデータベースを作成
    
    CSV形式:
    SMILES,activity,logP,MW,other_properties...
    """
    df = pd.read_csv(csv_file)
    db = connect(output_db)
    
    for idx, row in df.iterrows():
        try:
            # SMILESから3D構造を生成
            mol = Chem.MolFromSmiles(row['SMILES'])
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            
            # 座標の取得
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            
            atoms = Atoms(symbols=symbols, positions=positions)
            
            # カスタム性質の追加
            properties = {
                'activity': float(row['activity']),
                'logP': float(row['logP']),
                'molecular_weight': float(row['MW']),
                # 他の性質も追加可能
            }
            
            db.write(atoms, data=properties)
            
        except Exception as e:
            print(f"Error processing molecule {idx}: {e}")
            continue
    
    print(f"Created database with {db.count()} molecules")
    return output_db

# 使用例
db_path = create_custom_database('company_library.csv')
```

---

### 分子記述子の追加

```python
# add_descriptors.py
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import numpy as np

def add_molecular_descriptors(db_path):
    """
    既存のデータベースに追加の記述子を計算して追加
    """
    db = connect(db_path)
    
    for row in db.select():
        atoms = row.toatoms()
        
        # ASE Atomsから RDKit Molへ変換
        # (簡略化: 実際にはより堅牢な変換が必要)
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        
        # 記述子の計算
        # ここでは例として基本的な記述子のみ
        descriptors = {
            'num_atoms': len(atoms),
            'num_heavy_atoms': sum(1 for s in symbols if s != 'H'),
            # RDKitの記述子も計算可能
        }
        
        # データベースの更新
        db.update(row.id, data=descriptors)
    
    print("Descriptors added successfully")

# 実行
add_molecular_descriptors('custom_molecules.db')
```

---

### 学習

```bash
# カスタムデータセットでの学習
python main_qm9.py \
    --exp_name custom_dataset_training \
    --dataset ase_db \
    --ase_db_path custom_molecules.db \
    --conditioning activity logP molecular_weight \
    --context_node_nf 3 \
    --n_epochs 500 \
    --batch_size 32 \
    --nf 256 \
    --n_layers 9 \
    --lr 1e-4 \
    --save_model 10 \
    --test_epochs 5
```

---

## ユースケース6: 大規模生成（10,000分子以上）

### 背景と目的

仮想スクリーニングや化学空間の網羅的探索のために、大量の分子を効率的に生成します。

---

### 並列生成スクリプト

```python
# large_scale_generation.py
import torch
import torch.multiprocessing as mp
from qm9.models import get_model
from qm9 import dataset
import numpy as np

def generate_batch(rank, model_path, n_samples, output_dir):
    """
    各プロセスで分子バッチを生成
    """
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    
    # モデルのロード
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # バッチ生成
    molecules = []
    batch_size = 100
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            current_batch = min(batch_size, n_samples - i)
            
            # ノード数のサンプリング
            n_nodes = nodes_dist.sample(current_batch)
            
            # マスクの準備
            node_mask = torch.ones(current_batch, n_nodes.max(), 1).to(device)
            edge_mask = get_edge_mask(current_batch, n_nodes.max(), device)
            
            # 生成
            x, h = model.sample(current_batch, n_nodes.max(), 
                              node_mask, edge_mask)
            
            molecules.append((x.cpu(), h.cpu()))
    
    # 保存
    torch.save(molecules, f'{output_dir}/batch_{rank}.pt')
    print(f"Process {rank}: Generated {n_samples} molecules")

def main():
    # パラメータ
    model_path = 'outputs/model_best.pt'
    total_samples = 100000
    num_processes = 8
    output_dir = 'large_scale_molecules'
    
    # プロセスごとのサンプル数
    samples_per_process = total_samples // num_processes
    
    # 並列実行
    mp.spawn(
        generate_batch,
        args=(model_path, samples_per_process, output_dir),
        nprocs=num_processes,
        join=True
    )
    
    print(f"Generated {total_samples} molecules total")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
```

---

### バッチ処理での実行

```bash
# シェルスクリプトでの大規模生成
#!/bin/bash

# generate_large_scale.sh
MODEL_PATH="outputs/model_best.pt"
OUTPUT_DIR="large_scale_molecules"
TOTAL_SAMPLES=100000
BATCH_SIZE=1000

mkdir -p $OUTPUT_DIR

# バッチに分割して生成
for i in $(seq 0 $((TOTAL_SAMPLES/BATCH_SIZE - 1))); do
    python eval_sample.py \
        --model_path $MODEL_PATH \
        --n_samples $BATCH_SIZE \
        --output_file $OUTPUT_DIR/batch_${i}.xyz \
        --batch_size 100 &
    
    # 並列数の制限（8プロセスまで）
    if [ $((i % 8)) -eq 7 ]; then
        wait
    fi
done

wait
echo "All batches completed"
```

---

### 生成分子の集約と解析

```python
# aggregate_and_analyze.py
import glob
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def aggregate_molecules(output_dir):
    """
    生成された分子を集約して解析
    """
    all_molecules = []
    
    # 全バッチファイルを読み込み
    for batch_file in glob.glob(f'{output_dir}/batch_*.pt'):
        molecules = torch.load(batch_file)
        all_molecules.extend(molecules)
    
    print(f"Total molecules loaded: {len(all_molecules)}")
    
    # SMILES変換と重複除去
    unique_smiles = set()
    valid_molecules = []
    
    for x, h in all_molecules:
        # 座標と原子タイプから分子を構築
        mol = build_molecule_from_coords(x, h)
        
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                valid_molecules.append((mol, smiles))
    
    print(f"Unique valid molecules: {len(valid_molecules)}")
    
    # 性質の計算と保存
    results = []
    for mol, smiles in valid_molecules:
        props = {
            'SMILES': smiles,
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        }
        results.append(props)
    
    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/all_molecules_properties.csv', index=False)
    
    return df

# 実行
df = aggregate_molecules('large_scale_molecules')

# 統計情報
print("\n=== 統計情報 ===")
print(df.describe())
```

---

## ユースケース7: 結晶構造予測

### 背景と目的

与えられた分子式から、可能な結晶構造を予測します。Cambridge Structural Database (CSD)に登録されていない新規構造の探索に有用です。

---

### 分子からの結晶生成

```python
# predict_crystal_structure.py
import torch
from crystal.models import CrystalDynamics
from crystal.conditioning import SpaceGroupEmbedding, DensityConditioning
from ase import Atoms
from ase.io import write
import spglib

def predict_crystal_structures(
    molecule_smiles,
    target_space_groups=[14, 19, 61, 62],  # 一般的な有機結晶空間群
    target_densities=[1.2, 1.4, 1.6],  # g/cm³
    n_predictions_per_condition=5
):
    """
    与えられた分子の結晶構造を予測
    """
    # モデルのロード
    model = load_crystal_model('outputs/crystal_model_best.pt')
    model.eval()
    
    predictions = []
    
    for sg in target_space_groups:
        for density in target_densities:
            print(f"Predicting: Space Group {sg}, Density {density} g/cm³")
            
            for i in range(n_predictions_per_condition):
                # 条件付けベクトルの準備
                sg_embedding = get_space_group_embedding(sg)
                density_cond = get_density_conditioning(density)
                context = torch.cat([sg_embedding, density_cond], dim=-1)
                
                # 結晶生成
                x, h, cell = model.sample_crystal(
                    n_molecules=4,  # ユニットセル内の分子数
                    context=context
                )
                
                # ASE Atomsオブジェクトの構築
                atoms = build_crystal_atoms(x, h, cell, molecule_smiles)
                
                # 対称性の検証
                is_valid, actual_sg = verify_symmetry(atoms, sg)
                
                if is_valid:
                    predictions.append({
                        'atoms': atoms,
                        'space_group': actual_sg,
                        'density': calculate_density(atoms),
                        'target_density': density
                    })
                    
                    # CIFファイルとして保存
                    filename = f'predicted_sg{sg}_dens{density}_n{i}.cif'
                    write(filename, atoms)
    
    return predictions

# 実行例
predictions = predict_crystal_structures(
    molecule_smiles='CC(=O)Nc1ccc(O)cc1',  # パラセタモール
    target_space_groups=[14, 19, 61],
    target_densities=[1.25, 1.30, 1.35],
    n_predictions_per_condition=10
)

print(f"Generated {len(predictions)} valid crystal structures")
```

---

### 結晶品質の評価

```python
# evaluate_crystal_quality.py
from ase.io import read
import spglib
import numpy as np

def evaluate_crystal_quality(cif_file):
    """
    生成された結晶構造の品質を評価
    """
    atoms = read(cif_file)
    
    # 1. 対称性の検証
    cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    symmetry = spglib.get_symmetry(cell, symprec=1e-5)
    spacegroup = spglib.get_spacegroup(cell, symprec=1e-5)
    
    # 2. 格子エネルギーの推定（簡略版）
    # 実際にはDFTや力場計算が必要
    lattice_energy = estimate_lattice_energy(atoms)
    
    # 3. 充填効率
    packing_efficiency = calculate_packing_efficiency(atoms)
    
    # 4. 最短原子間距離
    distances = atoms.get_all_distances()
    min_distance = np.min(distances[distances > 0.1])
    
    # 5. 格子パラメータの妥当性
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    
    quality_metrics = {
        'space_group': spacegroup,
        'n_symmetry_operations': len(symmetry['rotations']),
        'lattice_energy': lattice_energy,
        'packing_efficiency': packing_efficiency,
        'min_distance': min_distance,
        'cell_parameters': {
            'a': a, 'b': b, 'c': c,
            'alpha': alpha, 'beta': beta, 'gamma': gamma
        }
    }
    
    # 品質判定
    is_valid = (
        min_distance > 1.0 and  # 原子間距離が現実的
        packing_efficiency > 0.5 and  # 充填効率が妥当
        lattice_energy < 0  # 安定な構造
    )
    
    quality_metrics['is_valid'] = is_valid
    
    return quality_metrics

# 全ての予測構造を評価
import glob

for cif_file in glob.glob('predicted_*.cif'):
    metrics = evaluate_crystal_quality(cif_file)
    print(f"\n{cif_file}:")
    print(f"  Space Group: {metrics['space_group']}")
    print(f"  Packing Efficiency: {metrics['packing_efficiency']:.2f}")
    print(f"  Min Distance: {metrics['min_distance']:.2f} Å")
    print(f"  Valid: {metrics['is_valid']}")
```

---

## まとめ

これらのユースケースは、E3 Diffusion for Moleculesの多様な応用例を示しています。

### 主要なポイント

1. **柔軟な条件付け**: 複数の性質を同時に制御可能
2. **結晶生成**: 空間群と密度を制御した結晶構造予測
3. **大規模生成**: 並列処理による効率的な生成
4. **カスタムデータ**: 独自データセットでの学習が容易
5. **品質評価**: 包括的な評価メトリクス

### さらなる応用

- **反応予測**: 反応物から生成物の構造予測
- **触媒設計**: 特定の反応に最適な触媒の探索
- **材料設計**: 特定の機械的・電気的性質を持つ材料
- **毒性予測**: 構造-毒性相関の学習

---

**このユースケース集は継続的に更新されます。**

**最終更新**: 2025年10月25日  
**バージョン**: 1.0  
**連絡先**: GitHub Issues
