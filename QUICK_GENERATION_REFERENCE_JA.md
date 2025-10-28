# 条件付き分子生成クイックリファレンス

## 問題ステートメントへの直接回答

**質問**: 条件付き学習で、`--conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding`で学習させたのち、[C,H,O,N]を含み、分子量が100、200で、官能基にOH、COOHを含み、π共役性が0.8、0.9、1.0の条件で分子生成させる場合の生成コマンドを示してください。

**回答**: 以下のコマンドを使用します。

---

## 個別コマンド（各条件を個別に実行）

### 分子量100の場合

```bash
# MW=100, π=0.8
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5

# MW=100, π=0.9
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5

# MW=100, π=1.0
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

### 分子量200の場合

```bash
# MW=200, π=0.8
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5

# MW=200, π=0.9
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5

# MW=200, π=1.0
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

---

## バッチ実行（すべての条件を一度に）

### 方法1: シェルスクリプトを使用

```bash
bash generate_all_conditions.sh outputs/molecular_descriptor_model
```

### 方法2: Pythonスクリプトを使用

```bash
python generate_molecules_with_conditions.py \
    --model_path outputs/molecular_descriptor_model \
    --molecular_weights 100 200 \
    --pi_conjugations 0.8 0.9 1.0 \
    --atom_types C H O N \
    --functional_groups OH COOH \
    --n_sweeps 5
```

### 方法3: ループを使用（Bash）

```bash
for MW in 100 200; do
    for PC in 0.8 0.9 1.0; do
        python eval_conditional_qm9.py \
            --generators_path outputs/molecular_descriptor_model \
            --task qualitative \
            --use_exact_conditions \
            --property_values "molecular_weight=${MW},pi_conjugation_ratio=${PC},atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]" \
            --n_sweeps 5
    done
done
```

---

## コマンドパラメータの説明

| パラメータ | 説明 | 値 |
|-----------|------|-----|
| `--generators_path` | 訓練済みモデルのパス | `outputs/molecular_descriptor_model` |
| `--task` | 生成タスクの種類 | `qualitative` |
| `--use_exact_conditions` | 厳密条件付き生成を有効化 | フラグ（値なし） |
| `--property_values` | 生成条件の指定 | 下記参照 |
| `--n_sweeps` | 各条件でのサンプル数 | `5`（推奨）、`10`など |

### property_values の形式

```
molecular_weight=<値>,pi_conjugation_ratio=<値>,atom_types_encoding=[<原子種>],functional_groups_encoding=[<官能基>]
```

**例**:
```
molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]
```

---

## 実行前の確認事項

### 1. モデルが存在するか確認

```bash
ls outputs/molecular_descriptor_model/
# 以下のファイルが存在するか確認:
# - args.pickle
# - generative_model.npy または generative_model_ema.npy
```

### 2. 学習時の条件を確認

```bash
python -c "import pickle; args = pickle.load(open('outputs/molecular_descriptor_model/args.pickle', 'rb')); print('Conditioning:', args.conditioning)"
```

出力例:
```
Conditioning: ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding']
```

### 3. 必要なPythonパッケージがインストールされているか確認

```bash
python -c "import torch; import ase; import rdkit; print('OK')"
```

---

## 実行例とその出力

### 実行コマンド

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

### 期待される出力

```
Loading model from outputs/molecular_descriptor_model...
Dataset: ase_db
Conditioning properties: ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding']
Using exact conditions mode
Parsed property values:
  molecular_weight: 100.0
  pi_conjugation_ratio: 0.8
  atom_types_encoding: ['C', 'H', 'O', 'N']
  functional_groups_encoding: ['OH', 'COOH']
Generating 5 samples...
[1/5] Generating...
[2/5] Generating...
[3/5] Generating...
[4/5] Generating...
[5/5] Generating...
Done! Samples saved to outputs/molecular_descriptor_model/generated_samples/
```

---

## トラブルシューティング

### エラー: "Property 'molecular_weight' was not used for conditioning during training"

**原因**: モデルが指定したプロパティで訓練されていない

**解決策**: モデルを正しいプロパティで再訓練するか、別のモデルを使用

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path /path/to/database.db \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
    --exp_name molecular_descriptor_model \
    --n_epochs 1000
```

### エラー: "Failed to parse property values"

**原因**: `property_values`の形式が正しくない

**解決策**: 
- シングルクォートで囲む: `'...'`
- カンマの位置を確認
- 角括弧が正しく閉じられているか確認

**正しい例**:
```bash
--property_values 'molecular_weight=100,atom_types_encoding=[C,H,O,N]'
```

**間違った例**:
```bash
--property_values molecular_weight=100, atom_types_encoding=[C,H,O,N]  # スペースとクォートなし
--property_values "molecular_weight=100,atom_types_encoding=[C,H,O,N"  # 閉じ括弧なし
```

### エラー: "No module named 'rdkit'"

**解決策**: RDKitをインストール

```bash
conda install -c conda-forge rdkit
```

または

```bash
pip install rdkit-pypi
```

---

## 高度な使用法

### GPUの使用

```bash
# デフォルトでGPUが利用可能な場合は自動的に使用されます
# CPUのみで実行したい場合:
python eval_conditional_qm9.py --no-cuda ...
```

### バッチサイズの変更

```bash
python eval_conditional_qm9.py \
    --batch_size 10 \
    ...
```

### より多くのサンプルを生成

```bash
python eval_conditional_qm9.py \
    --n_sweeps 50 \
    ...
```

---

## まとめ

問題ステートメントで要求された条件で分子を生成するには、以下のいずれかの方法を使用します：

1. **個別実行**: 各条件（MW=100/200 × π=0.8/0.9/1.0）ごとに`eval_conditional_qm9.py`を実行
2. **バッチ実行**: `generate_all_conditions.sh`または`generate_molecules_with_conditions.py`を使用
3. **ループ実行**: Bashのforループで自動化

すべての方法で、以下の条件が指定されます：
- ✅ 原子種: C, H, O, N
- ✅ 分子量: 100, 200
- ✅ 官能基: OH, COOH
- ✅ π共役性: 0.8, 0.9, 1.0

詳細な説明は `GENERATION_COMMAND_GUIDE_JA.md` を参照してください。
