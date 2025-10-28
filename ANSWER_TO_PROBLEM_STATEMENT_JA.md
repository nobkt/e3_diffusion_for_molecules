# 問題ステートメントへの回答

## 質問

条件付き学習で、`--conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding`で学習させたのち、[C,H,O,N]を含み、分子量が100、200で、官能基にOH、COOHを含み、π共役性が0.8、0.9、1.0の条件で分子生成させる場合の生成コマンドを示してください。

## 回答

### 生成コマンド（6パターン）

以下のコマンドを使用して、指定された条件で分子を生成できます：

#### 分子量100の場合

**1. MW=100, π=0.8:**
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

**2. MW=100, π=0.9:**
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

**3. MW=100, π=1.0:**
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

#### 分子量200の場合

**4. MW=200, π=0.8:**
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

**5. MW=200, π=0.9:**
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

**6. MW=200, π=1.0:**
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

### 一括実行方法

すべての組み合わせを自動的に実行する場合は、以下のいずれかの方法を使用できます：

**方法1: シェルスクリプト**
```bash
bash generate_all_conditions.sh outputs/molecular_descriptor_model
```

**方法2: Pythonスクリプト**
```bash
python generate_molecules_with_conditions.py \
    --model_path outputs/molecular_descriptor_model \
    --molecular_weights 100 200 \
    --pi_conjugations 0.8 0.9 1.0 \
    --atom_types C H O N \
    --functional_groups OH COOH \
    --n_sweeps 5
```

### コマンドパラメータの説明

- `--generators_path`: 訓練済みモデルが保存されているディレクトリ
- `--task qualitative`: 定性的な生成タスク（複数の条件で生成）
- `--use_exact_conditions`: 厳密な条件付き生成を有効化するフラグ
- `--property_values`: 生成条件を指定する文字列
  - `molecular_weight=<値>`: 分子量（原子質量単位）
  - `pi_conjugation_ratio=<値>`: π共役性の比率（0.0～1.0）
  - `atom_types_encoding=[<原子種>]`: 含まれる原子種のリスト
  - `functional_groups_encoding=[<官能基>]`: 含まれる官能基のリスト
- `--n_sweeps`: 各条件で生成するサンプル数

### 前提条件

これらのコマンドを実行する前に、以下の条件でモデルが訓練されている必要があります：

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path /path/to/your/database.db \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
    --exp_name molecular_descriptor_model \
    --n_epochs 1000 \
    --batch_size 32 \
    --lr 1e-4 \
    --nf 192 \
    --n_layers 9 \
    --save_model True
```

### 参考資料

詳細な説明やトラブルシューティングについては、以下のドキュメントを参照してください：

- [QUICK_GENERATION_REFERENCE_JA.md](QUICK_GENERATION_REFERENCE_JA.md) - クイックリファレンス
- [GENERATION_COMMAND_GUIDE_JA.md](GENERATION_COMMAND_GUIDE_JA.md) - 詳細ガイド
- [EXACT_CONDITIONAL_GENERATION.md](EXACT_CONDITIONAL_GENERATION.md) - 技術ドキュメント（英語）

---

**まとめ**: 上記の6つのコマンド、またはバッチ実行スクリプトを使用することで、指定された全ての条件（分子量100/200、π共役性0.8/0.9/1.0、原子種C/H/O/N、官能基OH/COOH）で分子を生成できます。
