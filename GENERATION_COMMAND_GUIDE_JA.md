# 条件付き分子生成コマンドガイド

## 概要

このガイドでは、`molecular_weight`、`pi_conjugation_ratio`、`atom_types_encoding`、`functional_groups_encoding`を使用した条件付き学習後の分子生成コマンドを詳しく説明します。

## 前提条件

学習時に以下のコマンドでモデルを訓練済みであることを前提とします：

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

## 問題ステートメントに対する回答

### 条件の詳細

以下の条件で分子を生成します：
- **原子種**: C, H, O, N を含む
- **分子量**: 100, 200
- **官能基**: OH (ヒドロキシル基), COOH (カルボキシル基)
- **π共役性**: 0.8, 0.9, 1.0

### 生成コマンド

#### 基本的な生成コマンド（全条件を一度に指定）

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 10
```

#### 条件を変えた複数の生成コマンド

##### 1. 分子量100、π共役性0.8
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

##### 2. 分子量100、π共役性0.9
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

##### 3. 分子量100、π共役性1.0
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

##### 4. 分子量200、π共役性0.8
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

##### 5. 分子量200、π共役性0.9
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

##### 6. 分子量200、π共役性1.0
```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=200,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5
```

## バッチ生成スクリプト

すべての組み合わせを一度に生成する場合は、付属の`generate_all_conditions.sh`スクリプトを使用できます：

```bash
bash generate_all_conditions.sh
```

または、Pythonスクリプトを使用：

```bash
python generate_molecules_with_conditions.py
```

## コマンドライン引数の説明

### 必須引数

- `--generators_path`: 訓練済みモデルが保存されているディレクトリパス
- `--task`: 生成タスクの種類（`qualitative`または`quantitative`）
- `--use_exact_conditions`: 厳密な条件付き生成を有効化するフラグ
- `--property_values`: 条件を指定する文字列

### property_values の形式

`property_values`引数は、カンマ区切りの`property=value`形式で指定します：

#### スカラー値（float/int）
```
molecular_weight=100
pi_conjugation_ratio=0.9
```

#### リスト値（原子種）
```
atom_types_encoding=[C,H,N,O]
```

#### リスト値（官能基）
```
functional_groups_encoding=[OH,COOH]
functional_groups_encoding=[OH,COOH,NH2]
```

#### 複合指定
```
molecular_weight=100,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]
```

### オプション引数

- `--n_sweeps`: 生成するサンプル数（デフォルト: 10）
- `--batch_size`: バッチサイズ（デフォルト: 1）
- `--no-cuda`: CPUで実行する場合に指定

## 官能基の指定方法

### サポートされる官能基名

システムは以下の一般的な官能基名をサポートしています：

- `OH`: ヒドロキシル基（水酸基）
- `COOH`: カルボキシル基（カルボン酸基）
- `NH2`: アミノ基
- `CHO`: アルデヒド基
- `COCH3`: アセチル基
- `carbonyl`: カルボニル基（一般）
- `hydroxyl`: ヒドロキシル基（別名）
- `amino`: アミノ基（別名）
- `carboxyl`: カルボキシル基（別名）

### SMARTS パターンでの指定（高度）

より詳細な制御が必要な場合、SMARTSパターンを使用できます：

```bash
--property_values 'functional_groups_encoding=[[CX3](=O)[OX2H1],[NX3;H2,H1;!$(NC=O)]]'
```

ただし、シェルでの特殊文字エスケープに注意してください。

## 出力

生成された分子は以下の場所に保存されます：

- **XYZファイル**: `outputs/molecular_descriptor_model/generated_samples/`
- **可視化画像**: `outputs/molecular_descriptor_model/visualizations/`
- **評価結果**: `outputs/molecular_descriptor_model/evaluation_results.txt`

## トラブルシューティング

### よくある問題

#### 1. "Property was not used for conditioning during training"

**原因**: 生成時に指定したプロパティが学習時に含まれていない

**解決策**: 学習時の`--conditioning`引数を確認し、同じプロパティを指定する

#### 2. "Failed to parse property values"

**原因**: `property_values`の形式が正しくない

**解決策**: 
- カンマの位置を確認
- 等号の周りにスペースがないか確認
- リストの角括弧が正しく閉じられているか確認

#### 3. モデルファイルが見つからない

**原因**: `--generators_path`が間違っている

**解決策**: 
```bash
ls outputs/
# 正しいモデルディレクトリ名を確認
```

## 応用例

### 1. 薬物様分子の生成

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=350,pi_conjugation_ratio=0.6,atom_types_encoding=[C,H,O,N,S]' \
    --n_sweeps 20
```

### 2. 小分子の系統的生成

分子量を50から200まで50ずつ変化させる：

```bash
for mw in 50 100 150 200; do
    python eval_conditional_qm9.py \
        --generators_path outputs/molecular_descriptor_model \
        --task qualitative \
        --use_exact_conditions \
        --property_values "molecular_weight=${mw},atom_types_encoding=[C,H,O,N]" \
        --n_sweeps 5
done
```

### 3. 高共役性分子の生成

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'pi_conjugation_ratio=0.95,atom_types_encoding=[C,H]' \
    --n_sweeps 10
```

## まとめ

本ガイドで示した生成コマンドを使用することで、以下が可能になります：

✅ 特定の分子量（100, 200）での分子生成
✅ 特定のπ共役性（0.8, 0.9, 1.0）での分子生成
✅ 特定の原子種（C, H, O, N）を含む分子の生成
✅ 特定の官能基（OH, COOH）を含む分子の生成
✅ これらの条件を組み合わせた厳密な条件付き生成

各コマンドは独立して実行でき、生成される分子は指定された条件を満たします。

## 参考資料

- [EXACT_CONDITIONAL_GENERATION.md](EXACT_CONDITIONAL_GENERATION.md) - 厳密な条件付き生成の詳細
- [README.md](README.md) - プロジェクトの概要
- [doc/complete_user_guide_ja.md](doc/complete_user_guide_ja.md) - 完全なユーザーガイド
