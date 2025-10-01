# CSV Export Feature Implementation Summary

## 概要 (Overview)

生成条件（molecular_weight, pi_conjugation_ratio, atom_types_encoding, functional_groups_encoding）を
データセットの全分子に対してCSV形式で出力する機能を実装しました。

Implemented a feature to export generation conditions (molecular_weight, pi_conjugation_ratio, 
atom_types_encoding, functional_groups_encoding) for all molecules in a dataset to CSV format.

## 実装内容 (Implementation Details)

### 1. 新しいコマンドライン引数 (New Command-Line Argument)

`main_qm9.py` に `--export_conditions_csv` オプションを追加:

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path ase.db \
    --export_conditions_csv ./output \
    --no_wandb \
    --remove_h
```

### 2. 出力されるCSVファイル (Output CSV Files)

4つのCSVファイルが生成されます:

#### (1) molecular_weight.csv
```csv
ID,分子の組成,分子量
0,C6H12O6,180.1559
1,C8H10N4O2,194.1906
```

- **ID**: 分子の識別番号
- **分子の組成**: 分子式（例: C6H12O6）
- **分子量**: 分子量（ダルトン単位）

#### (2) pi_conjugation_ratio.csv
```csv
ID,分子の組成,π共役比率
0,C6H12O6,0.1667
1,C8H10N4O2,0.5556
```

- **ID**: 分子の識別番号
- **分子の組成**: 分子式
- **π共役比率**: π共役結合の比率（0.0～1.0）

#### (3) atom_types_encoding.csv
```csv
ID,分子の組成,C,H,N,O,F
0,C6H12O6,1,1,0,1,0
1,C8H10N4O2,1,1,1,1,0
```

- **ID**: 分子の識別番号
- **分子の組成**: 分子式
- **原子種の列**: 各原子種の有無（1=存在、0=不在）

#### (4) functional_groups_encoding.csv
```csv
ID,分子の組成,carbonyl,hydroxyl,amine
0,C6H12O6,1,1,0
1,C8H10N4O2,1,0,1
```

- **ID**: 分子の識別番号
- **分子の組成**: 分子式
- **官能基の列**: 各官能基の有無（1=存在、0=不在）

### 3. 動作 (Behavior)

- `--export_conditions_csv` オプションが指定されると:
  1. データセットを読み込む
  2. 全分子（train/valid/testの全スプリット）の生成条件を計算
  3. 4つのCSVファイルを出力
  4. プログラムを終了（訓練は行わない）

- オプションが指定されない場合:
  - 通常通り訓練が実行される

## ファイル構成 (File Structure)

```
e3_diffusion_for_molecules/
├── main_qm9.py                      # 修正: --export_conditions_csv オプション追加
├── qm9/
│   └── dataset.py                   # 修正: export_generation_conditions_to_csv 関数追加
├── CSV_EXPORT_USAGE.md              # 新規: 詳細な使用方法ドキュメント
├── csv_export_example.py            # 新規: 使用例スクリプト
├── demo_csv_export.py               # 新規: 出力フォーマットのデモ
└── test_csv_export.py               # 新規: バリデーションテスト
```

## テスト結果 (Test Results)

全てのバリデーションテストが成功しました:

```
✓ PASS: Syntax
✓ PASS: Main arguments
✓ PASS: Export function
✓ PASS: CSV headers
✓ PASS: Documentation

Total: 5/5 tests passed
```

## 使用例 (Usage Examples)

### 例1: ASEデータベースからのエクスポート

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path ase.db \
    --export_conditions_csv ./output \
    --no_wandb \
    --remove_h
```

### 例2: QM9データセットからのエクスポート

```bash
python main_qm9.py \
    --dataset qm9 \
    --datadir qm9/temp \
    --export_conditions_csv ./qm9_output \
    --no_wandb
```

### 例3: サンプルスクリプトの使用

```bash
python csv_export_example.py \
    --dataset ase_db \
    --ase_db_path ase.db \
    --output ./csv_output \
    --remove_h
```

## データ分析例 (Data Analysis Example)

生成されたCSVファイルはPandasで簡単に分析できます:

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('output/molecular_weight.csv')

# 統計情報を表示
print(df['分子量'].describe())

# 分布をプロット
df['分子量'].hist(bins=50)
plt.xlabel('Molecular Weight (Da)')
plt.ylabel('Frequency')
plt.title('Molecular Weight Distribution')
plt.show()

# 特定の範囲の分子を検索
target_molecules = df[(df['分子量'] >= 150) & (df['分子量'] <= 200)]
print(f'Found {len(target_molecules)} molecules in target range')
```

## 主な特徴 (Key Features)

✓ 全分子（train/valid/testスプリット）のデータを統合して出力
✓ 日本語のカラムヘッダーで直感的に理解可能
✓ UTF-8エンコーディングでExcel/LibreOffice互換
✓ カテゴリカル特徴（原子種、官能基）のバイナリエンコーディング
✓ 分子式による分子の識別が容易

## ユースケース (Use Cases)

1. **データセット分析**
   - 分子の多様性を理解
   - 一般的な官能基を特定
   - 分子量分布を分析

2. **条件付き生成**
   - 生成のための目標条件を選択
   - 条件付けデータを準備
   - 生成された分子が条件に合致することを検証

3. **データ探索**
   - 特性分布を可視化
   - 特定の特性を持つ分子を検索
   - データセットを比較

## ドキュメント (Documentation)

詳細なドキュメントは以下を参照してください:

- **CSV_EXPORT_USAGE.md**: 機能の詳細な説明と使用方法
- **csv_export_example.py**: 実行可能な使用例
- **demo_csv_export.py**: 出力フォーマットのデモンストレーション
- **test_csv_export.py**: 機能の検証テスト

## 技術詳細 (Technical Details)

### 実装場所

1. **main_qm9.py**: 
   - Line 135-139: `--export_conditions_csv` 引数の定義
   - Line 214-231: CSV エクスポートロジック

2. **qm9/dataset.py**:
   - Line 911-1138: `export_generation_conditions_to_csv()` 関数

### 依存関係

- 既存の依存関係のみを使用（新しい依存関係は不要）
- PyTorch, NumPy, CSV標準ライブラリ
- ASEデータベースの場合: OpenBabel（分子記述子の計算に使用）

## まとめ (Summary)

本実装により、データセット内の全分子の生成条件を簡単にCSV形式で出力できるようになりました。
これにより、データセットの分析、条件付き生成の準備、データ探索が容易になります。

This implementation allows easy export of generation conditions for all molecules in a dataset to CSV format.
This facilitates dataset analysis, preparation for conditional generation, and data exploration.
