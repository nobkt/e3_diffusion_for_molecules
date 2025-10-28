# 条件付き分子生成機能 - 実装完了サマリー

## 問題ステートメント

> 条件付き学習で、`--conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding`で学習させたのち、[C,H,O,N]を含み、分子量が100、200で、官能基にOH、COOHを含み、π共役性が0.8、0.9、1.0の条件で分子生成させる場合の生成コマンドを示してください。

## 解決策の概要

問題ステートメントに対する完全な解決策として、以下のドキュメントとスクリプトを作成しました：

### 📚 ドキュメント

1. **ANSWER_TO_PROBLEM_STATEMENT_JA.md**
   - 問題ステートメントへの直接的な回答
   - 6つの生成コマンドを明示
   - 最も簡潔で分かりやすい回答

2. **QUICK_GENERATION_REFERENCE_JA.md**
   - クイックリファレンス
   - コマンドパラメータの詳細説明
   - トラブルシューティング

3. **GENERATION_COMMAND_GUIDE_JA.md**
   - 包括的なガイド
   - 使用例と応用例
   - 詳細な説明とベストプラクティス

4. **EXACT_CONDITIONAL_GENERATION.md** (既存)
   - 技術的な実装詳細（英語）
   - システムアーキテクチャ
   - API リファレンス

### 🛠️ 自動化スクリプト

1. **show_generation_commands.sh**
   ```bash
   bash show_generation_commands.sh
   ```
   - すべての生成コマンドを表示（実行はしない）
   - 問題ステートメントの要求に対する最も直接的な回答

2. **generate_all_conditions.sh**
   ```bash
   bash generate_all_conditions.sh outputs/molecular_descriptor_model
   ```
   - 6つの条件すべてを自動的に実行
   - ログファイルとサマリーを生成
   - 実行時間とステータスを記録

3. **generate_molecules_with_conditions.py**
   ```bash
   python generate_molecules_with_conditions.py \
       --model_path outputs/molecular_descriptor_model \
       --molecular_weights 100 200 \
       --pi_conjugations 0.8 0.9 1.0 \
       --atom_types C H O N \
       --functional_groups OH COOH
   ```
   - Pythonベースの柔軟なスクリプト
   - コマンドライン引数でカスタマイズ可能
   - 詳細なログとエラーハンドリング

### 📝 README更新

README.mdに新しいセクション「Exact Conditional Generation」を追加：
- クイック使用例
- ドキュメントへのリンク
- 自動化スクリプトの使用方法

## 生成コマンドの例

### 個別実行（6パターン）

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

### バッチ実行

```bash
# シェルスクリプトで一括実行
bash generate_all_conditions.sh outputs/molecular_descriptor_model

# Pythonスクリプトで一括実行
python generate_molecules_with_conditions.py --model_path outputs/molecular_descriptor_model
```

## 技術的特徴

### 厳密な条件付け

- ✅ **分子量**: 正確な値を指定（100, 200）
- ✅ **π共役性**: 0.0～1.0の範囲で指定（0.8, 0.9, 1.0）
- ✅ **原子種**: リストで指定（[C,H,O,N]）
- ✅ **官能基**: リストで指定（[OH,COOH]）

### 使用可能なフラグ

- `--use_exact_conditions`: 厳密な条件付き生成を有効化
- `--property_values`: 条件を文字列で指定
- `--n_sweeps`: 各条件での生成数
- `--task qualitative`: 定性的生成タスク

### プロパティ値の形式

```
property1=value1,property2=value2,property3=[list],property4=[list]
```

例：
```
molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]
```

## 使用方法

### 1. モデルの訓練

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

### 2. 分子の生成

```bash
# 方法1: 個別コマンド
python eval_conditional_qm9.py \
    --generators_path outputs/molecular_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \
    --n_sweeps 5

# 方法2: 自動化スクリプト
bash generate_all_conditions.sh outputs/molecular_descriptor_model

# 方法3: Pythonスクリプト
python generate_molecules_with_conditions.py --model_path outputs/molecular_descriptor_model
```

### 3. コマンドの確認

実行せずにコマンドを確認：
```bash
bash show_generation_commands.sh
```

## ファイル一覧

### ドキュメント
- `ANSWER_TO_PROBLEM_STATEMENT_JA.md` - 問題ステートメントへの直接回答
- `QUICK_GENERATION_REFERENCE_JA.md` - クイックリファレンス
- `GENERATION_COMMAND_GUIDE_JA.md` - 詳細ガイド
- `EXACT_CONDITIONAL_GENERATION.md` - 技術ドキュメント（英語）
- `README.md` - 更新済み（新セクション追加）

### スクリプト
- `show_generation_commands.sh` - コマンド表示スクリプト
- `generate_all_conditions.sh` - バッチ実行スクリプト（Bash）
- `generate_molecules_with_conditions.py` - バッチ実行スクリプト（Python）

### 既存の関連ファイル
- `eval_conditional_qm9.py` - メイン評価スクリプト
- `exact_conditioning_demo.py` - デモスクリプト
- `qm9/sampling.py` - サンプリング実装

## トラブルシューティング

### よくある問題

1. **モデルが見つからない**
   ```bash
   # モデルディレクトリを確認
   ls outputs/molecular_descriptor_model/
   ```

2. **プロパティが学習時に含まれていない**
   ```bash
   # 学習時の設定を確認
   python -c "import pickle; args = pickle.load(open('outputs/molecular_descriptor_model/args.pickle', 'rb')); print(args.conditioning)"
   ```

3. **property_valuesの形式エラー**
   - シングルクォートで囲む
   - カンマの位置を確認
   - 角括弧が正しく閉じられているか確認

## まとめ

問題ステートメントに対する完全な解決策を提供：

✅ **6つの生成コマンド** - すべての条件の組み合わせ  
✅ **3つの自動化スクリプト** - 簡単なバッチ実行  
✅ **4つのドキュメント** - 日本語と英語の詳細ガイド  
✅ **README更新** - 使用方法のクイックスタート  

最も簡単な使用方法：
1. `ANSWER_TO_PROBLEM_STATEMENT_JA.md` を読む
2. `bash show_generation_commands.sh` でコマンドを確認
3. `bash generate_all_conditions.sh outputs/molecular_descriptor_model` で実行

---

**作成日**: 2025-10-28  
**対応問題**: 条件付き分子生成コマンドの提示
