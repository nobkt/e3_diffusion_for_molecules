# エラー修正の概要 (日本語)

## 問題
条件付き学習中に以下のエラーが発生しました:

```
RuntimeError: shape '[11, 27]' is invalid for input of size 187
```

このエラーは、`--conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding` を使用した場合に、Epoch 145の iteration 3300で発生しました。

## 原因
`qm9/utils.py`の`prepare_context()`関数に、特徴量の種類を判定する際のバグがありました:

```python
if properties.size(1) == n_nodes:
    # ノード特徴量として扱う
```

`atom_types_encoding`が17個の特徴量を持ち、バッチ内の分子が17個のノードを持つ場合、本来はグローバル特徴量(分子全体の特徴)として扱うべきところを、誤ってノード特徴量として扱っていました。

### 結果として:
- 期待されるcontext形状: `(batch_size, n_nodes, 27)` 
  - 27 = 1 (molecular_weight) + 1 (pi_conjugation_ratio) + 17 (atom_types) + 8 (functional_groups)
- 実際のcontext形状: `(batch_size, n_nodes, 11)`
  - 11 = 1 + 1 + 1 (atom_typesが誤って1特徴量として扱われた) + 8

モデルは27個の特徴量を期待しているのに、実際には11個しか受け取れず、形状の不一致エラーが発生しました。

## 修正内容

### 1. 特徴量分類の修正 (qm9/utils.py)
常にグローバル特徴量として扱うべき特徴量のリストを明示的に定義:

```python
global_features = {'atom_types_encoding', 'functional_groups_encoding', 
                  'molecular_weight', 'pi_conjugation_ratio'}

if key in global_features:
    # 次元に関係なく、常にグローバル特徴量として扱う
    n_features = properties.size(1)
    reshaped = properties.view(batch_size, 1, n_features).repeat(1, n_nodes, 1)
    context_node_nf += n_features
```

### 2. 検証チェックの追加
- 条件付けキーがminibatchとproperty_normsの両方に存在することを確認
- 最終的なcontext形状が期待される次元と一致することを検証
- デバッグのための明確なエラーメッセージを提供

## テスト方法

元のコマンドで実行できます:

```bash
python main_qm9.py --exp_name exp_cond_molecular_descriptors \
  --model egnn_dynamics --lr 1e-4 --nf 256 --n_layers 9 \
  --save_model True --diffusion_steps 1000 --sin_embedding False \
  --n_epochs 200 --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --dequantization deterministic --include_charges False \
  --diffusion_loss_type l2 --batch_size 16 \
  --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
  --dataset ase_db --ase_db_path ase.db \
  --test_epochs 10 --no_wandb
```

## 期待される動作
✅ 形状不一致エラーなしで学習が進行します
✅ Contextが正しい形状 `(batch_size, n_nodes, 27)` で作成されます
✅ モデルがcontextを正常に処理できます

## 修正されたファイル
1. `qm9/utils.py` - prepare_context関数の修正
2. `PREPARE_CONTEXT_FIX.md` - 詳細な説明(英語)
3. `CONTEXT_SHAPE_FIX_SUMMARY.md` - 修正の概要(英語)
4. `test_context_fix.py` - 修正を実証するテスト
5. `TESTING_GUIDE.md` - テスト手順書(英語)

## 影響範囲
この修正により、以下の条件付け特徴量を正しく使用できるようになります:
- `atom_types_encoding` - 分子中の原子タイプのマルチホットエンコーディング
- `functional_groups_encoding` - 官能基のマルチホットエンコーディング
- `molecular_weight` - 分子量(原子質量単位)
- `pi_conjugation_ratio` - π結合の比率

これらの特徴量を組み合わせて使用しても、形状の不一致が発生しなくなり、より高度な条件付き分子生成が可能になります。

## 後方互換性
標準的なQM9の特徴量(homo, lumo, gapなど)との後方互換性は維持されています。これらはスカラー特徴量(1次元)なので、修正の影響を受けません。
