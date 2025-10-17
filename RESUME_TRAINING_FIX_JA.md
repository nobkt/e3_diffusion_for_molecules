# Resume Training Fix - Summary (日本語)

## 問題の内容

`example_resume_training.sh`を実行して継続学習を実行したところ、以下のエラーが発生しました：

```
RuntimeError: Error(s) in loading state_dict for EnVariationalDiffusion:
	size mismatch for dynamics.egnn.embedding.weight: copying a param with shape torch.Size([256, 39]) from checkpoint, the shape in current model is torch.Size([256, 33]).
```

## 原因

継続学習を実行する際、以下の流れで問題が発生していました：

1. 保存された`args.pickle`から学習パラメータをロード
2. しかし、その後データセットから`context_node_nf`を**再計算**
3. 再計算された値が元の学習時と異なる場合がある
4. 新しい`context_node_nf`でモデルを作成
5. しかしチェックポイントは古い`context_node_nf`で保存されている
6. 結果：埋め込み層のサイズが一致せずエラー（39 vs 33特徴量）

具体的には：
- チェックポイント：11原子タイプ + 1時間 + 27コンテキスト特徴 = 39特徴量
- 再作成されたモデル：11原子タイプ + 1時間 + 21コンテキスト特徴 = 33特徴量（異なる！）

## 解決策

`main_qm9.py`を修正して、継続学習時には保存された`context_node_nf`を使用するようにしました：

### 修正内容

```python
# 継続学習時は、保存されたcontext_node_nfを使用
if args.resume is not None and hasattr(args, 'context_node_nf'):
    # チェックポイントから保存されたcontext_node_nfを使用
    context_node_nf = args.context_node_nf
    print(f'Resuming training: using saved context_node_nf = {context_node_nf}')
    
    # プロパティの正規化は計算が必要
    if len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
else:
    # 新規学習：データから計算（従来通り）
    if len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
```

### 修正のポイント

1. **継続学習時**：`args.resume`が設定されており、`args.context_node_nf`が存在する場合
   - 保存された`context_node_nf`をそのまま使用
   - これにより、モデルアーキテクチャがチェックポイントと一致

2. **新規学習時**：`args.resume`が設定されていない場合
   - データから`context_node_nf`を計算（従来通り）
   - 動作は変更なし

3. **後方互換性**：古いチェックポイント（`context_node_nf`が保存されていない）
   - `hasattr`チェックによりフォールバック
   - データから再計算される

## テスト結果

すべてのテストが成功しました：

```
✓ context_node_nf preservation: PASSED
✓ context_node_nf calculation for new training: PASSED  
✓ Backward compatibility: PASSED
✓ Bug scenario reproduction: PASSED
✓ All existing resume tests: PASSED
✓ Integration test: PASSED
  - 修正なし: 39特徴量 vs 33期待値 → サイズ不一致 ✗
  - 修正あり: 39特徴量 vs 39期待値 → 成功 ✓
```

## 使い方

修正後は、以下のコマンドで継続学習が正常に動作します：

```bash
# チェックポイントディレクトリから継続
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --resume outputs/exp_cond_molecular_descriptors \
    --n_epochs 500 \
    --no_wandb

# または example_resume_training.sh を使用
bash example_resume_training.sh
```

## 影響範囲

- ✅ 継続学習のエラーを修正
- ✅ 古いチェックポイントとの互換性を維持
- ✅ 新規学習の動作は変更なし
- ✅ 既存のすべてのテストが成功

## 技術的詳細

### モデルアーキテクチャ

```
EGNN埋め込み層の入力特徴数 = dynamics_in_node_nf + context_node_nf

where:
- in_node_nf = len(atom_decoder)  # 原子タイプ数（例：11）
- dynamics_in_node_nf = in_node_nf + 1 (if condition_time)  # 時間条件を含む
- context_node_nf = prepare_context()の出力次元  # コンディショニング特徴

例：
- in_node_nf = 11 (H, C, N, O, F, Si, P, S, Cl, Br, I)
- dynamics_in_node_nf = 12 (11 + 1 for time)
- context_node_nf = 27 (molecular_weight + pi_conjugation_ratio + atom_types_encoding + functional_groups_encoding)
- total = 12 + 27 = 39
```

### エラーが発生する理由

1. 学習時：データAで`context_node_nf = 27`、モデル保存
2. 継続時：データBで`context_node_nf = 21`（異なる値）を再計算
3. 新モデル作成：`total = 12 + 21 = 33`特徴量
4. チェックポイントロード：`39`特徴量を期待
5. エラー：`33 != 39`

### 修正の効果

継続学習時に保存された`context_node_nf = 27`を使用することで：
- 新モデル：`total = 12 + 27 = 39`特徴量
- チェックポイント：`39`特徴量
- 成功：`39 == 39` ✓

## 関連ファイル

- `main_qm9.py`: 主な修正箇所（267-290行目）
- `test_resume_context_nf.py`: コンテキストノード特徴保存のテスト
- `test_resume_integration.py`: 統合テスト（実際のバグシナリオを再現）
- `test_resume.py`: 既存の継続学習テスト

## 追加情報

この修正は、モデルアーキテクチャの整合性を保証する重要な修正です。特に以下のシナリオで重要：

1. 異なるデータセットでの継続学習
2. コンディショニング設定が変更された場合
3. データセットの統計が変化した場合

修正により、チェックポイントから継続する際は常に保存時のアーキテクチャが使用され、エラーを防ぎます。
