# 継続学習エラー修正の完了報告

## 問題の要約

`example_resume_training.sh`を実行した際に発生したサイズミスマッチエラー：

```
RuntimeError: Error(s) in loading state_dict for EnVariationalDiffusion:
	size mismatch for dynamics.egnn.embedding.weight: copying a param with shape torch.Size([256, 39]) from checkpoint, the shape in current model is torch.Size([256, 33]).
```

## 根本原因

PR#137とPR#139の修正では`args.context_node_nf`の保存・復元に焦点を当てていましたが、**コンテキスト特徴量のサイズがデータベースの実際の内容に依存する**という根本的な問題に対処できていませんでした。

具体的には：
- `atom_types_encoding`のサイズは、データベース内の一意な原子タイプの数に依存
- `functional_groups_encoding`のサイズは、データベース内の一意な官能基の数に依存
- データベースが変更されると、これらの特徴量のサイズが変わり、`context_node_nf`が変化

## 実装した解決策

### 1. データセット設定の保存と復元
- `dataset_info.pickle`を新たに保存（`atom_encoder`、`atom_decoder`などを含む）
- 継続時に保存された原子タイプ設定を復元
- モデルが元のトレーニングと同じ原子タイプで作成されることを保証

### 2. コンテキスト特徴量サイズの検証
- 継続時に現在のデータベースから`context_node_nf`を計算
- 保存された値と比較
- 不一致の場合、明確なエラーメッセージを表示

### 3. デバッグログの追加
- `context_node_nf`の値を各段階で追跡
- 問題発生時の診断を容易に

## 変更されたファイル

**main_qm9.py**:
- 204-222行: 継続時に`dataset_info.pickle`を読み込み
- 262-267行: 保存された原子タイプを復元
- 294-333行: `context_node_nf`を検証し、明確なエラーメッセージを提供
- 486-488行: `dataset_info.pickle`を保存
- 497-499行: エポック固有の`dataset_info.pickle`を保存

**追加されたドキュメント**:
- `RESUME_CONTEXT_MISMATCH_COMPLETE_FIX.md`: 詳細な英語ドキュメント
- `RESUME_CONTEXT_MISMATCH_COMPLETE_FIX_JA.md`: 詳細な日本語ドキュメント

## 使用方法

### 既存のチェックポイント（`dataset_info.pickle`なし）

元のトレーニングで使用したのと**全く同じデータベースファイル**を使用してください：

```bash
python main_qm9.py \
    --resume outputs/exp_cond_molecular_descriptors \
    --n_epochs 500 \
    --no_wandb
```

### 期待される動作

**ケース1: データベース未変更**
```
Loading dataset_info from outputs/exp_name/dataset_info.pickle
...
Resuming training: using saved context_node_nf = 27
Current database produces context_node_nf = 27
✓ コンテキスト特徴量が一致！
✓ トレーニングが正常に再開
```

**ケース2: データベース変更**
```
Resuming training: using saved context_node_nf = 27
Current database produces context_node_nf = 21

エラー: コンテキスト特徴量サイズの不一致！
チェックポイントは context_node_nf = 27 でトレーニングされました
しかし現在のデータベースは context_node_nf = 21 を生成します

修正方法:
  - 元のトレーニングで使用したのと同じデータベースファイルを使用してください
  - データベースに同じ分子/プロパティがあることを確認してください
```

## 今後のトレーニング

今後、すべてのチェックポイントは自動的に`dataset_info.pickle`を含むため、継続トレーニングがより確実になります。

## 重要なポイント

1. **継続トレーニングには元のデータベースが必要**: モデルアーキテクチャは、元のトレーニング時のデータベースの内容（原子タイプ、官能基など）に依存します。

2. **明確なエラーメッセージ**: データベースが変更されている場合、修正が必要なことが明確に通知されます。

3. **後方互換性**: 古いチェックポイント（`dataset_info.pickle`なし）でも、元のデータベースを使用すれば継続可能です。

4. **前方互換性**: 新しいチェックポイントは自動的に必要な情報をすべて保存します。

## テスト状況

- ✅ 構文チェック合格
- ✅ ロジック検証合格（テストスクリプト）
- ⚠️ 実際の継続シナリオでのテストは、チェックポイントが利用可能になり次第実施可能

## 詳細ドキュメント

完全な技術詳細、使用例、トラブルシューティングについては、以下のドキュメントをご参照ください：
- 日本語: `RESUME_CONTEXT_MISMATCH_COMPLETE_FIX_JA.md`
- English: `RESUME_CONTEXT_MISMATCH_COMPLETE_FIX.md`

## まとめ

この修正により：
- ✅ サイズミスマッチエラーの根本原因を特定
- ✅ 包括的な解決策を実装
- ✅ 明確なエラーメッセージで問題診断を容易に
- ✅ 将来の継続トレーニングをより確実に

PR#137とPR#139で試みられた修正を完全に補完し、継続トレーニングを確実に機能させる解決策を提供しました。
