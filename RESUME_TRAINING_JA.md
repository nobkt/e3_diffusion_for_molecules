# 学習の再開方法 (How to Resume Training)

## 概要

`main_qm9.py`で学習を途中から再開する方法を説明します。

## 基本的な使い方

### 方法1: チェックポイントディレクトリから再開（推奨）

```bash
python main_qm9.py \
    --exp_name your_experiment_name \
    --resume outputs/your_experiment_name
```

これにより以下が自動的に読み込まれます：
- モデルの重み (`generative_model_ema.npy` または `generative_model.npy`)
- オプティマイザの状態 (`optim.npy`)
- 学習パラメータ (`args.pickle`)
- 保存されていたエポック番号から自動的に再開

### 方法2: 特定のモデルファイルから再開

```bash
python main_qm9.py \
    --exp_name your_experiment_name \
    --resume outputs/your_experiment_name/generative_model_ema.npy
```

### 方法3: 特定のエポックから再開

```bash
python main_qm9.py \
    --exp_name your_experiment_name \
    --resume outputs/your_experiment_name \
    --start_epoch 150
```

## 質問のケースへの回答

以下のコマンドで200エポック学習したが、学習が不十分だったため継続したい場合：

```bash
# 元のコマンド（200エポックで停止）
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --model egnn_dynamics \
    --lr 1e-4 \
    --nf 256 \
    --n_layers 9 \
    --save_model True \
    --diffusion_steps 1000 \
    --sin_embedding False \
    --n_epochs 200 \
    --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --dequantization deterministic \
    --include_charges False \
    --diffusion_loss_type l2 \
    --batch_size 16 \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding functional_groups_encoding \
    --dataset ase_db \
    --ase_db_path ase.db \
    --test_epochs 10 \
    --include_charges False \
    --no_wandb

# 学習を再開して合計500エポックまで学習する場合
python main_qm9.py \
    --exp_name exp_cond_molecular_descriptors \
    --resume outputs/exp_cond_molecular_descriptors \
    --n_epochs 500 \
    --no_wandb
```

## 重要な注意点

1. **自動的に読み込まれるもの**
   - モデルのアーキテクチャ（レイヤー数、特徴量次元など）
   - 学習率などのハイパーパラメータ
   - オプティマイザの状態（学習の進行状態）
   - 前回保存したエポック番号

2. **上書きできるパラメータ**
   - `--n_epochs`: より多くのエポック数を指定して学習を継続できます
   - `--start_epoch`: 特定のエポックから開始したい場合に指定できます

3. **新しい実験名**
   - 再開した学習は `元の実験名_resume` という名前で保存されます
   - 例: `exp_cond_molecular_descriptors_resume`

4. **必要なファイル**
   チェックポイントディレクトリに以下のファイルが必要です：
   - `generative_model.npy` または `generative_model_ema.npy`（必須）
   - `optim.npy`（推奨：オプティマイザの状態を保持）
   - `args.pickle`（推奨：学習設定を保持）

5. **チェックポイントの確認**
   ```bash
   ls -la outputs/exp_cond_molecular_descriptors/
   ```
   
   以下のファイルがあることを確認してください：
   ```
   args.pickle
   generative_model.npy
   generative_model_ema.npy  # EMA（指数移動平均）版のモデル
   optim.npy
   ```

## トラブルシューティング

### エラー: "No model checkpoint found"

**原因**: チェックポイントディレクトリにモデルファイルが見つからない

**解決方法**:
1. ディレクトリパスが正しいか確認
   ```bash
   ls -la outputs/exp_cond_molecular_descriptors/
   ```

2. `--save_model True` で学習していたか確認

### エラー: "args.pickle not found"

**原因**: 古いバージョンで保存されたチェックポイントの可能性

**解決方法**:
- 警告が表示されますが、現在のコマンドライン引数を使用して学習が継続されます
- すべてのパラメータを再度コマンドラインで指定することを推奨

### 学習が最初からやり直される

**原因**: `start_epoch` が正しく設定されていない

**解決方法**:
- `--start_epoch` パラメータを指定せずに実行してください
- チェックポイントから自動的にエポック番号が読み込まれます

## より詳細な情報

英語版のドキュメント `doc/user_manual.md` の "Resuming Training" セクションを参照してください。

## テスト

再開機能が正しく動作するかテストする場合：

```bash
python test_resume.py
```

このテストスクリプトは以下を検証します：
- チェックポイントディレクトリからの読み込み
- 特定のモデルファイルからの読み込み
- 旧形式（flow.npy）との後方互換性
- 既存のチェックポイントの読み込み
