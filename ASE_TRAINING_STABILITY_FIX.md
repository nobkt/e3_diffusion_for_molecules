# ASE Database Training Stability Fix

## 問題の概要 (Problem Overview)

ASE databaseでの訓練において、以下の問題が発生していました：

1. **非常に大きな勾配ノルム**: 1412.8, 12436.3などの極端に大きな勾配が発生
2. **molecular_weightの正規化警告**: "Large normalized values in 'molecular_weight': max_abs = 3.86"
3. **頻繁な勾配クリッピング警告**: 毎イテレーションで勾配がクリップされ、学習が不安定

## 根本原因の分析 (Root Cause Analysis)

### 1. 勾配クリッピングの初期値が低すぎる
- 初期max_grad_norm = 1.0は、ASE databaseの勾配パターンには不適切
- 大きな勾配値（1000+）に対して適応的な調整が必要

### 2. Molecular Weightの正規化が不適切
- 分子量は幅広い範囲（50-500+）を持つため、通常のMAD計算では不十分
- 正規化後の値が±3.0を超えて警告が頻発

### 3. 学習率が高すぎる
- デフォルトの2e-4は、ASE databaseには高すぎる可能性

## 実装した解決策 (Implemented Solutions)

### 1. 改善された勾配クリッピング (`utils.py`)

```python
# Before: 初期値が低すぎる
max_grad_norm = 1.0

# After: より適切な初期値
max_grad_norm = 10.0

# ノイズ削減: 大きな勾配のみ警告表示
if float(grad_norm) > max_grad_norm * 2.0:
    print(f'Clipped gradient...')
```

**改善点:**
- 初期max_grad_norm: 1.0 → 10.0
- 警告表示の条件を厳格化（ノイズ削減）
- より適切な境界値設定: max(1.0, min(max_grad_norm, 100.0))

### 2. Molecular Weight正規化の改善 (`qm9/utils.py`)

```python
# Molecular weight専用のMAD計算
if property_key == 'molecular_weight':
    min_mad = max(abs(float(mean)) * 0.25, mad * 0.5)
    print(f"Debug: Molecular weight normalization - adjusted MAD: {min_mad:.1f}")

# より大きなクランプ範囲
if key == 'molecular_weight':
    properties = torch.clamp(properties, min=-4.0, max=4.0)

# 調整された警告閾値
warning_threshold = 3.5 if key == 'molecular_weight' else 2.5
```

**改善点:**
- Molecular weight専用のMAD計算（平均の25%）
- クランプ範囲: ±3.0 → ±4.0 (molecular weightの場合)
- 警告閾値: 3.0 → 3.5 (molecular weightの場合)

### 3. 自動学習率調整 (`main_qm9.py`)

```python
# ASE databaseに対する自動学習率調整
if 'ase_db' in args.dataset and args.lr >= 2e-4:
    args.lr = 1e-4  # より安定した学習率
    print(f"Adjusted learning rate for ASE database stability")
```

### 4. 改善された勾配キュー初期化 (`main_qm9.py`)

```python
# Before: 単一の小さな値
gradnorm_queue.add(10.0)

# After: 複数の適切な値で初期化
gradnorm_queue.add(50.0)
gradnorm_queue.add(30.0)
gradnorm_queue.add(20.0)
gradnorm_queue.add(40.0)
gradnorm_queue.add(60.0)  # 5つの値で適応的クリッピングを即座に有効化
```

## 期待される改善効果 (Expected Improvements)

### 1. 勾配安定性の向上
- ✅ 極端に大きな勾配ノルム（1000+）の削減
- ✅ 勾配クリッピング警告の大幅な削減
- ✅ より安定した学習過程

### 2. 正規化警告の削減
- ✅ "Large normalized values in 'molecular_weight'"警告の削減
- ✅ より適切な特徴量正規化
- ✅ 数値的安定性の向上

### 3. 全体的な学習安定性
- ✅ より予測可能な学習曲線
- ✅ 数値的不安定性の削減
- ✅ バランスの取れた要素分布

## 使用方法 (Usage)

### 基本コマンド
```bash
python main_qm9.py \
  --dataset ase_db \
  --ase_db_path your_database.db \
  --batch_size 16 \
  --n_epochs 200 \
  --conditioning molecular_weight
```

### 推奨設定
```bash
python main_qm9.py \
  --dataset ase_db \
  --ase_db_path your_database.db \
  --batch_size 16 \
  --lr 1e-4 \
  --n_epochs 200 \
  --conditioning molecular_weight pi_conjugation_ratio \
  --normalize_factors [1, 1.0, 1]
```

## 修正の検証 (Validation)

修正が正しく適用されているかは以下のコマンドで確認できます：

```bash
python test_gradient_stability_fix.py
```

## トラブルシューティング (Troubleshooting)

### まだ勾配が不安定な場合
1. 学習率をさらに下げる: `--lr 5e-5`
2. バッチサイズを減らす: `--batch_size 8`
3. 条件付け特徴量を減らす: `--conditioning molecular_weight`のみ

### 警告が続く場合
1. 正規化因子を調整: `--normalize_factors [1, 0.5, 1]`
2. より少ない要素を含むデータセットを使用
3. 前処理でoutlierを除去

## 技術的詳細 (Technical Details)

### 勾配クリッピングアルゴリズム
```python
if len(gradnorm_queue) < 5:
    max_grad_norm = 10.0  # 改善された初期値
else:
    max_grad_norm = 1.5 * queue_mean + 2 * queue_std
    max_grad_norm = max(1.0, min(max_grad_norm, 100.0))
```

### Molecular Weight正規化
```python
normalized = (molecular_weight - mean) / adjusted_mad
clamped = torch.clamp(normalized, min=-4.0, max=4.0)
```

これらの改善により、ASE databaseでの学習がより安定し、数値的不安定性が大幅に削減されることが期待されます。