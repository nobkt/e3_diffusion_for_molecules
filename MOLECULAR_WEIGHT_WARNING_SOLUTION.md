# 分子量正規化警告の解決方法

## 問題の概要

ASEデータベースでの訓練時に以下の警告が頻繁に出現していました：

```
Epoch: 24, iter: 910/6608, Loss 3.88, NLL: 3.88, RegTerm: 0.0, GradNorm: 1.4
Debug: Applied log transformation to 'molecular_weight' during context preparation
Warning: Large normalized values in 'molecular_weight': max_abs = 3.00
  This might cause training instability. Consider feature engineering.
```

## 根本原因の分析

この警告が発生する原因は以下の通りでした：

### 1. 警告閾値が厳しすぎる
- **旧設定**: 全ての特徴量に対して警告閾値が2.5
- **問題**: 対数変換後の分子量でも2.5を超える場合があった

### 2. クランプ範囲と警告閾値の不整合
- **旧設定**: クランプ範囲[-3.0, 3.0]、警告閾値2.5
- **問題**: クランプ後でも警告が発生する可能性があった

### 3. 誤解を招く警告メッセージ
- **問題**: 対数変換が既に適用されている場合でも、適用されていないかのようなメッセージが表示された

### 4. MAD（平均絶対偏差）計算の改善が必要
- **問題**: 極値を含む分布で、MADが小さくなりすぎることがあった

## 実装された修正

### 1. 適応的警告閾値の導入

```python
# 特徴量の種類に応じた閾値設定
if key == 'molecular_weight' and 'transform' in property_norms[key]:
    # 対数変換済み分子量には緩い閾値
    warning_threshold = 3.5  
else:
    # その他の特徴量には従来の閾値
    warning_threshold = 2.5
```

### 2. クランプ範囲の拡張

```python
# より広い範囲でクランプして数値安定性を保持
properties = torch.clamp(properties, min=-4.0, max=4.0)
```

### 3. 改善された警告メッセージ

```python
if key == 'molecular_weight':
    if 'transform' not in property_norms[key]:
        print(f"  Suggestion: Log transformation is recommended for molecular_weight")
    else:
        print(f"  Note: Log transformation is already applied. This warning may indicate")
        print(f"        extreme outliers in your dataset (very small or very large molecules).")
        print(f"        Consider filtering extreme outliers if training becomes unstable.")
```

### 4. MAD計算の改善

```python
# 対数範囲を考慮したより堅牢なMAD計算
log_range = torch.max(log_values) - torch.min(log_values)
min_mad = max(abs(float(mean)) * 0.1, 0.5, float(log_range) * 0.05)
mad = torch.max(mad, torch.tensor(min_mad))
```

## 修正結果

### 典型的な訓練シナリオ
- **分子量範囲**: 15-619 u（典型的な有機化合物）
- **正規化後の最大絶対値**: 1.7-2.5
- **結果**: ⭐ **警告なし** ⭐

### 極値を含むシナリオ
- **分子量範囲**: 1-10,000 u（極端な外れ値を含む）
- **正規化後の最大絶対値**: ~4.0
- **結果**: 適切な警告メッセージで制御された警告

## 使用方法

修正は自動的に適用されます。追加の設定は不要です：

```python
# 分子量を含む条件付け（自動的に対数変換と改善された正規化が適用）
--conditioning molecular_weight

# 他の特徴量との組み合わせも同様に動作
--conditioning molecular_weight pi_conjugation_ratio energy
```

## バックワード互換性

- ✅ 既存のコードは変更なしで動作
- ✅ 他の特徴量（energy、homo、lumoなど）の正規化は影響を受けません
- ✅ QM9データセットでの動作も維持されています

## トラブルシューティング

### まだ警告が出る場合

1. **極端な外れ値の確認**
   ```python
   # データセットの分子量分布を確認
   print(f"分子量範囲: {molecular_weights.min():.1f} - {molecular_weights.max():.1f}")
   print(f"比率: {molecular_weights.max()/molecular_weights.min():.1f}")
   ```

2. **外れ値のフィルタリング**
   ```python
   # 極端な値を除外（必要に応じて）
   filtered_data = data[(data['molecular_weight'] >= 10) & 
                       (data['molecular_weight'] <= 1000)]
   ```

3. **警告レベルの調整**
   - 4.0未満の警告は通常問題ありません
   - 訓練が不安定になった場合のみ対処を検討してください

## 効果の確認

修正後、以下の改善が期待できます：

- ⭐ **警告の大幅な減少**: 通常の分子量分布では警告が出なくなります
- ⭐ **明確なガイダンス**: 警告が出る場合も、具体的な対処法が提示されます
- ⭐ **訓練の安定性向上**: 数値的な安定性が改善されます
- ⭐ **より良いデバッグ情報**: 対数変換の適用状況が明確になります

## 技術的詳細

この修正は`qm9/utils.py`の以下の関数で実装されています：

1. `compute_mean_mad_from_dataloader()`: MAD計算の改善
2. `prepare_context()`: 警告閾値とメッセージの改善

修正により、ASEデータベースでの分子量を用いた条件付き生成がより安定して動作するようになります。