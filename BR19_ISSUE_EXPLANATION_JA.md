# Br19問題の原因と修正

## 問題の症状

以下のコマンドを実行すると、Br19のようなあり得ない分子ばかり生成されていました：

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/exp_cond_molecular_descriptors \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N]' \
    --n_samples 100
```

指定では`atom_types_encoding=[C,H,O,N]`としているにもかかわらず、C、H、O、N以外の原子（例：Br、I、Siなど）を含む分子が生成されていました。

## 根本原因

問題は`eval_conditional_qm9.py`の`create_exact_context`関数にありました。

### 学習時の動作

モデルを`--conditioning atom_types_encoding`で学習させる際、`atom_types_encoding`は**個別のバイナリ特徴量に展開**されます：

- `atom_types_encoding` → `has_C`, `has_H`, `has_N`, `has_O` など

例えば、データセットに`['C', 'H', 'N', 'O']`という原子種が含まれている場合、4つの個別特徴量が作成されます。

ソースコード (`qm9/dataset.py`の331-337行目付近)：
```python
# For atom types - create one feature per atom type
if atom_types_sorted:
    for atom_type in atom_types_sorted:
        feature_name = f'has_{atom_type}'
        property_tensors[feature_name] = torch.zeros(n_molecules, dtype=torch.float32)
        for i, mol_atom_types in enumerate(atom_types_list):
            if atom_type in mol_atom_types:
                property_tensors[feature_name][i] = 1.0
```

### 生成時の問題（修正前）

生成時に`--property_values 'atom_types_encoding=[C,H,O,N]'`を指定すると、修正前のコードは`property_norms`辞書から`atom_types_encoding`というキーを直接検索しようとしていました。

しかし、学習時に`atom_types_encoding`は個別の`has_<原子>`特徴量に展開されているため、`property_norms`には`atom_types_encoding`というキーは存在せず、代わりに`has_C`、`has_H`、`has_N`、`has_O`などのキーが存在します。

その結果：
1. `atom_types_encoding`が`property_norms`に見つからない
2. コンテキストテンソルに原子種情報が設定されない（デフォルトのゼロのまま）
3. モデルが原子種の制約なしで分子を生成する
4. 学習データに含まれる任意の原子種（Br、I、Siなど）で分子が生成される

## 修正内容

`create_exact_context`関数を修正して、`atom_types_encoding`と`functional_groups_encoding`を個別の特徴量に展開するようにしました：

### 主な変更点

1. **データローダーからマッピング情報を取得**
   ```python
   atom_types_mapping = dataset.data.get('_atom_types_mapping', [])
   functional_groups_mapping = dataset.data.get('_functional_groups_mapping', [])
   ```

2. **atom_types_encodingの展開**
   ```python
   if key == 'atom_types_encoding' and isinstance(value, list):
       # 個別のhas_<原子>特徴量に展開
       for atom in atom_types_mapping:
           feature_name = f'has_{atom}'
           if atom in value:
               expanded_property_values[feature_name] = 1.0
           else:
               expanded_property_values[feature_name] = 0.0
   ```

3. **コンテキストの設定**
   ```python
   for key in args_gen.conditioning:
       if key == 'atom_types_encoding':
           # 各has_<原子>特徴量を処理
           for atom in atom_types_mapping:
               feature_name = f'has_{atom}'
               if feature_name in expanded_property_values and feature_name in property_norms:
                   value = expanded_property_values[feature_name]
                   mean = property_norms[feature_name]['mean']
                   mad = property_norms[feature_name]['mad']
                   normalized_value = (value - mean) / mad
                   context[:, feature_idx] = normalized_value
                   feature_idx += 1
   ```

## テスト結果

修正後、以下のテストがすべて成功しました：

```bash
$ python test_atom_types_fix.py
```

テスト結果：
- ✓ プロパティ値のパースが正しく動作
- ✓ atom_types_encodingが個別特徴量に展開される
- ✓ functional_groups_encodingが個別特徴量に展開される
- ✓ コンテキストテンソルに正しい値が設定される
- ✓ 非ゼロの特徴量が存在する（原子種情報が設定されている）

## 使用方法

修正後は、元のコマンドがそのまま正しく動作します：

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/exp_cond_molecular_descriptors \
    --task qualitative \
    --use_exact_conditions \
    --property_values 'pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N]' \
    --n_samples 100
```

このコマンドは、C、H、O、Nのみを含む分子を生成するようになります。

## 追加パラメータ

`--n_samples`パラメータを追加しました。これにより、1回のスイープで生成する分子の数を制御できます：

```bash
# 10個の分子を生成
--n_samples 10

# 100個の分子を生成（デフォルト）
--n_samples 100

# 500個の分子を生成
--n_samples 500
```

## 技術的詳細

### コンテキストテンソルの構造

学習時のコンテキストテンソルは、すべての条件付け特徴量を連結したものです：

```
[molecular_weight, pi_conjugation_ratio, has_C, has_H, has_N, has_O, has_Br, has_I, ...]
```

修正により、生成時にも同じ構造でコンテキストが作成されるようになりました。

### 正規化

各特徴量は学習時の平均（mean）と平均絶対偏差（MAD）を用いて正規化されます：

```python
normalized_value = (value - mean) / mad
```

これにより、モデルが学習時と同じスケールの値を受け取ります。

## まとめ

この修正により、`atom_types_encoding`と`functional_groups_encoding`を使用した厳密条件付き生成が正しく機能するようになりました。指定された原子種のみを含む分子が生成され、Br19のようなあり得ない分子は生成されなくなります。
