# トラブルシューティングガイド拡張版

**最終更新**: 2025年10月25日  
**対象**: E3 Diffusion for Molecules ユーザー  
**目的**: よくあるエラーの診断と解決方法

---

## 目次

1. [よくあるエラーと解決方法](#よくあるエラーと解決方法)
2. [デバッグ手法](#デバッグ手法)
3. [パフォーマンス問題の診断](#パフォーマンス問題の診断)
4. [環境別の注意事項](#環境別の注意事項)
5. [データ関連の問題](#データ関連の問題)

---

## よくあるエラーと解決方法

### エラー1: ImportError: No module named 'rdkit'

#### 症状
```
ImportError: No module named 'rdkit'
```

#### 原因
RDKitパッケージがインストールされていません。

#### 解決方法

**方法1: condaでインストール（推奨）**
```bash
conda install -c conda-forge rdkit
```

**方法2: pipでインストール**
```bash
pip install rdkit
```

#### 検証
```python
from rdkit import Chem
print(Chem.__version__)  # 2025.9.1 または類似
```

#### 関連情報
- RDKitドキュメント: https://www.rdkit.org/docs/
- チュートリアル05で使用
- 分子の有効性検証に必要

---

### エラー2: ImportError: No module named 'ase'

#### 症状
```
ImportError: No module named 'ase'
```

#### 原因
ASE (Atomic Simulation Environment) パッケージがインストールされていません。

#### 解決方法

**pipでインストール**
```bash
pip install ase
```

#### 検証
```python
from ase import Atoms
from ase.db import connect
print("ASE successfully imported")
```

#### 関連情報
- ASEドキュメント: https://wiki.fysik.dtu.dk/ase/
- チュートリアル04で使用
- カスタムデータセット作成に必須

---

### エラー3: ImportError: No module named 'spglib'

#### 症状
```
ImportError: No module named 'spglib'
```

#### 原因
spglibパッケージがインストールされていません。

#### 解決方法

**pipでインストール**
```bash
pip install spglib
```

#### 検証
```python
import spglib
print(f"spglib version: {spglib.__version__}")
```

#### 関連情報
- spglibドキュメント: https://spglib.github.io/spglib/
- チュートリアル03, 06で使用
- 結晶の空間群解析に必須

---

### エラー4: RuntimeError: CUDA out of memory

#### 症状
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity; 12.34 GiB already allocated; 1.45 GiB free; 13.56 GiB reserved in total by PyTorch)
```

#### 原因
GPUメモリが不足しています。

#### 解決方法

**方法1: バッチサイズを減らす**
```python
# 変更前
batch_size = 64

# 変更後
batch_size = 32  # または 16
```

**方法2: モデルサイズを小さくする**
```python
# 変更前
nf = 256

# 変更後
nf = 128
```

**方法3: 勾配累積を使用**
```python
# 実質的なバッチサイズを維持しながらメモリ使用量を削減
accumulation_steps = 4
batch_size = 16  # 実質的なバッチサイズ: 16 * 4 = 64

for i, batch in enumerate(train_loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**方法4: 混合精度訓練を有効化**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

#### 検証
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

---

### エラー5: FileNotFoundError: No such file or directory: 'qm9.db'

#### 症状
```
FileNotFoundError: [Errno 2] No such file or directory: 'qm9.db'
```

#### 原因
データセットファイルが存在しないか、パスが間違っています。

#### 解決方法

**QM9データセットのダウンロードと配置**
```bash
# QM9データセットをダウンロード
python build_geom_dataset.py --dataset qm9

# または、既存のデータベースを使用
ls -la *.db  # 利用可能なデータベースを確認
```

**パスの明示的な指定**
```python
from qm9.dataset import retrieve_dataloaders

# 相対パス
dataloaders = retrieve_dataloaders('./qm9.db', batch_size=64)

# 絶対パス
dataloaders = retrieve_dataloaders('/path/to/qm9.db', batch_size=64)
```

#### 検証
```python
import os
db_path = './qm9.db'
if os.path.exists(db_path):
    print(f"Database found: {db_path}")
else:
    print(f"Database not found: {db_path}")
```

---

### エラー6: ValueError: Input contains NaN

#### 症状
```
ValueError: Input contains NaN, infinity or a value too large for dtype('float32')
```

#### 原因
データに異常な値（NaN、無限大）が含まれています。

#### 解決方法

**データのクリーニング**
```python
import torch
import numpy as np

def check_and_clean_data(x, h):
    """データの異常値をチェックして除去"""
    # NaNチェック
    if torch.isnan(x).any():
        print(f"Warning: NaN detected in positions")
        x = torch.nan_to_num(x, nan=0.0)
    
    if torch.isnan(h).any():
        print(f"Warning: NaN detected in features")
        h = torch.nan_to_num(h, nan=0.0)
    
    # 無限大チェック
    if torch.isinf(x).any():
        print(f"Warning: Inf detected in positions")
        x = torch.nan_to_num(x, posinf=1e6, neginf=-1e6)
    
    if torch.isinf(h).any():
        print(f"Warning: Inf detected in features")
        h = torch.nan_to_num(h, posinf=1e6, neginf=-1e6)
    
    return x, h

# 使用例
x, h = check_and_clean_data(positions, features)
```

**数値安定性の確保**
```python
# 正規化時の安定化
def stable_normalize(x, eps=1e-8):
    """数値的に安定な正規化"""
    mean = x.mean()
    std = x.std()
    # ゼロ除算を防ぐ
    std = torch.clamp(std, min=eps)
    return (x - mean) / std
```

#### 検証
```python
def validate_tensors(*tensors):
    """テンソルの健全性を検証"""
    for i, tensor in enumerate(tensors):
        if torch.isnan(tensor).any():
            raise ValueError(f"Tensor {i} contains NaN")
        if torch.isinf(tensor).any():
            raise ValueError(f"Tensor {i} contains Inf")
    print("All tensors are valid")
```

---

### エラー7: KeyError: 'alpha'

#### 症状
```
KeyError: 'alpha'
```

#### 原因
指定した性質名がデータセットに存在しません。

#### 解決方法

**利用可能な性質を確認**
```python
from ase.db import connect

db = connect('qm9.db')
row = db.get(1)
available_properties = list(row.data.keys())
print(f"Available properties: {available_properties}")
```

**正しい性質名を使用**
```python
# QM9データセットの一般的な性質名
valid_properties = [
    'mu', 'alpha', 'homo', 'lumo', 'gap', 
    'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
]

# 性質の存在を確認
property_name = 'alpha'
if property_name in available_properties:
    print(f"Property '{property_name}' is available")
else:
    print(f"Property '{property_name}' not found")
    print(f"Available properties: {available_properties}")
```

---

### エラー8: RuntimeError: Expected all tensors to be on the same device

#### 症状
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

#### 原因
テンソルが異なるデバイス（CPUとGPU）に配置されています。

#### 解決方法

**全テンソルを同じデバイスに移動**
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルをデバイスに移動
model = model.to(device)

# データをデバイスに移動
x = x.to(device)
h = h.to(device)
node_mask = node_mask.to(device)
edge_mask = edge_mask.to(device)

# 推論時も同様
context = context.to(device)
```

**デバイス確認ヘルパー関数**
```python
def check_device_consistency(model, *tensors):
    """モデルとテンソルのデバイス一貫性を確認"""
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    
    for i, tensor in enumerate(tensors):
        print(f"Tensor {i} device: {tensor.device}")
        if tensor.device != model_device:
            raise RuntimeError(
                f"Device mismatch: model on {model_device}, "
                f"tensor {i} on {tensor.device}"
            )
```

---

### エラー9: AttributeError: 'NoneType' object has no attribute 'shape'

#### 症状
```
AttributeError: 'NoneType' object has no attribute 'shape'
```

#### 原因
変数が初期化されていないか、関数がNoneを返しています。

#### 解決方法

**明示的なNoneチェック**
```python
def safe_processing(data):
    """安全なデータ処理"""
    if data is None:
        raise ValueError("Data cannot be None")
    
    # 処理を続行
    return data.shape

# 使用例
try:
    result = safe_processing(my_data)
except ValueError as e:
    print(f"Error: {e}")
    # デフォルト値を使用するか、エラーを再発生
```

**モデル出力の検証**
```python
x, h = model.sample(n_samples, n_nodes, node_mask, edge_mask, context)

# 出力を検証
if x is None:
    raise RuntimeError("Model returned None for positions")
if h is None:
    raise RuntimeError("Model returned None for features")

print(f"Generated positions shape: {x.shape}")
print(f"Generated features shape: {h.shape}")
```

---

### エラー10: torch.cuda.OutOfMemoryError during evaluation

#### 症状
評価時にGPUメモリ不足エラーが発生します。

#### 原因
評価時に勾配計算が不要なのに有効になっています。

#### 解決方法

**torch.no_grad()を使用**
```python
import torch

model.eval()  # 評価モードに切り替え

with torch.no_grad():  # 勾配計算を無効化
    for batch in test_loader:
        x, h = model.sample(
            n_samples=batch_size,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=context
        )
        # 評価メトリクスの計算
```

**バッチ単位の処理**
```python
def evaluate_in_batches(model, data, batch_size=16):
    """大規模データを小さなバッチで評価"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            output = model(batch)
            results.append(output.cpu())  # GPUメモリを解放
    
    return torch.cat(results, dim=0)
```

---

### エラー11: ValueError: not enough values to unpack

#### 症状
```
ValueError: not enough values to unpack (expected 2, got 1)
```

#### 原因
関数の戻り値の数が期待と異なります。

#### 解決方法

**戻り値の確認**
```python
# 誤った例
x, h = model.sample()  # modelは1つの値のみ返す

# 正しい例
result = model.sample()
if isinstance(result, tuple):
    x, h = result
else:
    x = result
    h = None
```

**明示的な戻り値処理**
```python
def safe_unpack(result, expected_count):
    """安全なアンパック"""
    if not isinstance(result, (tuple, list)):
        result = (result,)
    
    if len(result) != expected_count:
        raise ValueError(
            f"Expected {expected_count} values, got {len(result)}"
        )
    
    return result

# 使用例
x, h = safe_unpack(model.sample(), 2)
```

---

### エラー12: RuntimeError: The size of tensor a must match the size of tensor b

#### 症状
```
RuntimeError: The size of tensor a (64) must match the size of tensor b (128) at non-singleton dimension 1
```

#### 原因
テンソルの形状が一致していません。

#### 解決方法

**形状の確認と調整**
```python
def ensure_compatible_shapes(a, b):
    """テンソルの形状を確認し、必要に応じて調整"""
    print(f"Tensor a shape: {a.shape}")
    print(f"Tensor b shape: {b.shape}")
    
    # ブロードキャスト可能か確認
    try:
        # ダミー演算で確認
        _ = a + torch.zeros_like(b)
        print("Shapes are compatible")
    except RuntimeError as e:
        print(f"Shape mismatch: {e}")
        raise

# 使用例
ensure_compatible_shapes(tensor_a, tensor_b)
```

**動的なサイズ調整**
```python
# 次元を追加
if a.dim() < b.dim():
    a = a.unsqueeze(-1)

# 特定の次元でリピート
if a.shape[1] != b.shape[1]:
    a = a.repeat(1, b.shape[1] // a.shape[1])
```

---

### エラー13: OSError: [Errno 28] No space left on device

#### 症状
```
OSError: [Errno 28] No space left on device
```

#### 原因
ディスク容量が不足しています。

#### 解決方法

**ディスク使用量の確認**
```bash
# Linuxの場合
df -h

# 現在のディレクトリのサイズ
du -sh .
du -sh outputs/
```

**不要なファイルの削除**
```bash
# チェックポイントの古いファイルを削除
find outputs/ -name "*.pkl" -mtime +7 -delete

# 一時ファイルを削除
rm -rf /tmp/*
```

**設定の調整**
```python
# チェックポイントの保存頻度を減らす
save_checkpoint_every = 1000  # 以前: 100

# 古いチェックポイントを自動削除
def cleanup_old_checkpoints(output_dir, keep_last=5):
    """古いチェックポイントを削除"""
    checkpoints = sorted(
        glob.glob(f"{output_dir}/*.pkl"),
        key=os.path.getmtime
    )
    
    for ckpt in checkpoints[:-keep_last]:
        os.remove(ckpt)
        print(f"Removed old checkpoint: {ckpt}")
```

---

### エラー14: ModuleNotFoundError: No module named 'wandb'

#### 症状
```
ModuleNotFoundError: No module named 'wandb'
```

#### 原因
Weights & Biasesパッケージがインストールされていません。

#### 解決方法

**方法1: wandbをインストール**
```bash
pip install wandb
```

**方法2: wandbを無効化**
```python
# コマンドライン引数で無効化
python main_qm9.py --no_wandb

# または、コード内で無効化
import os
os.environ['WANDB_MODE'] = 'disabled'
```

#### 検証
```python
try:
    import wandb
    print("wandb is available")
except ImportError:
    print("wandb is not installed")
```

---

### エラー15: RuntimeError: Expected scalar type Double but found Float

#### 症状
```
RuntimeError: Expected scalar type Double but found Float
```

#### 原因
テンソルのデータ型が一致していません。

#### 解決方法

**データ型を統一**
```python
# float32に統一（推奨）
x = x.float()  # または .to(torch.float32)
h = h.float()

# float64に統一（高精度が必要な場合）
x = x.double()  # または .to(torch.float64)
h = h.double()
```

**モデル全体のデータ型を変更**
```python
# モデルをfloat32に変換
model = model.float()

# モデルをfloat64に変換
model = model.double()
```

**データ型確認関数**
```python
def check_dtype_consistency(*tensors):
    """テンソルのデータ型一貫性を確認"""
    dtypes = [t.dtype for t in tensors]
    if len(set(dtypes)) > 1:
        print(f"Warning: Multiple dtypes found: {dtypes}")
    else:
        print(f"All tensors have dtype: {dtypes[0]}")
```

---

### エラー16: IndexError: index out of range in self

#### 症状
```
IndexError: index 5 is out of bounds for dimension 0 with size 5
```

#### 原因
インデックスが配列の範囲を超えています。

#### 解決方法

**インデックスの範囲チェック**
```python
def safe_index(array, index):
    """安全なインデックスアクセス"""
    if index < 0 or index >= len(array):
        raise IndexError(
            f"Index {index} out of range for array of size {len(array)}"
        )
    return array[index]
```

**動的なインデックス調整**
```python
# インデックスをクリップ
index = min(max(0, index), len(array) - 1)
value = array[index]

# またはモジュロ演算を使用
index = index % len(array)
value = array[index]
```

---

### エラー17: RuntimeError: one of the variables needed for gradient computation has been modified

#### 症状
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

#### 原因
逆伝播に必要な変数がインプレース操作で変更されています。

#### 解決方法

**インプレース操作を避ける**
```python
# 誤った例（インプレース操作）
x += 1
x *= 2
x[mask] = 0

# 正しい例（新しいテンソルを作成）
x = x + 1
x = x * 2
x = x.masked_fill(mask, 0)
```

**detach()を使用**
```python
# 勾配計算から切り離す
x_detached = x.detach()
x_detached += 1  # これは安全
```

---

### エラー18: ValueError: Target size must be same as input size

#### 症状
```
ValueError: Target size (torch.Size([64, 10])) must be the same as input size (torch.Size([64, 5]))
```

#### 原因
損失関数の入力とターゲットの形状が一致していません。

#### 解決方法

**形状の確認と調整**
```python
def compute_loss_safely(prediction, target):
    """安全な損失計算"""
    print(f"Prediction shape: {prediction.shape}")
    print(f"Target shape: {target.shape}")
    
    # 形状が一致しない場合の処理
    if prediction.shape != target.shape:
        if prediction.shape[0] == target.shape[0]:
            # バッチサイズは一致、その他の次元を調整
            if len(target.shape) < len(prediction.shape):
                target = target.view_as(prediction)
            else:
                raise ValueError(
                    f"Cannot reconcile shapes: "
                    f"prediction {prediction.shape}, target {target.shape}"
                )
    
    loss = torch.nn.functional.mse_loss(prediction, target)
    return loss
```

---

### エラー19: RuntimeError: CUDA error: device-side assert triggered

#### 症状
```
RuntimeError: CUDA error: device-side assert triggered
```

#### 原因
GPU上で不正な操作が実行されました（範囲外のインデックスアクセスなど）。

#### 解決方法

**CPUで実行してデバッグ**
```python
# CPUモードで実行
device = torch.device('cpu')  # GPUの代わりにCPU
model = model.to(device)
data = data.to(device)

# エラーが再現されれば、詳細なエラーメッセージが得られる
```

**範囲チェックの追加**
```python
# インデックスの検証
def validate_indices(indices, max_value):
    """インデックスが有効範囲内か確認"""
    if (indices < 0).any() or (indices >= max_value).any():
        raise ValueError(
            f"Invalid indices detected: min={indices.min()}, "
            f"max={indices.max()}, allowed range=[0, {max_value})"
        )
```

**CUDA同期デバッグ**
```python
import os
# 同期モードを有効化（デバッグ用、遅くなる）
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

---

### エラー20: MemoryError: Unable to allocate array

#### 症状
```
MemoryError: Unable to allocate 8.00 GiB for an array with shape (1000000000,) and data type float64
```

#### 原因
システムメモリ（RAM）が不足しています。

#### 解決方法

**データをストリーミング処理**
```python
def process_large_dataset_streaming(dataset, process_fn, chunk_size=1000):
    """大規模データセットをチャンク単位で処理"""
    results = []
    
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        result = process_fn(chunk)
        results.append(result)
        
        # メモリ解放
        del chunk
        import gc
        gc.collect()
    
    return results
```

**データ型を軽量化**
```python
# float64 → float32に変更（メモリ使用量が半分に）
data = data.astype(np.float32)

# または、より小さい整数型を使用
indices = indices.astype(np.int32)  # int64の代わり
```

**メモリマップファイルを使用**
```python
import numpy as np

# ディスク上のファイルとして配列を作成
large_array = np.memmap(
    'temp_array.dat',
    dtype='float32',
    mode='w+',
    shape=(1000000000,)
)
```

---

## デバッグ手法

### 1. ロギングの設定

**基本的なロギング設定**
```python
import logging

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 使用例
logger.info("Starting training")
logger.debug(f"Batch size: {batch_size}")
logger.warning("GPU memory usage high")
logger.error("Failed to load checkpoint")
```

**モジュール別のログレベル**
```python
# 特定のモジュールのログレベルを変更
logging.getLogger('qm9').setLevel(logging.DEBUG)
logging.getLogger('crystal').setLevel(logging.INFO)
logging.getLogger('torch').setLevel(logging.WARNING)
```

---

### 2. デバッグモードの有効化

**詳細なエラー情報を取得**
```python
import torch

# 異常検出を有効化
torch.autograd.set_detect_anomaly(True)

# PyTorch内部のデバッグ情報を有効化
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 再現性の確保
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

---

### 3. 中間出力の確認

**テンソルの統計情報を表示**
```python
def print_tensor_stats(name, tensor):
    """テンソルの統計情報を表示"""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")
    print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
    print(f"  Inf count: {torch.isinf(tensor).sum().item()}")

# 使用例
print_tensor_stats("positions", x)
print_tensor_stats("features", h)
```

**勾配の確認**
```python
def check_gradients(model):
    """モデルの勾配を確認"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.4f}")
            
            if torch.isnan(param.grad).any():
                print(f"  WARNING: NaN gradient detected in {name}")
            if grad_norm > 1000:
                print(f"  WARNING: Large gradient in {name}")
```

---

### 4. ブレークポイントの使用

**pdbデバッガー**
```python
import pdb

# ブレークポイントを設定
def train_step(batch):
    x, h = batch
    
    # デバッグポイント
    pdb.set_trace()  # ここで実行が停止
    
    output = model(x, h)
    return output

# Python 3.7以降は breakpoint() も使用可能
def train_step(batch):
    x, h = batch
    breakpoint()  # ここで実行が停止
    output = model(x, h)
    return output
```

**条件付きブレークポイント**
```python
def train_loop(dataloader):
    for i, batch in enumerate(dataloader):
        loss = train_step(batch)
        
        # 特定の条件でブレーク
        if torch.isnan(loss):
            print(f"NaN loss detected at iteration {i}")
            breakpoint()
        
        if loss > 1000:
            print(f"Abnormally large loss at iteration {i}")
            breakpoint()
```

---

### 5. Weights & Biasesでの可視化

**W&Bの設定**
```python
import wandb

# プロジェクトの初期化
wandb.init(
    project="e3-diffusion-debug",
    config={
        "learning_rate": 2e-4,
        "batch_size": 64,
        "n_layers": 9
    }
)

# メトリクスのログ
wandb.log({
    "train_loss": loss.item(),
    "grad_norm": grad_norm,
    "lr": optimizer.param_groups[0]['lr']
})

# テンソルのヒストグラムをログ
wandb.log({"position_hist": wandb.Histogram(x.cpu().numpy())})
```

**カスタム可視化**
```python
import matplotlib.pyplot as plt

def visualize_molecule(x, h):
    """分子構造を可視化"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原子タイプで色分け
    colors = ['r', 'g', 'b', 'y', 'c']
    atom_types = h.argmax(dim=-1)
    
    for atom_type in range(5):
        mask = (atom_types == atom_type)
        if mask.any():
            positions = x[mask].cpu().numpy()
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=colors[atom_type],
                s=100,
                label=f'Type {atom_type}'
            )
    
    ax.legend()
    wandb.log({"molecule_structure": wandb.Image(fig)})
    plt.close()
```

---

## パフォーマンス問題の診断

### 1. メモリ使用量の確認

**GPUメモリの監視**
```python
import torch

def monitor_gpu_memory():
    """GPUメモリ使用量を監視"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")
        
        # メモリ使用率
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Usage: {allocated/total_memory*100:.1f}%")

# 定期的に監視
import time

while training:
    monitor_gpu_memory()
    time.sleep(60)  # 1分ごと
```

**システムメモリの監視**
```python
import psutil

def monitor_system_memory():
    """システムメモリ使用量を監視"""
    memory = psutil.virtual_memory()
    print(f"System Memory:")
    print(f"  Total: {memory.total / 1024**3:.2f} GB")
    print(f"  Available: {memory.available / 1024**3:.2f} GB")
    print(f"  Used: {memory.used / 1024**3:.2f} GB")
    print(f"  Usage: {memory.percent}%")
```

---

### 2. GPU使用率の監視

**nvidia-smiを使用**
```bash
# GPUステータスの確認
nvidia-smi

# 継続的な監視（1秒ごと）
watch -n 1 nvidia-smi

# 特定のGPUを監視
nvidia-smi -i 0 -l 1
```

**Pythonからの監視**
```python
import subprocess

def get_gpu_usage():
    """GPU使用率を取得"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
         '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )
    
    gpu_util, mem_used, mem_total = map(float, result.stdout.strip().split(','))
    
    print(f"GPU Utilization: {gpu_util}%")
    print(f"Memory: {mem_used}/{mem_total} MB ({mem_used/mem_total*100:.1f}%)")
    
    return gpu_util, mem_used, mem_total
```

---

### 3. ボトルネックの特定

**時間計測**
```python
import time

class Timer:
    """コードブロックの実行時間を計測"""
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"{self.name}: {self.elapsed:.4f} seconds")

# 使用例
with Timer("Data loading"):
    batch = next(iter(train_loader))

with Timer("Forward pass"):
    output = model(batch)

with Timer("Backward pass"):
    loss.backward()

with Timer("Optimizer step"):
    optimizer.step()
```

**プロファイリング**
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # プロファイルするコード
    for i in range(10):
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 結果を表示
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Chrome traceとして保存
prof.export_chrome_trace("trace.json")
```

---

### 4. プロファイリングツールの使用

**cProfileを使用**
```python
import cProfile
import pstats

def train_one_epoch():
    # 訓練コード
    pass

# プロファイリング実行
profiler = cProfile.Profile()
profiler.enable()

train_one_epoch()

profiler.disable()

# 結果を表示
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 上位20項目
```

**line_profilerを使用**
```bash
# インストール
pip install line_profiler

# 使用例
# @profile デコレータを追加
@profile
def train_step(batch):
    # 訓練コード
    pass

# 実行
kernprof -l -v script.py
```

---

## 環境別の注意事項

### Linux環境

**CUDA設定の確認**
```bash
# CUDA バージョン
nvcc --version

# CUDA ライブラリパス
echo $LD_LIBRARY_PATH

# 必要に応じて設定
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**マルチGPU環境**
```python
import torch

# 利用可能なGPU数
n_gpus = torch.cuda.device_count()
print(f"Available GPUs: {n_gpus}")

# 特定のGPUを使用
torch.cuda.set_device(0)

# DataParallelを使用
if n_gpus > 1:
    model = torch.nn.DataParallel(model)
```

---

### macOS環境

**MPS (Metal Performance Shaders) の使用**
```python
import torch

# M1/M2 Macの場合
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) backend")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

model = model.to(device)
```

**RDKitのインストール**
```bash
# Homebrewを使用（推奨）
brew install rdkit

# または conda
conda install -c conda-forge rdkit
```

---

### Windows環境

**パスの問題**
```python
import os

# Windows形式のパスを適切に処理
db_path = os.path.join('data', 'qm9.db')  # 推奨
# 生のパスは避ける: db_path = 'data\qm9.db'

# パスが存在するか確認
if not os.path.exists(db_path):
    print(f"Path not found: {db_path}")
```

**長いパス名の問題**
```python
# Windows のパス長制限（260文字）を回避
import pathlib

# パスを短縮
output_dir = pathlib.Path("outputs")
output_dir.mkdir(exist_ok=True)

# 相対パスを使用
checkpoint_path = output_dir / "model.pkl"
```

---

### クラウド環境（AWS, GCP, Azure）

**インスタンスタイプの選択**
```
推奨構成:
- GPU: NVIDIA V100, A100, または T4
- VRAM: 16GB以上
- RAM: 32GB以上
- Storage: 100GB以上 SSD
```

**コスト最適化**
```python
# スポットインスタンスの使用
# チェックポイントを頻繁に保存
save_checkpoint_every = 100  # 100イテレーションごと

# 中断可能な設計
def save_checkpoint(model, optimizer, epoch, path):
    """チェックポイントを保存"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")

# 定期的に保存
if iteration % save_checkpoint_every == 0:
    save_checkpoint(model, optimizer, epoch, f"ckpt_{iteration}.pkl")
```

---

## データ関連の問題

### 1. データセットの破損

**データベースの整合性チェック**
```python
from ase.db import connect
import sqlite3

def check_database_integrity(db_path):
    """データベースの整合性を確認"""
    try:
        # SQLite整合性チェック
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        
        if result[0] == 'ok':
            print(f"Database integrity: OK")
        else:
            print(f"Database integrity: FAILED - {result}")
            return False
        
        conn.close()
        
        # ASEでの読み込みテスト
        db = connect(db_path)
        n_rows = len(db)
        print(f"Number of rows: {n_rows}")
        
        # 最初の行を読み込み
        row = db.get(1)
        print(f"First row successfully loaded")
        
        return True
        
    except Exception as e:
        print(f"Database check failed: {e}")
        return False

# 使用例
if not check_database_integrity('qm9.db'):
    print("Database is corrupted, please re-download")
```

---

### 2. カスタムデータの形式エラー

**データ形式の検証**
```python
def validate_molecule_data(atoms, properties):
    """分子データを検証"""
    # 原子数のチェック
    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ValueError("No atoms in molecule")
    if n_atoms > 200:
        print(f"Warning: Large molecule with {n_atoms} atoms")
    
    # 座標のチェック
    positions = atoms.get_positions()
    if positions.shape != (n_atoms, 3):
        raise ValueError(f"Invalid positions shape: {positions.shape}")
    
    # 原子番号のチェック
    atomic_numbers = atoms.get_atomic_numbers()
    if (atomic_numbers <= 0).any() or (atomic_numbers > 118).any():
        raise ValueError(f"Invalid atomic numbers: {atomic_numbers}")
    
    # 性質のチェック
    for prop_name, prop_value in properties.items():
        if not isinstance(prop_value, (int, float)):
            raise TypeError(
                f"Property '{prop_name}' must be numeric, got {type(prop_value)}"
            )
        if not np.isfinite(prop_value):
            raise ValueError(
                f"Property '{prop_name}' has non-finite value: {prop_value}"
            )
    
    print("Validation passed")
    return True
```

---

### 3. ASEデータベースの問題

**データベースの修復**
```python
from ase.db import connect

def repair_database(source_db, target_db):
    """破損したデータベースを修復"""
    source = connect(source_db)
    target = connect(target_db)
    
    n_total = len(source)
    n_success = 0
    n_failed = 0
    
    for i, row in enumerate(source.select(), 1):
        try:
            # データの読み込みと検証
            atoms = row.toatoms()
            data = row.data
            
            # 新しいデータベースに書き込み
            target.write(atoms, data=data)
            n_success += 1
            
        except Exception as e:
            print(f"Failed to copy row {i}: {e}")
            n_failed += 1
        
        if i % 1000 == 0:
            print(f"Progress: {i}/{n_total} ({n_success} success, {n_failed} failed)")
    
    print(f"Repair complete: {n_success}/{n_total} rows recovered")
    return n_success, n_failed
```

---

### 4. 性質データの欠損

**欠損値の処理**
```python
def handle_missing_properties(db_path, properties):
    """欠損性質を処理"""
    db = connect(db_path)
    
    for row in db.select():
        data = row.data.copy()
        
        # 欠損値を確認
        missing = []
        for prop in properties:
            if prop not in data:
                missing.append(prop)
        
        if missing:
            print(f"Row {row.id}: missing {missing}")
            
            # オプション1: その行をスキップ
            continue
            
            # オプション2: デフォルト値を設定（非推奨）
            # for prop in missing:
            #     data[prop] = 0.0
            
            # オプション3: 平均値で補完（慎重に使用）
            # for prop in missing:
            #     data[prop] = compute_mean(db, prop)
```

**完全なデータのみを使用**
```python
from qm9.dataset import retrieve_dataloaders

def filter_complete_data(db_path, required_properties):
    """完全なデータのみをフィルタ"""
    from ase.db import connect
    
    source = connect(db_path)
    filtered = connect('filtered.db')
    
    n_total = 0
    n_complete = 0
    
    for row in source.select():
        n_total += 1
        
        # 全ての必須性質が存在するか確認
        if all(prop in row.data for prop in required_properties):
            filtered.write(row.toatoms(), data=row.data)
            n_complete += 1
    
    print(f"Complete data: {n_complete}/{n_total} ({n_complete/n_total*100:.1f}%)")
    return 'filtered.db'

# 使用例
filtered_db = filter_complete_data('qm9.db', ['alpha', 'homo', 'lumo'])
dataloaders = retrieve_dataloaders(filtered_db, batch_size=64)
```

---

## まとめ

このトラブルシューティングガイドでは、E3 Diffusion for Moleculesの使用時に遭遇する可能性のある一般的な問題と解決方法を網羅しました。

### 重要なポイント

1. **エラーメッセージを注意深く読む**: 多くの場合、解決のヒントが含まれています
2. **環境を確認**: GPU、CUDA、依存関係のバージョンを確認
3. **データを検証**: 入力データの形状、型、範囲を確認
4. **デバッグツールを活用**: ロギング、プロファイリング、可視化
5. **段階的にテスト**: 小さなデータセットで動作確認してからスケールアップ

### さらなるサポート

- **GitHubのIssues**: https://github.com/nobkt/e3_diffusion_for_molecules/issues
- **チュートリアル**: `tutorials/README_JA.md`
- **APIリファレンス**: `doc/api_reference_ja.md`
- **ユースケース**: `doc/use_cases_ja.md`

---

**最終更新**: 2025年10月25日  
**バージョン**: 1.0
