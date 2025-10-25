# FAQ（よくある質問）

**最終更新**: 2025年10月25日  
**対象**: E3 Diffusion for Molecules ユーザー  
**目的**: プロジェクトに関するよくある質問と回答

---

## 目次

1. [一般的な質問](#一般的な質問)
2. [技術的な質問](#技術的な質問)
3. [ベストプラクティス](#ベストプラクティス)
4. [トラブルシューティング](#トラブルシューティング)

---

## 一般的な質問

### Q1: このプロジェクトは何をするものですか？

**A**: E3 Diffusion for Moleculesは、E(3)同変拡散モデルを使用して3D分子と結晶構造を生成するPythonライブラリです。主な特徴:

- **E(3)同変性**: 分子の向きや位置に依存しない生成
- **条件付き生成**: 特定の物理化学的性質を持つ分子を生成
- **結晶構造生成**: 230種類の空間群に対応した結晶生成
- **数学的厳密性**: 理論的に健全な実装、フォールバックなし

**コード例**:
```python
from qm9.models import get_model
from qm9.dataset import retrieve_dataloaders

# データセットをロード
dataloaders = retrieve_dataloaders('qm9.db', batch_size=64)

# モデルを構築
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, train_loader)

# 分子を生成
x, h = model.sample(n_samples=100, n_nodes=19, node_mask=node_mask, 
                    edge_mask=edge_mask, context=context)
```

---

### Q2: どのような研究分野に応用できますか？

**A**: 以下の分野で広く応用可能です:

#### 1. 医薬品設計
- 新規薬物候補の生成
- 薬物動態（ADME）特性の最適化
- 多形予測と結晶工学

**例**:
```python
# Lipinskiルールを満たす分子を生成
from qm9.utils import prepare_context

properties = {
    'molecular_weight': 350.0,  # < 500 Da
    'logP': 2.5,                # < 5
    'TPSA': 80.0,               # < 140 Å²
}

context, _ = prepare_context(properties, batch, property_norms)
x, h = model.sample(n_samples=100, context=context)
```

#### 2. 材料科学
- 有機半導体の探索
- 触媒材料の設計
- エネルギー貯蔵材料

**例**:
```python
# HOMO-LUMOギャップを制御
target_gap = 3.5  # eV（有機太陽電池に適した値）
context = prepare_context({'gap': target_gap}, batch, property_norms)
```

#### 3. 結晶工学
- 結晶多形の予測
- 新規結晶構造の探索
- 相転移の研究

#### 4. 化学情報学
- 仮想スクリーニング
- 構造-活性相関の研究
- 化学空間の探索

---

### Q3: 必要な環境とハードウェアは？

**A**: 

#### 最小構成
- **OS**: Linux, macOS, Windows
- **Python**: 3.7以上
- **CPU**: 任意（Intel, AMD, Apple Silicon）
- **RAM**: 8GB以上
- **Storage**: 20GB以上

#### 推奨構成
- **OS**: Linux（Ubuntu 20.04以降）
- **Python**: 3.9-3.10
- **GPU**: NVIDIA GPU（8GB VRAM以上）
  - 推奨: RTX 3080, V100, A100
- **RAM**: 32GB以上
- **Storage**: 100GB以上 SSD

#### GPU vs CPU

| 項目 | GPU | CPU |
|------|-----|-----|
| 学習時間（10K分子） | 2-3時間 | 20-40時間 |
| 生成時間（100分子） | 10秒 | 2-3分 |
| バッチサイズ | 64-128 | 8-16 |
| 推奨用途 | 学習、大規模生成 | 小規模実験、推論 |

**環境構築**:
```bash
# Condaで環境作成（推奨）
conda create -n e3-diffusion python=3.9
conda activate e3-diffusion

# PyTorch（GPU版）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 依存パッケージ
pip install ase rdkit spglib wandb
```

---

### Q4: 学習にどのくらい時間がかかりますか？

**A**: データセットサイズとハードウェアに依存します。

#### QM9データセット（約13万分子）

| 構成 | 時間 | 備考 |
|------|------|------|
| 1x V100 GPU | 8-12時間 | 推奨 |
| 1x RTX 3090 | 10-15時間 | 個人研究に最適 |
| 1x T4 GPU | 15-20時間 | クラウド環境 |
| 8-core CPU | 5-7日 | 非推奨 |

#### カスタムデータセット

**小規模**（1,000-5,000分子）:
- GPU: 30分-2時間
- CPU: 4-12時間

**中規模**（10,000-50,000分子）:
- GPU: 3-8時間
- CPU: 1-3日

**大規模**（100,000+分子）:
- GPU: 12-24時間
- CPU: 5-10日

**高速化のヒント**:
```python
# 1. バッチサイズを最大化（GPUメモリの許す限り）
batch_size = 128  # デフォルト: 64

# 2. 混合精度訓練を使用
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. DataLoaderのワーカー数を増やす
dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)

# 4. コンパイルモードを使用（PyTorch 2.0以降）
model = torch.compile(model)
```

---

### Q5: 商用利用は可能ですか？

**A**: プロジェクトのライセンスを確認してください。一般的に:

- **研究目的**: 自由に使用可能
- **商用目的**: ライセンス条項に従う必要があります
- **派生作品**: ライセンスに準拠する必要があります

**重要**: 生成された分子の化学的・生物学的安全性は使用者の責任で確認してください。

---

### Q6: どのようなサポートがありますか？

**A**: 

#### ドキュメント
- **チュートリアル**: `tutorials/README_JA.md`（6つの包括的なチュートリアル）
- **APIリファレンス**: `doc/api_reference_ja.md`（50以上のAPI）
- **ユースケース**: `doc/use_cases_ja.md`（7つの実用例）
- **トラブルシューティング**: `doc/troubleshooting_extended_ja.md`

#### コミュニティ
- **GitHubのIssues**: バグレポート、機能リクエスト
- **GitHubのDiscussions**: 質問、議論
- **論文**: 理論的背景の詳細

#### 推奨学習パス
1. チュートリアル01（基本的な分子生成）
2. チュートリアル02（条件付き生成）
3. チュートリアル04（カスタムデータセット）
4. ユースケース集で応用例を学ぶ
5. APIリファレンスで詳細を確認

---

### Q7: 他の生成モデルとの違いは？

**A**: 

| 特徴 | E3 Diffusion | VAE | GAN |
|------|--------------|-----|-----|
| E(3)同変性 | ✅ 厳密 | ❌ なし | ❌ なし |
| 学習の安定性 | ✅ 高い | ✅ 高い | ❌ 低い |
| 多様性 | ✅ 高い | ⚠️ 中程度 | ✅ 高い |
| 生成品質 | ✅ 高い | ⚠️ 中程度 | ✅ 高い |
| 条件付き生成 | ✅ ネイティブ | ✅ 可能 | ⚠️ 複雑 |
| 理論的保証 | ✅ あり | ✅ あり | ❌ なし |

**E(3)同変性の重要性**:
```python
# 同変性テスト
import torch

# ランダムな回転
R = random_rotation_matrix()

# 元の分子を生成
x_original, h_original = model.sample(...)

# 回転後に生成
model_rotated = apply_rotation_to_model(model, R)
x_rotated, h_rotated = model_rotated.sample(...)

# 検証: x_rotated ≈ x_original @ R.T
assert torch.allclose(x_rotated, x_original @ R.T, atol=1e-5)
```

---

### Q8: 論文として発表されていますか？

**A**: このプロジェクトは以下の研究に基づいています:

**主要論文**:
1. **E(n) Equivariant Graph Neural Networks** (ICML 2021)
   - Satorras et al.
   - https://arxiv.org/abs/2102.09844

2. **Equivariant Diffusion for Molecule Generation** (ICML 2022)
   - Hoogeboom et al.
   - 本プロジェクトの主要アルゴリズム

3. **Denoising Diffusion Probabilistic Models** (NeurIPS 2020)
   - Ho et al.
   - https://arxiv.org/abs/2006.11239

詳細は `doc/papers_ja.md` を参照してください。

---

### Q9: モデルの事前学習済み重みは提供されていますか？

**A**: プロジェクトの配布方針によります。一般的に:

**提供される可能性があるもの**:
- QM9データセットで訓練済みのモデル
- GEOMDrugsデータセットで訓練済みのモデル
- ベースライン結晶生成モデル

**使用方法**:
```python
import torch

# 事前学習済みモデルをロード
checkpoint = torch.load('pretrained_qm9.pkl')
model.load_state_dict(checkpoint['model_state_dict'])

# そのまま使用、または微調整
model.eval()  # 推論モード
x, h = model.sample(n_samples=100)

# 微調整
model.train()  # 学習モード
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

---

### Q10: プロジェクトに貢献できますか？

**A**: はい！貢献は歓迎されます。

**貢献方法**:
1. **バグレポート**: GitHubのIssuesで報告
2. **機能リクエスト**: Issuesで提案
3. **コード貢献**: Pull Requestを作成
4. **ドキュメント改善**: 誤字脱字の修正、説明の改善
5. **ユースケースの共有**: 成功事例を共有

**Pull Request のガイドライン**:
```bash
# 1. リポジトリをフォーク
git clone https://github.com/YOUR_USERNAME/e3_diffusion_for_molecules.git

# 2. ブランチを作成
git checkout -b feature/your-feature-name

# 3. 変更を実装
# - コードスタイルを統一
# - テストを追加
# - ドキュメントを更新

# 4. テストを実行
python -m pytest tests/

# 5. Pull Requestを作成
git push origin feature/your-feature-name
```

---

## 技術的な質問

### Q11: GPUは必須ですか？

**A**: 必須ではありませんが、強く推奨します。

**CPUでの実行**:
```python
# デバイスを自動選択
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# モデルをデバイスに移動
model = model.to(device)

# データもデバイスに移動
x = x.to(device)
h = h.to(device)
```

**CPUで高速化する方法**:
1. バッチサイズを小さく（8-16）
2. モデルサイズを縮小（nf=128）
3. 拡散ステップ数を減らす（500-700）
4. 並列処理を活用

```python
# PyTorchのスレッド数を設定
import torch
torch.set_num_threads(8)  # CPUコア数に応じて調整
```

---

### Q12: どのくらいのデータが必要ですか？

**A**: タスクの複雑さに依存します。

#### 最小データ量
- **無条件生成**: 1,000分子
- **条件付き生成**: 3,000-5,000分子
- **結晶生成**: 500-1,000結晶

#### 推奨データ量
- **高品質な無条件生成**: 10,000分子以上
- **複数性質の条件付き生成**: 20,000分子以上
- **空間群特化の結晶生成**: 1,000結晶以上

**データ拡張の活用**:
```python
def augment_molecule(x, h):
    """分子のデータ拡張"""
    # ランダムな回転
    R = random_rotation_matrix()
    x_rot = x @ R.T
    
    # ランダムな並進
    t = torch.randn(1, 3) * 0.1
    x_trans = x_rot + t
    
    # ランダムな反転
    if random.random() > 0.5:
        x_trans = -x_trans
    
    return x_trans, h

# データ拡張を適用
for x, h in dataloader:
    x_aug, h_aug = augment_molecule(x, h)
    # 訓練に使用
```

**転移学習の活用**:
```python
# QM9で事前学習したモデルを小規模データセットで微調整
pretrained = torch.load('qm9_pretrained.pkl')
model.load_state_dict(pretrained['model_state_dict'])

# 低い学習率で微調整
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 少量のデータで訓練
for epoch in range(50):
    train(model, small_dataset, optimizer)
```

---

### Q13: カスタムデータセットを使用できますか？

**A**: はい、ASEデータベース形式でデータを準備すれば使用可能です。

**ステップ1: ASEデータベースの作成**
```python
from ase import Atoms
from ase.db import connect

# データベースを作成
db = connect('custom_molecules.db')

# 分子を追加
for smiles, properties in your_data:
    # SMILESから3D構造を生成（RDKit使用）
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # ASE Atomsオブジェクトに変換
    positions = mol.GetConformer().GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    
    # 性質と共にデータベースに書き込み
    db.write(atoms, data=properties)
```

**ステップ2: データのロード**
```python
from qm9.dataset import retrieve_dataloaders

# カスタムデータセットをロード
dataloaders = retrieve_dataloaders(
    'custom_molecules.db',
    batch_size=64,
    num_workers=4
)

train_loader, val_loader, test_loader = dataloaders
```

**ステップ3: 訓練**
```python
# 通常通り訓練
from qm9.models import get_model

model, nodes_dist, prop_dist = get_model(args, device, dataset_info, train_loader)

for epoch in range(n_epochs):
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

詳細は **チュートリアル04** および **ユースケース5** を参照してください。

---

### Q14: 複数のGPUで訓練できますか？

**A**: はい、PyTorchのDataParallelまたはDistributedDataParallelを使用できます。

**DataParallel（シンプル）**:
```python
import torch
import torch.nn as nn

# 複数GPUでモデルをラップ
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to('cuda')

# 通常通り訓練
for batch in train_loader:
    batch = batch.to('cuda')
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**DistributedDataParallel（推奨、高速）**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """分散訓練の初期化"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # モデルをGPUに配置
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 分散サンプラー
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    
    # 訓練
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for batch in train_loader:
            # 通常通り訓練
            pass
    
    dist.destroy_process_group()

# 起動
import torch.multiprocessing as mp
world_size = torch.cuda.device_count()
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

---

### Q15: 生成された分子の品質を評価する方法は？

**A**: 複数の評価指標があります。

**1. 有効性 (Validity)**
```python
from rdkit import Chem

def compute_validity(molecules):
    """分子の有効性を計算"""
    valid_count = 0
    for mol in molecules:
        if Chem.MolFromSmiles(mol) is not None:
            valid_count += 1
    return valid_count / len(molecules)

validity = compute_validity(generated_smiles)
print(f"Validity: {validity:.2%}")
```

**2. 一意性 (Uniqueness)**
```python
def compute_uniqueness(molecules):
    """一意性を計算"""
    unique_molecules = set(molecules)
    return len(unique_molecules) / len(molecules)

uniqueness = compute_uniqueness(generated_smiles)
print(f"Uniqueness: {uniqueness:.2%}")
```

**3. 新規性 (Novelty)**
```python
def compute_novelty(generated, training_set):
    """新規性を計算"""
    training_set = set(training_set)
    novel_count = sum(1 for mol in generated if mol not in training_set)
    return novel_count / len(generated)

novelty = compute_novelty(generated_smiles, training_smiles)
print(f"Novelty: {novelty:.2%}")
```

**4. 分子安定性**
```python
from qm9.analyze import check_stability

stability = check_stability(positions, atom_types)
print(f"Stable molecules: {stability['mol_stable']}/{stability['n_molecules']}")
```

**5. 性質分布の一致**
```python
import matplotlib.pyplot as plt

# 訓練データと生成データの性質分布を比較
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

properties = ['homo', 'lumo', 'gap']
for ax, prop in zip(axes, properties):
    ax.hist(train_properties[prop], bins=50, alpha=0.5, label='Training')
    ax.hist(generated_properties[prop], bins=50, alpha=0.5, label='Generated')
    ax.set_xlabel(prop)
    ax.legend()

plt.tight_layout()
plt.savefig('property_distribution.png')
```

詳細は **チュートリアル05** および **APIリファレンス** の `qm9.analyze` セクションを参照してください。

---

### Q16: メモリ不足エラーが発生します

**A**: メモリ使用量を削減する方法はいくつかあります。

**方法1: バッチサイズを減らす**
```python
# 推奨値: GPU 8GB → batch_size=16, GPU 16GB → batch_size=32-64
batch_size = 16  # デフォルトの64から削減
```

**方法2: モデルサイズを縮小**
```python
# モデルパラメータを調整
nf = 128           # デフォルト: 256
n_layers = 6       # デフォルト: 9
attention_nf = 32  # デフォルト: 64
```

**方法3: 勾配累積を使用**
```python
# 実質的なバッチサイズを維持しながらメモリを節約
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps  # 16 * 4 = 64

optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**方法4: 混合精度訓練**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():  # 自動的にfloat16を使用
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**方法5: チェックポイント手法**
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        # メモリを節約するために勾配チェックポイントを使用
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

---

### Q17: 訓練が収束しません

**A**: いくつかの原因が考えられます。

**原因1: 学習率が高すぎる**
```python
# 学習率を下げる
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 2e-4から削減

# または学習率スケジューラを使用
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    verbose=True
)

for epoch in range(n_epochs):
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)
```

**原因2: 勾配爆発**
```python
# 勾配クリッピングを使用
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 訓練ループ内で
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**原因3: データの正規化が不適切**
```python
from qm9.utils import compute_mean_mad

# 性質を正規化
property_norms = compute_mean_mad(dataloaders, properties, 'qm9')

# コンテキストを準備する際に正規化を使用
context, _ = prepare_context(properties, batch, property_norms)
```

**原因4: モデルが複雑すぎる**
```python
# よりシンプルなモデルから開始
nf = 128        # 256から削減
n_layers = 6    # 9から削減

# 十分に収束してから複雑なモデルに移行
```

**診断ツール**:
```python
# 損失の推移をプロット
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')  # 対数スケール
plt.savefig('loss_curve.png')

# 勾配ノルムを監視
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms.append(param.grad.norm().item())

print(f"Max gradient norm: {max(grad_norms):.4f}")
print(f"Mean gradient norm: {sum(grad_norms)/len(grad_norms):.4f}")
```

---

### Q18: 特定の性質を持つ分子を生成するには？

**A**: 条件付き生成を使用します。

**単一性質の制御**:
```python
from qm9.utils import prepare_context, compute_mean_mad

# 性質の正規化情報を計算
property_norms = compute_mean_mad(dataloaders, ['homo'], 'qm9')

# ターゲット性質を設定
target_homo = -5.0  # eV

# コンテキストを準備
context_dict = {'homo': target_homo}
context, _ = prepare_context(['homo'], context_dict, property_norms)

# 生成
x, h = model.sample(
    n_samples=100,
    n_nodes=19,
    node_mask=node_mask,
    edge_mask=edge_mask,
    context=context
)
```

**複数性質の制御**:
```python
# 複数の性質を同時に制御
properties = ['homo', 'lumo', 'gap']
property_norms = compute_mean_mad(dataloaders, properties, 'qm9')

# ターゲット値を設定
targets = {
    'homo': -5.0,   # eV
    'lumo': -1.0,   # eV
    'gap': 4.0,     # eV
}

context, _ = prepare_context(properties, targets, property_norms)
x, h = model.sample(n_samples=100, context=context, ...)
```

**Classifier-Free Guidanceで精度向上**:
```python
# ガイダンススケールを調整
guidance_scale = 2.0  # 大きいほど条件への従属度が高い

# サンプリング時にガイダンスを適用
x, h = model.sample_with_guidance(
    n_samples=100,
    context=context,
    guidance_scale=guidance_scale,
    ...
)
```

詳細は **チュートリアル02** を参照してください。

---

### Q19: 結晶構造を生成するには？

**A**: 結晶生成用のモデルを使用します。

**基本的な結晶生成**:
```python
from crystal.models import CrystalDynamics
from crystal.conditioning import SpaceGroupEmbedding

# 結晶モデルを構築
crystal_model = CrystalDynamics(
    n_dims=3,
    in_node_nf=n_atom_types,
    context_node_nf=context_nf,
    hidden_nf=256,
    n_layers=9
)

# 空間群条件付け
space_group = 14  # P21/c
sg_embedding = SpaceGroupEmbedding(space_group)
context = sg_embedding.get_embedding()

# 結晶を生成
lattice, positions, atom_types = crystal_model.sample(
    n_atoms=20,
    space_group=space_group,
    context=context
)
```

**空間群を指定**:
```python
# 特定の空間群で生成
space_groups = [14, 15, 61, 19]  # よく使われる空間群

for sg in space_groups:
    lattice, positions, atom_types = crystal_model.sample(
        n_atoms=20,
        space_group=sg
    )
    print(f"Generated crystal in space group {sg}")
```

**Wyckoff位置を制約**:
```python
from crystal.utils import generate_wyckoff_positions

# 特定のWyckoff位置に原子を配置
wyckoff_sites = generate_wyckoff_positions(
    space_group=14,
    n_atoms=20,
    site_preference={'4e': 16, '2a': 2, '2b': 2}
)

lattice, positions, atom_types = crystal_model.sample(
    n_atoms=20,
    space_group=14,
    wyckoff_sites=wyckoff_sites
)
```

**密度制約**:
```python
from crystal.conditioning import DensityConditioning

# ターゲット密度を設定
target_density = 1.5  # g/cm³

density_cond = DensityConditioning(target_density)
context = density_cond.get_conditioning()

lattice, positions, atom_types = crystal_model.sample(
    n_atoms=20,
    space_group=14,
    context=context
)
```

詳細は **チュートリアル03** および **チュートリアル06** を参照してください。

---

### Q20: 生成速度を向上させるには？

**A**: いくつかの最適化手法があります。

**方法1: 拡散ステップ数を減らす**
```python
# 品質と速度のトレードオフ
diffusion_steps = 500  # デフォルト: 1000

# DDIMサンプリングを使用（より少ないステップで高品質）
from equivariant_diffusion.sampling import ddim_sample

x, h = ddim_sample(
    model,
    n_samples=100,
    n_steps=50,  # 大幅に少ないステップ数
    ...
)
```

**方法2: バッチ生成**
```python
# 複数分子を同時に生成
batch_size = 128  # GPUメモリが許す限り大きく

x, h = model.sample(
    n_samples=batch_size,  # 一度に多数生成
    ...
)
```

**方法3: 並列生成**
```python
import multiprocessing as mp
from functools import partial

def generate_batch(batch_id, model, n_samples):
    """バッチを生成"""
    x, h = model.sample(n_samples=n_samples, ...)
    return x, h

# 複数プロセスで並列生成
n_processes = 4
n_samples_per_process = 250

with mp.Pool(n_processes) as pool:
    generate_fn = partial(generate_batch, model=model, n_samples=n_samples_per_process)
    results = pool.map(generate_fn, range(n_processes))

# 結果を結合
all_x = torch.cat([r[0] for r in results], dim=0)
all_h = torch.cat([r[1] for r in results], dim=0)
```

**方法4: モデルの最適化**
```python
# PyTorch 2.0以降: compilationを使用
model = torch.compile(model, mode='max-autotune')

# TorchScript（より広範な互換性）
model_scripted = torch.jit.script(model)
model_scripted.save('model_scripted.pt')
```

**方法5: キャッシングを活用**
```python
# 頻繁に使用される計算結果をキャッシュ
from functools import lru_cache

@lru_cache(maxsize=1000)
def compute_edge_mask(n_atoms):
    """エッジマスクをキャッシュ"""
    node_mask = torch.ones(1, n_atoms)
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    return edge_mask

# 使用
edge_mask = compute_edge_mask(n_atoms=19)
```

---

## ベストプラクティス

### Q21: 学習時のハイパーパラメータはどう設定すべきですか？

**A**: データセットとタスクに応じて調整しますが、以下が良い出発点です。

**QM9データセット（小分子）**:
```python
# モデルアーキテクチャ
nf = 256                    # 隠れ層の次元数
n_layers = 9                # EGNNレイヤー数
attention = True            # 注意機構を使用
tanh = True                 # tanh活性化を使用

# 訓練設定
batch_size = 64             # GPU 16GB以上
lr = 2e-4                   # 学習率
n_epochs = 1000             # エポック数
ema_decay = 0.999           # 指数移動平均

# 拡散設定
diffusion_steps = 1000      # 拡散ステップ数
diffusion_noise_schedule = 'polynomial_2'  # ノイズスケジュール
```

**カスタムデータセット（中規模）**:
```python
# 小さめから開始
nf = 128
n_layers = 6
batch_size = 32
lr = 1e-4

# 性能を確認してから増やす
```

**大規模データセット**:
```python
# より大きなモデル
nf = 384
n_layers = 12
batch_size = 128  # 複数GPU推奨

# 長時間訓練
n_epochs = 2000
```

**ハイパーパラメータチューニング**:
```python
import optuna

def objective(trial):
    """Optunaによるハイパーパラメータチューニング"""
    # パラメータを提案
    nf = trial.suggest_categorical('nf', [128, 256, 384])
    n_layers = trial.suggest_int('n_layers', 6, 12)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    
    # モデルを訓練
    model = build_model(nf=nf, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    val_loss = train_and_validate(model, optimizer, n_epochs=100)
    
    return val_loss

# 最適化を実行
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(f"Best hyperparameters: {study.best_params}")
```

---

### Q22: 生成品質を向上させるには？

**A**: 複数のアプローチがあります。

**1. より多くのデータで訓練**
```python
# データ拡張
def augment_data(x, h):
    # 回転
    R = random_rotation_matrix()
    x_aug = x @ R.T
    
    # ノイズ追加
    x_aug = x_aug + torch.randn_like(x_aug) * 0.01
    
    return x_aug, h

# 訓練ループで使用
for x, h in train_loader:
    x_aug, h_aug = augment_data(x, h)
    loss = model(x_aug, h_aug)
```

**2. 拡散ステップ数を増やす**
```python
# より多くのステップで滑らかな生成
diffusion_steps = 1500  # デフォルト: 1000

model = EnVariationalDiffusion(
    dynamics=dynamics,
    timesteps=diffusion_steps,
    ...
)
```

**3. モデルの深さを増やす**
```python
# より表現力のあるモデル
n_layers = 12  # デフォルト: 9
nf = 384       # デフォルト: 256

# ただし、過学習に注意
```

**4. Classifier-Free Guidanceのスケール調整**
```python
# ガイダンススケールの影響をテスト
guidance_scales = [0.5, 1.0, 2.0, 5.0]

for scale in guidance_scales:
    x, h = model.sample_with_guidance(
        guidance_scale=scale,
        ...
    )
    # 品質を評価
    validity = compute_validity(x, h)
    print(f"Scale {scale}: Validity = {validity:.2%}")
```

**5. EMA（指数移動平均）を使用**
```python
from torch_ema import ExponentialMovingAverage

# EMAを初期化
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

# 訓練ループ
for batch in train_loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # EMAを更新
    ema.update()

# 推論時はEMAモデルを使用
with ema.average_parameters():
    x, h = model.sample(...)
```

**6. アーリーストッピング**
```python
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 訓練を停止
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

# 使用
early_stopping = EarlyStopping(patience=50)

for epoch in range(n_epochs):
    val_loss = validate(model, val_loader)
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

### Q23: 再現性を確保するには？

**A**: 乱数シードを固定し、決定論的な動作を設定します。

**完全な再現性のセットアップ**:
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """全ての乱数生成器のシードを固定"""
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 複数GPU
    
    # CuDNNを決定論的にする
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Pythonハッシュシードを固定
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

# プログラムの最初で呼び出す
set_seed(42)
```

**注意事項**:
- 決定論的モードは実行速度が遅くなる可能性があります
- 完全な再現性は同じハードウェアとPyTorchバージョンでのみ保証されます

**実験管理**:
```python
import wandb

# 実験設定を記録
config = {
    'seed': 42,
    'batch_size': 64,
    'lr': 2e-4,
    'n_layers': 9,
    'nf': 256,
    # ... その他のハイパーパラメータ
}

# W&Bで記録
wandb.init(project='molecule-generation', config=config)

# コードのバージョンも記録
import subprocess
git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
wandb.config.update({'git_commit': git_commit})
```

---

### Q24: 効率的なデータローディングの方法は？

**A**: PyTorchのDataLoaderを最適化します。

**基本的な設定**:
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # CPUコア数に応じて調整
    pin_memory=True,      # GPUメモリへの転送を高速化
    persistent_workers=True,  # ワーカーを再利用
    prefetch_factor=2     # 先読みバッファ
)
```

**カスタムcollate_fn**:
```python
def custom_collate(batch):
    """バッチを効率的にパック"""
    # 最大原子数を見つける
    max_n_atoms = max([item['n_atoms'] for item in batch])
    
    # パディング
    positions = torch.zeros(len(batch), max_n_atoms, 3)
    features = torch.zeros(len(batch), max_n_atoms, n_features)
    node_mask = torch.zeros(len(batch), max_n_atoms)
    
    for i, item in enumerate(batch):
        n = item['n_atoms']
        positions[i, :n] = item['positions']
        features[i, :n] = item['features']
        node_mask[i, :n] = 1.0
    
    return {
        'positions': positions,
        'features': features,
        'node_mask': node_mask
    }

train_loader = DataLoader(
    train_dataset,
    collate_fn=custom_collate,
    ...
)
```

**大規模データセットの処理**:
```python
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    """データをストリーミングで読み込む"""
    def __init__(self, db_path):
        self.db_path = db_path
    
    def __iter__(self):
        from ase.db import connect
        db = connect(self.db_path)
        
        for row in db.select():
            atoms = row.toatoms()
            data = row.data
            
            yield {
                'positions': torch.tensor(atoms.get_positions()),
                'atomic_numbers': torch.tensor(atoms.get_atomic_numbers()),
                'properties': data
            }

# 使用
streaming_dataset = StreamingDataset('large_dataset.db')
train_loader = DataLoader(streaming_dataset, batch_size=64, num_workers=4)
```

---

### Q25: モデルのチェックポイントを管理するには？

**A**: 定期的に保存し、ベストモデルを保持します。

**基本的なチェックポイント**:
```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    """チェックポイントを保存"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config,  # モデル設定も保存
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None):
    """チェックポイントをロード"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    return checkpoint['epoch'], checkpoint['loss']
```

**ベストモデルの保持**:
```python
best_val_loss = float('inf')

for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # 定期的に保存
    if epoch % 10 == 0:
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            f'checkpoints/epoch_{epoch}.pkl'
        )
    
    # ベストモデルを保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            'checkpoints/best_model.pkl'
        )
        print(f"New best model at epoch {epoch}")
```

**古いチェックポイントの自動削除**:
```python
import glob
import os

def cleanup_old_checkpoints(checkpoint_dir, keep_last=5):
    """古いチェックポイントを削除"""
    checkpoints = glob.glob(f"{checkpoint_dir}/epoch_*.pkl")
    
    # 時刻でソート
    checkpoints.sort(key=os.path.getmtime)
    
    # 古いものを削除（最新のkeep_last個を残す）
    for ckpt in checkpoints[:-keep_last]:
        os.remove(ckpt)
        print(f"Removed old checkpoint: {ckpt}")

# 定期的に実行
if epoch % 50 == 0:
    cleanup_old_checkpoints('checkpoints', keep_last=5)
```

---

## トラブルシューティング

### Q26: "CUDA out of memory"エラーが出ます

**A**: メモリ使用量を削減する必要があります。

→ **Q16: メモリ不足エラーが発生します** を参照してください。

追加の対策:
```python
# GPUメモリをクリア
torch.cuda.empty_cache()

# 使用していない変数を削除
del x, h, loss
torch.cuda.empty_cache()
```

---

### Q27: 生成された分子が不安定です

**A**: いくつかの原因が考えられます。

**原因1: 学習が不十分**
```python
# 損失をプロット
plt.plot(train_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')

# 収束していない場合は追加訓練
```

**原因2: 拡散ステップ数が少ない**
```python
# ステップ数を増やす
diffusion_steps = 1500  # 1000から増加
```

**原因3: データセットに不安定な分子が含まれる**
```python
# データをフィルタリング
from qm9.analyze import check_stability

filtered_molecules = []
for mol in dataset:
    stability = check_stability(mol['positions'], mol['atom_types'])
    if stability['mol_stable']:
        filtered_molecules.append(mol)

print(f"Filtered: {len(filtered_molecules)}/{len(dataset)} molecules")
```

**原因4: 後処理が必要**
```python
# 生成後に構造最適化
from ase.optimize import BFGS
from ase.calculators.emt import EMT

atoms = Atoms(...)
atoms.set_calculator(EMT())
opt = BFGS(atoms)
opt.run(fmax=0.05)

optimized_positions = atoms.get_positions()
```

---

### Q28: 条件付き生成が機能しません

**A**: 以下を確認してください。

**チェック1: 性質の正規化**
```python
# 正規化情報を確認
from qm9.utils import compute_mean_mad

property_norms = compute_mean_mad(dataloaders, ['homo'], 'qm9')
print(f"HOMO mean: {property_norms['homo']['mean']}")
print(f"HOMO MAD: {property_norms['homo']['mad']}")

# コンテキストが正しく正規化されているか確認
context, _ = prepare_context(['homo'], batch, property_norms)
print(f"Context shape: {context.shape}")
print(f"Context range: [{context.min():.3f}, {context.max():.3f}]")
```

**チェック2: モデルが条件付きで訓練されているか**
```python
# 訓練時にcontextを使用しているか確認
loss = model(x, h, node_mask, edge_mask, context=context)  # contextを渡す

# contextがNoneになっていないか確認
assert context is not None, "Context is None!"
```

**チェック3: ガイダンススケール**
```python
# ガイダンススケールを調整
for scale in [0.5, 1.0, 2.0, 5.0]:
    x, h = model.sample_with_guidance(
        guidance_scale=scale,
        context=context,
        ...
    )
    # 生成された性質を評価
```

---

### Q29: 訓練が遅すぎます

**A**: 複数の最適化が可能です。

→ **Q20: 生成速度を向上させるには？** の方法を訓練にも適用できます。

追加の最適化:
```python
# 1. DataLoaderを最適化（Q24参照）
train_loader = DataLoader(..., num_workers=8, pin_memory=True)

# 2. 混合精度訓練
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. 勾配累積（バッチサイズを大きくする効果）
accumulation_steps = 4

# 4. モデルのコンパイル（PyTorch 2.0+）
model = torch.compile(model)

# 5. プロファイリングでボトルネックを特定
with torch.profiler.profile() as prof:
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        loss = model(batch)
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

### Q30: 結果が再現できません

**A**: 再現性の設定を確認してください。

→ **Q23: 再現性を確保するには？** を参照してください。

追加のチェックポイント:
```python
# 1. シードが設定されているか
set_seed(42)

# 2. 同じデータセット順序か
train_loader = DataLoader(..., shuffle=False)  # 検証時

# 3. 同じハードウェアとソフトウェアか
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 4. ドロップアウトなどが無効化されているか
model.eval()  # 推論モード
with torch.no_grad():
    x, h = model.sample(...)
```

---

## まとめ

このFAQでは、E3 Diffusion for Moleculesに関するよくある質問とその回答をまとめました。

### さらなる情報源

- **チュートリアル**: `tutorials/README_JA.md`
- **APIリファレンス**: `doc/api_reference_ja.md`
- **ユースケース**: `doc/use_cases_ja.md`
- **トラブルシューティング**: `doc/troubleshooting_extended_ja.md`
- **用語集**: `doc/glossary_ja.md`

### サポート

質問や問題がある場合:
1. まずこのFAQとトラブルシューティングガイドを確認
2. GitHubのIssuesを検索
3. 新しいIssueを作成（再現可能な例を含める）
4. GitHubのDiscussionsで議論

---

**最終更新**: 2025年10月25日  
**バージョン**: 1.0  
**対象プロジェクト**: E3 Diffusion for Molecules
