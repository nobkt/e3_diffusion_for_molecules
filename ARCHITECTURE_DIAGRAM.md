# 分子性結晶生成システム アーキテクチャ図
# Molecular Crystal Generation System Architecture Diagram

## システム全体図 (Overall System Architecture)

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                      Molecular Crystal Generation System                       ║
║                    分子性結晶生成システム (E3DM Extended)                        ║
╚════════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER (入力層)                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┏━━━━━━━━━━━━━━━━━━┓  ┏━━━━━━━━━━━━━━━━━━┓  ┏━━━━━━━━━━━━━━━━━━┓         │
│  ┃ ASE Database    ┃  ┃ QM9 Dataset     ┃  ┃ GEOM Dataset    ┃         │
│  ┃ (.db files)     ┃  ┃ (molecules)     ┃  ┃ (molecules)     ┃         │
│  ┗━━━━━━━━━━━━━━━━━━┛  ┗━━━━━━━━━━━━━━━━━━┛  ┗━━━━━━━━━━━━━━━━━━┛         │
│         │                     │                      │                        │
│         └─────────────────────┴──────────────────────┘                        │
│                               │                                               │
└───────────────────────────────┼───────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER (データ処理層)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────┐         ┌──────────────────────┐                   │
│  │ CrystalDataset      │◄────────┤ PeriodicBoundary     │                   │
│  │ - Load structures   │         │ Handler              │                   │
│  │ - Convert coords    │         │ - Min image distance │                   │
│  │ - Extract cell info │         │ - Neighbor list      │                   │
│  └─────────────────────┘         │ - Wrap positions     │                   │
│           │                      └──────────────────────┘                   │
│           │                                                                   │
│           ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │ Coordinate Transformations                              │                │
│  │ • Cartesian ↔ Fractional                               │                │
│  │ • Cell Vectors ↔ Cell Parameters (a,b,c,α,β,γ)        │                │
│  └─────────────────────────────────────────────────────────┘                │
│           │                                                                   │
└───────────┼───────────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER (モデル層)                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Diffusion Process (拡散プロセス)                     │ │
│  │  ┌──────────────────────────┐    ┌──────────────────────────┐         │ │
│  │  │   Position Diffusion     │    │   Lattice Diffusion      │         │ │
│  │  │   (原子座標の拡散)        │    │   (格子パラメータの拡散)   │         │ │
│  │  │                          │    │                          │         │ │
│  │  │  • SE(3) Equivariant    │    │  • GL(3) Covariant      │         │ │
│  │  │  • Periodic boundaries  │    │  • Physical constraints │         │ │
│  │  └──────────────────────────┘    └──────────────────────────┘         │ │
│  │             │                              │                           │ │
│  │             └──────────────┬───────────────┘                           │ │
│  │                            ▼                                           │ │
│  │              ┌──────────────────────────┐                              │ │
│  │              │  Crystal Dynamics Model  │                              │ │
│  │              │  (結晶動力学モデル)       │                              │ │
│  │              └──────────────────────────┘                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                 │
│                            ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Periodic E(3)-EGNN                                  │ │
│  │                    (周期的等変グラフニューラルネットワーク)               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │ │
│  │  │ Message Pass │  │ Coord Update │  │ Node Update  │                │ │
│  │  │ (periodic)   │─▶│ (periodic)   │─▶│              │                │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │ │
│  │         │                 │                  │                         │ │
│  │         └─────────────────┴──────────────────┘                         │ │
│  │                            │                                            │ │
│  │                  [Repeated n_layers times]                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                 │
└────────────────────────────┼─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   CONDITIONING LAYER (条件付け層)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Space Group   │  │ Density       │  │ Lattice      │  │ Molecular    │ │
│  │ Embedding     │  │ Conditioning  │  │ Params       │  │ Properties   │ │
│  │ (空間群)       │  │ (密度)         │  │ (格子定数)    │  │ (分子特性)    │ │
│  └───────────────┘  └───────────────┘  └──────────────┘  └──────────────┘ │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                               │                                             │
│                               ▼                                             │
│                  ┌─────────────────────────┐                                │
│                  │  Context Vector         │                                │
│                  │  (条件付けベクトル)       │                                │
│                  └─────────────────────────┘                                │
│                               │                                             │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER (出力層)                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   Generated Crystal Structure                          │ │
│  │                   (生成された結晶構造)                                   │ │
│  │  • Atomic positions (原子座標)                                          │ │
│  │  • Atom types (原子種)                                                  │ │
│  │  • Cell parameters (格子パラメータ)                                      │ │
│  │  • Space group (空間群) [optional]                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                               │                                              │
│                     ┌─────────┴─────────┐                                    │
│                     ▼                   ▼                                    │
│         ┌──────────────────┐  ┌──────────────────┐                          │
│         │ Post-processing  │  │ Validation       │                          │
│         │ • Apply symmetry │  │ • Check validity │                          │
│         │ • Optimize cell  │  │ • Compute metrics│                          │
│         └──────────────────┘  └──────────────────┘                          │
│                     │                   │                                    │
│                     └─────────┬─────────┘                                    │
│                               ▼                                              │
│         ┌──────────────────────────────────────────┐                         │
│         │          Export Formats                  │                         │
│         │  ┌──────────┐ ┌──────┐ ┌─────────────┐ │                         │
│         │  │ ASE DB   │ │ CIF  │ │ XYZ (super) │ │                         │
│         │  └──────────┘ └──────┘ └─────────────┘ │                         │
│         └──────────────────────────────────────────┘                         │
│                               │                                              │
└───────────────────────────────┼──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 EVALUATION LAYER (評価層)                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Structural       │  │ Symmetry         │  │ Physical         │          │
│  │ Metrics          │  │ Metrics          │  │ Metrics          │          │
│  │ • Lattice MAE    │  │ • Space group    │  │ • Stability      │          │
│  │ • Density error  │  │   accuracy       │  │ • Packing        │          │
│  │ • RDF similarity │  │ • Symmetry score │  │ • Energy         │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## データフロー詳細 (Detailed Data Flow)

### 1. Training Phase (学習フェーズ)

```
┌─────────────┐
│   Dataset   │  ASE DB with crystal structures
│   (結晶DB)   │  • Positions (座標)
└──────┬──────┘  • Cell (単位格子)
       │         • PBC (周期境界条件)
       ▼
┌─────────────────────────┐
│   DataLoader            │
│   • Batch crystals      │
│   • Apply padding       │
│   • Convert coordinates │
└──────┬──────────────────┘
       │
       ▼
┌───────────────────────────────────────┐
│  Forward Diffusion (xt, cell_t)      │
│  • Add noise to positions             │
│  • Add noise to lattice               │
└──────┬────────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────┐
│  Crystal Dynamics Model               │
│  ┌─────────────────────────────────┐  │
│  │ Periodic EGNN                   │  │
│  │ • Compute minimum image         │  │
│  │ • Build neighbor list           │  │
│  │ • Message passing               │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │ Lattice Diffusion               │  │
│  │ • Predict lattice noise         │  │
│  └─────────────────────────────────┘  │
└──────┬────────────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Loss Computation       │
│  • Position loss        │
│  • Lattice loss         │
│  • Combined loss        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Optimizer Step         │
│  • Backpropagation      │
│  • Update weights       │
└─────────────────────────┘
```

### 2. Sampling Phase (サンプリングフェーズ)

```
┌─────────────┐
│ Random Noise│  x_T ~ N(0, I)
│ (ランダム)   │  cell_T ~ N(μ, σ)
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│ Reverse Diffusion    │  For t = T, T-1, ..., 1:
│ (逆拡散プロセス)      │
└──────┬───────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Crystal Dynamics Model                │
│  • Predict velocity: v_t              │
│  • Predict lattice velocity: v_cell_t │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Update Step                           │
│  • x_{t-1} = x_t - dt * v_t           │
│  • cell_{t-1} = cell_t - dt * v_cell_t│
└──────┬─────────────────────────────────┘
       │
       │ (Repeat until t = 0)
       │
       ▼
┌─────────────┐
│   x_0       │  Generated crystal structure
│   cell_0    │  • Clean positions
└──────┬──────┘  • Clean lattice
       │
       ▼
┌─────────────────────────┐
│  Post-processing        │
│  • Wrap to unit cell    │
│  • Apply symmetry (opt) │
│  • Validate structure   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────┐
│   Output    │  ASE Atoms object / CIF / XYZ
└─────────────┘
```

---

## モジュール間の依存関係 (Module Dependencies)

```
                        main_crystal.py
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
        crystal/data   crystal/models  crystal/conditioning
                │             │             │
                │     ┌───────┴───────┐     │
                │     │               │     │
                ▼     ▼               ▼     ▼
          periodic_utils        crystal_dynamics
                │                     │
                │              ┌──────┴──────┐
                │              │             │
                ▼              ▼             ▼
          coordinate_transform  periodic_egnn  lattice_diffusion
                                     │
                                     ▼
                            equivariant_diffusion/
                                 en_diffusion.py
```

### 依存関係の詳細:

1. **crystal/data/**
   - `crystal_loader.py` → `periodic_utils.py`
   - `periodic_utils.py` → 独立（外部依存: torch, numpy）

2. **crystal/models/**
   - `crystal_dynamics.py` → `periodic_egnn.py`, `lattice_diffusion.py`
   - `periodic_egnn.py` → `crystal/data/periodic_utils.py`
   - `lattice_diffusion.py` → `crystal/data/periodic_utils.py`

3. **crystal/conditioning/**
   - 各モジュール → 独立（torch.nnのみ依存）

4. **crystal/evaluation/**
   - `crystal_metrics.py` → `crystal/data/periodic_utils.py`

---

## 座標系の変換フロー (Coordinate System Transformation Flow)

```
                    ┌──────────────────┐
                    │  ASE Atoms       │
                    │  (Cartesian)     │
                    └────────┬─────────┘
                             │
                  ┌──────────▼──────────┐
                  │ cartesian_to_        │
                  │ fractional()         │
                  └──────────┬──────────┘
                             │
                    ┌────────▼────────┐
                    │  Fractional     │◄────┐
                    │  Coordinates    │     │
                    └────────┬────────┘     │
                             │              │
                  ┌──────────▼──────────┐   │
                  │  Model Processing   │   │
                  │  (Diffusion)        │   │
                  └──────────┬──────────┘   │
                             │              │
                    ┌────────▼────────┐     │
                    │  Fractional     │─────┘
                    │  Coordinates    │
                    └────────┬────────┘
                             │
                  ┌──────────▼──────────┐
                  │ fractional_to_       │
                  │ cartesian()          │
                  └──────────┬──────────┘
                             │
                    ┌────────▼────────┐
                    │  Cartesian      │
                    │  Coordinates    │
                    └─────────────────┘
```

---

## 周期境界条件の処理 (Periodic Boundary Condition Handling)

```
           Atom i                    Atom j
             (●)                      (●)
              │                        │
              │    Direct Distance     │
              │◄──────────────────────►│
              │         d_direct        │
              │                        │
    ┌─────────┴────────────────────────┴─────────┐
    │           Unit Cell (単位格子)              │
    │                                             │
    │    (●)                              (●)    │
    │   Image                          Image     │
    │                                             │
    └─────────────────────────────────────────────┘
              │                        │
              │◄──────────────────────►│
              │    Periodic Distance    │
              │         d_min           │
              │                        │
              ▼                        ▼
    
    Minimum Image Convention:
    d_min = min(d_direct, d_periodic)
    
    where d_periodic considers all 26 neighboring cells:
    • 6 face neighbors
    • 12 edge neighbors  
    • 8 corner neighbors
```

---

## 格子パラメータの表現 (Lattice Parameter Representations)

```
┌──────────────────────────────────────────────────────────────┐
│                  Lattice Representations                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Cell Parameters (格子パラメータ)                           │
│     ┌─────────────────────────────────────┐                  │
│     │  a, b, c  (lengths in Å)            │                  │
│     │  α, β, γ  (angles in degrees)       │                  │
│     └─────────────────────────────────────┘                  │
│                      │                                        │
│                      ▼ cell_params_to_vectors()              │
│                      │                                        │
│  2. Cell Vectors (格子ベクトル)                               │
│     ┌─────────────────────────────────────┐                  │
│     │  a_vec = [ax, ay, az]               │                  │
│     │  b_vec = [bx, by, bz]               │                  │
│     │  c_vec = [cx, cy, cz]               │                  │
│     │                                     │                  │
│     │  Matrix form:                       │                  │
│     │  ┌                    ┐             │                  │
│     │  │ ax  ay  az         │             │                  │
│     │  │ bx  by  bz         │             │                  │
│     │  │ cx  cy  cz         │             │                  │
│     │  └                    ┘             │                  │
│     └─────────────────────────────────────┘                  │
│                      │                                        │
│                      ▼ Used for coordinate transformation    │
│                      │                                        │
│  3. Normalized Representation (正規化表現)                     │
│     ┌─────────────────────────────────────┐                  │
│     │  log(a), log(b), log(c)             │                  │
│     │  sin(α), cos(α), ...                │                  │
│     └─────────────────────────────────────┘                  │
│                Used in neural network                         │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 実装の流れ (Implementation Flow)

```
Week 1-2: Data Foundation
    │
    ├─► crystal/data/periodic_utils.py ✓
    │   • Coordinate conversions
    │   • Minimum image distance
    │   • Cell parameter conversions
    │
    └─► crystal/data/crystal_loader.py ✓
        • CrystalDataset class
        • Batch collation
        
Week 3-5: Model Core
    │
    ├─► crystal/models/periodic_egnn.py ✓
    │   • Periodic message passing
    │   • E(3) equivariant updates
    │
    ├─► crystal/models/lattice_diffusion.py ✓
    │   • Lattice parameter evolution
    │   • Physical constraints
    │
    └─► crystal/models/crystal_dynamics.py ✓
        • Integration of position + lattice
        
Week 6-7: Diffusion Integration
    │
    └─► equivariant_diffusion/en_diffusion.py (modify)
        • Add crystal mode support
        • Joint loss function
        
Week 8-9: Conditioning & Evaluation
    │
    ├─► crystal/conditioning/*.py ✓
    │   • Space group embedding
    │   • Density conditioning
    │
    └─► crystal/evaluation/*.py ✓
        • Metrics computation
        • Validity checking
        
Week 10-11: Integration & Testing
    │
    ├─► main_crystal.py ✓
    │   • Training script
    │
    ├─► eval_crystal.py ✓
    │   • Evaluation script
    │
    └─► tests/*.py ✓
        • Unit tests
        • Integration tests
```

---

## まとめ (Summary)

このアーキテクチャは、以下の設計原則に基づいています:

1. **モジュール性**: 各コンポーネントは独立して開発・テスト可能
2. **既存システムとの互換性**: 分子生成機能を破壊しない
3. **段階的実装**: 各フェーズで動作確認可能
4. **拡張性**: 新しい条件付けや評価指標を容易に追加可能
5. **物理的正確性**: 周期境界条件と格子力学の正しい実装

詳細な実装については、**MOLECULAR_CRYSTAL_DESIGN.md** を参照してください。
