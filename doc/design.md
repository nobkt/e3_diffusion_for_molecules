# E3 Equivariant Diffusion for Molecules: Design Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [Module Specifications](#module-specifications)
4. [Data Pipeline](#data-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [Extension Points](#extension-points)
8. [Performance Considerations](#performance-considerations)

---

## System Overview

### Design Principles

The system follows strict design principles established across PR#124-130:

1. **No Fallback Heuristics (ごまかしのためのfallbackは絶対にしない)**
   - All operations explicitly handle edge cases
   - No silent corrections or default values
   - Clear error messages for invalid inputs

2. **Modularity**
   - Independent, reusable components
   - Clear interfaces and contracts
   - Minimal coupling between modules

3. **Theoretical Soundness**
   - E(3) equivariance maintained throughout
   - Proper handling of periodic boundary conditions
   - Physical constraints enforced rigorously

4. **Extensibility**
   - New conditioning types easily added
   - Support for different datasets
   - Pluggable model components

### System Capabilities

**Molecule Generation**:
- QM9 dataset (small molecules)
- GEOM-Drugs dataset (drug-like molecules)
- ASE database format (custom datasets)
- Conditional generation on properties
- Exact conditional generation with molecular descriptors

**Crystal Generation**:
- Homocrystal generation from single molecules
- Space group conditioning
- Density conditioning
- Unit cell parameter learning
- Periodic boundary condition support

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│  (main_qm9.py, main_geom_drugs.py, main_crystal.py)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Training/Inference Loop                    │
│            (train_test.py, train_test_crystal.py)           │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                      ▼
┌────────────────────────┐          ┌────────────────────────┐
│    Data Loading        │          │   Model Components     │
│  - Dataset classes     │          │  - EGNN                │
│  - DataLoaders         │          │  - Diffusion           │
│  - Preprocessing       │          │  - Conditioning        │
└────────────────────────┘          └────────────────────────┘
           │                                      │
           └──────────────────┬──────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation & Output                        │
│  - Metrics computation                                       │
│  - Structure validation                                      │
│  - CIF export                                               │
│  - Visualization                                            │
└─────────────────────────────────────────────────────────────┘
```

### Module Organization

```
e3_diffusion_for_molecules/
├── egnn/                           # Core EGNN implementation
│   ├── egnn_new.py                 # EGNN with sinusoidal embeddings
│   └── dynamics.py                 # Dynamics wrapper
│
├── equivariant_diffusion/          # Diffusion framework
│   ├── en_diffusion.py             # Main diffusion model
│   ├── utils.py                    # Diffusion utilities
│   └── crystal_distributions.py   # Crystal size distributions
│
├── qm9/                            # QM9/GEOM dataset support
│   ├── dataset.py                  # Dataset classes
│   ├── models.py                   # Model builders
│   ├── losses.py                   # Loss functions
│   ├── sampling.py                 # Sampling utilities
│   └── utils.py                    # General utilities
│
├── crystal/                        # Crystal extension (PR#124-130)
│   ├── data/                       # Crystal data handling
│   │   ├── molecule_loader.py     # Single molecule dataset
│   │   ├── crystal_loader.py      # Crystal dataset
│   │   ├── molecule_crystal_mapper.py  # Mapping system
│   │   └── periodic_utils.py      # PBC utilities
│   │
│   ├── models/                     # Crystal-specific models
│   │   ├── molecular_encoder.py   # Molecule feature extraction
│   │   ├── periodic_egnn.py       # EGNN with PBC
│   │   ├── lattice_diffusion.py   # Cell parameter learning
│   │   ├── crystal_dynamics.py    # Unified crystal model
│   │   └── crystal_diffusion.py   # Crystal diffusion sampling
│   │
│   ├── conditioning/               # Conditioning modules
│   │   ├── molecular_conditioning.py  # PRIMARY conditioning
│   │   ├── space_group_embedding.py   # Space group conditioning
│   │   └── density_conditioning.py    # Density conditioning
│   │
│   ├── evaluation/                 # Evaluation tools
│   │   ├── crystal_metrics.py      # Metric computation
│   │   ├── structure_validator.py  # Validation
│   │   └── symmetry_analyzer.py    # Symmetry analysis
│   │
│   ├── utils/                      # Utility functions
│   │   ├── cif_writer.py          # CIF export
│   │   ├── cell_operations.py     # Cell manipulation
│   │   └── neighbor_list.py       # Neighbor search
│   │
│   └── sampling.py                 # Crystal sampling
│
├── configs/                        # Configuration files
│   └── datasets_config.py         # Dataset configurations
│
└── tests/                          # Comprehensive test suite
    ├── test_*.py                   # Unit tests
    └── test_*_integration.py       # Integration tests
```

---

## Module Specifications

### 1. EGNN Module (`egnn/`)

#### Purpose
Implements E(3) equivariant graph neural networks for molecular representation.

#### Key Classes

**EGNN** (`egnn/egnn_new.py`):
```python
class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf: int,      # Input node features
        hidden_nf: int,        # Hidden dimension
        out_node_nf: int,      # Output node features
        in_edge_nf: int = 0,   # Edge features
        n_layers: int = 4,     # Number of layers
        attention: bool = False,  # Use attention
        normalize: bool = False,  # Normalize coordinates
        tanh: bool = False,    # Use tanh activation
        coords_agg: str = 'mean',  # Coordinate aggregation
        norm_constant: float = 1,  # Normalization constant
    )
    
    def forward(
        self,
        h: Tensor,             # [B*N, in_node_nf]
        x: Tensor,             # [B*N, 3]
        edge_index: Tensor,    # [2, E]
        node_mask: Tensor = None,  # [B*N, 1]
        edge_mask: Tensor = None,  # [E]
        edge_attr: Tensor = None,  # [E, in_edge_nf]
    ) -> Tuple[Tensor, Tensor]:  # (h_out, x_out)
```

**Design Features**:
- **Sinusoidal Distance Embedding**: Multi-scale distance representation
- **Attention Mechanism**: Optional message weighting
- **Coordinate Aggregation**: Mean/sum aggregation with normalization
- **Masking Support**: Proper handling of variable-size molecules

#### Invariants Maintained
- Sum of coordinate updates is zero (center-of-mass)
- E(3) equivariance for all operations
- Message passing preserves graph structure

---

### 2. Diffusion Module (`equivariant_diffusion/`)

#### Purpose
Implements variational diffusion for molecule generation.

#### Key Classes

**EnVariationalDiffusion** (`equivariant_diffusion/en_diffusion.py`):
```python
class EnVariationalDiffusion(nn.Module):
    def __init__(
        self,
        dynamics: nn.Module,          # Denoising network (EGNN-based)
        in_node_nf: int,              # Input node features
        n_dims: int,                  # Coordinate dimensions (3)
        timesteps: int = 1000,        # Number of diffusion steps
        noise_schedule: str = 'polynomial_2',
        noise_precision: float = 1e-5,
        loss_type: str = 'l2',        # 'l2' or 'vlb'
        norm_values: Tuple = (1., 1., 1.),  # Normalization factors
        norm_biases: Tuple = (0., 0., 0.),
    )
    
    def forward(
        self,
        x: Tensor,                    # Positions
        h: Dict[str, Tensor],         # Features (categorical, integer)
        node_mask: Tensor,
        edge_mask: Tensor,
        context: Tensor = None,
    ) -> Tuple[Tensor, Dict]:  # (loss, info_dict)
    
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_nodes: int,
        node_mask: Tensor,
        edge_mask: Tensor,
        context: Tensor = None,
        fix_noise: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:  # (x, h)
```

**Noise Schedule**:
```python
def gamma(self, t: Tensor) -> Tensor:
    """
    Returns log SNR: γ(t) = -log(σ²/α²)
    
    polynomial_2: γ(t) = -log[(t/T)² + precision]
    cosine: γ(t) = -2*log[cos(πt/2T)]
    """
```

**Design Features**:
- **Flexible Noise Schedules**: Polynomial, cosine, learned
- **Loss Types**: L2 (simple) or VLB (rigorous)
- **Normalization**: Per-feature normalization with learned biases
- **Context Support**: Conditional generation via context vectors

#### Sampling Algorithm
```
1. Initialize from noise: z_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. Predict noise: ε_θ(z_t, t, context)
   b. Compute mean: μ_θ(z_t, t) using ε_θ
   c. Sample: z_{t-1} ~ N(μ_θ, σ²_t I)
3. Return: (x_0, h_0)
```

---

### 3. Crystal Module (`crystal/`)

Implements crystal-specific extensions across 7 phases (PR#124-130).

#### Phase 2: Model Core (PR#124-125)

**MolecularEncoder** (`crystal/models/molecular_encoder.py`):
```python
class MolecularEncoder(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        n_layers: int = 4,
        attention: bool = True,
    )
    
    def forward(
        self,
        h: Tensor,              # [N, in_node_nf]
        x: Tensor,              # [N, 3]
        edge_index: Tensor,     # [2, E]
        node_mask: Tensor = None,
    ) -> MolecularFeatures:
        """
        Returns:
            node_features: [N, hidden_nf]
            global_features: [hidden_nf]
            mol_size: float
            mol_volume: float
            principal_axes: [3, 3]
        """
```

**PeriodicEGNN** (`crystal/models/periodic_egnn.py`):
```python
class PeriodicEGNN(nn.Module):
    def forward(
        self,
        h: Tensor,              # Node features
        x: Tensor,              # Positions (fractional or Cartesian)
        cell: Tensor,           # [6] or [3, 3] cell parameters
        pbc: Tensor,            # [3] Boolean PBC flags
        node_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:  # (h_out, x_out)
```

**LatticeEncoding** (`crystal/models/lattice_diffusion.py`):
```python
class LatticeEncoding(nn.Module):
    """Encodes cell parameters for neural networks."""
    
    def forward(
        self,
        cell_params: Tensor,    # [B, 6] (a,b,c,α,β,γ)
    ) -> Tensor:                # [B, encoding_dim]
```

**CrystalDynamics** (`crystal/models/crystal_dynamics.py`):
```python
class CrystalDynamics(nn.Module):
    """Unified model for position and lattice diffusion."""
    
    def forward(
        self,
        t: Tensor,              # Timestep
        xh: Tuple[Tensor, Dict],  # (positions, features)
        cell: Tensor,           # Cell parameters
        pbc: Tensor,            # PBC flags
        node_mask: Tensor,
        context: Tensor = None,
    ) -> Tuple[Tensor, Dict, Tensor]:  # (vel_x, vel_h, vel_cell)
```

#### Phase 3: Conditioning (PR#126)

**MolecularConditioning** (`crystal/conditioning/molecular_conditioning.py`):
```python
class MolecularConditioning(nn.Module):
    """PRIMARY conditioning method."""
    
    def __init__(
        self,
        molecular_feature_dim: int,
        conditioning_dim: int,
        use_geometry: bool = True,  # Include size, volume, axes
    )
    
    def forward(
        self,
        molecular_features: MolecularFeatures,
    ) -> Tensor:  # [conditioning_dim]
```

**SpaceGroupEmbedding** (`crystal/conditioning/space_group_embedding.py`):
```python
class SpaceGroupEmbedding(nn.Module):
    """Embedding for 230 crystallographic space groups."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_space_groups: int = 230,
    )
    
    def forward(
        self,
        space_group: Tensor,    # [B] integers in [1, 230]
    ) -> Tensor:                # [B, embedding_dim]
```

**DensityConditioning** (`crystal/conditioning/density_conditioning.py`):
```python
class DensityConditioning(nn.Module):
    """Conditioning on crystal density (g/cm³)."""
    
    def __init__(
        self,
        conditioning_dim: int,
        min_density: float = 0.5,
        max_density: float = 5.0,
    )
    
    def forward(
        self,
        density: Tensor,        # [B] density values
    ) -> Tensor:                # [B, conditioning_dim]
```

#### Phase 4: Evaluation (PR#127)

**CrystalMetrics** (`crystal/evaluation/crystal_metrics.py`):
```python
class CrystalMetrics:
    """Comprehensive crystal evaluation metrics."""
    
    def compute_all_metrics(
        self,
        generated: List[Dict],
        reference: List[Dict] = None,
    ) -> Dict[str, float]:
        """
        Returns metrics:
            - validity_ratio
            - volume_mean, volume_std
            - density_mean, density_std
            - volume_wasserstein (if reference provided)
            - density_wasserstein
            - lattice_parameter_errors
        """
```

**StructureValidator** (`crystal/evaluation/structure_validator.py`):
```python
class StructureValidator:
    """Validates crystal structures."""
    
    def validate_structure(
        self,
        structure: Dict,
    ) -> Tuple[bool, List[str]]:
        """
        Returns:
            is_valid: bool
            errors: List of error messages
        
        Checks:
            - Data format
            - Cell parameter bounds
            - Coordinate consistency
            - Minimum distances with PBC
        """
```

**SymmetryAnalyzer** (`crystal/evaluation/symmetry_analyzer.py`):
```python
class SymmetryAnalyzer:
    """Analyzes crystal symmetry."""
    
    def detect_space_group(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        atom_types: np.ndarray,
        symprec: float = 1e-5,
    ) -> Optional[int]:
        """Detect space group (requires spglib)."""
    
    def compute_structure_fingerprint(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        r_max: float = 10.0,
        n_bins: int = 100,
    ) -> np.ndarray:
        """RDF-based fingerprint."""
```

#### Phase 5: Visualization & Output (PR#128)

**CIFWriter** (`crystal/utils/cif_writer.py`):
```python
class CIFWriter:
    """IUCr-compliant CIF file writer."""
    
    def __init__(self, dataset_info: Dict)
    
    def write_cif(
        self,
        crystal: Dict,
        filename: str,
        compound_name: str = 'Generated',
        space_group: int = 1,
    ) -> None:
        """Write crystal to CIF file."""
    
    def write_batch(
        self,
        crystals: List[Dict],
        output_dir: str,
        prefix: str = 'crystal',
    ) -> List[str]:
        """Write multiple CIF files."""
```

**CellOperations** (`crystal/utils/cell_operations.py`):
```python
class CellOperations:
    """Cell parameter manipulation utilities."""
    
    @staticmethod
    def params_to_matrix(
        a: float, b: float, c: float,
        alpha: float, beta: float, gamma: float,
    ) -> np.ndarray:  # [3, 3]
        """Convert parameters to matrix."""
    
    @staticmethod
    def matrix_to_params(
        matrix: np.ndarray,  # [3, 3]
    ) -> Tuple[float, float, float, float, float, float]:
        """Convert matrix to parameters."""
    
    @staticmethod
    def compute_volume(
        cell_params: Union[np.ndarray, Tuple] = None,
        cell_matrix: np.ndarray = None,
    ) -> float:
        """Compute cell volume."""
```

**NeighborList** (`crystal/utils/neighbor_list.py`):
```python
class NeighborList:
    """Periodic neighbor search."""
    
    def __init__(self, cutoff: float)
    
    def build(
        self,
        positions: Tensor,      # [N, 3] Cartesian
        cell_vectors: Tensor,   # [3, 3]
        pbc: Tensor = None,     # [3] Boolean
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            edge_index: [2, E]
            edge_shift: [E, 3] Lattice shifts
        """
    
    def compute_distances(
        self,
        positions: Tensor,
        cell_vectors: Tensor,
        edge_index: Tensor,
        edge_shift: Tensor,
    ) -> Tensor:  # [E]
        """Compute distances with PBC."""
```

#### Phase 6: Training Loop (PR#129)

**Training Functions** (`train_test_crystal.py`):
```python
def prepare_crystal_context(
    args: argparse.Namespace,
    data: Dict,
    mol_encoder: nn.Module,
    mol_conditioning: nn.Module,
    sg_embedding: nn.Module = None,
    density_conditioning: nn.Module = None,
) -> Tensor:
    """Prepare multi-modal conditioning."""

def train_epoch_crystal(
    args: argparse.Namespace,
    loader: DataLoader,
    epoch: int,
    model: nn.Module,
    model_dp: nn.Module,
    model_ema: nn.Module,
    ema: ExponentialMovingAverage,
    device: torch.device,
    dtype: torch.dtype,
    optim: torch.optim.Optimizer,
    nodes_dist: nn.Module,
    gradnorm_queue: Queue,
    dataset_info: Dict,
    mol_encoder: nn.Module,
    conditioning_modules: Dict,
) -> None:
    """Crystal-specific training epoch."""

def test_crystal(
    args: argparse.Namespace,
    loader: DataLoader,
    epoch: int,
    model: nn.Module,
    nodes_dist: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    dataset_info: Dict,
    mol_encoder: nn.Module,
    conditioning_modules: Dict,
    partition: str = 'Test',
) -> float:
    """Crystal-specific validation."""

def analyze_and_save_crystal(
    epoch: int,
    model_sample: nn.Module,
    nodes_dist: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dataset_info: Dict,
    n_samples: int,
    conditioning_modules: Dict = None,
) -> Tuple[Dict, Dict]:
    """Structure analysis and metrics."""
```

#### Phase 7: Crystal Diffusion (PR#130)

**CrystalDiffusion** (`crystal/models/crystal_diffusion.py`):
```python
class CrystalDiffusion(EnVariationalDiffusion):
    """Crystal-specific diffusion sampling."""
    
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_nodes: int,
        node_mask: Tensor,
        edge_mask: Tensor,
        context: Tensor = None,
        cell_params: Tensor = None,
        pbc: Tensor = None,
        fix_noise: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """
        Returns:
            x: [B, N, 3] Positions
            h: Dict with features
            cell_params: [B, 6] Cell parameters
        """
    
    @torch.no_grad()
    def sample_chain(
        self,
        n_samples: int,
        n_nodes: int,
        node_mask: Tensor,
        edge_mask: Tensor,
        context: Tensor = None,
        cell_params: Tensor = None,
        pbc: Tensor = None,
        keep_frames: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample with intermediate states."""
```

**CrystalNodesDistribution** (`equivariant_diffusion/crystal_distributions.py`):
```python
class CrystalNodesDistribution(nn.Module):
    """Distribution over crystal sizes."""
    
    @classmethod
    def from_dataset(
        cls,
        dataset: CrystalDataset,
        min_nodes: int = None,
        max_nodes: int = None,
    ) -> 'CrystalNodesDistribution':
        """Learn distribution from data."""
    
    def sample(self, n_samples: int) -> Tensor:
        """Sample number of atoms."""
    
    def log_prob(self, x: Tensor) -> Tensor:
        """Log probability."""
```

---

## Data Pipeline

### Data Flow

```
Raw Data
   │
   ├─► QM9/GEOM Dataset
   │   └─► MoleculeDataset
   │       └─► (x, h, charges, properties)
   │
   ├─► ASE Database
   │   └─► ASEDataset
   │       └─► (x, h, charges, descriptors)
   │
   └─► Crystal Database
       ├─► MoleculeDataset (single molecules)
       ├─► CrystalDataset (crystal structures)
       └─► MoleculeCrystalMapper (relationships)
           └─► (molecule_data, crystal_data, mapping)
```

### Dataset Classes

**QM9Dataset** (`qm9/dataset.py`):
```python
class QM9Dataset(torch.utils.data.Dataset):
    def __getitem__(self, idx) -> Dict:
        return {
            'positions': Tensor,      # [N, 3]
            'one_hot': Tensor,        # [N, num_atom_types]
            'charges': Tensor,        # [N, 1]
            'atom_mask': Tensor,      # [N, 1]
            'edge_mask': Tensor,      # [E]
            'n_nodes': int,
            **properties,             # e.g., 'alpha', 'homo', 'lumo'
        }
```

**ASEDataset** (via `qm9/dataset.py` with `dataset='ase_db'`):
```python
# Same structure as QM9Dataset, but loaded from ASE database
# Additional molecular descriptors available:
#   - molecular_weight
#   - pi_conjugation_ratio
#   - atom_types_encoding
#   - functional_groups_encoding
```

**MoleculeDataset** (`crystal/data/molecule_loader.py`):
```python
class MoleculeDataset(torch.utils.data.Dataset):
    """Single molecule dataset for crystal conditioning."""
    
    def __getitem__(self, idx) -> Dict:
        return {
            'molecule_id': str,
            'positions': Tensor,      # [N, 3]
            'one_hot': Tensor,        # [N, num_atom_types]
            'atom_mask': Tensor,      # [N, 1]
            'edge_index': Tensor,     # [2, E]
            'n_nodes': int,
        }
```

**CrystalDataset** (`crystal/data/crystal_loader.py`):
```python
class CrystalDataset(torch.utils.data.Dataset):
    """Crystal structure dataset."""
    
    def __getitem__(self, idx) -> Dict:
        return {
            'crystal_id': str,
            'molecule_id': str,       # For homocrystal
            'positions': Tensor,      # [N, 3] Cartesian
            'one_hot': Tensor,        # [N, num_atom_types]
            'atom_mask': Tensor,      # [N, 1]
            'cell_params': Tensor,    # [6] (a,b,c,α,β,γ)
            'pbc': Tensor,            # [3] Boolean
            'space_group': int,       # 1-230
            'density': float,         # g/cm³
            'n_nodes': int,
        }
```

### Data Preprocessing

**Normalization**:
```python
# Positions: center-of-mass to origin
x_centered = x - x.mean(dim=0, keepdim=True)

# Properties: standardization
property_norm = (property - mean) / mad

# Cell parameters: log-space for lengths
a_norm = log(a)
alpha_norm = alpha  # radians
```

**Augmentation** (optional):
```python
# Random rotation (preserves E(3) invariance)
R = random_rotation_matrix()
x_aug = x @ R.T

# Random noise (for robustness)
x_aug = x + noise * torch.randn_like(x)
```

---

## Training Pipeline

### Training Loop

```python
def main():
    # 1. Setup
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load data
    dataloaders = get_dataloaders(args)
    dataset_info = get_dataset_info(args.dataset)
    
    # 3. Create model
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info)
    optimizer = get_optimizer(args, model)
    
    # 4. Optional: Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 5. Training loop
    best_val_loss = float('inf')
    for epoch in range(args.start_epoch, args.n_epochs):
        # Train
        train_epoch(args, dataloaders['train'], epoch, model, 
                   optimizer, device, ...)
        
        # Validate
        if epoch % args.test_epochs == 0:
            val_loss = test(args, dataloaders['valid'], epoch, 
                          model, device, ...)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, 'best')
            
            # Analyze samples
            analyze_and_save(epoch, model, args, device, ...)
    
    # 6. Final evaluation
    test_loss = test(args, dataloaders['test'], epoch, model, ...)
    print(f'Final test loss: {test_loss}')
```

### Loss Computation

**For Molecules**:
```python
def compute_loss(model, data, nodes_dist):
    x, h = data['positions'], data['features']
    node_mask = data['atom_mask']
    edge_mask = data['edge_mask']
    context = data.get('context', None)
    
    # Forward pass
    nll, info = model(x, h, node_mask, edge_mask, context)
    
    # Aggregate losses
    loss = nll.mean()
    
    return loss, info
```

**For Crystals** (additional cell loss):
```python
def compute_loss_crystal(model, data, nodes_dist):
    x, h = data['positions'], data['features']
    cell = data['cell_params']
    pbc = data['pbc']
    node_mask = data['atom_mask']
    context = data['context']
    
    # Forward pass
    nll, nll_cell, info = model(x, h, cell, pbc, node_mask, context)
    
    # Combined loss
    loss = nll.mean() + args.cell_weight * nll_cell.mean()
    
    return loss, info
```

### Gradient Updates

```python
def training_step(model, optimizer, loss, args):
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    if args.clip_grad:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.clip_grad_norm
        )
    
    # Update
    optimizer.step()
    
    return grad_norm
```

### EMA Updates

```python
class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                      for name, param in model.named_parameters()}
    
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] +
                (1 - self.decay) * param.data
            )
```

---

## Inference Pipeline

### Sampling

```python
@torch.no_grad()
def sample_molecules(model, n_samples, n_nodes, device, context=None):
    # 1. Create masks
    node_mask = torch.ones(n_samples, n_nodes, 1, device=device)
    edge_mask = torch.ones(n_samples, n_nodes, n_nodes, device=device)
    
    # 2. Sample
    x, h = model.sample(
        n_samples=n_samples,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        context=context,
    )
    
    # 3. Post-process
    molecules = []
    for i in range(n_samples):
        mol = {
            'positions': x[i],
            'atom_types': h['categorical'][i].argmax(dim=-1),
            'charges': h['integer'][i] if 'integer' in h else None,
        }
        molecules.append(mol)
    
    return molecules
```

### Evaluation

```python
def evaluate_samples(molecules, reference_set, dataset_info):
    from qm9.analyze import analyze_stability_for_molecules
    from rdkit import Chem
    
    # 1. Stability
    stability_dict = analyze_stability_for_molecules(
        molecules, dataset_info
    )
    
    # 2. Validity (RDKit)
    valid_mols = []
    for mol in molecules:
        rdkit_mol = build_rdkit_mol(mol, dataset_info)
        if rdkit_mol is not None:
            valid_mols.append(mol)
    
    validity = len(valid_mols) / len(molecules)
    
    # 3. Uniqueness
    smiles_set = set()
    for mol in valid_mols:
        rdkit_mol = build_rdkit_mol(mol, dataset_info)
        smiles = Chem.MolToSmiles(rdkit_mol)
        smiles_set.add(smiles)
    
    uniqueness = len(smiles_set) / len(valid_mols)
    
    # 4. Novelty
    reference_smiles = get_reference_smiles(reference_set)
    novel_smiles = smiles_set - reference_smiles
    novelty = len(novel_smiles) / len(smiles_set)
    
    return {
        'stability': stability_dict,
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
    }
```

---

## Extension Points

### Adding New Conditioning Types

1. **Create Conditioning Module**:
```python
# crystal/conditioning/my_conditioning.py
class MyConditioning(nn.Module):
    def __init__(self, conditioning_dim: int, ...):
        super().__init__()
        self.mlp = nn.Sequential(...)
    
    def forward(self, my_property: Tensor) -> Tensor:
        # Validate input (no fallback!)
        if my_property is None:
            raise ValueError("my_property cannot be None")
        
        # Process
        return self.mlp(my_property)
```

2. **Integrate in Training**:
```python
# train_test_crystal.py
def prepare_crystal_context(..., my_conditioning=None):
    contexts = []
    
    # ... existing conditioning ...
    
    if args.condition_on_my_property:
        if my_conditioning is None:
            raise ValueError("my_conditioning required but not provided")
        my_context = my_conditioning(data['my_property'])
        contexts.append(my_context)
    
    return torch.cat(contexts, dim=-1) if contexts else None
```

3. **Add to Main Script**:
```python
# main_crystal.py
parser.add_argument('--condition_on_my_property', type=bool, default=False)

if args.condition_on_my_property:
    my_conditioning = MyConditioning(...).to(device)
else:
    my_conditioning = None
```

### Adding New Datasets

1. **Create Dataset Class**:
```python
# qm9/my_dataset.py
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, ...):
        self.data = self.load_data(data_path)
    
    def __getitem__(self, idx) -> Dict:
        return {
            'positions': ...,
            'one_hot': ...,
            'atom_mask': ...,
            # ... required fields ...
        }
```

2. **Register in Config**:
```python
# configs/datasets_config.py
def get_dataset_info(dataset_name):
    if dataset_name == 'my_dataset':
        return {
            'atom_encoder': {...},
            'max_n_nodes': ...,
            'n_nodes': {...},
            ...
        }
```

3. **Add Dataloader**:
```python
# qm9/dataset.py
def retrieve_dataloaders(args):
    if args.dataset == 'my_dataset':
        dataset = MyDataset(args.data_path)
        # ... split and create loaders ...
```

---

## Performance Considerations

### Memory Optimization

1. **Gradient Checkpointing**:
```python
# For large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x, h, ...):
    return checkpoint(self._forward, x, h, ...)
```

2. **Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model(x, h, ...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

3. **Batch Size Scaling**:
```python
# Accumulate gradients for effective larger batch
effective_batch_size = 256
accumulation_steps = effective_batch_size // args.batch_size

for i, data in enumerate(loader):
    loss = model(data) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Computational Efficiency

1. **Sparse Neighbor Lists**:
```python
# Instead of fully connected
def build_sparse_edges(x, cutoff=10.0):
    dist = torch.cdist(x, x)
    edge_index = (dist < cutoff).nonzero(as_tuple=False).T
    return edge_index
```

2. **Caching**:
```python
# Cache expensive computations
@lru_cache(maxsize=1000)
def compute_cell_matrix(a, b, c, alpha, beta, gamma):
    # ... cell matrix computation ...
```

3. **Vectorization**:
```python
# Vectorize batch operations
def batch_minimum_image(positions, cell, pbc):
    # Process all atoms at once
    frac = torch.einsum('bij,bj->bi', positions, cell.inverse())
    frac = frac - torch.floor(frac + 0.5)
    return torch.einsum('bij,bj->bi', frac, cell)
```

---

## Design Validation

### Checklist for New Features

- [ ] **No Fallback Heuristics**: All error cases explicitly handled
- [ ] **E(3) Equivariance**: Verified for geometric operations
- [ ] **Physical Validity**: Constraints enforced
- [ ] **Unit Tests**: All public methods tested
- [ ] **Integration Tests**: End-to-end workflows verified
- [ ] **Documentation**: Docstrings and examples provided
- [ ] **Performance**: Profiled and optimized if needed

### Code Review Guidelines

1. **Check Error Handling**:
   - No silent corrections
   - Clear error messages
   - Proper exception types

2. **Verify Equivariance**:
   - Rotation/translation invariant where required
   - Coordinate operations use proper conventions

3. **Validate Physics**:
   - Constraints enforced
   - Units consistent
   - Boundary conditions correct

4. **Test Coverage**:
   - Happy path tested
   - Edge cases covered
   - Error paths verified

---

**Version**: 1.0  
**Date**: 2025-10-13  
**Status**: Complete design documentation for PR#124-130
