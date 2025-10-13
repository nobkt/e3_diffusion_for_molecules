# Phase 7 Quick Start Guide

## Getting Started with Crystal Diffusion Sampling

This guide provides quick examples to get you started with Phase 7 crystal sampling functionality.

---

## Prerequisites

Ensure you have completed Phases 1-6:
- Phase 1: Data loading with periodic boundaries
- Phase 2: Crystal dynamics models
- Phase 3: Conditioning modules
- Phase 4: Evaluation metrics
- Phase 5: CIF output
- Phase 6: Training loop

---

## Quick Example 1: Sample a Single Crystal

```python
import torch
from crystal.models import CrystalDynamics, CrystalDiffusion

# Create dynamics model
dynamics = CrystalDynamics(
    in_node_nf=5,          # 5 atom types (H, C, N, O, F)
    hidden_nf=128,         # Hidden dimension
    n_layers=6,            # Network depth
    learn_lattice=True     # Enable lattice learning
)

# Wrap in diffusion model
diffusion = CrystalDiffusion(
    dynamics=dynamics,
    in_node_nf=5,
    n_dims=3,
    timesteps=500,         # Diffusion steps
    learn_lattice=True
)

# Setup sampling
n_nodes = 50  # Number of atoms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_mask = torch.ones(1, n_nodes, 1, device=device)
edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0).view(-1, 1).to(device)
cell_params = torch.tensor([[15., 15., 15., 90., 90., 90.]], device=device)

# Sample
x, h, cell = diffusion.sample(
    n_samples=1,
    n_nodes=n_nodes,
    node_mask=node_mask,
    edge_mask=edge_mask,
    context=None,
    cell_params=cell_params
)

print(f"Generated crystal with {n_nodes} atoms")
print(f"Cell parameters: {cell}")
```

---

## Quick Example 2: Sample with Molecular Conditioning

```python
from crystal.models import MolecularEncoder, MolecularConditioning
from crystal.data import MoleculeDataset

# Load molecule dataset
mol_dataset = MoleculeDataset(db_path='molecules.db')
molecule = mol_dataset[0]

# Create molecular encoder
mol_encoder = MolecularEncoder(
    in_node_nf=5,
    hidden_nf=128,
    n_layers=4
)

# Encode molecule
mol_features, mol_geometry = mol_encoder(
    molecule['one_hot'],
    molecule['positions'],
    molecule['atom_mask'].unsqueeze(2)
)

# Create conditioning
mol_cond = MolecularConditioning(
    molecular_feature_dim=128,
    conditioning_dim=256,
    use_geometry=True
)
context = mol_cond(mol_features, mol_geometry)

# Expand context to all atoms in crystal
n_nodes = 50
context = context.unsqueeze(1).repeat(1, n_nodes, 1)

# Sample with conditioning
x, h, cell = diffusion.sample(
    n_samples=1,
    n_nodes=n_nodes,
    node_mask=node_mask,
    edge_mask=edge_mask,
    context=context,  # ← Use molecular context
    cell_params=cell_params
)
```

---

## Quick Example 3: Learn Crystal Size Distribution

```python
from crystal.data import CrystalDataset, MoleculeCrystalMapper
from equivariant_diffusion.crystal_distributions import CrystalNodesDistribution

# Load datasets
mol_dataset = MoleculeDataset(db_path='molecules.db')
mapper = MoleculeCrystalMapper()
mapper.build_from_databases('molecules.db', 'crystals.db')

crystal_dataset = CrystalDataset(
    db_path='crystals.db',
    molecule_dataset=mol_dataset,
    molecule_crystal_mapper=mapper
)

# Learn distribution from data
nodes_dist = CrystalNodesDistribution.from_dataset(
    crystal_dataset,
    min_nodes=20,
    max_nodes=150
)

# Print statistics
info = nodes_dist.log_info()
# Output:
# Crystal Nodes Distribution:
#   min_nodes: 20.00
#   max_nodes: 150.00
#   mean: 75.23
#   std: 28.45
#   mode: 68.00

# Sample crystal sizes
sizes = nodes_dist.sample(n_samples=10)
print(f"Sampled sizes: {sizes}")
# Output: tensor([72, 45, 89, 68, 101, 55, 82, 68, 93, 47])
```

---

## Quick Example 4: Train with Crystal Loss

```python
from qm9.losses import compute_loss_and_nll_crystal
from torch.optim import Adam

# Setup training
optimizer = Adam(diffusion.parameters(), lr=1e-4)

for epoch in range(n_epochs):
    for batch in train_loader:
        # Extract batch data
        x = batch['positions'].to(device)
        h = {
            'categorical': batch['one_hot'].to(device),
            'integer': batch['charges'].to(device)
        }
        node_mask = batch['atom_mask'].unsqueeze(2).to(device)
        edge_mask = batch['edge_mask'].to(device)
        cell_params = batch['cell_params'].to(device)
        pbc = batch['pbc'].to(device)
        
        # Prepare context (if using conditioning)
        context = prepare_context(batch)  # Your context function
        
        # Compute loss
        nll, reg_term, _ = compute_loss_and_nll_crystal(
            args,
            diffusion,
            nodes_dist,
            x, h, node_mask, edge_mask, context,
            cell_params=cell_params,
            pbc=pbc
        )
        
        # Total loss with regularization
        loss = nll + args.ode_regularization * reg_term
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

---

## Quick Example 5: Generate and Save to CIF

```python
from crystal.utils import CIFWriter
from crystal.data.periodic_utils import cell_params_to_vectors, cell_vectors_to_params

# Sample crystal
x, h, cell_params = diffusion.sample(
    n_samples=1,
    n_nodes=50,
    node_mask=node_mask,
    edge_mask=edge_mask,
    context=context,
    cell_params=torch.tensor([[15., 15., 15., 90., 90., 90.]], device=device)
)

# Convert to numpy
positions = x.squeeze(0).cpu().numpy()  # [n_atoms, 3]
atom_types = torch.argmax(h['categorical'], dim=-1).squeeze(0).cpu().numpy()  # [n_atoms]
cell_params_np = cell_params.squeeze(0).cpu().numpy()  # [6]

# Filter by mask
n_atoms = int(node_mask.sum().item())
positions = positions[:n_atoms]
atom_types = atom_types[:n_atoms]

# Write to CIF
dataset_info = {
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
}

cif_writer = CIFWriter(dataset_info)
cif_writer.write_cif(
    positions=positions,
    atom_types=atom_types,
    cell_params=cell_params_np,
    filename='generated_crystal.cif',
    compound_name='Generated Crystal',
    space_group='P1'  # or detected space group
)

print("Crystal saved to generated_crystal.cif")
```

---

## Quick Example 6: Sample Multiple Crystals of Different Sizes

```python
from crystal.sampling import sample_different_crystal_sizes

# Sample with varying sizes
crystals = sample_different_crystal_sizes(
    model=diffusion,
    nodes_dist=nodes_dist,
    args=args,
    device=device,
    dataset_info=dataset_info,
    mol_encoder=mol_encoder,
    conditioning_modules={'molecular': mol_cond},
    n_samples=20,
    batch_size=4
)

print(f"Generated {len(crystals['x'])} crystals")
print(f"Sizes: {[int(m.sum().item()) for m in crystals['node_mask']]}")
```

---

## Quick Example 7: Visualize Sampling Chain

```python
# Sample with trajectory
chain, cell_chain = diffusion.sample_chain(
    n_samples=1,
    n_nodes=30,
    node_mask=torch.ones(1, 30, 1, device=device),
    edge_mask=edge_mask,
    context=None,
    cell_params=torch.tensor([[15., 15., 15., 90., 90., 90.]], device=device),
    keep_frames=50  # Save 50 intermediate frames
)

print(f"Chain shape: {chain.shape}")  # [50, 30, 8] (50 frames, 30 atoms, 3 pos + 5 features)
print(f"Cell chain: {cell_chain.shape}")  # [50, 1, 6]

# Extract frames for visualization
for i in range(0, 50, 10):
    frame = chain[i * 1 + 0]  # First sample, frame i
    positions = frame[:, :3].cpu().numpy()
    # Visualize positions...
```

---

## Running Tests

```bash
# Test CrystalDiffusion
PYTHONPATH=.:$PYTHONPATH python -c "
from tests.test_phase7_diffusion import TestCrystalDiffusion
test = TestCrystalDiffusion()
test.test_init()
test.test_sample_cell_noise()
print('✓ CrystalDiffusion tests passed')
"

# Test distributions
PYTHONPATH=.:$PYTHONPATH python -c "
from tests.test_phase7_diffusion import TestCrystalNodesDistribution
test = TestCrystalNodesDistribution()
test.test_from_dataset()
test.test_sample()
print('✓ Distribution tests passed')
"

# Test loss computation
PYTHONPATH=.:$PYTHONPATH python -c "
from tests.test_phase7_diffusion import TestCrystalLossComputation
test = TestCrystalLossComputation()
test.test_compute_loss_basic()
print('✓ Loss computation tests passed')
"

# Run all tests
PYTHONPATH=.:$PYTHONPATH python tests/test_phase7_diffusion.py
```

---

## Troubleshooting

### Issue: "CrystalDiffusion object has no attribute 'lattice_diffusion'"

**Solution**: Your dynamics model must support lattice learning.

```python
# Wrong:
dynamics = PeriodicEGNN(...)  # Doesn't have lattice_diffusion

# Correct:
dynamics = CrystalDynamics(..., learn_lattice=True)
```

### Issue: "Mean is not zero" assertion error

**Solution**: This is expected for some crystal systems due to PBC. The assertion has been relaxed in sampling.

### Issue: Sampling takes too long

**Solution**: Reduce the number of timesteps or use a simpler model for testing.

```python
# Fast sampling for testing
diffusion = CrystalDiffusion(
    dynamics=dynamics,
    timesteps=10,  # Much faster
    ...
)
```

### Issue: Cell parameters out of range

**Solution**: Ensure cell parameter regularization is enabled in loss.

```python
# Check regularization
nll, reg_term, _ = compute_loss_and_nll_crystal(
    ...,
    cell_params=cell_params  # Must be provided
)

# Check that reg_term > 0 for bad cells
print(f"Regularization: {reg_term.item()}")
```

---

## Common Parameters

### CrystalDiffusion
- `timesteps`: 500 (production), 10-50 (testing)
- `learn_lattice`: True (enable cell learning)
- `noise_schedule`: 'polynomial_2' (recommended)

### CrystalDynamics
- `hidden_nf`: 128-256 (model capacity)
- `n_layers`: 6-9 (network depth)
- `attention`: True (better performance)

### Node Distribution
- `min_nodes`: 10-20 (minimum crystal size)
- `max_nodes`: 100-200 (maximum crystal size)

### Loss Computation
- `ode_regularization`: 0.01-0.1 (regularization weight)
- Cell regularization: 0.01 (built-in weight)

---

## Next Steps

1. **Read Full Documentation**: `PHASE7_CRYSTAL_SAMPLING_SUMMARY.md`
2. **Review Tests**: `tests/test_phase7_diffusion.py`
3. **Check Integration**: See how Phase 7 works with Phases 1-6
4. **Experiment**: Try different hyperparameters and model configurations

---

**For more details, see**:
- `PHASE7_CRYSTAL_SAMPLING_SUMMARY.md` - Complete technical documentation
- `tests/test_phase7_diffusion.py` - Comprehensive test examples
- `crystal/models/crystal_diffusion.py` - Implementation details
- `PROJECT_COMPLETE_SUMMARY.md` - Overall system overview
