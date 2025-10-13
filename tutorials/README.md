# E3 Diffusion for Molecules - Tutorials

This directory contains Jupyter notebook tutorials for the E3 Equivariant Diffusion Model for molecule and crystal generation.

## Tutorial Overview

| Tutorial | Topic | Time | Prerequisites |
|----------|-------|------|---------------|
| 01 | Basic Molecule Generation | 15-20 min | None |
| 02 | Conditional Generation | 20-30 min | Tutorial 01 |
| 03 | Crystal Generation | 30 min | Tutorial 01 |
| 04 | Molecular Descriptors & ASE | 20 min | Tutorial 01 |
| 05 | Evaluation & Analysis | 25 min | Tutorial 01 |
| 06 | Advanced Crystal Conditioning | 30 min | Tutorial 03 |

## Quick Start

### Setup Environment

```bash
# Clone repository
git clone https://github.com/nobkt/e3_diffusion_for_molecules.git
cd e3_diffusion_for_molecules

# Install dependencies
pip install -r requirements.txt

# Optional: RDKit for visualization
conda install -c conda-forge rdkit

# Launch Jupyter
jupyter notebook tutorials/
```

### Run Tutorials

1. **Start with Tutorial 01** to understand basics
2. **Follow numbered order** for progressive learning
3. **Complete code cells sequentially** from top to bottom
4. **Adjust parameters** for experimentation

## Tutorial Details

### Tutorial 01: Basic Molecule Generation

**Learn**:
- Load QM9 dataset
- Create diffusion model
- Train basic model
- Generate molecules
- Evaluate quality

**Key Code**:
```python
from qm9 import dataset
from qm9.models import get_model

dataloaders = dataset.retrieve_dataloaders(args)
model, nodes_dist, _ = get_model(args, device, dataset_info)

# Generate
x, h = model.sample(n_samples, n_nodes, node_mask, edge_mask)
```

**Output**: Generated 3D molecules with stability metrics

---

### Tutorial 02: Conditional Generation

**Learn**:
- Property normalization
- Conditional training
- Target property generation
- Property sweeps
- Multiple properties

**Key Code**:
```python
from qm9.utils import prepare_context, compute_mean_mad

# Normalize properties
property_norms = compute_mean_mad(dataloaders, ['alpha'], 'qm9')

# Condition on target value
context = prepare_context(['alpha'], data, property_norms)
x, h = model.sample(..., context=context)
```

**Output**: Molecules with controlled properties

---

### Tutorial 03: Crystal Generation

**Learn**:
- Crystal data with ASE
- Molecular feature extraction
- Multi-modal conditioning
- Unit cell learning
- CIF export

**Key Code**:
```python
from crystal.models import MolecularEncoder, CrystalDynamics
from crystal.conditioning import MolecularConditioning
from crystal.utils import CIFWriter

# Extract molecular features
mol_features = mol_encoder(h, x, edge_index)
context = mol_conditioning(mol_features)

# Generate crystal
crystal = crystal_model.sample(..., context=context, pbc=[True,True,True])

# Export
writer.write_cif(crystal, 'output.cif', space_group=14)
```

**Output**: Crystal structures in CIF format

---

### Tutorial 04: Molecular Descriptors & ASE

**Learn**:
- Create ASE databases
- Extract molecular descriptors
- Exact conditional generation
- SMARTS patterns

**Key Code**:
```python
from ase import Atoms
from ase.db import connect

# Create database
db = connect('molecules.db')
atoms = Atoms('C6H6', positions=...)
db.write(atoms, data={'molecular_weight': 78.0})

# Train with descriptors
# --conditioning molecular_weight pi_conjugation_ratio
```

**Output**: Descriptor-conditioned generation

---

### Tutorial 05: Evaluation & Analysis

**Learn**:
- Stability analysis
- RDKit validation
- Uniqueness/novelty metrics
- Property distributions
- Comprehensive evaluation

**Key Code**:
```python
from qm9.analyze import check_stability, analyze_stability_for_molecules

# Check stability
atom_stable, mol_stable, validity = check_stability(
    positions, atom_types, charges, dataset_info
)

# Comprehensive metrics
python eval_analyze.py --model_path outputs/model --n_samples 10000
```

**Output**: Quality metrics and visualizations

---

### Tutorial 06: Advanced Crystal Conditioning

**Learn**:
- Conditioning hierarchy
- Space group selection
- Density targeting
- Polymorph generation
- Strict validation

**Key Code**:
```python
# Combined conditioning
mol_context = mol_conditioning(mol_features)
sg_context = sg_embedding(space_group)
dens_context = density_conditioning(density)
context = torch.cat([mol_context, sg_context, dens_context], dim=-1)

# Generate polymorph
crystal = model.sample(..., context=context)
```

**Output**: Controlled crystal structures

---

## Features Covered

### Molecule Generation (PR#124-128)
- ✅ Basic unconditional generation
- ✅ Property-conditioned generation
- ✅ Multi-property conditioning
- ✅ Exact conditional generation
- ✅ Molecular descriptor conditioning
- ✅ QM9, GEOM-Drugs, ASE datasets

### Crystal Generation (PR#129-130)
- ✅ Homocrystal generation
- ✅ Molecular feature extraction
- ✅ Space group conditioning (230 groups)
- ✅ Density conditioning
- ✅ Unit cell parameter learning
- ✅ Periodic boundary conditions
- ✅ CIF file export

### Evaluation
- ✅ Stability metrics
- ✅ RDKit validity
- ✅ Uniqueness
- ✅ Novelty
- ✅ Structure validation
- ✅ Crystal metrics

## Design Principles

All tutorials follow the core principles:

### 1. No Fallback Heuristics (ごまかしのためのfallbackは絶対にしない)
- All error cases explicitly handled
- Clear error messages
- No silent corrections
- Strict input validation

### 2. Theoretical Soundness
- E(3) equivariance maintained
- Proper PBC handling
- Physical constraints enforced
- Mathematically rigorous

### 3. Practical Usability
- Executable code snippets
- Clear explanations
- Real examples
- Performance tips

## Common Issues

### CUDA Out of Memory
```python
# Reduce batch size
args.batch_size = 16  # or smaller

# Reduce model size
args.nf = 64
args.n_layers = 4
```

### RDKit Not Found
```bash
# Install with conda
conda install -c conda-forge rdkit

# Or continue without RDKit (some features disabled)
```

### Slow Training
```python
# Use fewer test samples
args.n_stability_samples = 100

# Less frequent evaluation
args.test_epochs = 10
```

## Tips for Learning

1. **Start Simple**: Begin with Tutorial 01, understand basics
2. **Run Code**: Execute each cell, observe outputs
3. **Experiment**: Modify parameters, see effects
4. **Read Comments**: Code comments explain key concepts
5. **Check Errors**: Error messages are informative, not fallbacks
6. **Scale Up**: Start with small models/datasets for speed

## Production Use

Tutorials use small datasets and models for quick learning. For production:

```bash
# Full QM9 training
python main_qm9.py \
    --exp_name production \
    --n_epochs 1000 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 9 \
    --diffusion_steps 1000 \
    --lr 1e-4 \
    --ema_decay 0.9999
```

**Training time**: 2-3 days on V100 GPU

## Additional Resources

- **Documentation**: `../doc/`
  - `theory.md`: Mathematical foundations
  - `design.md`: Architecture and implementation
  - `user_manual.md`: Complete usage guide

- **Example Scripts**:
  - `../main_qm9.py`: QM9 training
  - `../main_crystal.py`: Crystal training
  - `../eval_analyze.py`: Evaluation
  - `../eval_conditional_qm9.py`: Conditional generation

- **Tests**: `../tests/`
  - Unit tests for all modules
  - Integration tests
  - Example usage patterns

## Support

For questions or issues:
1. Check tutorial comments and documentation
2. Review error messages (they're designed to be helpful)
3. Consult user manual in `../doc/user_manual.md`
4. Open GitHub issue with details

## Citation

If you use this code in research, please cite:

```bibtex
@article{hoogeboom2022equivariant,
  title={Equivariant Diffusion for Molecule Generation in 3D},
  author={Hoogeboom, Emiel and Satorras, V{\'\i}ctor Garcia and Vignac, Cl{\'e}ment and Welling, Max},
  journal={ICML},
  year={2022}
}
```

---

**Version**: 1.0  
**Date**: 2025-10-13  
**Status**: Complete tutorials for PR#124-130

**Happy Learning! 🚀**
