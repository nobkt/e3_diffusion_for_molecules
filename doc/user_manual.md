# E3 Equivariant Diffusion for Molecules: User Manual

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Molecule Generation](#molecule-generation)
4. [Crystal Generation](#crystal-generation)
5. [Conditional Generation](#conditional-generation)
6. [Evaluation and Analysis](#evaluation-and-analysis)
7. [Command-Line Reference](#command-line-reference)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

---

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+ (for GPU support)

### Basic Installation

```bash
# Clone repository
git clone https://github.com/nobkt/e3_diffusion_for_molecules.git
cd e3_diffusion_for_molecules

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

**For molecule visualization and validation**:
```bash
conda create -c conda-forge -n edm-env rdkit
conda activate edm-env
pip install -r requirements.txt
```

**For crystal analysis**:
```bash
pip install spglib  # Space group detection
pip install ase openbabel-wheel  # ASE database support
```

### Verify Installation

```python
python -c "import torch; import rdkit; print('Installation successful!')"
```

---

## Quick Start

### 1. Train a Basic Model (QM9)

```bash
python main_qm9.py \
    --exp_name my_first_model \
    --n_epochs 100 \
    --batch_size 64 \
    --nf 128 \
    --n_layers 6
```

**What this does**:
- Trains a diffusion model on QM9 dataset
- Saves checkpoints to `outputs/my_first_model/`
- Generates samples every test epoch

### 2. Generate Molecules

```bash
python eval_sample.py \
    --model_path outputs/my_first_model \
    --n_samples 100
```

**Output**:
- Molecular structures in `outputs/my_first_model/samples/`
- Visualization images (if RDKit installed)

### 3. Evaluate Quality

```bash
python eval_analyze.py \
    --model_path outputs/my_first_model \
    --n_samples 1000
```

**Metrics computed**:
- Validity: % of chemically valid molecules
- Uniqueness: % of unique structures
- Novelty: % not in training set
- Stability: Atom/molecule stability

---

## Molecule Generation

### Datasets

#### QM9 Dataset

**Description**: 134k small organic molecules with quantum properties.

**Usage**:
```bash
python main_qm9.py \
    --dataset qm9 \
    --exp_name qm9_experiment
```

**Properties available**: `alpha`, `gap`, `homo`, `lumo`, `mu`, `Cv`

#### GEOM-Drugs Dataset

**Description**: Drug-like molecules with conformers.

**Setup**:
```bash
# Follow instructions in data/geom/README.md
cd data/geom
python download_dataset.py
```

**Usage**:
```bash
python main_geom_drugs.py \
    --exp_name geom_experiment \
    --n_epochs 1000 \
    --batch_size 32
```

#### ASE Database

**Description**: Custom molecular datasets in ASE format.

**Creating ASE database**:
```python
from ase import Atoms
from ase.db import connect

db = connect('my_molecules.db', append=False)

# Add molecules
mol = Atoms('H2O', positions=[[0,0,0], [1,0,0], [0,1,0]])
db.write(mol, data={'energy': -76.4, 'molecular_weight': 18.0})
```

**Usage**:
```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path my_molecules.db \
    --exp_name ase_experiment
```

### Training Parameters

#### Essential Parameters

```bash
--exp_name NAME              # Experiment name (required)
--dataset DATASET            # qm9 | geom_drugs | ase_db
--n_epochs N                 # Number of training epochs (default: 200)
--batch_size B               # Batch size (default: 128)
--lr RATE                    # Learning rate (default: 2e-4)
```

#### Model Architecture

```bash
--model MODEL                # egnn_dynamics (default)
--nf FEATURES                # Hidden features (default: 128)
--n_layers LAYERS            # Number of EGNN layers (default: 6)
--attention                  # Use attention mechanism (default: True)
--normalization_factor F     # Coordinate normalization (default: 1)
```

#### Diffusion Parameters

```bash
--diffusion_steps STEPS              # Number of diffusion steps (default: 500)
--diffusion_noise_schedule SCHEDULE  # polynomial_2 | cosine
--diffusion_noise_precision PREC     # Noise precision (default: 1e-5)
--diffusion_loss_type TYPE           # l2 | vlb
```

#### Advanced Options

```bash
--ema_decay DECAY            # EMA decay rate (default: 0.999)
--clip_grad                  # Enable gradient clipping
--normalize_factors [X,H,C]  # Normalization factors for [positions, features, charges]
--remove_h                   # Remove hydrogen atoms
--include_charges            # Include atomic charges
```

### Example: Full Training Configuration

```bash
python main_qm9.py \
    --exp_name production_model \
    --dataset qm9 \
    --n_epochs 3000 \
    --batch_size 64 \
    --lr 1e-4 \
    --nf 256 \
    --n_layers 9 \
    --attention True \
    --diffusion_steps 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_loss_type l2 \
    --ema_decay 0.9999 \
    --normalize_factors [1,4,10] \
    --test_epochs 20 \
    --n_stability_samples 1000 \
    --save_model True
```

**Training time**: ~2-3 days on single GPU (V100)

### Resuming Training

```bash
python main_qm9.py \
    --exp_name my_model \
    --resume outputs/my_model/generative_model.npy \
    --start_epoch 100
```

---

## Crystal Generation

### Overview

Crystal generation extends molecule generation to periodic systems with learned unit cell parameters.

### Data Preparation

#### Creating Crystal Database

```python
from ase import Atoms
from ase.db import connect

# Create crystal database
db = connect('crystals.db', append=False)

# Example: Benzene crystal
positions = [...]  # Atomic positions
cell = [5.0, 5.0, 5.0, 90, 90, 90]  # Cell parameters (a,b,c,α,β,γ)

atoms = Atoms('C6H6', positions=positions, cell=cell, pbc=True)
db.write(atoms, data={
    'molecule_id': 'benzene_001',
    'space_group': 14,  # P2_1/c
    'density': 1.08,    # g/cm³
})
```

#### Creating Molecule Database

```python
# Single molecule database for conditioning
mol_db = connect('molecules.db', append=False)

mol = Atoms('C6H6', positions=[...])  # Single benzene molecule
mol_db.write(mol, data={'molecule_id': 'benzene_001'})
```

### Training Crystal Model

```bash
python main_crystal.py \
    --exp_name crystal_experiment \
    --crystal_db_path data/crystals.db \
    --molecule_db_path data/molecules.db \
    --condition_on_molecule True \
    --batch_size 16 \
    --n_epochs 200 \
    --lr 1e-4 \
    --nf 128 \
    --n_layers 6 \
    --learn_lattice True
```

### Conditioning Options

#### Molecular Conditioning (Required)

Conditions crystal generation on molecular features:

```bash
--condition_on_molecule True        # Enable molecular conditioning
--molecular_feature_dim 128         # Dimension of molecular features
--use_molecular_geometry True       # Include size, volume, axes
```

**What it does**:
- Extracts EGNN features from single molecule
- Computes geometric properties (size, volume, principal axes)
- Uses as primary conditioning signal

#### Space Group Conditioning (Optional)

Conditions on crystallographic space group:

```bash
--condition_on_space_group True     # Enable space group conditioning
--space_group_embedding_dim 64      # Embedding dimension
```

**Valid values**: 1-230 (International Tables space groups)

#### Density Conditioning (Optional)

Conditions on crystal density:

```bash
--condition_on_density True         # Enable density conditioning
--density_min 0.5                   # Minimum density (g/cm³)
--density_max 5.0                   # Maximum density (g/cm³)
```

### Example: Full Crystal Training

```bash
python main_crystal.py \
    --exp_name benzene_crystals \
    --crystal_db_path data/benzene_crystals.db \
    --molecule_db_path data/benzene_molecules.db \
    --condition_on_molecule True \
    --condition_on_space_group True \
    --condition_on_density True \
    --batch_size 16 \
    --n_epochs 500 \
    --lr 1e-4 \
    --nf 192 \
    --n_layers 9 \
    --learn_lattice True \
    --lattice_hidden_dim 128 \
    --diffusion_steps 1000 \
    --save_cif True \
    --test_epochs 10
```

### Generating Crystals

```python
from crystal.models import CrystalDiffusion, MolecularEncoder
from crystal.conditioning import MolecularConditioning
import torch

# Load model
model = CrystalDiffusion.from_checkpoint('outputs/benzene_crystals/')
mol_encoder = MolecularEncoder(...)
mol_conditioning = MolecularConditioning(...)

# Load single molecule
molecule = load_molecule('benzene')

# Extract features
with torch.no_grad():
    mol_features = mol_encoder(molecule['h'], molecule['x'], ...)
    context = mol_conditioning(mol_features)

# Generate crystal
crystal = model.sample(
    n_samples=1,
    n_nodes=100,
    context=context,
    cell_params=None,  # Will be generated
    pbc=torch.tensor([True, True, True])
)

# Export to CIF
from crystal.utils import CIFWriter
writer = CIFWriter(dataset_info)
writer.write_cif(crystal, 'generated_crystal.cif')
```

---

## Conditional Generation

### Property Conditioning (QM9)

#### Training

Train a model conditioned on a property:

```bash
python main_qm9.py \
    --exp_name cond_alpha \
    --conditioning alpha \
    --dataset qm9 \
    --n_epochs 1000 \
    --batch_size 64 \
    --lr 1e-4 \
    --nf 192 \
    --n_layers 9
```

**Available properties**:
- `alpha`: Polarizability
- `gap`: HOMO-LUMO gap
- `homo`: HOMO energy
- `lumo`: LUMO energy
- `mu`: Dipole moment
- `Cv`: Heat capacity

#### Generating with Property Sweeps

Generate molecules across property range:

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/cond_alpha \
    --property alpha \
    --task qualitative \
    --n_sweeps 10
```

**What this does**:
- Divides property range into 10 bins
- Generates molecules for each bin
- Saves samples organized by property value

### Molecular Descriptor Conditioning (ASE)

#### Available Descriptors

**Molecular Weight**:
```bash
--conditioning molecular_weight
```
- **Type**: Scalar
- **Units**: Atomic mass units (u)
- **Use case**: Control molecule size

**π Conjugation Ratio**:
```bash
--conditioning pi_conjugation_ratio
```
- **Type**: Scalar in [0, 1]
- **Definition**: (# double bonds + # aromatic bonds) / (# total bonds)
- **Use case**: Control aromaticity/conjugation

**Atom Types Encoding**:
```bash
--conditioning atom_types_encoding
```
- **Type**: Binary vector
- **Format**: Presence/absence of each element type
- **Use case**: Control elemental composition

**Functional Groups Encoding**:
```bash
--conditioning functional_groups_encoding
```
- **Type**: Binary vector
- **Format**: Presence/absence of SMARTS patterns
- **Example**: `[CX3](=O)[OX2H1]` (carboxylic acid)
- **Use case**: Control functional group presence

#### Training with Multiple Descriptors

```bash
python main_qm9.py \
    --dataset ase_db \
    --ase_db_path my_database.db \
    --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding \
    --exp_name multi_descriptor_model \
    --n_epochs 1000
```

#### Exact Conditional Generation

Generate molecules with exact descriptor values:

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/multi_descriptor_model \
    --task qualitative \
    --use_exact_conditions \
    --property_values "molecular_weight=50.0,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,N,O]" \
    --n_sweeps 5
```

**Format for property_values**:
- Comma-separated key=value pairs
- Arrays use brackets: `[C,H,N,O]`
- SMARTS patterns: `functional_groups_encoding=[[CX3](=O)[OX2H1]]`

### Training Property Classifier

For evaluation, train a classifier:

```bash
cd qm9/property_prediction

python main_qm9_prop.py \
    --property alpha \
    --exp_name classifier_alpha \
    --model_name egnn \
    --batch_size 96 \
    --lr 5e-4 \
    --num_workers 2
```

**For ASE descriptors**:
```bash
python main_qm9_prop.py \
    --dataset ase_db \
    --ase_db_path ../../my_database.db \
    --property molecular_weight \
    --exp_name classifier_mw \
    --model_name egnn
```

### Evaluating Conditional Model

```bash
python eval_conditional_qm9.py \
    --generators_path outputs/cond_alpha \
    --classifiers_path qm9/property_prediction/outputs/classifier_alpha \
    --property alpha \
    --task edm \
    --iterations 100 \
    --batch_size 100
```

**Metrics computed**:
- MAE between target and predicted property
- Property distribution plots
- Sample quality metrics

---

## Evaluation and Analysis

### Computing Metrics

#### Basic Evaluation

```bash
python eval_analyze.py \
    --model_path outputs/my_model \
    --n_samples 10000
```

**Outputs**:
- `eval_log.txt`: Summary statistics
- Stability metrics
- RDKit validity metrics
- Uniqueness and novelty

#### Detailed Metrics

**Stability**:
- `atom_stable`: % atoms with valid valence
- `mol_stable`: % molecules with all atoms stable
- `atm_valency_MAE`: Mean absolute error in valence

**Validity**:
- `valid`: % molecules passing RDKit sanitization
- `connected`: % molecules with single connected component

**Uniqueness**:
- `unique`: % unique SMILES among valid molecules

**Novelty**:
- `novel`: % molecules not in training set

### Visualization

#### Generate Visualizations

```bash
python eval_sample.py \
    --model_path outputs/my_model \
    --n_samples 100 \
    --visualize
```

**Creates**:
- `samples/molecule_*.png`: 2D depictions
- `samples/molecule_*.xyz`: 3D coordinates
- `samples/molecule_*.sdf`: SDF files

#### Custom Visualization

```python
from qm9.visualizer import MoleculeVisualizer
from rdkit import Chem

# Load molecule
positions = ...  # [N, 3]
atom_types = ...  # [N]

# Create RDKit molecule
mol = build_rdkit_molecule(positions, atom_types, dataset_info)

# Visualize
visualizer = MoleculeVisualizer()
visualizer.draw_molecule_2d(mol, 'output.png')
visualizer.draw_molecule_3d(mol, 'output.html')
```

### Crystal Analysis

#### Validating Crystals

```python
from crystal.evaluation import StructureValidator

validator = StructureValidator()

for crystal in generated_crystals:
    is_valid, errors = validator.validate_structure(crystal)
    if not is_valid:
        print(f"Invalid crystal: {errors}")
```

**Validation checks**:
- Cell parameter bounds (positive lengths, valid angles)
- Coordinate consistency
- Minimum inter-atomic distances
- PBC correctness

#### Computing Crystal Metrics

```python
from crystal.evaluation import CrystalMetrics

metrics_obj = CrystalMetrics(dataset_info)
metrics = metrics_obj.compute_all_metrics(
    generated_crystals,
    reference_crystals=test_set
)

print(f"Validity: {metrics['validity_ratio']:.2%}")
print(f"Volume MAE: {metrics['volume_mean']:.2f} ų")
print(f"Density MAE: {metrics['density_mean']:.2f} g/cm³")
```

#### Symmetry Analysis

```python
from crystal.evaluation import SymmetryAnalyzer

analyzer = SymmetryAnalyzer()

# Detect space group (requires spglib)
space_group = analyzer.detect_space_group(
    positions, cell, atom_types, symprec=1e-5
)
print(f"Detected space group: {space_group}")

# Compute structure fingerprint
fingerprint = analyzer.compute_structure_fingerprint(
    positions, cell, r_max=10.0
)

# Compare structures
similarity = analyzer.compare_fingerprints(fp1, fp2)
print(f"Structure similarity: {similarity:.3f}")
```

---

## Command-Line Reference

### Main Training Scripts

#### `main_qm9.py`

**Purpose**: Train molecule generation model on QM9/ASE datasets.

**Required Arguments**:
```
--exp_name NAME              Experiment name
```

**Dataset Arguments**:
```
--dataset {qm9,ase_db}       Dataset type (default: qm9)
--ase_db_path PATH           Path to ASE database (for ase_db dataset)
--split_ratios R1 R2 R3      Train/val/test split (default: 0.8 0.1 0.1)
```

**Model Arguments**:
```
--model MODEL                Model type (default: egnn_dynamics)
--nf FEATURES                Hidden features (default: 128)
--n_layers LAYERS            EGNN layers (default: 6)
--attention {True,False}     Use attention (default: True)
--tanh {True,False}          Use tanh in coord MLP (default: True)
--norm_constant CONST        Normalization constant (default: 1)
--normalization_factor F     Sum aggregation normalization (default: 1)
```

**Training Arguments**:
```
--n_epochs N                 Training epochs (default: 200)
--batch_size B               Batch size (default: 128)
--lr RATE                    Learning rate (default: 2e-4)
--ema_decay DECAY            EMA decay (default: 0.999, 0 = off)
--clip_grad {True,False}     Enable gradient clipping (default: True)
--test_epochs N              Epochs between evaluation (default: 1)
```

**Diffusion Arguments**:
```
--diffusion_steps N          Diffusion steps (default: 500)
--diffusion_noise_schedule S Noise schedule (default: polynomial_2)
--diffusion_noise_precision P Noise precision (default: 1e-5)
--diffusion_loss_type T      Loss type (default: l2)
```

**Conditioning Arguments**:
```
--conditioning PROPS         Properties to condition on (space-separated)
                             For QM9: alpha gap homo lumo mu Cv
                             For ASE: molecular_weight pi_conjugation_ratio 
                                     atom_types_encoding functional_groups_encoding
```

**Data Arguments**:
```
--normalize_factors [X,H,C]  Normalization for [pos, feat, charge] (default: [1,4,1])
--remove_h                   Remove hydrogen atoms
--include_charges {True,False} Include charges (default: True)
--augment_noise NOISE        Data augmentation noise (default: 0)
```

**Checkpoint Arguments**:
```
--resume PATH                Resume from checkpoint
--start_epoch N              Starting epoch (default: 0)
--save_model {True,False}    Save model (default: True)
```

**Evaluation Arguments**:
```
--n_stability_samples N      Samples for stability (default: 500)
--visualize_every_batch N    Visualization frequency (default: 1e8)
```

#### `main_geom_drugs.py`

Similar to `main_qm9.py` but for GEOM-Drugs dataset. Key differences:

```
--data_dir PATH              GEOM data directory
--remove_h                   Remove hydrogens (recommended)
--conditioning PROPS         Properties for conditioning
```

#### `main_crystal.py`

**Purpose**: Train crystal generation model.

**Required Arguments**:
```
--exp_name NAME              Experiment name
--crystal_db_path PATH       Crystal database path
--molecule_db_path PATH      Molecule database path (for conditioning)
```

**Crystal-Specific Arguments**:
```
--learn_lattice {True,False}      Learn cell parameters (default: True)
--lattice_hidden_dim DIM          Lattice encoding dimension (default: 128)
--save_cif {True,False}           Export CIF files (default: False)
```

**Conditioning Arguments**:
```
--condition_on_molecule {True,False}     Molecular conditioning (default: True)
--condition_on_space_group {True,False}  Space group conditioning (default: False)
--condition_on_density {True,False}      Density conditioning (default: False)
--molecular_feature_dim DIM              Molecular feature dimension (default: 128)
--conditioning_dim DIM                   Total conditioning dimension (default: 256)
--use_molecular_geometry {True,False}    Include geometric properties (default: True)
```

**Other arguments**: Similar to `main_qm9.py`

### Evaluation Scripts

#### `eval_analyze.py`

**Purpose**: Compute comprehensive evaluation metrics.

**Required Arguments**:
```
--model_path PATH            Path to trained model
```

**Optional Arguments**:
```
--n_samples N                Number of samples (default: 10000)
--device {cpu,cuda}          Device (default: cuda if available)
--batch_size B               Batch size (default: 100)
--save_molecules {True,False} Save molecule files (default: True)
```

#### `eval_sample.py`

**Purpose**: Generate and visualize samples.

**Required Arguments**:
```
--model_path PATH            Path to trained model
```

**Optional Arguments**:
```
--n_samples N                Number of samples (default: 100)
--visualize {True,False}     Create visualizations (default: True)
--output_dir PATH            Output directory (default: model_path/samples)
```

#### `eval_conditional_qm9.py`

**Purpose**: Evaluate conditional generation.

**Required Arguments**:
```
--generators_path PATH       Path to trained generator
--property PROP              Property name
```

**Task Arguments**:
```
--task {qualitative,edm,mflow} Evaluation task
  qualitative: Generate property sweeps
  edm: Evaluate with classifier
  mflow: Compare to MoFlow baseline
```

**Qualitative Task**:
```
--n_sweeps N                 Number of property bins (default: 10)
--use_exact_conditions       Use exact property values
--property_values VALUES     Exact values (comma-separated key=value)
```

**EDM Task**:
```
--classifiers_path PATH      Path to trained classifier
--iterations N               Number of iterations (default: 100)
--batch_size B               Batch size (default: 100)
```

### Utility Scripts

#### Creating ASE Database

```bash
python create_test_db.py \
    --output my_database.db \
    --n_molecules 1000
```

#### Exporting Molecules to CSV

```bash
python csv_export_example.py \
    --model_path outputs/my_model \
    --n_samples 1000 \
    --output results.csv
```

**CSV columns**:
- `smiles`: SMILES string
- `positions`: Atomic positions (JSON)
- `atom_types`: Atom type indices
- `valid`: Validity flag
- `stable`: Stability flag
- Properties (if conditional model)

---

## Troubleshooting

### Common Errors

#### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```bash
   --batch_size 32  # or 16
   ```

2. Reduce model size:
   ```bash
   --nf 64 --n_layers 4
   ```

3. Use gradient accumulation:
   ```python
   # In training loop
   for i, data in enumerate(loader):
       loss = model(data) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

#### RDKit Import Error

**Error**: `ModuleNotFoundError: No module named 'rdkit'`

**Solution**:
```bash
conda install -c conda-forge rdkit
# OR
conda create -c conda-forge -n edm-env rdkit
conda activate edm-env
```

**Note**: Code works without RDKit, but some visualizations unavailable.

#### ASE Database Not Found

**Error**: `FileNotFoundError: ASE database not found`

**Solution**:
```bash
# Check path
--ase_db_path /absolute/path/to/database.db

# Create database if missing
python create_test_db.py --output database.db
```

#### Space Group Validation Error

**Error**: `ValueError: space_group must be in [1, 230]`

**Solution**: Ensure space group numbers are valid:
```python
# Valid
space_group = 14  # P2_1/c

# Invalid
space_group = 0   # Not a valid space group
space_group = 231  # Exceeds maximum
```

**No fallback**: Must provide valid space group or set `--condition_on_space_group False`.

#### Cell Parameter Constraints

**Error**: `ValueError: Cell parameters violate physical constraints`

**Solution**: Check cell parameters:
```python
# Valid
cell = [5.0, 6.0, 7.0, 90.0, 95.0, 100.0]  # Positive lengths, valid angles

# Invalid
cell = [-1.0, 5.0, 6.0, 90.0, 90.0, 90.0]  # Negative length
cell = [5.0, 6.0, 7.0, 0.0, 90.0, 90.0]    # Zero angle
cell = [5.0, 6.0, 7.0, 200.0, 90.0, 90.0]  # Angle > 180°
```

### Performance Issues

#### Slow Training

**Symptoms**: Training much slower than expected.

**Diagnostics**:
```python
# Check dataloader
import time
for i, data in enumerate(loader):
    if i == 0:
        t0 = time.time()
    if i == 10:
        print(f"10 batches in {time.time() - t0:.2f}s")
        break

# Check forward pass
model.train()
t0 = time.time()
loss = model(x, h, node_mask, edge_mask)
print(f"Forward pass: {time.time() - t0:.3f}s")
```

**Solutions**:
1. Increase num_workers:
   ```bash
   --num_workers 4
   ```

2. Use fully connected graphs (for small molecules):
   ```bash
   --brute_force True
   ```

3. Reduce evaluation frequency:
   ```bash
   --test_epochs 10  # Instead of 1
   ```

#### High Memory Usage

**Symptoms**: CPU/GPU memory filling up.

**Solutions**:
1. Enable garbage collection:
   ```python
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   ```

2. Use EMA less frequently:
   ```bash
   --ema_decay 0  # Disable EMA
   ```

3. Reduce stability sample size:
   ```bash
   --n_stability_samples 100  # Instead of 500
   ```

### Validation Issues

#### Low Stability

**Symptoms**: Many molecules have invalid valences.

**Solutions**:
1. Train longer:
   ```bash
   --n_epochs 3000  # More training
   ```

2. Use valence loss:
   ```bash
   --normalize_factors [1,4,10]  # Higher weight on features
   ```

3. Check data quality:
   ```python
   # Validate training data
   from qm9.analyze import check_valency
   for data in dataset:
       valid = check_valency(data['one_hot'], data['charges'])
       if not valid:
           print("Invalid training example!")
   ```

#### Low Validity

**Symptoms**: RDKit can't sanitize generated molecules.

**Causes**:
- Wrong atom types
- Invalid connectivity
- Unrealistic geometries

**Solutions**:
1. Use more EGNN layers:
   ```bash
   --n_layers 9  # Instead of 6
   ```

2. Enable attention:
   ```bash
   --attention True
   ```

3. Use smaller diffusion step:
   ```bash
   --diffusion_steps 1000  # Instead of 500
   ```

---

## Advanced Usage

### Custom Datasets

#### Creating Dataset Class

```python
# my_dataset.py
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = self.load_data(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        mol = self.data[idx]
        
        # Required fields
        item = {
            'positions': torch.tensor(mol['positions'], dtype=torch.float32),
            'one_hot': torch.tensor(mol['one_hot'], dtype=torch.float32),
            'charges': torch.tensor(mol['charges'], dtype=torch.float32),
            'atom_mask': torch.ones(len(mol['positions']), 1),
            'n_nodes': len(mol['positions']),
        }
        
        # Optional: properties
        if 'energy' in mol:
            item['energy'] = torch.tensor([mol['energy']])
        
        # Apply transforms
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def load_data(self, path):
        # Load your data format
        # Return list of molecule dictionaries
        pass
```

#### Registering Dataset

```python
# configs/datasets_config.py

def get_dataset_info(dataset_name, remove_h=False):
    if dataset_name == 'my_dataset':
        return {
            'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
            'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
            'max_n_nodes': 29,
            'n_nodes': {'mean': 18.0, 'std': 4.5},
            'max_weight': 150,
        }
    # ... existing datasets ...
```

#### Using Custom Dataset

```bash
python main_qm9.py \
    --dataset my_dataset \
    --data_path /path/to/data \
    --exp_name my_experiment
```

### Custom Conditioning

#### Implementing Conditioning Module

```python
# custom_conditioning.py
import torch
import torch.nn as nn

class MyConditioning(nn.Module):
    def __init__(self, conditioning_dim):
        super().__init__()
        self.conditioning_dim = conditioning_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, conditioning_dim)
        )
    
    def forward(self, my_property):
        """
        Args:
            my_property: [B, 1] Custom property values
        
        Returns:
            conditioning: [B, conditioning_dim]
        """
        # NO FALLBACK! Validate input
        if my_property is None:
            raise ValueError("my_property cannot be None when conditioning is enabled")
        
        if torch.isnan(my_property).any():
            raise ValueError("my_property contains NaN values")
        
        # Process
        return self.mlp(my_property)
```

#### Integrating in Training

```python
# In main script
if args.condition_on_my_property:
    my_conditioning = MyConditioning(args.conditioning_dim).to(device)
else:
    my_conditioning = None

# In training loop
def prepare_context(data):
    contexts = []
    
    # Existing conditioning
    # ...
    
    # Custom conditioning
    if args.condition_on_my_property:
        if my_conditioning is None:
            raise ValueError("my_conditioning required but not provided")
        my_context = my_conditioning(data['my_property'])
        contexts.append(my_context)
    
    return torch.cat(contexts, dim=-1) if contexts else None
```

### Distributed Training

#### DataParallel (Single Node, Multiple GPUs)

```python
# Already enabled by default
# main_qm9.py uses nn.DataParallel
model_dp = torch.nn.DataParallel(model)
```

#### DistributedDataParallel (Multiple Nodes)

```python
# Initialize distributed
import torch.distributed as dist
dist.init_process_group(backend='nccl')

# Wrap model
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler, ...)
```

**Launch**:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main_qm9.py --exp_name distributed_training
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data in loader:
    optimizer.zero_grad()
    
    # Forward with autocast
    with autocast():
        loss = model(data)
    
    # Backward with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- ~2x speedup on modern GPUs (V100, A100)
- Reduced memory usage
- Same accuracy

### Hyperparameter Tuning

#### Using Weights & Biases

```python
import wandb

# Initialize
wandb.init(project='edm', config=args)

# Log metrics
wandb.log({
    'train_loss': loss,
    'val_loss': val_loss,
    'stability': stability_dict['mol_stable'],
})

# Log artifacts
wandb.save('model.npy')
wandb.log({"examples": [wandb.Image(img) for img in images]})
```

**Launch sweep**:
```yaml
# sweep.yaml
program: main_qm9.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    min: 1e-5
    max: 1e-3
  nf:
    values: [128, 192, 256]
  n_layers:
    values: [6, 9, 12]
```

```bash
wandb sweep sweep.yaml
wandb agent <sweep_id>
```

### Checkpointing Strategy

#### Save Best Model

```python
best_val_loss = float('inf')

for epoch in range(n_epochs):
    # Training
    # ...
    
    # Validation
    val_loss = test(model, val_loader)
    
    # Save if best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_model.pt')
```

#### Periodic Checkpoints

```python
if epoch % args.checkpoint_freq == 0:
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f'checkpoint_epoch_{epoch}.pt')
```

#### Resume Training

```bash
python main_qm9.py \
    --exp_name my_experiment \
    --resume outputs/my_experiment/generative_model.npy \
    --start_epoch 100
```

---

## Appendix

### File Structure

```
outputs/
└── my_experiment/
    ├── args.pickle              # Saved arguments
    ├── generative_model.npy     # Final model (no EMA)
    ├── generative_model_ema.npy # Final model (with EMA)
    ├── eval_log.txt             # Evaluation results
    ├── samples/
    │   ├── epoch_100/
    │   │   ├── molecule_0.png
    │   │   ├── molecule_0.xyz
    │   │   └── ...
    │   └── ...
    └── images_gen/
        ├── epoch_100.png
        └── ...
```

### Environment Variables

```bash
# CUDA device selection
export CUDA_VISIBLE_DEVICES=0,1

# OMP threads (for CPU parallelism)
export OMP_NUM_THREADS=8

# PyTorch determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Useful Code Snippets

#### Convert to RDKit Molecule

```python
from qm9.analyze import build_molecule

rdkit_mol = build_molecule(
    positions,      # [N, 3]
    atom_types,     # [N]
    dataset_info
)

# Get SMILES
from rdkit import Chem
smiles = Chem.MolToSmiles(rdkit_mol)
```

#### Compute Molecular Properties

```python
from rdkit.Chem import Descriptors

mol = build_molecule(...)

# Basic properties
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)

# Lipinski's Rule of Five
ro5 = (
    mw <= 500 and
    logp <= 5 and
    Descriptors.NumHDonors(mol) <= 5 and
    Descriptors.NumHAcceptors(mol) <= 10
)
```

#### Export Molecules

```python
# SDF format
from rdkit import Chem

writer = Chem.SDWriter('molecules.sdf')
for mol in molecules:
    rdkit_mol = build_molecule(mol, dataset_info)
    writer.write(rdkit_mol)
writer.close()

# XYZ format
def write_xyz(positions, atom_types, filename):
    with open(filename, 'w') as f:
        f.write(f"{len(positions)}\n\n")
        for pos, atom in zip(positions, atom_types):
            f.write(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
```

---

**Version**: 1.0  
**Date**: 2025-10-13  
**Status**: Complete user manual for PR#124-130

**For questions or issues, please open a GitHub issue.**
