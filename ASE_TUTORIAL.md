# Complete Tutorial: E3 Diffusion for Molecules with ASE Databases

This tutorial provides a comprehensive guide for using E3 Diffusion for Molecules (EDM) with ASE (Atomic Simulation Environment) databases, offering the same functionality as the QM9 and GEOM drug workflows.

## Overview

The ASE integration allows you to:

1. **Train EDM models** on custom molecular datasets stored in ASE database format
2. **Generate new molecules** using the trained models
3. **Evaluate sample quality** using validity, uniqueness, and novelty metrics
4. **Visualize molecules** in 3D and analyze molecular properties
5. **Use any molecular dataset** that can be represented in ASE format

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setting Up ASE Databases](#setting-up-ase-databases)
3. [Training EDM with ASE Data](#training-edm-with-ase-data)
4. [Generating Molecules](#generating-molecules)
5. [Evaluating Sample Quality](#evaluating-sample-quality)
6. [Visualization and Analysis](#visualization-and-analysis)
7. [Complete Examples](#complete-examples)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### Installation

```bash
# Install PyTorch and basic dependencies
pip install torch torchvision numpy scipy matplotlib tqdm wandb

# Install ASE and OpenBabel for molecular operations
pip install ase openbabel-wheel

# Optional: Install RDKit for additional molecular functionality
conda install -c conda-forge rdkit
```

### Understanding ASE Databases

ASE databases are a convenient way to store molecular structures with associated properties. Each entry in the database represents a molecule with:
- Atomic positions (3D coordinates)
- Atomic symbols/numbers
- Optional properties (energy, forces, etc.)
- Optional metadata

## Setting Up ASE Databases

### Creating an ASE Database from Scratch

```python
#!/usr/bin/env python3
"""Create a sample ASE database for training."""

from ase import Atoms
from ase.db import connect
import numpy as np

# Create or connect to database
db = connect('my_molecules.db')

# Example: Add small molecules
molecules = [
    {
        'symbols': ['O', 'H', 'H'],
        'positions': [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
        'name': 'water',
        'energy': -76.3  # Example energy in eV
    },
    {
        'symbols': ['C', 'H', 'H', 'H', 'H'],
        'positions': [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [-0.36, 1.03, 0.0], 
                      [-0.36, -0.52, 0.89], [-0.36, -0.52, -0.89]],
        'name': 'methane',
        'energy': -40.5
    },
    # Add more molecules...
]

# Add molecules to database
for mol_data in molecules:
    atoms = Atoms(symbols=mol_data['symbols'], positions=mol_data['positions'])
    db.write(atoms, name=mol_data['name'], energy=mol_data['energy'])

print(f"Created database with {len(db)} molecules")
```

### Converting from Other Formats

#### From XYZ Files

```python
from ase.io import read
from ase.db import connect

# Read molecules from XYZ files
db = connect('molecules.db')
xyz_files = ['mol1.xyz', 'mol2.xyz', 'mol3.xyz']

for xyz_file in xyz_files:
    atoms = read(xyz_file)
    db.write(atoms, source_file=xyz_file)
```

#### From SMILES (using RDKit)

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.db import connect
import numpy as np

def smiles_to_ase_db(smiles_list, db_path):
    db = connect(db_path)
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
            
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Convert to ASE Atoms
        conf = mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = [[conf.GetAtomPosition(i).x, 
                     conf.GetAtomPosition(i).y, 
                     conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]
        
        atoms = Atoms(symbols=symbols, positions=positions)
        db.write(atoms, smiles=smiles)

# Example usage
smiles_list = ['CCO', 'CCC', 'C=C', 'c1ccccc1']  # ethanol, propane, ethene, benzene
smiles_to_ase_db(smiles_list, 'from_smiles.db')
```

## Training EDM with ASE Data

### Basic Training Script

Create `train_ase.py`:

```python
#!/usr/bin/env python3
"""Training script for EDM with ASE databases."""

import argparse
import torch
import wandb
from qm9.dataset import retrieve_dataloaders
from qm9.models import get_model, get_optim
from qm9.utils import prepare_context, compute_mean_mad
from equivariant_diffusion import en_diffusion
from train_test import train_epoch, test, analyze_and_save
import utils
import time
import pickle
from os.path import join
import copy
from equivariant_diffusion import utils as flow_utils

def main():
    parser = argparse.ArgumentParser(description='EDM Training with ASE Database')
    
    # Dataset parameters
    parser.add_argument('--ase_db_path', type=str, required=True,
                        help='Path to ASE database file')
    parser.add_argument('--exp_name', type=str, default='edm_ase',
                        help='Experiment name')
    parser.add_argument('--max_atoms', type=int, default=100,
                        help='Maximum number of atoms per molecule')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                        help='Model type')
    parser.add_argument('--nf', type=int, default=256,
                        help='Number of features')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of layers')
    
    # Diffusion parameters
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='Number of diffusion steps')
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='Noise schedule type')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                        help='Noise precision')
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='Loss type')
    
    # Other parameters
    parser.add_argument('--test_epochs', type=int, default=10,
                        help='Test every N epochs')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--normalize_factors', type=float, nargs=3, default=[1, 4, 10],
                        help='Normalization factors for [coordinates, features, charges]')
    parser.add_argument('--n_stability_samples', type=int, default=500,
                        help='Number of samples for stability analysis')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Configure dataset loading
    class Config:
        dataset = 'ase_custom'
        ase_db_path = args.ase_db_path
        batch_size = args.batch_size
        num_workers = 4
        max_atoms = args.max_atoms
        include_charges = True
        datadir = './data'
        remove_h = False
        filter_n_atoms = None
        sequential = False
        device = device
    
    config = Config()
    
    # Load dataset
    print(f"Loading ASE database: {args.ase_db_path}")
    dataloaders, charge_scale = retrieve_dataloaders(config)
    dataset_info = dataloaders['train'].dataset_info
    
    print(f"Dataset loaded:")
    print(f"  Atom types: {dataset_info['atom_decoder']}")
    print(f"  Max atoms: {dataset_info['max_n_nodes']}")
    print(f"  Training samples: {len(dataloaders['train'].dataset)}")
    print(f"  Validation samples: {len(dataloaders['valid'].dataset)}")
    print(f"  Test samples: {len(dataloaders['test'].dataset)}")
    
    # Set up context and normalization
    context, property_norms = prepare_context(args, dataloaders, dataset_info)
    
    # Initialize node distribution
    nodes_dist = flow_utils.DistributionNodes(dataset_info['n_nodes'])
    
    # Initialize property distribution (if needed)
    prop_dist = None
    
    # Initialize model
    model, gradnorm_queue = get_model(args, device, dataset_info, charge_scale)
    optim = get_optim(args, model)
    
    print(f"Model initialized:")
    print(f"  Type: {args.model}")
    print(f"  Features: {args.nf}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize wandb
    wandb.init(project='edm_ase', name=args.exp_name, config=args)
    
    # Create output directory
    import os
    os.makedirs(f'outputs/{args.exp_name}', exist_ok=True)
    
    # Initialize EMA
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)
    else:
        ema = None
        model_ema = model
    
    # Training loop
    best_nll_val = 1e8
    best_nll_test = 1e8
    
    print("Starting training...")
    for epoch in range(args.n_epochs):
        start_time = time.time()
        
        # Train epoch
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, 
                   model=model, model_dp=model, model_ema=model_ema, ema=ema,
                   device=device, dtype=dtype, property_norms=property_norms,
                   nodes_dist=nodes_dist, dataset_info=dataset_info,
                   gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch:4d} took {epoch_time:.1f}s")
        
        # Evaluation
        if epoch % args.test_epochs == 0:
            # Log diffusion info
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)
            
            # Analyze stability and save samples
            analyze_and_save(args=args, epoch=epoch, model_sample=model_ema,
                           nodes_dist=nodes_dist, dataset_info=dataset_info,
                           device=device, prop_dist=prop_dist,
                           n_samples=args.n_stability_samples)
            
            # Validation and test
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch,
                          eval_model=model_ema, partition='Val', device=device,
                          dtype=dtype, nodes_dist=nodes_dist,
                          property_norms=property_norms)
            
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch,
                           eval_model=model_ema, partition='Test', device=device,
                           dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            
            # Save best model
            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                
                if args.save_model:
                    utils.save_model(optim, f'outputs/{args.exp_name}/optim.npy')
                    utils.save_model(model, f'outputs/{args.exp_name}/generative_model.npy')
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, f'outputs/{args.exp_name}/generative_model_ema.npy')
                    with open(f'outputs/{args.exp_name}/args.pickle', 'wb') as f:
                        pickle.dump(args, f)
            
            print(f'Val loss: {nll_val:.4f} | Test loss: {nll_test:.4f}')
            print(f'Best val: {best_nll_val:.4f} | Best test: {best_nll_test:.4f}')
            
            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "val_loss": nll_val,
                "test_loss": nll_test,
                "best_test_loss": best_nll_test
            })
    
    print("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()
```

### Running Training

```bash
# Basic training with ASE database
python train_ase.py --ase_db_path molecules.db --exp_name my_experiment

# Advanced training with custom parameters
python train_ase.py \
    --ase_db_path molecules.db \
    --exp_name advanced_experiment \
    --n_epochs 2000 \
    --batch_size 64 \
    --lr 1e-4 \
    --nf 256 \
    --n_layers 9 \
    --diffusion_steps 1000 \
    --test_epochs 20 \
    --save_model
```

## Generating Molecules

### Sample Generation Script

Create `sample_ase.py`:

```python
#!/usr/bin/env python3
"""Generate molecules using trained ASE model."""

import argparse
import torch
import pickle
from qm9.models import get_model
from qm9.sampling import sample
from qm9.utils import prepare_context
from equivariant_diffusion import utils as flow_utils
from qm9 import visualizer as vis
import os

def main():
    parser = argparse.ArgumentParser(description='Generate molecules from trained ASE model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='generated_molecules',
                        help='Output directory for generated molecules')
    
    args = parser.parse_args()
    
    # Load model arguments
    with open(os.path.join(args.model_path, 'args.pickle'), 'rb') as f:
        model_args = pickle.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset info from the ASE database
    from qm9.ase_dataset import get_ase_dataset_info
    dataset_info = get_ase_dataset_info(model_args.ase_db_path, 
                                        max_atoms=model_args.max_atoms)
    
    # Load model
    model, _ = get_model(model_args, device, dataset_info, None)
    model_state = torch.load(os.path.join(args.model_path, 'generative_model_ema.npy'), 
                             map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Initialize node distribution
    nodes_dist = flow_utils.DistributionNodes(dataset_info['n_nodes'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.n_samples} molecules...")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Generate molecules
    generated_molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(n_batches):
        current_batch_size = min(args.batch_size, args.n_samples - batch_idx * args.batch_size)
        
        # Sample node counts
        nodesxsample = nodes_dist.sample(current_batch_size)
        
        # Generate molecules
        one_hot, charges, x, node_mask = sample(
            model_args, device, model, dataset_info, 
            prop_dist=None, nodesxsample=nodesxsample
        )
        
        # Store results
        generated_molecules['one_hot'].append(one_hot.detach().cpu())
        generated_molecules['x'].append(x.detach().cpu())
        generated_molecules['node_mask'].append(node_mask.detach().cpu())
        
        print(f"Generated batch {batch_idx + 1}/{n_batches}")
    
    # Concatenate all batches
    for key in generated_molecules:
        generated_molecules[key] = torch.cat(generated_molecules[key], dim=0)
    
    # Save molecules in various formats
    print("Saving generated molecules...")
    
    # Save as PyTorch tensors
    torch.save(generated_molecules, os.path.join(args.output_dir, 'generated_molecules.pt'))
    
    # Save as XYZ files
    vis.save_xyz_file(
        args.output_dir, 
        generated_molecules['one_hot'], 
        torch.zeros_like(generated_molecules['node_mask'].unsqueeze(-1)),  # No charges
        generated_molecules['x'], 
        dataset_info, 
        id_from=0, 
        name='generated'
    )
    
    print(f"Generated {args.n_samples} molecules saved to {args.output_dir}")

if __name__ == "__main__":
    main()
```

### Usage

```bash
# Generate molecules from trained model
python sample_ase.py --model_path outputs/my_experiment --n_samples 1000

# Generate with custom parameters
python sample_ase.py \
    --model_path outputs/my_experiment \
    --n_samples 5000 \
    --batch_size 200 \
    --output_dir my_generated_molecules
```

## Evaluating Sample Quality

### Evaluation Script

Create `evaluate_ase.py`:

```python
#!/usr/bin/env python3
"""Evaluate quality of generated molecules from ASE model."""

import argparse
import torch
import os
from qm9.analyze import analyze_stability_for_molecules
from qm9.ase_dataset import get_ase_dataset_info
from qm9.openbabel_functions import BasicMolecularMetricsOB

def main():
    parser = argparse.ArgumentParser(description='Evaluate generated molecules')
    parser.add_argument('--generated_molecules', type=str, required=True,
                        help='Path to generated molecules (.pt file)')
    parser.add_argument('--reference_db', type=str, required=True,
                        help='Path to reference ASE database')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("Loading generated molecules...")
    generated_molecules = torch.load(args.generated_molecules)
    
    print("Loading reference dataset info...")
    dataset_info = get_ase_dataset_info(args.reference_db)
    
    print("Evaluating molecular stability...")
    # Analyze stability
    validity_dict, metrics = analyze_stability_for_molecules(generated_molecules, dataset_info)
    
    print("Computing molecular metrics...")
    # Compute validity, uniqueness, novelty using OpenBabel
    molecule_list = []
    n_molecules = generated_molecules['one_hot'].shape[0]
    
    for i in range(n_molecules):
        # Extract molecule data
        one_hot = generated_molecules['one_hot'][i]
        positions = generated_molecules['x'][i]
        mask = generated_molecules['node_mask'][i]
        
        # Convert to atom types
        atom_types = torch.argmax(one_hot, dim=-1)
        
        # Apply mask
        n_atoms = int(mask.sum())
        if n_atoms > 0:
            molecule_list.append((positions[:n_atoms], atom_types[:n_atoms]))
    
    # Compute metrics
    ob_metrics = BasicMolecularMetricsOB(dataset_info)
    validity, uniqueness, novelty = ob_metrics.evaluate(molecule_list)
    
    # Print results
    results = f"""
Evaluation Results
==================

Dataset Information:
- Reference database: {args.reference_db}
- Atom types: {dataset_info['atom_decoder']}
- Generated molecules: {n_molecules}

Stability Analysis:
- Molecular stability: {validity_dict.get('mol_stable', 0):.2%}
- Atomic stability: {validity_dict.get('atm_stable', 0):.2%}

Quality Metrics:
- Validity: {validity:.2%}
- Uniqueness: {uniqueness:.2%}
- Novelty: {novelty:.2%}

Summary:
- Valid molecules: {int(validity * len(molecule_list))}
- Unique valid molecules: {int(validity * uniqueness * len(molecule_list))}
- Novel molecules: {int(validity * novelty * len(molecule_list))}
"""
    
    print(results)
    
    # Save results
    with open(args.output_file, 'w') as f:
        f.write(results)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
```

### Usage

```bash
# Evaluate generated molecules
python evaluate_ase.py \
    --generated_molecules generated_molecules/generated_molecules.pt \
    --reference_db molecules.db \
    --output_file evaluation_results.txt
```

## Visualization and Analysis

### 3D Visualization Script

Create `visualize_ase.py`:

```python
#!/usr/bin/env python3
"""Visualize generated molecules from ASE model."""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qm9.ase_dataset import get_ase_dataset_info
import os

def plot_molecule_3d(positions, atom_types, dataset_info, title="Molecule"):
    """Plot a single molecule in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    atom_decoder = dataset_info['atom_decoder']
    colors = dataset_info.get('colors_dic', ['red', 'blue', 'green', 'yellow', 'orange'])
    
    # Plot atoms
    for i, (pos, atom_type) in enumerate(zip(positions, atom_types)):
        if atom_type < len(atom_decoder):
            element = atom_decoder[atom_type]
            color = colors[atom_type % len(colors)]
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, alpha=0.8, label=element)
    
    # Set labels and title
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    
    # Add legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    return fig

def analyze_molecular_properties(generated_molecules, dataset_info):
    """Analyze properties of generated molecules."""
    atom_decoder = dataset_info['atom_decoder']
    
    # Extract molecule sizes
    molecule_sizes = []
    atom_counts = {element: 0 for element in atom_decoder}
    
    n_molecules = generated_molecules['one_hot'].shape[0]
    
    for i in range(n_molecules):
        mask = generated_molecules['node_mask'][i]
        one_hot = generated_molecules['one_hot'][i]
        
        n_atoms = int(mask.sum())
        molecule_sizes.append(n_atoms)
        
        # Count atom types
        if n_atoms > 0:
            atom_types = torch.argmax(one_hot[:n_atoms], dim=-1)
            for atom_type in atom_types:
                if atom_type < len(atom_decoder):
                    atom_counts[atom_decoder[atom_type]] += 1
    
    return molecule_sizes, atom_counts

def main():
    parser = argparse.ArgumentParser(description='Visualize generated molecules')
    parser.add_argument('--generated_molecules', type=str, required=True,
                        help='Path to generated molecules (.pt file)')
    parser.add_argument('--reference_db', type=str, required=True,
                        help='Path to reference ASE database')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for plots')
    parser.add_argument('--n_examples', type=int, default=10,
                        help='Number of example molecules to plot')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    generated_molecules = torch.load(args.generated_molecules)
    dataset_info = get_ase_dataset_info(args.reference_db)
    
    print("Analyzing molecular properties...")
    molecule_sizes, atom_counts = analyze_molecular_properties(generated_molecules, dataset_info)
    
    # Plot molecule size distribution
    plt.figure(figsize=(10, 6))
    plt.hist(molecule_sizes, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.title('Distribution of Molecule Sizes')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'molecule_sizes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot atom type distribution
    plt.figure(figsize=(10, 6))
    elements = list(atom_counts.keys())
    counts = list(atom_counts.values())
    plt.bar(elements, counts, alpha=0.7, edgecolor='black')
    plt.xlabel('Element')
    plt.ylabel('Count')
    plt.title('Distribution of Atom Types')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(args.output_dir, 'atom_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plotting {args.n_examples} example molecules...")
    # Plot example molecules
    n_molecules = generated_molecules['one_hot'].shape[0]
    example_indices = np.random.choice(n_molecules, min(args.n_examples, n_molecules), replace=False)
    
    for i, mol_idx in enumerate(example_indices):
        one_hot = generated_molecules['one_hot'][mol_idx]
        positions = generated_molecules['x'][mol_idx]
        mask = generated_molecules['node_mask'][mol_idx]
        
        # Get valid atoms
        n_atoms = int(mask.sum())
        if n_atoms > 0:
            valid_positions = positions[:n_atoms]
            atom_types = torch.argmax(one_hot[:n_atoms], dim=-1)
            
            # Plot molecule
            fig = plot_molecule_3d(valid_positions, atom_types, dataset_info, 
                                 title=f"Generated Molecule {mol_idx} ({n_atoms} atoms)")
            fig.savefig(os.path.join(args.output_dir, f'molecule_{mol_idx:04d}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # Print summary
    print(f"\nGenerated Molecules Summary:")
    print(f"Total molecules: {n_molecules}")
    print(f"Average size: {np.mean(molecule_sizes):.1f} atoms")
    print(f"Size range: {min(molecule_sizes)} - {max(molecule_sizes)} atoms")
    print(f"\nAtom counts:")
    for element, count in atom_counts.items():
        print(f"  {element}: {count}")
    
    print(f"\nVisualizations saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
```

### Usage

```bash
# Visualize generated molecules
python visualize_ase.py \
    --generated_molecules generated_molecules/generated_molecules.pt \
    --reference_db molecules.db \
    --output_dir visualizations \
    --n_examples 20
```

## Complete Examples

### End-to-End Example: Small Molecules

Here's a complete example workflow for training on small molecules:

```bash
#!/bin/bash
# complete_example.sh - Complete workflow for EDM with ASE

# 1. Create sample database
python -c "
from qm9.ase_dataset import create_sample_ase_db
create_sample_ase_db('small_molecules.db', num_molecules=1000)
print('Created sample database with 1000 molecules')
"

# 2. Train model
python train_ase.py \
    --ase_db_path small_molecules.db \
    --exp_name small_molecules_experiment \
    --n_epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --nf 128 \
    --n_layers 4 \
    --test_epochs 10 \
    --save_model

# 3. Generate molecules
python sample_ase.py \
    --model_path outputs/small_molecules_experiment \
    --n_samples 1000 \
    --output_dir generated_small_molecules

# 4. Evaluate quality
python evaluate_ase.py \
    --generated_molecules generated_small_molecules/generated_molecules.pt \
    --reference_db small_molecules.db \
    --output_file small_molecules_evaluation.txt

# 5. Visualize results
python visualize_ase.py \
    --generated_molecules generated_small_molecules/generated_molecules.pt \
    --reference_db small_molecules.db \
    --output_dir small_molecules_viz \
    --n_examples 15

echo "Complete workflow finished!"
echo "Check results in:"
echo "  - outputs/small_molecules_experiment/ (model)"
echo "  - generated_small_molecules/ (generated molecules)"
echo "  - small_molecules_evaluation.txt (evaluation results)"
echo "  - small_molecules_viz/ (visualizations)"
```

### Large-Scale Example: Custom Dataset

For larger datasets, you might want to use more sophisticated parameters:

```bash
# Large-scale training example
python train_ase.py \
    --ase_db_path large_dataset.db \
    --exp_name large_scale_experiment \
    --n_epochs 2000 \
    --batch_size 64 \
    --lr 1e-4 \
    --nf 256 \
    --n_layers 9 \
    --diffusion_steps 1000 \
    --test_epochs 20 \
    --ema_decay 0.9999 \
    --save_model \
    --normalize_factors 1 4 10 \
    --n_stability_samples 1000
```

## Advanced Usage

### Custom Dataset Configuration

You can customize the dataset configuration by modifying the dataset info:

```python
from qm9.ase_dataset import get_ase_dataset_info

# Get default configuration
dataset_info = get_ase_dataset_info('molecules.db')

# Customize colors for visualization
dataset_info['colors_dic'] = ['red', 'blue', 'green', 'orange', 'purple']

# Customize atomic radii for visualization  
dataset_info['radius_dic'] = [0.5, 0.7, 0.6, 0.8, 0.9]

# Add custom properties
dataset_info['custom_property'] = 'my_property'
```

### Conditional Generation

To enable conditional generation based on molecular properties:

```python
# Modify training script to include property conditioning
parser.add_argument('--conditioning', type=str, default=None,
                    help='Property to condition on (e.g., "energy", "size")')

# In the dataset loading section, extract the property values
if args.conditioning:
    # Extract property values from ASE database
    from ase.db import connect
    db = connect(args.ase_db_path)
    property_values = []
    for row in db.select():
        if args.conditioning in row.data:
            property_values.append(row.data[args.conditioning])
    # Use these values for conditional training
```

### Multi-GPU Training

For large-scale training on multiple GPUs:

```python
# Add to training script
parser.add_argument('--dp', action='store_true',
                    help='Enable data parallel training')

# In main function
if args.dp and torch.cuda.device_count() > 1:
    print(f'Training using {torch.cuda.device_count()} GPUs')
    model = torch.nn.DataParallel(model)
    model = model.cuda()
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

If you encounter out-of-memory errors:

```bash
# Reduce batch size
--batch_size 16

# Reduce model size
--nf 128 --n_layers 4

# Reduce maximum atoms
--max_atoms 50
```

#### 2. Database Loading Issues

```python
# Check database format
from ase.db import connect
db = connect('molecules.db')
print(f"Database contains {len(db)} entries")
for i, row in enumerate(db.select(limit=3)):
    atoms = row.toatoms()
    print(f"Entry {i}: {len(atoms)} atoms, symbols: {atoms.get_chemical_symbols()}")
```

#### 3. Model Loading Issues

```python
# Check model files
import os
model_path = 'outputs/my_experiment'
required_files = ['generative_model_ema.npy', 'args.pickle']
for file in required_files:
    filepath = os.path.join(model_path, file)
    if os.path.exists(filepath):
        print(f"✓ {file} exists")
    else:
        print(f"✗ {file} missing")
```

#### 4. OpenBabel Issues

```python
# Test OpenBabel installation
try:
    from openbabel import openbabel
    print("✓ OpenBabel imported successfully")
except ImportError:
    print("✗ OpenBabel import failed")
    print("Install with: pip install openbabel-wheel")
```

### Performance Tips

1. **Dataset Size**: Start with smaller datasets (< 10k molecules) for initial experiments
2. **Batch Size**: Use batch sizes that are multiples of 8 for optimal GPU utilization
3. **Model Size**: Balance between model capacity and training time
4. **Evaluation Frequency**: Test less frequently during long training runs

### Getting Help

If you encounter issues:

1. Check the console output for specific error messages
2. Verify your ASE database format and content
3. Ensure all dependencies are properly installed
4. Start with the provided example scripts
5. Check the original QM9 and GEOM examples for reference

## Summary

This tutorial provides a complete workflow for using E3 Diffusion for Molecules with ASE databases:

1. **Setup**: Install dependencies and create ASE databases
2. **Training**: Use the training script to learn molecular distributions
3. **Generation**: Sample new molecules from trained models
4. **Evaluation**: Assess quality using stability and molecular metrics
5. **Visualization**: Create plots and 3D visualizations

The ASE integration allows you to work with any molecular dataset that can be stored in ASE format, providing the same capabilities as the QM9 and GEOM workflows while supporting a much broader range of molecular systems.

For more details on specific components, refer to the existing documentation and code examples in the repository.