#!/usr/bin/env python3
"""
Example script demonstrating how to use ASE databases with the diffusion model.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qm9.ase_database import ASEDatabaseReader
from qm9 import dataset
from configs.datasets_config import get_dataset_info


def create_sample_ase_database(db_path, n_molecules=100):
    """Create a small sample ASE database for demonstration."""
    from ase import Atoms
    from ase.db import connect
    import numpy as np
    
    db = connect(db_path)
    
    # Create some simple molecules
    molecules = [
        # Water
        Atoms('H2O', positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]),
        # Methane  
        Atoms('CH4', positions=[[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], 
                                [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]]),
        # Ammonia
        Atoms('NH3', positions=[[0, 0, 0], [0.94, 0, 0], [-0.47, 0.81, 0], [-0.47, -0.81, 0]]),
    ]
    
    # Add random variations
    np.random.seed(42)
    for i in range(n_molecules):
        base_mol = molecules[i % len(molecules)]
        
        # Add small random perturbations to positions
        perturbed_pos = base_mol.positions + np.random.normal(0, 0.1, base_mol.positions.shape)
        
        mol = Atoms(symbols=base_mol.get_chemical_symbols(), positions=perturbed_pos)
        
        # Add some properties
        # For ASE, energy should be passed as a separate argument, not in kvp
        mol.calc = None  # Remove any calculator
        db.write(mol, mol_energy=np.random.uniform(-5, 0), custom_prop=i)
        
    print(f"Created sample ASE database with {len(db)} molecules at {db_path}")


def example_ase_usage():
    """Demonstrate ASE database usage."""
    
    # 1. Create a sample database
    sample_db_path = './example_molecules.db'
    create_sample_ase_database(sample_db_path, n_molecules=50)
    
    # 2. Create ASE reader
    ase_reader = ASEDatabaseReader(sample_db_path)
    
    print("ASE Database Info:")
    print(f"  Atom types: {ase_reader.atom_types}")
    print(f"  Atom encoder: {ase_reader.atom_encoder}")
    
    # 3. Get dataset info
    dataset_info = ase_reader.get_dataset_info('example_molecules', with_h=True)
    print(f"  Max nodes: {dataset_info['max_n_nodes']}")
    print(f"  Node distribution: {dataset_info['n_nodes']}")
    
    # 4. Create train/validation/test splits
    data_splits = ase_reader.create_splits(train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1)
    
    for split_name, split_data in data_splits.items():
        print(f"{split_name} split: {len(split_data['positions'])} molecules")
        
    # 5. Show how to use with dataloader
    class ASEConfig:
        def __init__(self):
            self.dataset = 'ase_example'
            self.ase_db_path = sample_db_path
            self.batch_size = 8
            self.include_charges = True
            self.remove_h = False
            self.datadir = './temp'
            self.num_workers = 0
            
    cfg = ASEConfig()
    
    try:
        dataloaders, charge_scale = dataset.retrieve_ase_dataloaders(cfg)
        
        print("\nDataloaders created successfully:")
        for split_name, dataloader in dataloaders.items():
            print(f"  {split_name}: {len(dataloader)} batches")
            
        # Test loading one batch
        train_loader = dataloaders['train']
        for batch in train_loader:
            print(f"\nSample batch:")
            print(f"  Positions shape: {batch['positions'].shape}")
            print(f"  Charges shape: {batch['charges'].shape}")
            print(f"  One-hot shape: {batch['one_hot'].shape}")
            print(f"  Atom mask shape: {batch['atom_mask'].shape}")
            break
            
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        
    # Clean up
    if os.path.exists(sample_db_path):
        os.remove(sample_db_path)
        

if __name__ == "__main__":
    example_ase_usage()