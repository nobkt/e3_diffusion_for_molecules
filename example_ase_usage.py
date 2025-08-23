#!/usr/bin/env python3
"""
Example usage of ASE dataset loading functionality.

This script demonstrates how to use the new ASE database loading feature
in the e3_diffusion_for_molecules repository.
"""

import os
import tempfile
import ase
import ase.db
from ase import Atoms
import numpy as np
import torch
from types import SimpleNamespace

from qm9.dataset import retrieve_dataloaders
from configs.datasets_config import get_dataset_info


def create_example_molecules_db(db_path):
    """Create an example ASE database with some realistic small molecules."""
    db = ase.db.connect(db_path)
    
    # Define some example molecules
    molecules = [
        # Water
        {'symbols': ['O', 'H', 'H'], 
         'positions': [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]},
        
        # Methane
        {'symbols': ['C', 'H', 'H', 'H', 'H'],
         'positions': [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629], 
                      [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629], 
                      [0.629, -0.629, -0.629]]},
        
        # Ammonia  
        {'symbols': ['N', 'H', 'H', 'H'],
         'positions': [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
                      [0.866, -0.5, 0.0], [-0.866, -0.5, 0.0]]},
        
        # Carbon dioxide
        {'symbols': ['C', 'O', 'O'],
         'positions': [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]},
    ]
    
    # Add some random variations of these molecules
    np.random.seed(42)
    for _ in range(10):
        for mol_template in molecules:
            symbols = mol_template['symbols']
            positions = np.array(mol_template['positions'])
            
            # Add small random perturbations
            positions += np.random.normal(0, 0.1, positions.shape)
            
            atoms = Atoms(symbols=symbols, positions=positions)
            db.write(atoms)
    
    print(f"Created example ASE database with {len(db)} molecules at {db_path}")
    return db_path


def example_usage():
    """Demonstrate how to use ASE dataset loading."""
    print("ASE Dataset Loading Example")
    print("=" * 40)
    
    # Create a temporary database for this example
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create example database
        create_example_molecules_db(db_path)
        print(f"Created example database: {db_path}")
        
        # Configure the dataset loading
        cfg = SimpleNamespace(
            dataset='ase',                    # Use ASE dataset
            ase_db_file=db_path,             # Path to ASE database file
            ase_max_entries=None,            # Load all entries (optional limit)
            remove_h=False,                  # Keep hydrogen atoms
            include_charges=False,           # Don't use charges
            device=torch.device('cpu'),      # Use CPU
            sequential=False,                # Use random batching
            batch_size=8,                    # Batch size
            filter_molecule_size=None        # No size filtering
        )
        
        print("\nConfiguration:")
        for key, value in vars(cfg).items():
            print(f"  {key}: {value}")
        
        # Load dataloaders
        print("\nLoading dataloaders...")
        dataloaders, charge_scale = retrieve_dataloaders(cfg)
        
        print(f"Successfully created dataloaders: {list(dataloaders.keys())}")
        print(f"Charge scale: {charge_scale}")
        
        # Demonstrate usage of dataloaders
        print("\nDataloader information:")
        for split_name, dataloader in dataloaders.items():
            print(f"  {split_name}: {len(dataloader.dataset)} molecules, {len(dataloader)} batches")
        
        # Get a sample batch
        print("\nSample batch from training data:")
        train_loader = dataloaders['train']
        sample_batch = next(iter(train_loader))
        
        print(f"  Batch keys: {list(sample_batch.keys())}")
        print(f"  Positions shape: {sample_batch['positions'].shape}")
        print(f"  One-hot encoding shape: {sample_batch['one_hot'].shape}")
        print(f"  Atom mask shape: {sample_batch['atom_mask'].shape}")
        print(f"  Edge mask shape: {sample_batch['edge_mask'].shape}")
        
        # Show molecular composition
        one_hot = sample_batch['one_hot']
        atom_types = torch.argmax(one_hot, dim=-1)
        atom_mask = sample_batch['atom_mask']
        
        dataset_info = get_dataset_info('ase', remove_h=False)
        atom_decoder = dataset_info['atom_decoder']
        
        print(f"\nMolecules in first batch:")
        batch_size = atom_types.shape[0]
        for mol_idx in range(min(3, batch_size)):  # Show first 3 molecules
            mol_atom_types = atom_types[mol_idx]
            mol_mask = atom_mask[mol_idx]
            valid_atoms = mol_atom_types[mol_mask.bool()]
            
            atom_symbols = [atom_decoder[idx.item()] for idx in valid_atoms]
            print(f"  Molecule {mol_idx + 1}: {' '.join(atom_symbols)}")
        
        print("\n✅ ASE dataset loading completed successfully!")
        
    finally:
        # Clean up temporary file
        if os.path.exists(db_path):
            os.unlink(db_path)


def usage_with_real_database():
    """Show how to use with a real ASE database file."""
    print("\nUsage with Real ASE Database:")
    print("=" * 40)
    
    example_code = '''
# Example configuration for using your own ASE database
from types import SimpleNamespace
from qm9.dataset import retrieve_dataloaders

# Configure for your ASE database
cfg = SimpleNamespace(
    dataset='ase',
    ase_db_file='/path/to/your/molecules.db',   # Your ASE database file
    ase_max_entries=10000,                      # Optional: limit number of molecules
    remove_h=False,                             # Whether to remove hydrogens
    include_charges=False,                      # Whether to include atomic charges
    device=torch.device('cuda'),                # Use GPU if available
    sequential=False,                           # Random vs sequential batching
    batch_size=32,                              # Adjust based on your memory
    filter_molecule_size=None                   # Optional: filter by atom count
)

# Load the dataloaders
dataloaders, charge_scale = retrieve_dataloaders(cfg)

# Use in training loop
for batch in dataloaders['train']:
    positions = batch['positions']      # Atomic positions [batch, n_atoms, 3]
    one_hot = batch['one_hot']         # Atom type encoding [batch, n_atoms, n_types]
    atom_mask = batch['atom_mask']     # Valid atom mask [batch, n_atoms]
    edge_mask = batch['edge_mask']     # Valid edge mask [batch*n_atoms*n_atoms, 1]
    
    # Your training code here...
    pass
'''
    
    print(example_code)
    
    print("Notes:")
    print("  - ASE database files typically have .db extension")
    print("  - The database should contain molecular structures with atomic positions")
    print("  - Supports automatic train/validation/test splitting")
    print("  - Compatible with existing diffusion model pipeline")


if __name__ == '__main__':
    example_usage()
    usage_with_real_database()