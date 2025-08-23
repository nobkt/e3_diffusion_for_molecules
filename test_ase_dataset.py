#!/usr/bin/env python3
"""
Simple test script for ASE dataset functionality.
Creates a test ASE database and validates the dataset loading pipeline.
"""

import os
import tempfile
import ase
import ase.db
from ase import Atoms
import numpy as np
import torch
from types import SimpleNamespace

import build_ase_dataset
from configs.datasets_config import get_dataset_info
from qm9.dataset import retrieve_dataloaders


def create_test_ase_db(db_path, n_molecules=20):
    """Create a small test ASE database with sample molecules."""
    db = ase.db.connect(db_path)
    
    np.random.seed(42)
    
    for i in range(n_molecules):
        # Create random small molecules (2-8 atoms)
        n_atoms = np.random.randint(2, 9)
        
        # Mix of common atom types
        atom_types = np.random.choice(['H', 'C', 'N', 'O'], size=n_atoms, 
                                    p=[0.4, 0.3, 0.2, 0.1])
        
        # Random positions in a small box
        positions = np.random.randn(n_atoms, 3) * 2.0
        
        # Create atoms object
        atoms = Atoms(symbols=atom_types, positions=positions)
        
        # Add to database
        db.write(atoms)
    
    print(f"Created test ASE database with {n_molecules} molecules at {db_path}")
    return db_path


def test_ase_data_loading():
    """Test ASE data loading functionality."""
    print("Testing ASE data loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test database
        db_path = os.path.join(temp_dir, "test_molecules.db")
        create_test_ase_db(db_path, n_molecules=20)
        
        # Test load_ase_data
        data_list = build_ase_dataset.load_ase_data(db_path, max_entries=10)
        assert len(data_list) == 10
        assert all(isinstance(mol, np.ndarray) for mol in data_list)
        assert all(mol.shape[1] == 4 for mol in data_list)  # [atomic_number, x, y, z]
        print(f"✓ load_ase_data: loaded {len(data_list)} molecules")
        
        # Test load_split_data
        train_data, val_data, test_data = build_ase_dataset.load_split_data(
            db_path, val_proportion=0.2, test_proportion=0.2, max_entries=15
        )
        assert len(train_data) + len(val_data) + len(test_data) == 15
        print(f"✓ load_split_data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Test dataset creation
        dataset_info = get_dataset_info('ase', remove_h=False)
        transform = build_ase_dataset.ASETransform(
            dataset_info, include_charges=False, device=torch.device('cpu'), sequential=False
        )
        
        dataset = build_ase_dataset.ASEDataset(train_data, transform=transform)
        assert len(dataset) == len(train_data)
        
        # Test getting a sample
        sample = dataset[0]
        assert 'positions' in sample
        assert 'one_hot' in sample
        assert 'charges' in sample
        assert 'atom_mask' in sample
        print(f"✓ ASEDataset: sample keys = {list(sample.keys())}")
        
        # Test dataloader
        dataloader = build_ase_dataset.ASEDataLoader(
            sequential=False, dataset=dataset, batch_size=2, shuffle=False
        )
        
        batch = next(iter(dataloader))
        assert 'positions' in batch
        assert 'edge_mask' in batch
        print(f"✓ ASEDataLoader: batch keys = {list(batch.keys())}")
        
        print("All ASE dataset loading tests passed!")


def test_retrieve_dataloaders():
    """Test integration with main retrieve_dataloaders function."""
    print("\nTesting retrieve_dataloaders integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test database
        db_path = os.path.join(temp_dir, "test_molecules.db")
        create_test_ase_db(db_path, n_molecules=50)
        
        # Create mock config
        cfg = SimpleNamespace(
            dataset='ase',
            ase_db_file=db_path,
            ase_max_entries=30,
            remove_h=False,
            include_charges=False,
            device=torch.device('cpu'),
            sequential=False,
            batch_size=4,
            filter_molecule_size=None
        )
        
        try:
            dataloaders, charge_scale = retrieve_dataloaders(cfg)
            
            assert 'train' in dataloaders
            assert 'valid' in dataloaders  
            assert 'test' in dataloaders
            assert charge_scale is None
            
            # Test that we can iterate through a dataloader
            train_loader = dataloaders['train']
            batch = next(iter(train_loader))
            assert 'positions' in batch
            assert 'one_hot' in batch
            assert 'edge_mask' in batch
            
            print(f"✓ retrieve_dataloaders: created dataloaders for {len(dataloaders)} splits")
            print(f"✓ train batch shape: positions={batch['positions'].shape}, one_hot={batch['one_hot'].shape}")
            
            print("Integration test passed!")
            
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            raise


if __name__ == '__main__':
    test_ase_data_loading()
    test_retrieve_dataloaders()
    print("\n🎉 All tests passed! ASE dataset loading is working correctly.")