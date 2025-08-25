#!/usr/bin/env python3
"""
Test script for ASE database support and OpenBabel functionality.
"""

import sys
import os
import torch
import numpy as np
import tempfile

# Add current directory to path
sys.path.insert(0, '/home/runner/work/e3_diffusion_for_molecules/e3_diffusion_for_molecules')

def test_openbabel_functions():
    """Test OpenBabel molecular functions."""
    print("Testing OpenBabel functions...")
    
    try:
        from qm9.openbabel_functions import build_molecule_ob, BasicMolecularMetricsOB
        from configs.datasets_config import qm9_with_h
        
        # Test with a simple water molecule
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        atom_types = torch.tensor([3, 0, 0])  # O, H, H (based on qm9_with_h encoding)
        
        dataset_info = qm9_with_h
        
        # Test molecule building
        mol = build_molecule_ob(positions, atom_types, dataset_info)
        
        if mol is not None:
            print(f"✓ Successfully built molecule with {mol.NumAtoms()} atoms")
        else:
            print("✗ Failed to build molecule")
            return False
        
        # Test molecular metrics
        metrics = BasicMolecularMetricsOB(dataset_info)
        generated = [(positions, atom_types)]
        
        result = metrics.evaluate(generated)
        if result is not None:
            print("✓ Successfully computed molecular metrics")
        else:
            print("✗ Failed to compute molecular metrics")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ OpenBabel functions test failed: {e}")
        return False


def test_ase_dataset():
    """Test ASE dataset loading."""
    print("\nTesting ASE dataset...")
    
    try:
        from qm9.ase_dataset import create_sample_ase_db, load_ase_dataset
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        # Create sample database
        create_sample_ase_db(db_path, num_molecules=10)
        
        # Test loading
        class TestConfig:
            batch_size = 2
            num_workers = 0
            max_atoms = 20
            include_charges = True
            dataset = 'ase_test'
            ase_db_path = db_path
        
        config = TestConfig()
        dataloaders, charge_scale = load_ase_dataset(db_path, config)
        
        print(f"✓ Successfully loaded ASE dataset with {len(dataloaders)} splits")
        
        # Test iteration
        for split, loader in dataloaders.items():
            for i, batch in enumerate(loader):
                print(f"✓ {split} batch {i}: positions {batch['positions'].shape}")
                break  # Only test first batch
        
        # Cleanup
        os.unlink(db_path)
        return True
        
    except Exception as e:
        print(f"✗ ASE dataset test failed: {e}")
        return False


def test_dataset_integration():
    """Test integration with dataset loading."""
    print("\nTesting dataset integration...")
    
    try:
        from qm9.ase_dataset import create_sample_ase_db
        from qm9.dataset import retrieve_dataloaders
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        create_sample_ase_db(db_path, num_molecules=10)
        
        # Test with dataset loader
        class Config:
            dataset = 'ase_test'
            batch_size = 2
            num_workers = 0
            include_charges = True
            ase_db_path = db_path
            max_atoms = 20
        
        config = Config()
        dataloaders, charge_scale = retrieve_dataloaders(config)
        
        print("✓ Successfully integrated ASE dataset with main dataset loader")
        
        # Test with analyze functions
        from qm9.analyze import analyze_stability_for_molecules
        
        # Get a sample from the dataset
        train_loader = dataloaders['train']
        for batch in train_loader:
            # Convert batch to molecule list format expected by analyze_stability_for_molecules
            batch_size = batch['positions'].shape[0]
            molecules = {
                'one_hot': [],
                'x': [],
                'node_mask': []
            }
            
            for i in range(batch_size):
                # Extract each molecule maintaining the right dimensions
                molecules['one_hot'].append(batch['one_hot'][i])  # Shape: [max_atoms, n_atom_types]
                molecules['x'].append(batch['positions'][i])      # Shape: [max_atoms, 3]
                molecules['node_mask'].append(batch['atom_mask'][i])  # Shape: [max_atoms]
            break
        
        # Get dataset info from loader
        dataset_info = train_loader.dataset_info
        
        validity_dict, metrics = analyze_stability_for_molecules(molecules, dataset_info)
        print("✓ Successfully analyzed molecules using OpenBabel/stability functions")
        
        # Cleanup
        os.unlink(db_path)
        return True
        
    except Exception as e:
        print(f"✗ Dataset integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running ASE DB and OpenBabel integration tests...")
    print("=" * 50)
    
    tests = [
        test_openbabel_functions,
        test_ase_dataset,
        test_dataset_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("Test Results:")
    test_names = [
        "OpenBabel Functions",
        "ASE Dataset Loading", 
        "Dataset Integration"
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)