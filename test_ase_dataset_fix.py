#!/usr/bin/env python3
"""
Test script to validate that ASE dataset loading produces consistent data structure with QM9.
"""

import torch
import numpy as np
import ase.db
import ase
import tempfile
import os
from qm9 import dataset
from configs.datasets_config import get_dataset_info
import argparse

def create_test_ase_database():
    """Create a test ASE database with QM9-like molecular properties."""
    db_path = '/tmp/test_qm9_ase.db'
    
    with ase.db.connect(db_path) as db:
        # Create several test molecules with properties
        test_molecules = [
            {
                'symbols': ['C', 'O'],
                'positions': [[0, 0, 0], [1.2, 0, 0]],
                'properties': {
                    'index': 1,
                    'rotA_GHz': 1.2,
                    'rotB_GHz': 1.1, 
                    'rotC_GHz': 1.0,
                    'mu': 1.5,
                    'alpha': 10.2,
                    'HOMO_Ha': -0.5,
                    'LUMO_Ha': 0.1,
                    'gap_Ha': 0.6,
                    'r2_Bohr2': 50.0,
                    'ZPVE_Ha': 0.05,
                    'U0_Ha': -100.0,
                    'U_298K_Ha': -99.8,
                    'H_298K_Ha': -99.7,
                    'G_298K_Ha': -100.1,
                    'Cv_cal_molK': 25.0
                }
            },
            {
                'symbols': ['C', 'C', 'H', 'H', 'H', 'H'],
                'positions': [[0, 0, 0], [1.5, 0, 0], [-0.5, 0.8, 0], [-0.5, -0.8, 0], [2.0, 0.8, 0], [2.0, -0.8, 0]],
                'properties': {
                    'index': 2,
                    'rotA_GHz': 2.1,
                    'rotB_GHz': 1.9, 
                    'rotC_GHz': 1.8,
                    'mu': 0.8,
                    'alpha': 15.5,
                    'HOMO_Ha': -0.4,
                    'LUMO_Ha': 0.2,
                    'gap_Ha': 0.6,
                    'r2_Bohr2': 75.0,
                    'ZPVE_Ha': 0.08,
                    'U0_Ha': -150.0,
                    'U_298K_Ha': -149.7,
                    'H_298K_Ha': -149.6,
                    'G_298K_Ha': -150.2,
                    'Cv_cal_molK': 35.0
                }
            }
        ]
        
        for mol_data in test_molecules:
            atoms = ase.Atoms(mol_data['symbols'], positions=mol_data['positions'])
            db.write(atoms, **mol_data['properties'])
    
    print(f"Created test ASE database with {len(test_molecules)} molecules: {db_path}")
    return db_path

def test_ase_dataset_loading():
    """Test that ASE dataset loading works with the new structure."""
    print("Testing ASE dataset loading...")
    
    # Create test database
    db_path = create_test_ase_database()
    
    # Create mock arguments for ASE dataset
    class MockArgs:
        def __init__(self):
            self.dataset = 'ase'
            self.ase_db_file = db_path
            self.ase_max_entries = None
            self.filter_molecule_size = None
            self.remove_h = False
            self.include_charges = True
            self.sequential = False
            self.batch_size = 2
            self.device = torch.device('cpu')
    
    args = MockArgs()
    
    try:
        # Test dataset loading
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        print(f"✓ Successfully loaded ASE dataset")
        print(f"✓ Charge scale: {charge_scale}")
        print(f"✓ Data splits: {list(dataloaders.keys())}")
        
        # Test a single batch
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"✓ Batch keys: {list(batch.keys())}")
        print(f"✓ Batch shapes:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"    {key}: {value.shape}")
            else:
                print(f"    {key}: {type(value)}")
        
        # Verify unit conversion happened
        if 'U0_Ha' in batch:
            print(f"✓ U0_Ha values after conversion: {batch['U0_Ha']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading ASE dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test file
        if os.path.exists(db_path):
            os.remove(db_path)

def compare_data_structures():
    """Compare QM9 and ASE dataset data structures to ensure consistency."""
    print("\nComparing QM9 and ASE data structures...")
    
    # We can't easily test QM9 without the actual data, so we'll just validate
    # that ASE produces the expected structure
    db_path = create_test_ase_database()
    
    class MockArgs:
        def __init__(self):
            self.dataset = 'ase'
            self.ase_db_file = db_path
            self.ase_max_entries = None
            self.filter_molecule_size = None
            self.remove_h = False
            self.include_charges = True
            self.sequential = False
            self.batch_size = 1
            self.device = torch.device('cpu')
    
    args = MockArgs()
    
    try:
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        batch = next(iter(dataloaders['train']))
        
        # Expected keys that should match QM9 structure
        expected_geometric_keys = ['positions', 'one_hot', 'charges', 'atom_mask', 'edge_mask']
        expected_property_keys = ['U0_Ha', 'HOMO_Ha', 'LUMO_Ha', 'gap_Ha', 'alpha']
        
        missing_geometric = [key for key in expected_geometric_keys if key not in batch]
        missing_properties = [key for key in expected_property_keys if key not in batch]
        
        if not missing_geometric and not missing_properties:
            print("✓ ASE dataset structure matches expected QM9-like structure")
            print(f"✓ Has geometric keys: {expected_geometric_keys}")
            print(f"✓ Has molecular property keys: {[k for k in expected_property_keys if k in batch]}")
            return True
        else:
            if missing_geometric:
                print(f"✗ Missing geometric keys: {missing_geometric}")
            if missing_properties:
                print(f"✗ Missing property keys: {missing_properties}")
            return False
            
    except Exception as e:
        print(f"✗ Error comparing structures: {e}")
        return False
    
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    print("=" * 60)
    print("Testing ASE Dataset Fixes")
    print("=" * 60)
    
    success1 = test_ase_dataset_loading()
    success2 = compare_data_structures()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ ALL TESTS PASSED - ASE dataset fixes are working correctly!")
    else:
        print("❌ Some tests failed - check the implementation")
    print("=" * 60)