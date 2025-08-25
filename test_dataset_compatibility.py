#!/usr/bin/env python3
"""
Test script to validate that ASE and QM9 datasets produce compatible data structures.
"""

import torch
import numpy as np
import ase.db
import ase
import tempfile
import os
from qm9 import dataset
from configs.datasets_config import get_dataset_info

def create_qm9_style_ase_database():
    """Create an ASE database that mimics QM9 molecules and properties."""
    db_path = '/tmp/qm9_style_ase.db'
    
    with ase.db.connect(db_path) as db:
        # QM9-style molecules (small organic molecules with H, C, N, O, F)
        test_molecules = [
            # Methane-like
            {
                'symbols': ['C', 'H', 'H', 'H', 'H'],
                'positions': [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]],
                'properties': {
                    'index': 1,
                    'rotA_GHz': 157.7,
                    'rotB_GHz': 157.7, 
                    'rotC_GHz': 157.7,
                    'mu': 0.0,
                    'alpha': 17.3,
                    'HOMO_Ha': -0.5998,
                    'LUMO_Ha': 0.0618,
                    'gap_Ha': 0.6616,
                    'r2_Bohr2': 35.36,
                    'ZPVE_Ha': 0.044749,
                    'U0_Ha': -40.47893,
                    'U_298K_Ha': -40.476062,
                    'H_298K_Ha': -40.475117,
                    'G_298K_Ha': -40.498597,
                    'Cv_cal_molK': 6.469
                }
            },
            # Water-like
            {
                'symbols': ['O', 'H', 'H'],
                'positions': [[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]],
                'properties': {
                    'index': 2,
                    'rotA_GHz': 835.51,
                    'rotB_GHz': 438.28, 
                    'rotC_GHz': 290.498,
                    'mu': 1.8546,
                    'alpha': 9.46,
                    'HOMO_Ha': -0.5875,
                    'LUMO_Ha': 0.0829,
                    'gap_Ha': 0.6704,
                    'r2_Bohr2': 23.22,
                    'ZPVE_Ha': 0.021375,
                    'U0_Ha': -76.404702,
                    'U_298K_Ha': -76.403446,
                    'H_298K_Ha': -76.402502,
                    'G_298K_Ha': -76.419745,
                    'Cv_cal_molK': 6.002
                }
            },
            # Ammonia-like
            {
                'symbols': ['N', 'H', 'H', 'H'],
                'positions': [[0, 0, 0], [0.9, 0.3, 0.3], [-0.3, 0.9, 0.3], [-0.3, -0.3, 0.9]],
                'properties': {
                    'index': 3,
                    'rotA_GHz': 298.198,
                    'rotB_GHz': 298.198, 
                    'rotC_GHz': 298.198,
                    'mu': 1.472,
                    'alpha': 14.8,
                    'HOMO_Ha': -0.4887,
                    'LUMO_Ha': 0.0278,
                    'gap_Ha': 0.5165,
                    'r2_Bohr2': 18.7,
                    'ZPVE_Ha': 0.034358,
                    'U0_Ha': -56.525887,
                    'U_298K_Ha': -56.523026,
                    'H_298K_Ha': -56.522082,
                    'G_298K_Ha': -56.544961,
                    'Cv_cal_molK': 6.316
                }
            }
        ]
        
        for mol_data in test_molecules:
            atoms = ase.Atoms(mol_data['symbols'], positions=mol_data['positions'])
            db.write(atoms, **mol_data['properties'])
    
    print(f"Created QM9-style ASE database with {len(test_molecules)} molecules: {db_path}")
    return db_path

def test_dataloader_consistency():
    """Test that ASE dataloaders produce data consistent with expected QM9 format."""
    print("Testing ASE dataloader consistency with QM9 format...")
    
    db_path = create_qm9_style_ase_database()
    
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
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        # Test multiple batches to ensure consistency
        train_loader = dataloaders['train']
        batches = []
        for i, batch in enumerate(train_loader):
            batches.append(batch)
            if i >= 1:  # Test first 2 batches
                break
        
        if not batches:
            print("✗ No batches generated")
            return False
        
        # Validate batch structure
        batch = batches[0]
        required_keys = ['positions', 'one_hot', 'charges', 'atom_mask', 'edge_mask']
        property_keys = ['U0', 'HOMO', 'LUMO', 'gap', 'alpha']  # Updated to QM9 format
        
        # Check required geometric keys
        for key in required_keys:
            if key not in batch:
                print(f"✗ Missing required key: {key}")
                return False
            if not torch.is_tensor(batch[key]):
                print(f"✗ Key {key} is not a tensor")
                return False
        
        # Check molecular property keys
        missing_props = [key for key in property_keys if key not in batch]
        if missing_props:
            print(f"✗ Missing molecular property keys: {missing_props}")
            return False
        
        # Validate tensor shapes and values
        batch_size = batch['positions'].shape[0]
        n_nodes = batch['positions'].shape[1]
        
        # Position shape: [batch_size, n_nodes, 3]
        if batch['positions'].shape != (batch_size, n_nodes, 3):
            print(f"✗ Unexpected positions shape: {batch['positions'].shape}")
            return False
        
        # One-hot encoding shape: [batch_size, n_nodes, n_atom_types]
        if len(batch['one_hot'].shape) != 3:
            print(f"✗ Unexpected one_hot shape: {batch['one_hot'].shape}")
            return False
        
        # Charges shape: [batch_size, n_nodes, 1]
        if batch['charges'].shape != (batch_size, n_nodes, 1):
            print(f"✗ Unexpected charges shape: {batch['charges'].shape}")
            return False
        
        # Atom mask shape: [batch_size, n_nodes]
        if batch['atom_mask'].shape != (batch_size, n_nodes):
            print(f"✗ Unexpected atom_mask shape: {batch['atom_mask'].shape}")
            return False
        
        # Edge mask shape: [batch_size * n_nodes * n_nodes, 1]
        expected_edge_shape = (batch_size * n_nodes * n_nodes, 1)
        if batch['edge_mask'].shape != expected_edge_shape:
            print(f"✗ Unexpected edge_mask shape: {batch['edge_mask'].shape}, expected: {expected_edge_shape}")
            return False
        
        # Molecular properties should be [batch_size] tensors
        for prop_key in property_keys:
            if prop_key in batch:
                if batch[prop_key].shape != (batch_size,):
                    print(f"✗ Unexpected shape for {prop_key}: {batch[prop_key].shape}")
                    return False
        
        # Validate unit conversion by checking reasonable value ranges
        # U0 should be in eV after conversion (negative values around -1000 to -4000 eV for small molecules)
        if 'U0' in batch:
            u0_values = batch['U0']
            if not torch.all((u0_values < 0) & (u0_values > -10000)):
                print(f"✗ U0 values seem unconverted or unreasonable: {u0_values}")
                return False
        
        # HOMO/LUMO should be in eV (HOMO negative, LUMO could be positive)
        if 'HOMO' in batch and 'LUMO' in batch:
            homo_values = batch['HOMO']
            lumo_values = batch['LUMO']
            if not torch.all(homo_values < 0):
                print(f"✗ HOMO values should be negative: {homo_values}")
                return False
        
        print("✓ All dataloader consistency tests passed")
        print(f"✓ Batch size: {batch_size}, Max nodes: {n_nodes}")
        print(f"✓ Charge scale: {charge_scale}")
        print(f"✓ Sample U0 values (eV): {batch['U0'][:min(3, batch_size)]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in dataloader consistency test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_property_scaling():
    """Test that property values are correctly scaled to match QM9 expectations."""
    print("\nTesting property scaling consistency...")
    
    db_path = create_qm9_style_ase_database()
    
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
        
        # Check specific conversion factors
        # Original U0_Ha: -40.47893 Ha -> should become -40.47893 * 27.2114 = -1101.6 eV
        # Original ZPVE_Ha: 0.044749 Ha -> should become 0.044749 * 27211.4 = 1217.8 cm⁻¹
        
        if 'U0' in batch:
            u0_val = batch['U0'][0].item()
            expected_range = (-3000, -500)  # Broader range for small molecules in eV
            if not (expected_range[0] < u0_val < expected_range[1]):
                print(f"✗ U0 value {u0_val} not in expected range {expected_range}")
                return False
        
        if 'zpve' in batch:
            zpve_val = batch['zpve'][0].item()
            expected_range = (400, 2000)  # Broader range for ZPVE in cm⁻¹ for small molecules
            if not (expected_range[0] < zpve_val < expected_range[1]):
                print(f"✗ zpve value {zpve_val} not in expected range {expected_range}")
                return False
        
        print("✓ Property scaling tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error in property scaling test: {e}")
        return False
    
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Dataset Compatibility: ASE vs QM9 Structure")
    print("=" * 70)
    
    success1 = test_dataloader_consistency()
    success2 = test_property_scaling()
    
    print("\n" + "=" * 70)
    if success1 and success2:
        print("✅ ALL COMPATIBILITY TESTS PASSED!")
        print("ASE datasets should now produce the same data structure and")
        print("value ranges as QM9 datasets, eliminating the 10x loss difference.")
    else:
        print("❌ Some compatibility tests failed - check the implementation")
    print("=" * 70)