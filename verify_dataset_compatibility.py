#!/usr/bin/env python3
"""
Focused verification script to check data structure compatibility between QM9 and ASE datasets.

This script creates a minimal test to identify the specific root causes of high loss values 
when using dataset=ase. It checks the key structural differences that affect training.
"""

import torch
import numpy as np
import ase.db
import ase
import tempfile
import os
from qm9 import dataset
from configs.datasets_config import get_dataset_info

def create_simple_ase_test():
    """Create a simple ASE test database."""
    db_path = '/tmp/simple_ase_test.db'
    
    with ase.db.connect(db_path) as db:
        # Simple water molecule
        atoms = ase.Atoms('OH2', positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        db.write(atoms, U0_Ha=-76.4, HOMO_Ha=-0.59, LUMO_Ha=0.08, gap_Ha=0.67, alpha=9.5)
        
        # Simple methane
        atoms = ase.Atoms('CH4', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-0.5, -0.5, -0.5]])
        db.write(atoms, U0_Ha=-40.5, HOMO_Ha=-0.60, LUMO_Ha=0.06, gap_Ha=0.66, alpha=17.3)
    
    print(f"Created simple ASE test database: {db_path}")
    return db_path

def get_ase_batch():
    """Get a batch from ASE dataset."""
    db_path = create_simple_ase_test()
    
    class ASEArgs:
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
    
    try:
        args = ASEArgs()
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        batch = next(iter(dataloaders['train']))
        return batch, 'ase'
    except Exception as e:
        print(f"Error loading ASE: {e}")
        return None, None
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def check_dataset_compatibility():
    """Main verification function."""
    print("="*70)
    print("QM9 vs ASE Dataset Structure Verification")
    print("="*70)
    
    # Get ASE batch
    ase_batch, dataset_name = get_ase_batch()
    if ase_batch is None:
        print("❌ Failed to load ASE dataset")
        return False
    
    print(f"\n✓ Loaded {dataset_name} dataset successfully")
    print(f"  Batch size: {ase_batch['positions'].shape[0]}")
    print(f"  Max nodes: {ase_batch['positions'].shape[1]}")
    
    # Check for critical structural issues
    issues = []
    
    # 1. Check one-hot encoding dimensions
    one_hot_shape = ase_batch['one_hot'].shape
    expected_atom_types = 5  # QM9 standard: H, C, N, O, F
    actual_atom_types = one_hot_shape[2]
    
    print(f"\n1. One-hot encoding check:")
    print(f"   Expected atom types (QM9): {expected_atom_types}")
    print(f"   Actual atom types (ASE): {actual_atom_types}")
    
    if actual_atom_types != expected_atom_types:
        issues.append(f"One-hot encoding mismatch: ASE has {actual_atom_types} atom types, QM9 expects {expected_atom_types}")
        print(f"   ❌ CRITICAL: Dimension mismatch!")
    else:
        print(f"   ✓ Match")
    
    # 2. Check property keys
    qm9_properties = ['U0', 'HOMO', 'LUMO', 'gap', 'alpha']
    ase_properties = ['U0_Ha', 'HOMO_Ha', 'LUMO_Ha', 'gap_Ha', 'alpha']
    
    print(f"\n2. Property key check:")
    for qm9_prop, ase_prop in zip(qm9_properties, ase_properties):
        qm9_exists = qm9_prop in ase_batch
        ase_exists = ase_prop in ase_batch
        print(f"   QM9 key '{qm9_prop}': {qm9_exists}, ASE key '{ase_prop}': {ase_exists}")
        
        if not qm9_exists and ase_exists:
            issues.append(f"Property key mismatch: found '{ase_prop}' but model expects '{qm9_prop}'")
    
    # 3. Check data types
    print(f"\n3. Data type check:")
    critical_tensors = ['positions', 'charges', 'one_hot']
    for tensor_name in critical_tensors:
        if tensor_name in ase_batch:
            dtype = ase_batch[tensor_name].dtype
            expected_dtype = torch.float32
            print(f"   {tensor_name}: {dtype} (expected: {expected_dtype})")
            
            if dtype != expected_dtype:
                issues.append(f"Data type mismatch for {tensor_name}: {dtype} vs expected {expected_dtype}")
    
    # 4. Check value ranges
    print(f"\n4. Value range check:")
    if 'charges' in ase_batch:
        charges = ase_batch['charges']
        charge_min, charge_max = charges.min().item(), charges.max().item()
        print(f"   Charges range: [{charge_min:.1f}, {charge_max:.1f}]")
        
        # QM9 without hydrogen should have charges 6-9 (C, N, O, F)
        # QM9 with hydrogen should have charges 1-9 (H, C, N, O, F)
        # Note: We'll check for zero charges only in real atoms later
        
        # Check if hydrogen is included when it shouldn't be
        has_hydrogen = (charges == 1).any()
        print(f"   Contains hydrogen: {has_hydrogen}")
    
    # 5. Property conversion check
    print(f"\n5. Property conversion check:")
    conversion_eV = 27.2114  # Hartree to eV conversion
    
    for prop in ['U0', 'HOMO', 'LUMO']:  # Use QM9 property names
        if prop in ase_batch:
            values = ase_batch[prop]
            print(f"   {prop}: min={values.min().item():.2f}, max={values.max().item():.2f}")
            
            # Check if values are in eV range (should be negative for U0, HOMO and larger in magnitude)
            if prop == 'U0':
                if values.max().item() > -100:  # Should be very negative in eV
                    issues.append(f"{prop} values appear unconverted (should be highly negative in eV)")
            elif prop == 'HOMO':
                if values.max().item() > -5:  # HOMO should be negative in eV
                    issues.append(f"{prop} values appear unconverted (should be negative in eV)")
        else:
            print(f"   {prop}: Not found in batch")
    
    # Additional debugging for charges
    if 'charges' in ase_batch:
        charges = ase_batch['charges']
        print(f"\n   Charge debugging:")
        print(f"   Unique charge values: {torch.unique(charges).tolist()}")
        print(f"   Charge shape: {charges.shape}")
        
        # Check if there are actual zero charges or just padding
        atom_mask = ase_batch.get('atom_mask', torch.ones_like(charges.squeeze(-1), dtype=torch.bool))
        masked_charges = charges[atom_mask.unsqueeze(-1)].squeeze()
        print(f"   Masked charge values (real atoms): {torch.unique(masked_charges).tolist()}")
        
        # Update the zero charge check to only consider real atoms
        if torch.any(masked_charges == 0):
            issues.append("Charges contain 0 values (invalid atomic numbers)")
        else:
            print("   ✓ No invalid charges found in real atoms")
    
    # Summary
    print(f"\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if issues:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n🔧 RECOMMENDED FIXES:")
        
        # Specific fix recommendations
        if any("One-hot encoding mismatch" in issue for issue in issues):
            print(f"  • Fix ASE dataset to use QM9-compatible atom encoding (5 types: H, C, N, O, F)")
        
        if any("Property key mismatch" in issue for issue in issues):
            print(f"  • Map ASE property keys to QM9 format or update model to expect ASE keys")
        
        if any("Data type mismatch" in issue for issue in issues):
            print(f"  • Ensure all ASE tensors use float32 dtype for consistency")
        
        if any("unconverted" in issue for issue in issues):
            print(f"  • Verify unit conversion from Hartree to eV is working correctly")
        
        print(f"\n⚡ These issues likely cause the high loss values (20-23) in ASE training!")
        return False
    else:
        print("✅ ALL CHECKS PASSED!")
        print("Data structures are compatible. High loss may be due to other factors:")
        print("  • Different normalization factors")
        print("  • Dataset distribution differences")
        print("  • Training hyperparameter mismatch")
        return True

if __name__ == "__main__":
    success = check_dataset_compatibility()
    print("="*70)
    exit(0 if success else 1)