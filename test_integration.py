#!/usr/bin/env python3
"""
Integration test for the ASE database conditional generation functionality
"""

import os
import sys
import tempfile
import numpy as np
import torch
from ase import Atoms
from ase.db import connect

def create_minimal_test_database(db_path):
    """Create a minimal test database"""
    db = connect(db_path)
    
    # Add just a few simple molecules
    water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    db.write(water, data={'energy': -76.4, 'homo': -12.6})
    
    methane = Atoms('CH4', positions=[
        [0, 0, 0], [1.089, 1.089, 1.089], [1.089, -1.089, -1.089], 
        [-1.089, 1.089, -1.089], [-1.089, -1.089, 1.089]
    ])
    db.write(methane, data={'energy': -40.5, 'homo': -14.4})
    
    return db_path

def test_ase_loading_with_conditioning():
    """Test that ASE database loading works with molecular descriptors"""
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create test database
        create_minimal_test_database(db_path)
        
        # Test dataset loading
        from qm9.dataset import load_ase_database
        
        datasets, num_species, charge_scale = load_ase_database(
            db_path, 
            split_ratios=(0.8, 0.1, 0.1), 
            include_charges=True, 
            remove_h=False
        )
        
        train_data = datasets['train'].data
        
        # Check that molecular descriptors are present
        required_properties = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding']
        for prop in required_properties:
            if prop not in train_data:
                raise ValueError(f"Missing required property: {prop}")
        
        print("✅ ASE database loading with molecular descriptors: PASSED")
        
        # Test property normalization
        from qm9.utils import compute_mean_mad_from_dataloader
        
        test_properties = ['molecular_weight', 'pi_conjugation_ratio']
        available_properties = [p for p in test_properties if p in train_data]
        
        if available_properties:
            # Create a mock dataloader-like object
            class MockDataloader:
                def __init__(self, data):
                    self.dataset = MockDataset(data)
            
            class MockDataset:
                def __init__(self, data):
                    self.data = data
            
            mock_dataloader = MockDataloader(train_data)
            property_norms = compute_mean_mad_from_dataloader(mock_dataloader, available_properties)
            
            for prop in available_properties:
                if prop not in property_norms:
                    raise ValueError(f"Failed to compute normalization for {prop}")
                if 'mean' not in property_norms[prop] or 'mad' not in property_norms[prop]:
                    raise ValueError(f"Missing mean or mad for {prop}")
            
            print("✅ Property normalization: PASSED")
        
        # Test context preparation
        from qm9.utils import prepare_context
        
        # Create a mock minibatch
        batch_size = 2
        max_atoms = 5
        
        mock_batch = {
            'positions': torch.zeros(batch_size, max_atoms, 3),
            'atom_mask': torch.ones(batch_size, max_atoms),  # Shape (batch_size, max_atoms) - will be unsqueezed in prepare_context
            'molecular_weight': torch.tensor([18.015, 16.043]),  # water, methane
            'pi_conjugation_ratio': torch.tensor([0.0, 0.0])
        }
        
        # Test with molecular weight conditioning
        if 'molecular_weight' in available_properties:
            property_norms_mw = {'molecular_weight': {'mean': torch.tensor(17.0), 'mad': torch.tensor(1.0)}}
            context = prepare_context(['molecular_weight'], mock_batch, property_norms_mw)
            
            expected_shape = (batch_size, max_atoms, 1)  # 1 for molecular_weight
            if context.shape != expected_shape:
                raise ValueError(f"Expected context shape {expected_shape}, got {context.shape}")
            
            print("✅ Context preparation with molecular weight: PASSED")
        
        # Test with multiple conditioning
        if len(available_properties) > 1:
            multi_norms = {}
            for prop in available_properties:
                multi_norms[prop] = {'mean': torch.tensor(0.0), 'mad': torch.tensor(1.0)}
            
            context = prepare_context(available_properties, mock_batch, multi_norms)
            expected_features = len(available_properties)
            expected_shape = (batch_size, max_atoms, expected_features)
            
            if context.shape != expected_shape:
                raise ValueError(f"Expected context shape {expected_shape}, got {context.shape}")
            
            print("✅ Context preparation with multiple conditions: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_molecular_descriptor_extraction():
    """Test the molecular descriptor extraction functions"""
    
    try:
        from qm9.openbabel_functions import extract_molecular_descriptors_ase_openbabel
        
        # Test with a simple molecule
        water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
        descriptors = extract_molecular_descriptors_ase_openbabel(water)
        
        required_keys = ['atom_types', 'molecular_weight', 'functional_groups', 'pi_conjugation_ratio']
        for key in required_keys:
            if key not in descriptors:
                raise ValueError(f"Missing descriptor: {key}")
        
        # Check values are reasonable
        if descriptors['molecular_weight'] < 10 or descriptors['molecular_weight'] > 30:
            raise ValueError(f"Unexpected molecular weight: {descriptors['molecular_weight']}")
        
        if descriptors['pi_conjugation_ratio'] < 0 or descriptors['pi_conjugation_ratio'] > 1:
            raise ValueError(f"Invalid π conjugation ratio: {descriptors['pi_conjugation_ratio']}")
        
        if 'H' not in descriptors['atom_types'] or 'O' not in descriptors['atom_types']:
            raise ValueError(f"Unexpected atom types: {descriptors['atom_types']}")
        
        print("✅ Molecular descriptor extraction: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Molecular descriptor extraction test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running integration tests for ASE database conditional generation...")
    
    test1 = test_molecular_descriptor_extraction()
    test2 = test_ase_loading_with_conditioning()
    
    if test1 and test2:
        print("\n🎉 All integration tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests FAILED!")
        sys.exit(1)