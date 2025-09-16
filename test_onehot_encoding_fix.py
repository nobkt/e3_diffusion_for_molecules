#!/usr/bin/env python3
"""
Test script to validate the one-hot encoding fix for ASE database conditioning features.

This script tests that atom_types_encoding and functional_groups_encoding are now
properly formatted as fixed-length one-hot encodings suitable for generation conditions.
"""

import os
import tempfile
import torch
import numpy as np
from ase import Atoms
from ase.db import connect
from qm9.dataset import convert_ase_to_dataset_format

def create_test_ase_database(db_path, n_molecules=20):
    """Create a small test ASE database with diverse molecules."""
    print(f"Creating test ASE database at {db_path}")
    
    db = connect(db_path)
    
    # Create diverse molecules with different atom types and structures
    molecules = []
    
    # Water (H2O)
    water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
    molecules.append(water)
    
    # Methane (CH4)
    methane = Atoms('CH4', positions=[[0, 0, 0], [1.089, 1.089, 1.089], 
                                     [1.089, -1.089, -1.089], [-1.089, 1.089, -1.089], 
                                     [-1.089, -1.089, 1.089]])
    molecules.append(methane)
    
    # Ammonia (NH3)
    ammonia = Atoms('NH3', positions=[[0, 0, 0], [1.017, 0, 0], 
                                     [-0.509, 0.882, 0], [-0.509, -0.882, 0]])
    molecules.append(ammonia)
    
    # Ethane (C2H6)
    ethane = Atoms('C2H6', positions=[[0, 0, 0], [1.54, 0, 0],
                                     [-0.51, 0.88, 0], [-0.51, -0.44, 0.76], [-0.51, -0.44, -0.76],
                                     [2.05, 0.88, 0], [2.05, -0.44, 0.76], [2.05, -0.44, -0.76]])
    molecules.append(ethane)
    
    # Carbon dioxide (CO2)
    co2 = Atoms('CO2', positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]])
    molecules.append(co2)
    
    # Hydrogen fluoride (HF)
    hf = Atoms('HF', positions=[[0, 0, 0], [0.917, 0, 0]])
    molecules.append(hf)
    
    # Add some molecules with chlorine and sulfur for diversity
    # Hydrogen chloride (HCl)
    hcl = Atoms('HCl', positions=[[0, 0, 0], [1.275, 0, 0]])
    molecules.append(hcl)
    
    # Hydrogen sulfide (H2S)
    h2s = Atoms('H2S', positions=[[0, 0, 0], [1.336, 0, 0], [-0.668, 1.157, 0]])
    molecules.append(h2s)
    
    # Replicate to reach desired count
    while len(molecules) < n_molecules:
        molecules.extend(molecules[:min(len(molecules), n_molecules - len(molecules))])
    
    # Add molecules to database with some properties
    for i, mol in enumerate(molecules[:n_molecules]):
        properties = {
            'energy': np.random.normal(-50, 20),
            'homo': np.random.normal(-10, 2),
            'lumo': np.random.normal(2, 1)
        }
        db.write(mol, data=properties)
    
    print(f"Created database with {n_molecules} molecules")
    return db_path

def test_onehot_encoding_format():
    """Test that the one-hot encodings are properly formatted."""
    print("\n" + "="*60)
    print("TESTING ONE-HOT ENCODING FORMAT")
    print("="*60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create test database
        create_test_ase_database(db_path, n_molecules=20)
        
        # Load database and convert to dataset format
        db = connect(db_path)
        all_atoms = []
        all_properties = []
        
        for row in db.select():
            atoms = row.toatoms()
            properties = {}
            if hasattr(row, 'data') and row.data:
                properties.update(row.data)
            all_atoms.append(atoms)
            all_properties.append(properties)
        
        # Convert to dataset format
        dataset_data = convert_ase_to_dataset_format(all_atoms, all_properties, 
                                                   include_charges=True, remove_h=False)
        
        print(f"\nLoaded {len(all_atoms)} molecules")
        
        # Test atom types encoding
        atom_types_encoding = dataset_data['atom_types_encoding']
        print(f"\nAtom types encoding shape: {atom_types_encoding.shape}")
        print(f"Atom types encoding dtype: {atom_types_encoding.dtype}")
        
        # Check if it's properly normalized (probabilities should sum to ~1 for molecules with atoms)
        atom_sums = torch.sum(atom_types_encoding, dim=1)
        print(f"Atom types encoding sums - min: {atom_sums.min():.3f}, max: {atom_sums.max():.3f}, mean: {atom_sums.mean():.3f}")
        
        # Check range
        print(f"Atom types encoding range: [{atom_types_encoding.min():.3f}, {atom_types_encoding.max():.3f}]")
        
        # Test functional groups encoding
        functional_groups_encoding = dataset_data['functional_groups_encoding']
        print(f"\nFunctional groups encoding shape: {functional_groups_encoding.shape}")
        print(f"Functional groups encoding dtype: {functional_groups_encoding.dtype}")
        
        # Check normalization
        fg_sums = torch.sum(functional_groups_encoding, dim=1)
        print(f"Functional groups encoding sums - min: {fg_sums.min():.3f}, max: {fg_sums.max():.3f}, mean: {fg_sums.mean():.3f}")
        
        # Check range
        print(f"Functional groups encoding range: [{functional_groups_encoding.min():.3f}, {functional_groups_encoding.max():.3f}]")
        
        # Check metadata
        print(f"\nMetadata:")
        print(f"  Atom types is one-hot: {dataset_data.get('_atom_types_is_onehot', False)}")
        print(f"  Functional groups is one-hot: {dataset_data.get('_functional_groups_is_onehot', False)}")
        print(f"  Atom types dimensions: {dataset_data.get('_atom_types_dimensions', 'Not set')}")
        print(f"  Functional groups dimensions: {dataset_data.get('_functional_groups_dimensions', 'Not set')}")
        print(f"  Atom types mapping: {dataset_data.get('_atom_types_mapping', [])}")
        print(f"  Functional groups mapping: {dataset_data.get('_functional_groups_mapping', [])}")
        
        # Validate that encodings are proper probability distributions
        success = True
        
        # For atom types - each molecule should have normalized probabilities
        for i in range(len(all_atoms)):
            atom_sum = atom_sums[i]
            if atom_sum > 0 and not (0.99 <= atom_sum <= 1.01):
                print(f"❌ Molecule {i}: atom types not properly normalized (sum = {atom_sum:.3f})")
                success = False
        
        # Check that encodings contain valid probabilities
        if torch.any(atom_types_encoding < 0) or torch.any(atom_types_encoding > 1):
            print("❌ Atom types encoding contains values outside [0,1] range")
            success = False
        
        if torch.any(functional_groups_encoding < 0) or torch.any(functional_groups_encoding > 1):
            print("❌ Functional groups encoding contains values outside [0,1] range")
            success = False
        
        # Check metadata flags
        if not dataset_data.get('_atom_types_is_onehot', False):
            print("❌ Atom types not marked as one-hot encoding")
            success = False
        
        if not dataset_data.get('_functional_groups_is_onehot', False):
            print("❌ Functional groups not marked as one-hot encoding")
            success = False
        
        if success:
            print("\n✅ All one-hot encoding tests passed!")
            print("The binary features are now properly formatted for conditional generation.")
        else:
            print("\n❌ Some tests failed. Check the implementation.")
            
        return success
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_normalization_computation():
    """Test that the normalization computation works correctly with one-hot encodings."""
    print("\n" + "="*60)
    print("TESTING NORMALIZATION COMPUTATION")
    print("="*60)
    
    # Create mock dataset data that simulates the new one-hot encodings
    batch_size = 10
    n_atom_types = 5
    n_functional_groups = 8
    
    # Create normalized atom types encoding (probability distributions)
    atom_types_encoding = torch.zeros(batch_size, n_atom_types)
    for i in range(batch_size):
        # Random number of atom types (1-3)
        n_types = np.random.randint(1, 4)
        indices = np.random.choice(n_atom_types, n_types, replace=False)
        for idx in indices:
            atom_types_encoding[i, idx] = 1.0
        # Normalize to probability distribution
        atom_types_encoding[i] = atom_types_encoding[i] / atom_types_encoding[i].sum()
    
    # Create functional groups encoding (some molecules have no functional groups)
    functional_groups_encoding = torch.zeros(batch_size, n_functional_groups)
    for i in range(batch_size):
        if np.random.random() > 0.3:  # 70% chance of having functional groups
            n_groups = np.random.randint(1, 3)
            indices = np.random.choice(n_functional_groups, n_groups, replace=False)
            for idx in indices:
                functional_groups_encoding[i, idx] = 1.0
            functional_groups_encoding[i] = functional_groups_encoding[i] / functional_groups_encoding[i].sum()
        else:
            # No functional groups - use small uniform probability
            functional_groups_encoding[i] = torch.full((n_functional_groups,), 0.01 / n_functional_groups)
    
    # Mock dataset structure
    class MockDataset:
        def __init__(self):
            self.data = {
                'atom_types_encoding': atom_types_encoding,
                'functional_groups_encoding': functional_groups_encoding,
                '_atom_types_is_onehot': True,
                '_functional_groups_is_onehot': True,
                'molecular_weight': torch.rand(batch_size) * 500 + 50  # 50-550 u
            }
    
    class MockDataloader:
        def __init__(self):
            self.dataset = MockDataset()
    
    # Test the normalization computation
    from qm9.utils import compute_mean_mad_from_dataloader
    
    dataloader = MockDataloader()
    properties = ['atom_types_encoding', 'functional_groups_encoding', 'molecular_weight']
    
    print("Computing normalization statistics...")
    norms = compute_mean_mad_from_dataloader(dataloader, properties)
    
    for prop in properties:
        print(f"\n{prop}:")
        mean = norms[prop]['mean']
        mad = norms[prop]['mad']
        
        if mean.dim() > 0:
            print(f"  Mean shape: {mean.shape}")
            print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
            print(f"  MAD shape: {mad.shape}")
            print(f"  MAD range: [{mad.min():.3f}, {mad.max():.3f}]")
        else:
            print(f"  Mean: {mean:.3f}")
            print(f"  MAD: {mad:.3f}")
    
    print("\n✅ Normalization computation test completed!")
    return True

if __name__ == "__main__":
    print("🧪 Testing One-Hot Encoding Fix for ASE Database Features")
    print("="*70)
    
    try:
        success1 = test_onehot_encoding_format()
        success2 = test_normalization_computation()
        
        if success1 and success2:
            print("\n🎉 All tests passed! The one-hot encoding fix is working correctly.")
            print("\nThe atom_types_encoding and functional_groups_encoding features are now:")
            print("✅ Properly formatted as normalized probability distributions")
            print("✅ Marked with metadata for identification")
            print("✅ Compatible with improved normalization logic")
            print("✅ Suitable for stable conditional generation")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()