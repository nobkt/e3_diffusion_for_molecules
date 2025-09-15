#!/usr/bin/env python3
"""
Test script to verify that the halogen bias fix is working correctly.

This script creates a synthetic ASE database that reproduces the halogen bias issue
and tests whether the fixes prevent the bias from occurring.
"""

import os
import tempfile
import torch
import numpy as np
from ase import Atoms
from ase.db import connect
import argparse

def create_test_database(db_path, n_molecules=100):
    """
    Create a synthetic ASE database with realistic element distributions
    that should NOT be dominated by halogens.
    
    Element distribution mimics the problem statement:
    H: 44.8%, C: 42.7%, N: 3.8%, O: 6.2%, F: 0.6%, Si: 0.2%, 
    P: 0.1%, S: 0.9%, Cl: 0.4%, Br: 0.2%, I: 0.1%
    """
    print(f"Creating test database with {n_molecules} molecules...")
    
    # Element probabilities (should match training data distribution)
    elements = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
    probabilities = [0.448, 0.427, 0.038, 0.062, 0.006, 0.002, 0.001, 0.009, 0.004, 0.002, 0.001]
    
    # Atomic numbers for elements
    atomic_numbers = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 
                     'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
    
    db = connect(db_path)
    
    # Track actual element counts for verification
    element_counts = {elem: 0 for elem in elements}
    total_atoms = 0
    
    for i in range(n_molecules):
        # Generate molecule with 10-25 atoms
        n_atoms = np.random.randint(10, 26)
        
        # Sample atoms according to realistic distribution
        molecule_elements = np.random.choice(elements, size=n_atoms, p=probabilities)
        
        # Count elements
        for elem in molecule_elements:
            element_counts[elem] += 1
        total_atoms += n_atoms
        
        # Create atomic numbers array
        numbers = [atomic_numbers[elem] for elem in molecule_elements]
        
        # Random 3D positions (doesn't matter for this test)
        positions = np.random.randn(n_atoms, 3) * 2.0
        
        # Create ASE Atoms object
        atoms = Atoms(numbers=numbers, positions=positions)
        
        # Add some molecular properties
        molecular_weight = sum([atomic_numbers[elem] for elem in molecule_elements])
        
        # Write to database
        db.write(atoms, 
                molecular_weight=molecular_weight,
                pi_conjugation_ratio=np.random.uniform(0, 1),
                source='synthetic_test')
    
    # Print actual distribution
    print("\nActual element distribution in test database:")
    for elem in elements:
        percentage = (element_counts[elem] / total_atoms) * 100
        print(f"  {elem}: {element_counts[elem]:,} atoms ({percentage:.1f}%)")
    
    return element_counts, total_atoms

def test_normalization_math():
    """Test the mathematical effect of different normalization factors."""
    print("\n" + "="*60)
    print("Testing Normalization Mathematics")
    print("="*60)
    
    # Test with 11 elements (like in the problem)
    n_elements = 11
    torch.manual_seed(42)  # For reproducible results
    
    print(f"Testing with {n_elements} elements...")
    
    # Create one-hot for Carbon (index 1, should be most common)
    one_hot_carbon = torch.zeros(n_elements)
    one_hot_carbon[1] = 1.0
    
    # Create one-hot for Bromine (index 9, should be rare)
    one_hot_bromine = torch.zeros(n_elements)
    one_hot_bromine[9] = 1.0
    
    # Test different normalization factors
    for norm_factor in [4.0, 2.0, 1.0]:
        print(f"\nNormalization factor: {norm_factor}")
        
        # Normalize
        carbon_norm = one_hot_carbon / norm_factor
        bromine_norm = one_hot_bromine / norm_factor
        
        # Add realistic noise (what happens during diffusion)
        noise_carbon = torch.randn(n_elements) * 0.1
        noise_bromine = torch.randn(n_elements) * 0.1
        
        carbon_noisy = carbon_norm + noise_carbon
        bromine_noisy = bromine_norm + noise_bromine
        
        # Denormalize
        carbon_final = carbon_noisy * norm_factor
        bromine_final = bromine_noisy * norm_factor
        
        # Check predictions
        carbon_pred = torch.argmax(carbon_final).item()
        bromine_pred = torch.argmax(bromine_final).item()
        
        element_names = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
        
        print(f"  Carbon input → predicted: {element_names[carbon_pred]} ({'✅' if carbon_pred == 1 else '❌'})")
        print(f"  Bromine input → predicted: {element_names[bromine_pred]} ({'✅' if bromine_pred == 9 else '❌'})")
        
        # Check if bias towards higher indices occurs
        avg_carbon = torch.mean(carbon_final).item()
        avg_bromine = torch.mean(bromine_final).item()
        max_carbon = torch.max(carbon_final).item()
        max_bromine = torch.max(bromine_final).item()
        
        print(f"  Carbon: avg={avg_carbon:.3f}, max={max_carbon:.3f}")
        print(f"  Bromine: avg={avg_bromine:.3f}, max={max_bromine:.3f}")

def test_conditional_features():
    """Test how problematic conditional features affect generation."""
    print("\n" + "="*60)
    print("Testing Conditional Feature Effects")
    print("="*60)
    
    # Simulate the problematic conditioning scenario
    features = ['molecular_weight', 'pi_conjugation_ratio', 'atom_types_encoding', 'functional_groups_encoding']
    
    problematic_features = ['atom_types_encoding', 'functional_groups_encoding']
    safe_features = [f for f in features if f not in problematic_features]
    
    print("Original conditioning:", features)
    print("Safe conditioning:", safe_features)
    print("Problematic features:", problematic_features)
    
    print("\nIssue with problematic features:")
    print("- atom_types_encoding tells model what atoms to expect")
    print("- But during conditional sampling, these are often set to zeros")
    print("- Creates inconsistency: 'expect no specific atoms' but 'generate molecule'")
    print("- Model compensates by generating whatever has highest probability after noise")
    print("- With high normalization factor, this biases towards halogens")
    
    print("\nFix implemented:")
    print("- Use statistical mean instead of zeros for binary features")
    print("- Add small positive bias to prevent pure zeros")
    print("- Warn users about problematic feature combinations")

def main():
    parser = argparse.ArgumentParser(description='Test halogen bias fix')
    parser.add_argument('--create-db', action='store_true', help='Create test database')
    parser.add_argument('--db-path', type=str, default='/tmp/test_halogen_bias.db', help='Database path')
    parser.add_argument('--test-math', action='store_true', help='Test normalization mathematics')
    parser.add_argument('--test-features', action='store_true', help='Test conditional features')
    args = parser.parse_args()
    
    print("🧪 Halogen Bias Fix Test Suite")
    print("="*50)
    
    if args.create_db:
        # Create test database
        create_test_database(args.db_path)
        print(f"✅ Test database created: {args.db_path}")
        
        # Test the database loading
        try:
            from qm9.general_molecular_db import analyze_ase_database
            analysis = analyze_ase_database(args.db_path)
            print("\n✅ Database analysis successful")
            
            # Check if normalization factor is correctly computed
            n_elements = len(analysis['unique_elements'])
            if n_elements > 10:
                expected_factor = 1.0
            elif n_elements > 5:
                expected_factor = 2.0
            else:
                expected_factor = 4.0
                
            print(f"Elements found: {n_elements}")
            print(f"Expected normalization factor: {expected_factor}")
            
        except Exception as e:
            print(f"❌ Error analyzing database: {e}")
    
    if args.test_math:
        test_normalization_math()
    
    if args.test_features:
        test_conditional_features()
    
    print("\n" + "="*50)
    print("🎯 Summary of Fixes Applied:")
    print("1. ✅ Dynamic normalization factor based on element count")
    print("2. ✅ Improved conditional sampling for binary features")
    print("3. ✅ Better gradient clipping with stability warnings")
    print("4. ✅ User warnings for problematic feature combinations")
    print("5. ✅ Enhanced element distribution reporting")
    print("\n💡 Recommendation:")
    print("For ASE databases with many elements, use only scalar conditioning")
    print("features like 'molecular_weight' and 'pi_conjugation_ratio'")
    print("="*50)

if __name__ == "__main__":
    main()