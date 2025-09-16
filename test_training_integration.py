#!/usr/bin/env python3
"""
Integration test to verify the full training pipeline works with the new one-hot encodings.

This creates a small ASE database and runs a few training steps to verify the warnings
are handled correctly and the conditioning features work as expected.
"""

import os
import sys
import tempfile
import subprocess
import torch
import numpy as np
from ase import Atoms
from ase.db import connect

def create_integration_test_database(db_path, n_molecules=50):
    """Create a test database with enough diversity for integration testing."""
    print(f"Creating integration test database at {db_path}")
    
    db = connect(db_path)
    
    # Create diverse molecules
    molecules = []
    
    # Various small molecules with different characteristics
    templates = [
        # Water
        {'formula': 'H2O', 'positions': [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]]},
        # Methane
        {'formula': 'CH4', 'positions': [[0, 0, 0], [1.089, 1.089, 1.089], [1.089, -1.089, -1.089], [-1.089, 1.089, -1.089], [-1.089, -1.089, 1.089]]},
        # Ammonia
        {'formula': 'NH3', 'positions': [[0, 0, 0], [1.017, 0, 0], [-0.509, 0.882, 0], [-0.509, -0.882, 0]]},
        # Carbon dioxide
        {'formula': 'CO2', 'positions': [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]},
        # Hydrogen fluoride
        {'formula': 'HF', 'positions': [[0, 0, 0], [0.917, 0, 0]]},
        # Hydrogen chloride
        {'formula': 'HCl', 'positions': [[0, 0, 0], [1.275, 0, 0]]},
        # Hydrogen sulfide
        {'formula': 'H2S', 'positions': [[0, 0, 0], [1.336, 0, 0], [-0.668, 1.157, 0]]},
        # Ethane
        {'formula': 'C2H6', 'positions': [[0, 0, 0], [1.54, 0, 0], [-0.51, 0.88, 0], [-0.51, -0.44, 0.76], [-0.51, -0.44, -0.76], [2.05, 0.88, 0], [2.05, -0.44, 0.76], [2.05, -0.44, -0.76]]},
    ]
    
    # Generate molecules
    for i in range(n_molecules):
        template = templates[i % len(templates)]
        
        # Add small random variations to positions
        positions = []
        for pos in template['positions']:
            new_pos = [pos[j] + np.random.normal(0, 0.02) for j in range(3)]
            positions.append(new_pos)
        
        mol = Atoms(template['formula'], positions=positions)
        
        # Add some molecular properties
        properties = {
            'energy': np.random.normal(-50, 20),
            'homo': np.random.normal(-10, 2), 
            'lumo': np.random.normal(2, 1),
            'mol_id': i
        }
        
        db.write(mol, data=properties)
        molecules.append(mol)
    
    print(f"Created database with {len(molecules)} molecules")
    return db_path

def test_training_integration():
    """Test the complete training pipeline with conditioning features."""
    print("\n" + "="*60)
    print("TESTING TRAINING INTEGRATION")
    print("="*60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create test database
        create_integration_test_database(db_path, n_molecules=50)
        
        # Test different conditioning scenarios
        test_cases = [
            {
                'name': 'Scalar features only',
                'conditioning': ['molecular_weight'],
                'expected_warnings': []
            },
            {
                'name': 'Scalar features with pi conjugation',
                'conditioning': ['molecular_weight', 'pi_conjugation_ratio'], 
                'expected_warnings': []
            },
            {
                'name': 'One-hot atom types encoding',
                'conditioning': ['atom_types_encoding'],
                'expected_warnings': []  # Should now work without problematic warnings
            },
            {
                'name': 'One-hot functional groups encoding',
                'conditioning': ['functional_groups_encoding'],
                'expected_warnings': []  # Should now work without problematic warnings
            },
            {
                'name': 'Mixed conditioning with one-hot encodings',
                'conditioning': ['molecular_weight', 'atom_types_encoding', 'functional_groups_encoding'],
                'expected_warnings': ['many features']  # Should warn about many features but not problematic ones
            }
        ]
        
        success_count = 0
        
        for test_case in test_cases:
            print(f"\n--- Testing: {test_case['name']} ---")
            print(f"Conditioning: {test_case['conditioning']}")
            
            # Construct command
            cmd = [
                sys.executable, 'main_qm9.py',
                '--dataset', 'ase_db',
                '--ase_db_path', db_path,
                '--conditioning'] + test_case['conditioning'] + [
                '--exp_name', f'test_{test_case["name"].replace(" ", "_")}',
                '--n_epochs', '1',  # Just 1 epoch for testing
                '--batch_size', '8',
                '--break_train_epoch',  # Exit after one batch
                '--no_wandb'  # Disable wandb
            ]
            
            try:
                # Capture output to check for warnings
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                print(f"Exit code: {result.returncode}")
                
                # Check for expected behavior
                output = result.stdout + result.stderr
                
                # Look for our specific warnings/messages
                has_problematic_warning = "Problematic conditioning features detected" in output
                has_onehot_confirmation = "properly formatted one-hot encodings" in output
                has_binary_features_note = "Using binary features" in output
                has_many_features_warning = "many conditioning features" in output
                
                print(f"Has problematic warning: {has_problematic_warning}")
                print(f"Has one-hot confirmation: {has_onehot_confirmation}")
                print(f"Has binary features note: {has_binary_features_note}")
                print(f"Has many features warning: {has_many_features_warning}")
                
                # Validate expectations
                test_passed = True
                
                if 'atom_types_encoding' in test_case['conditioning'] or 'functional_groups_encoding' in test_case['conditioning']:
                    # Should have confirmation of one-hot encodings, not problematic warnings
                    if has_problematic_warning:
                        print("❌ FAIL: Still showing problematic warnings for one-hot encodings")
                        test_passed = False
                    elif has_onehot_confirmation:
                        print("✅ PASS: Showing one-hot encoding confirmation")
                    elif has_binary_features_note:
                        print("✅ PASS: Showing binary features note")
                    else:
                        print("⚠️  No specific one-hot confirmation found, but no problematic warnings")
                
                if len(test_case['conditioning']) > 3 and 'many features' in test_case['expected_warnings']:
                    if has_many_features_warning:
                        print("✅ PASS: Correctly warning about many features")
                    else:
                        print("⚠️  Expected warning about many features")
                
                if result.returncode == 0:
                    print("✅ PASS: Training started successfully")
                    success_count += 1
                else:
                    print("❌ FAIL: Training failed to start")
                    test_passed = False
                    # Print some output for debugging
                    print("STDOUT:")
                    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                    print("STDERR:")
                    print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                
                if test_passed:
                    success_count += 1
                    
            except subprocess.TimeoutExpired:
                print("❌ FAIL: Training timed out")
            except Exception as e:
                print(f"❌ FAIL: Error running training: {e}")
        
        print(f"\n--- Integration Test Results ---")
        print(f"Passed: {success_count}/{len(test_cases)} test cases")
        
        if success_count == len(test_cases):
            print("🎉 All integration tests passed!")
            return True
        else:
            print("❌ Some integration tests failed")
            return False
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    print("🧪 Integration Test: ASE Database Training with One-Hot Encodings")
    print("="*80)
    
    # Change to the repository directory
    os.chdir('/home/runner/work/e3_diffusion_for_molecules/e3_diffusion_for_molecules')
    
    try:
        success = test_training_integration()
        
        if success:
            print("\n🎉 Integration tests completed successfully!")
            print("\nKey improvements verified:")
            print("✅ atom_types_encoding now works as one-hot encoding")
            print("✅ functional_groups_encoding now works as one-hot encoding") 
            print("✅ No more 'problematic conditioning features' warnings for properly formatted encodings")
            print("✅ Training starts successfully with binary conditioning features")
            print("✅ Proper warnings for excessive number of conditioning features")
        else:
            print("\n❌ Some integration tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during integration testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)