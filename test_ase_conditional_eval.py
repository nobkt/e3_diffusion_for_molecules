#!/usr/bin/env python3
"""
Test script for the updated eval_conditional_qm9.py to verify it works with 
ASE database and new molecular descriptor conditioning.

This script tests:
1. Creating a sample ASE database
2. Training a minimal model with molecular descriptor conditioning
3. Using eval_conditional_qm9.py to evaluate the model
4. Property classifier training and evaluation
"""

import os
import sys
import tempfile
import shutil
import subprocess
import numpy as np
from ase import Atoms
from ase.db import connect

def create_test_ase_database(db_path, n_molecules=20):
    """Create a small test ASE database for training and evaluation."""
    
    print(f"Creating test ASE database at {db_path} with {n_molecules} molecules...")
    
    db = connect(db_path)
    molecules = []
    
    # Create diverse small molecules for testing
    
    # 1. Water molecules
    for i in range(n_molecules // 4):
        positions = [
            [0, 0, 0], 
            [0.757 + np.random.normal(0, 0.02), 0.586 + np.random.normal(0, 0.02), 0], 
            [-0.757 + np.random.normal(0, 0.02), 0.586 + np.random.normal(0, 0.02), 0]
        ]
        water = Atoms('H2O', positions=positions)
        molecules.append((water, {
            'energy': -76.4 + np.random.normal(0, 0.1),
            'homo': -12.6 + np.random.normal(0, 0.1),
            'lumo': 1.4 + np.random.normal(0, 0.1)
        }))
    
    # 2. Methane molecules
    for i in range(n_molecules // 4):
        positions = [
            [0, 0, 0],  # C center
            [1.089, 1.089, 1.089] + np.random.normal(0, 0.02, 3),
            [1.089, -1.089, -1.089] + np.random.normal(0, 0.02, 3),
            [-1.089, 1.089, -1.089] + np.random.normal(0, 0.02, 3),
            [-1.089, -1.089, 1.089] + np.random.normal(0, 0.02, 3)
        ]
        methane = Atoms('CH4', positions=positions)
        molecules.append((methane, {
            'energy': -40.5 + np.random.normal(0, 0.1),
            'homo': -14.4 + np.random.normal(0, 0.1),
            'lumo': 6.0 + np.random.normal(0, 0.1)
        }))
    
    # 3. Ammonia molecules
    for i in range(n_molecules // 4):
        positions = [
            [0, 0, 0],  # N center
            [1.017, 0, 0] + np.random.normal(0, 0.02, 3),
            [-0.509, 0.882, 0] + np.random.normal(0, 0.02, 3),
            [-0.509, -0.882, 0] + np.random.normal(0, 0.02, 3)
        ]
        ammonia = Atoms('NH3', positions=positions)
        molecules.append((ammonia, {
            'energy': -56.5 + np.random.normal(0, 0.1),
            'homo': -10.8 + np.random.normal(0, 0.1),
            'lumo': 2.1 + np.random.normal(0, 0.1)
        }))
    
    # 4. CO2 molecules
    for i in range(n_molecules - len(molecules)):
        positions = [
            [0, 0, 0],      # C center
            [1.16, 0, 0] + np.random.normal(0, 0.02, 3),   # O1
            [-1.16, 0, 0] + np.random.normal(0, 0.02, 3)   # O2
        ]
        co2 = Atoms('CO2', positions=positions)
        molecules.append((co2, {
            'energy': -188.6 + np.random.normal(0, 0.1),
            'homo': -13.8 + np.random.normal(0, 0.1),
            'lumo': 4.0 + np.random.normal(0, 0.1)
        }))
    
    # Add molecules to database
    for atoms, properties in molecules:
        db.write(atoms, data=properties)
    
    print(f"Created database with {len(molecules)} molecules")
    return db_path

def test_training_command(db_path, output_dir):
    """Test training a model with molecular descriptor conditioning."""
    
    print("\n" + "="*60)
    print("TESTING MODEL TRAINING")
    print("="*60)
    
    # Test with multiple molecular descriptor conditioning
    cmd = [
        'python', 'main_qm9.py',
        '--dataset', 'ase_db',
        '--ase_db_path', db_path,
        '--conditioning', 'molecular_weight', 'atom_types_encoding',
        '--exp_name', 'test_ase_multi_cond',
        '--n_epochs', '2',  # Very few epochs for testing
        '--batch_size', '8',
        '--lr', '1e-3',
        '--nf', '32',  # Small network for testing
        '--n_layers', '3',
        '--save_model', 'True',
        '--no_wandb'  # Disable wandb for testing
    ]
    
    print("Running training command:")
    print(' '.join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print("Training failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print("Training completed successfully!")
        return True

def test_qualitative_evaluation(generators_path):
    """Test qualitative conditional generation."""
    
    print("\n" + "="*60)
    print("TESTING QUALITATIVE EVALUATION")
    print("="*60)
    
    cmd = [
        'python', 'eval_conditional_qm9.py',
        '--generators_path', generators_path,
        '--task', 'qualitative',
        '--n_sweeps', '2',  # Few sweeps for testing
        '--exp_name', 'test_qualitative_eval'
    ]
    
    print("Running qualitative evaluation command:")
    print(' '.join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print("Qualitative evaluation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print("Qualitative evaluation completed successfully!")
        return True

def test_property_classifier_training(db_path):
    """Test training a property classifier."""
    
    print("\n" + "="*60)
    print("TESTING PROPERTY CLASSIFIER TRAINING")
    print("="*60)
    
    cmd = [
        'python', 'qm9/property_prediction/main_qm9_prop.py',
        '--dataset', 'ase_db',
        '--ase_db_path', db_path,
        '--property', 'molecular_weight',
        '--exp_name', 'test_class_molecular_weight',
        '--epochs', '2',  # Few epochs for testing
        '--batch_size', '8',
        '--lr', '1e-3',
        '--model_name', 'egnn',
        '--save_model', 'True'
    ]
    
    print("Running property classifier training command:")
    print(' '.join(cmd))
    
    os.chdir('qm9/property_prediction')
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.chdir('../..')
    
    if result.returncode != 0:
        print("Property classifier training failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print("Property classifier training completed successfully!")
        return True

def test_edm_evaluation(generators_path, classifiers_path):
    """Test EDM evaluation with property classifier."""
    
    print("\n" + "="*60)
    print("TESTING EDM EVALUATION")
    print("="*60)
    
    cmd = [
        'python', 'eval_conditional_qm9.py',
        '--generators_path', generators_path,
        '--classifiers_path', classifiers_path,
        '--property', 'molecular_weight',
        '--task', 'edm',
        '--iterations', '5',  # Few iterations for testing
        '--batch_size', '4'
    ]
    
    print("Running EDM evaluation command:")
    print(' '.join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print("EDM evaluation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print("EDM evaluation completed successfully!")
        return True

def main():
    """Main testing function."""
    
    print("="*60)
    print("ASE CONDITIONAL EVALUATION TEST")
    print("="*60)
    
    # Create temporary directory for test
    test_dir = tempfile.mkdtemp(prefix='ase_test_')
    print(f"Test directory: {test_dir}")
    
    try:
        # Create test database
        db_path = os.path.join(test_dir, 'test_molecules.db')
        create_test_ase_database(db_path, n_molecules=24)
        
        # Test training
        if not test_training_command(db_path, test_dir):
            print("❌ Training test failed")
            return False
        
        generators_path = 'outputs/test_ase_multi_cond'
        
        # Test qualitative evaluation
        if not test_qualitative_evaluation(generators_path):
            print("❌ Qualitative evaluation test failed")
            return False
        
        # Test property classifier training
        if not test_property_classifier_training(db_path):
            print("❌ Property classifier training test failed")
            return False
        
        classifiers_path = 'qm9/property_prediction/outputs/test_class_molecular_weight'
        
        # Test EDM evaluation
        if not test_edm_evaluation(generators_path, classifiers_path):
            print("❌ EDM evaluation test failed")
            return False
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print()
        print("The updated eval_conditional_qm9.py successfully supports:")
        print("✅ ASE database datasets")
        print("✅ New molecular descriptor conditioning")
        print("✅ Qualitative conditional generation")
        print("✅ Property classifier training")
        print("✅ EDM evaluation with classifiers")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)