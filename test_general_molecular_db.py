#!/usr/bin/env python3
"""
Test script for general molecular database support.

This script tests the new functionality for supporting arbitrary molecular
databases like PubChem with automatic configuration generation.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from ase import Atoms
from ase.db import connect

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qm9.general_molecular_db import (
    analyze_ase_database,
    create_optimal_dataset_config,
    validate_molecular_database,
    suggest_training_parameters,
    create_general_ase_config
)
from configs.datasets_config import get_dataset_info


def create_diverse_test_database(db_path, n_molecules=50):
    """Create a test database with diverse molecular types."""
    
    print(f"Creating diverse test database: {db_path}")
    
    db = connect(db_path)
    molecules = []
    
    # 1. Simple hydrocarbons
    for i in range(n_molecules // 5):
        # Methane variations
        positions = [
            [0, 0, 0],  # C
            [1.089 + np.random.normal(0, 0.01), 1.089 + np.random.normal(0, 0.01), 1.089 + np.random.normal(0, 0.01)],
            [1.089 + np.random.normal(0, 0.01), -1.089 + np.random.normal(0, 0.01), -1.089 + np.random.normal(0, 0.01)],
            [-1.089 + np.random.normal(0, 0.01), 1.089 + np.random.normal(0, 0.01), -1.089 + np.random.normal(0, 0.01)],
            [-1.089 + np.random.normal(0, 0.01), -1.089 + np.random.normal(0, 0.01), 1.089 + np.random.normal(0, 0.01)]
        ]
        mol = Atoms('CH4', positions=positions)
        molecules.append((mol, {
            'energy': -40.5 + np.random.normal(0, 0.5),
            'homo': -14.4 + np.random.normal(0, 0.5),
            'lumo': 6.0 + np.random.normal(0, 0.5)
        }))
    
    # 2. Oxygen-containing compounds
    for i in range(n_molecules // 5):
        # Water and alcohols
        if i % 2 == 0:
            # Water
            positions = [
                [0, 0, 0],
                [0.757 + np.random.normal(0, 0.02), 0.586 + np.random.normal(0, 0.02), 0],
                [-0.757 + np.random.normal(0, 0.02), 0.586 + np.random.normal(0, 0.02), 0]
            ]
            mol = Atoms('H2O', positions=positions)
        else:
            # Simple alcohol (methanol-like)
            positions = [
                [0, 0, 0],      # C
                [1.4, 0, 0],    # O
                [2.2, 0, 0],    # H (OH)
                [0, 1.1, 0],    # H
                [0, -0.55, 0.95], # H
                [0, -0.55, -0.95] # H
            ]
            mol = Atoms('CH4O', positions=positions)
        
        molecules.append((mol, {
            'energy': -76.4 + np.random.normal(0, 2.0),
            'homo': -12.6 + np.random.normal(0, 1.0),
            'lumo': 1.4 + np.random.normal(0, 1.0)
        }))
    
    # 3. Nitrogen-containing compounds
    for i in range(n_molecules // 5):
        # Ammonia and amines
        positions = [
            [0, 0, 0],  # N
            [1.017 + np.random.normal(0, 0.02), 0, 0],
            [-0.509 + np.random.normal(0, 0.02), 0.882 + np.random.normal(0, 0.02), 0],
            [-0.509 + np.random.normal(0, 0.02), -0.882 + np.random.normal(0, 0.02), 0]
        ]
        mol = Atoms('NH3', positions=positions)
        molecules.append((mol, {
            'energy': -56.5 + np.random.normal(0, 1.0),
            'homo': -10.8 + np.random.normal(0, 0.8),
            'lumo': 2.1 + np.random.normal(0, 0.5)
        }))
    
    # 4. Halogenated compounds
    for i in range(n_molecules // 5):
        # Different halides
        halogens = ['F', 'Cl', 'Br', 'I']
        halogen = halogens[i % len(halogens)]
        
        # Simple halide (HX)
        positions = [
            [0, 0, 0],      # H
            [1.2, 0, 0]     # X
        ]
        mol = Atoms(f'H{halogen}', positions=positions)
        molecules.append((mol, {
            'energy': -50.0 + np.random.normal(0, 5.0),
            'homo': -15.0 + np.random.normal(0, 2.0),
            'lumo': 3.0 + np.random.normal(0, 1.0)
        }))
    
    # 5. Larger/more complex molecules
    for i in range(n_molecules - len(molecules)):
        # Benzene-like or larger molecules
        if i % 3 == 0:
            # Benzene (C6H6)
            angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
            radius = 1.4
            positions = []
            symbols = []
            
            # Carbon ring
            for angle in angles:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append([x, y, 0])
                symbols.append('C')
            
            # Hydrogen atoms
            for angle in angles:
                x = (radius + 1.1) * np.cos(angle)
                y = (radius + 1.1) * np.sin(angle)
                positions.append([x, y, 0])
                symbols.append('H')
            
            mol = Atoms(''.join(symbols), positions=positions)
        else:
            # Random larger molecule
            n_atoms = np.random.randint(8, 20)
            symbols = np.random.choice(['C', 'N', 'O', 'H'], size=n_atoms, p=[0.4, 0.1, 0.1, 0.4])
            positions = np.random.normal(0, 2.0, size=(n_atoms, 3))
            mol = Atoms(symbols, positions=positions)
        
        molecules.append((mol, {
            'energy': -200.0 + np.random.normal(0, 20.0),
            'homo': -8.0 + np.random.normal(0, 2.0),
            'lumo': 2.0 + np.random.normal(0, 1.0),
            'gap': np.random.uniform(5.0, 15.0),
            'dipole_moment': np.random.uniform(0.0, 5.0)
        }))
    
    # Write to database
    for atoms, properties in molecules:
        db.write(atoms, data=properties)
    
    print(f"Created database with {len(molecules)} diverse molecules")
    return db_path


def test_database_analysis():
    """Test database analysis functionality."""
    print("\n" + "="*60)
    print("TEST: Database Analysis")
    print("="*60)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        create_diverse_test_database(db_path, n_molecules=30)
        
        # Test analysis
        analysis = analyze_ase_database(db_path)
        
        # Validate analysis results
        assert analysis['total_molecules'] == 30
        assert len(analysis['unique_elements']) >= 4  # Should have H, C, N, O at minimum
        assert analysis['max_atoms'] > 0
        assert len(analysis['available_properties']) >= 3  # energy, homo, lumo
        
        print("✅ Database analysis test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database analysis test failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_config_generation():
    """Test automatic configuration generation."""
    print("\n" + "="*60)
    print("TEST: Configuration Generation")
    print("="*60)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        create_diverse_test_database(db_path, n_molecules=25)
        
        # Test configuration generation
        config = create_general_ase_config(db_path, remove_h=False)
        
        # Validate configuration
        assert 'atom_encoder' in config
        assert 'atom_decoder' in config
        assert 'max_n_nodes' in config
        assert len(config['atom_encoder']) == len(config['atom_decoder'])
        assert len(config['colors_dic']) == len(config['atom_decoder'])
        assert len(config['radius_dic']) == len(config['atom_decoder'])
        assert config['with_h'] == True
        
        # Test with hydrogen removal
        config_no_h = create_general_ase_config(db_path, remove_h=True)
        assert config_no_h['with_h'] == False
        assert 'H' not in config_no_h['atom_decoder']
        
        print("✅ Configuration generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration generation test failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_dataset_info_integration():
    """Test integration with get_dataset_info function."""
    print("\n" + "="*60)
    print("TEST: Dataset Info Integration")
    print("="*60)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        create_diverse_test_database(db_path, n_molecules=20)
        
        # Test get_dataset_info with automatic config generation
        config = get_dataset_info('ase_db', remove_h=False, ase_db_path=db_path)
        
        # Validate that we got a proper config
        assert 'atom_encoder' in config
        assert 'atom_decoder' in config
        assert len(config['atom_decoder']) > 0
        
        # Test fallback behavior
        config_fallback = get_dataset_info('ase_db', remove_h=False, ase_db_path=None)
        assert 'is_fallback_config' in config_fallback
        
        print("✅ Dataset info integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dataset info integration test failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_validation():
    """Test database validation functionality."""
    print("\n" + "="*60)
    print("TEST: Database Validation")
    print("="*60)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        create_diverse_test_database(db_path, n_molecules=25)
        
        # Test validation
        is_valid, issues, recommendations = validate_molecular_database(db_path)
        
        # Should be valid for a well-formed database
        assert is_valid, f"Database should be valid, but got issues: {issues}"
        
        print("✅ Database validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database validation test failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_training_suggestions():
    """Test training parameter suggestions."""
    print("\n" + "="*60)
    print("TEST: Training Suggestions")
    print("="*60)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        create_diverse_test_database(db_path, n_molecules=30)
        
        # Analyze and get suggestions
        analysis = analyze_ase_database(db_path)
        suggestions = suggest_training_parameters(analysis)
        
        # Validate suggestions
        required_params = ['batch_size', 'n_epochs', 'nf', 'n_layers', 'lr', 'diffusion_steps']
        for param in required_params:
            assert param in suggestions, f"Missing required parameter: {param}"
            assert isinstance(suggestions[param], (int, float)), f"Parameter {param} should be numeric"
        
        print("✅ Training suggestions test passed")
        return True
        
    except Exception as e:
        print(f"❌ Training suggestions test failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_pubchem_like_database():
    """Test with a PubChem-like database containing many different elements."""
    print("\n" + "="*60)
    print("TEST: PubChem-like Database")
    print("="*60)
    
    # Create test database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = connect(db_path)
        
        # Create molecules with many different elements (PubChem-like)
        diverse_molecules = [
            # Organic compounds
            (['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], "ethane"),
            (['C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], "propanol"),
            (['C', 'C', 'N', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], "ethylamine"),
            
            # Inorganic compounds
            (['Na', 'Cl'], "sodium chloride"),
            (['K', 'Br'], "potassium bromide"),
            (['Ca', 'F', 'F'], "calcium fluoride"),
            
            # Organometallic compounds
            (['Fe', 'C', 'O', 'C', 'O', 'C', 'O'], "iron carbonyl"),
            (['Zn', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], "zinc compound"),
            
            # Silicon compounds
            (['Si', 'O', 'Si', 'O', 'H', 'H'], "silicate"),
            (['Si', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], "silane"),
            
            # Phosphorus compounds
            (['P', 'O', 'O', 'O', 'O', 'H', 'H', 'H'], "phosphoric acid"),
            
            # Sulfur compounds
            (['S', 'O', 'O'], "sulfur dioxide"),
            (['C', 'S', 'H', 'H', 'H', 'H'], "methanethiol"),
        ]
        
        for i, (symbols, name) in enumerate(diverse_molecules):
            n_atoms = len(symbols)
            positions = np.random.normal(0, 1.5, size=(n_atoms, 3))  # Random positions
            
            mol = Atoms(symbols, positions=positions)
            properties = {
                'name': name,
                'energy': np.random.normal(-100, 50),
                'formation_energy': np.random.normal(-50, 25),
                'molecular_weight': sum([1 if s == 'H' else 12 if s == 'C' else 16 if s == 'O' else 14 if s == 'N' else 35 for s in symbols])
            }
            
            db.write(mol, data=properties)
        
        print(f"Created PubChem-like database with {len(diverse_molecules)} molecules")
        
        # Test analysis on this diverse database
        analysis = analyze_ase_database(db_path)
        print(f"Found elements: {', '.join(analysis['unique_elements'])}")
        
        # Should handle many different elements
        assert len(analysis['unique_elements']) >= 8, "Should find many different elements"
        
        # Test configuration generation
        config = create_general_ase_config(db_path, remove_h=False)
        
        # Should handle all elements properly
        assert len(config['atom_encoder']) >= 8, "Should handle many different atom types"
        assert all(elem in config['atom_encoder'] for elem in analysis['unique_elements'])
        
        print("✅ PubChem-like database test passed")
        return True
        
    except Exception as e:
        print(f"❌ PubChem-like database test failed: {e}")
        return False
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests():
    """Run all tests."""
    print("GENERAL MOLECULAR DATABASE SUPPORT TESTS")
    print("=" * 80)
    
    tests = [
        test_database_analysis,
        test_config_generation,
        test_dataset_info_integration,
        test_validation,
        test_training_suggestions,
        test_pubchem_like_database
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("General molecular database support is working correctly.")
        return True
    else:
        print(f"\n💥 {failed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)