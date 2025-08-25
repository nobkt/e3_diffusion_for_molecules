#!/usr/bin/env python3
"""
Simple test script to verify ASE and OpenBabel integration works.
"""

import sys
import os
import logging
import torch
import numpy as np

# Add the parent directory to the path so we can import qm9 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported."""
    logging.info("Testing imports...")
    
    try:
        import ase
        logging.info(f"✓ ASE imported successfully (version: {ase.__version__})")
    except ImportError as e:
        logging.error(f"✗ Failed to import ASE: {e}")
        return False
    
    try:
        from openbabel import pybel
        logging.info("✓ OpenBabel (pybel) imported successfully")
    except ImportError as e:
        logging.error(f"✗ Failed to import OpenBabel: {e}")
        return False
    
    try:
        from qm9.ase_interface import ASEDatasetInterface
        logging.info("✓ ASE interface imported successfully")
    except ImportError as e:
        logging.error(f"✗ Failed to import ASE interface: {e}")
        return False
    
    try:
        from qm9.openbabel_functions import OpenBabelMolecularMetrics, build_molecule_openbabel
        logging.info("✓ OpenBabel functions imported successfully")
    except ImportError as e:
        logging.error(f"✗ Failed to import OpenBabel functions: {e}")
        return False
    
    return True


def test_openbabel_basic():
    """Test basic OpenBabel functionality."""
    logging.info("Testing basic OpenBabel functionality...")
    
    try:
        from qm9.openbabel_functions import build_molecule_openbabel, mol2smiles_openbabel
        
        # Test building a simple molecule (ethane: C2H6)
        positions = torch.tensor([
            [0.0, 0.0, 0.0],      # C
            [1.54, 0.0, 0.0],     # C  
            [-0.51, 0.89, 0.0],   # H
            [-0.51, -0.89, 0.0],  # H
            [-0.51, 0.0, 0.89],   # H
            [2.05, 0.89, 0.0],    # H
            [2.05, -0.89, 0.0],   # H
            [2.05, 0.0, 0.89],    # H
        ]).float()
        
        atom_types = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1])  # C, C, H, H, H, H, H, H
        
        dataset_info = {
            'atom_decoder': {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'},
            'name': 'test'
        }
        
        # Build molecule
        mol = build_molecule_openbabel(positions, atom_types, dataset_info)
        
        if mol is None:
            logging.error("✗ Failed to build molecule")
            return False
        
        logging.info("✓ Successfully built molecule")
        
        # Convert to SMILES
        smiles = mol2smiles_openbabel(mol)
        if smiles:
            logging.info(f"✓ Generated SMILES: {smiles}")
        else:
            logging.warning("Could not generate SMILES, but molecule was built")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ OpenBabel test failed: {e}")
        return False


def test_ase_interface():
    """Test ASE database interface with synthetic data."""
    logging.info("Testing ASE database interface...")
    
    try:
        from qm9.ase_interface import ASEDatasetInterface
        
        # Create test database path
        test_db_path = '/tmp/test_ase.db'
        
        # Remove existing test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        # Create synthetic molecular data
        synthetic_data = {
            'train': {
                'positions': torch.tensor([
                    [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 0.0]],  # Molecule 1 (padded)
                    [[0.0, 0.0, 0.0], [0.0, 1.4, 0.0], [1.4, 1.4, 0.0]]   # Molecule 2
                ]).float(),
                'charges': torch.tensor([
                    [6, 6, 0],  # C-C (padded)
                    [8, 1, 1]   # O-H-H (water)
                ]).long(),
                'num_atoms': torch.tensor([2, 3]).long(),
                'U0': torch.tensor([-1.5, -2.3]).float(),
                'gap': torch.tensor([0.5, 0.8]).float()
            }
        }
        
        # Test saving to ASE DB
        ase_interface = ASEDatasetInterface(test_db_path)
        ase_interface.save_molecules_to_db(synthetic_data)
        ase_interface.close()
        
        if not os.path.exists(test_db_path):
            logging.error("✗ ASE database was not created")
            return False
        
        logging.info("✓ Successfully saved data to ASE database")
        
        # Test loading from ASE DB
        ase_interface = ASEDatasetInterface(test_db_path)
        loaded_data = ase_interface.load_molecules_from_db(['train'])
        ase_interface.close()
        
        if 'train' not in loaded_data:
            logging.error("✗ Failed to load data from ASE database")
            return False
        
        # Verify data consistency
        original_num_atoms = synthetic_data['train']['num_atoms']
        loaded_num_atoms = loaded_data['train']['num_atoms']
        
        if not torch.equal(original_num_atoms, loaded_num_atoms):
            logging.error("✗ Loaded data does not match original")
            return False
        
        logging.info("✓ Successfully loaded and verified data from ASE database")
        
        # Clean up
        os.remove(test_db_path)
        
        return True
        
    except Exception as e:
        logging.error(f"✗ ASE interface test failed: {e}")
        return False


def main():
    """Run all tests."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Starting integration tests...")
    
    tests = [
        ("Import test", test_imports),
        ("OpenBabel basic test", test_openbabel_basic),
        ("ASE interface test", test_ase_interface),
    ]
    
    results = []
    for test_name, test_func in tests:
        logging.info(f"\n{'='*50}")
        logging.info(f"Running: {test_name}")
        logging.info('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logging.info(f"✓ {test_name} PASSED")
            else:
                logging.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logging.error(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logging.info(f"\n{'='*50}")
    logging.info("TEST SUMMARY")
    logging.info('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logging.info(f"{test_name}: {status}")
    
    logging.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logging.info("🎉 All tests passed!")
        return 0
    else:
        logging.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())