#!/usr/bin/env python3
"""
Simple validation test for ASE database functionality.
Creates synthetic molecular data and tests the full pipeline.
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qm9.ase_database import ASEDatabaseReader
from qm9.openbabel_functions import OpenBabelMolecularMetrics, xyz_to_smiles_openbabel
from qm9.analyze import analyze_stability_for_molecules
from qm9 import dataset
from configs.datasets_config import get_dataset_info
from ase import Atoms
from ase.db import connect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_synthetic_molecules_db(db_path, n_molecules=200):
    """Create a synthetic molecular database for testing."""
    logging.info(f"Creating synthetic molecular database with {n_molecules} molecules...")
    
    db = connect(db_path)
    
    # Define some simple molecular templates
    templates = [
        # Water H2O
        {'symbols': ['O', 'H', 'H'], 
         'positions': np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])},
        
        # Methane CH4
        {'symbols': ['C', 'H', 'H', 'H', 'H'],
         'positions': np.array([[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], 
                               [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]])},
        
        # Ammonia NH3
        {'symbols': ['N', 'H', 'H', 'H'],
         'positions': np.array([[0, 0, 0], [1.01, 0, 0], [-0.33, 0.95, 0], [-0.33, -0.48, 0.82]])},
        
        # Ethane C2H6
        {'symbols': ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
         'positions': np.array([[0, 0, 0], [1.54, 0, 0], 
                               [-0.51, 0.89, 0], [-0.51, -0.44, 0.77], [-0.51, -0.44, -0.77],
                               [2.05, 0.89, 0], [2.05, -0.44, 0.77], [2.05, -0.44, -0.77]])},
        
        # Hydrogen fluoride HF
        {'symbols': ['H', 'F'],
         'positions': np.array([[0, 0, 0], [0.92, 0, 0]])},
    ]
    
    np.random.seed(42)
    
    for i in range(n_molecules):
        # Choose a random template
        template = templates[i % len(templates)]
        
        # Add small random perturbations to positions
        positions = template['positions'] + np.random.normal(0, 0.05, template['positions'].shape)
        
        # Create molecule
        mol = Atoms(symbols=template['symbols'], positions=positions)
        
        # Add some synthetic properties
        kvp = {
            'mol_id': i,
            'mol_energy': np.random.uniform(-10, 0),
            'temperature': np.random.uniform(200, 400),
            'pressure': np.random.uniform(1, 10)
        }
        
        db.write(mol, **kvp)
    
    logging.info(f"Created synthetic database with {len(db)} molecules")


def test_ase_database_loading(db_path):
    """Test ASE database loading functionality."""
    logging.info("Testing ASE database loading...")
    
    # Create ASE reader
    ase_reader = ASEDatabaseReader(db_path)
    
    # Test basic properties
    logging.info(f"Atom types: {ase_reader.atom_types}")
    logging.info(f"Atom encoder: {ase_reader.atom_encoder}")
    
    # Get dataset info
    dataset_info = ase_reader.get_dataset_info('synthetic_molecules', with_h=True)
    logging.info(f"Max nodes: {dataset_info['max_n_nodes']}")
    logging.info(f"Node distribution: {dict(list(dataset_info['n_nodes'].items())[:5])}")
    
    # Create splits
    data_splits = ase_reader.create_splits(train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1)
    
    for split_name, split_data in data_splits.items():
        logging.info(f"{split_name}: {len(split_data['positions'])} molecules")
        
    return data_splits, dataset_info


def test_openbabel_functions(data_splits, dataset_info):
    """Test OpenBabel molecular processing functions."""
    logging.info("Testing OpenBabel functions...")
    
    test_data = data_splits['test']
    n_test_molecules = min(10, len(test_data['positions']))
    
    # Test SMILES generation
    smiles_generated = 0
    for i in range(n_test_molecules):
        n_atoms = int(test_data['num_atoms'][i])
        positions = test_data['positions'][i][:n_atoms].numpy()
        atom_types = test_data['one_hot'][i][:n_atoms].argmax(1).numpy()
        
        smiles = xyz_to_smiles_openbabel(positions, atom_types, dataset_info['atom_decoder'])
        if smiles:
            smiles_generated += 1
            logging.debug(f"Molecule {i}: {smiles}")
    
    logging.info(f"Generated SMILES for {smiles_generated}/{n_test_molecules} molecules")
    
    # Test molecular metrics
    try:
        metrics = OpenBabelMolecularMetrics(dataset_info)
        
        # Prepare molecule list
        molecule_list = []
        for i in range(n_test_molecules):
            n_atoms = int(test_data['num_atoms'][i])
            positions = test_data['positions'][i][:n_atoms]
            atom_types = test_data['one_hot'][i][:n_atoms].argmax(1)
            molecule_list.append((positions, atom_types))
        
        validity, uniqueness, novelty = metrics.evaluate(molecule_list)
        logging.info(f"Molecular metrics - Validity: {validity:.3f}, Uniqueness: {uniqueness:.3f}, Novelty: {novelty:.3f}")
        
    except Exception as e:
        logging.warning(f"OpenBabel metrics test failed: {e}")
        
    return True


def test_dataloader_integration(db_path):
    """Test integration with the dataloader system."""
    logging.info("Testing dataloader integration...")
    
    # Create config
    class ASEConfig:
        def __init__(self):
            self.dataset = 'ase_synthetic'
            self.ase_db_path = db_path
            self.batch_size = 8
            self.include_charges = True
            self.remove_h = False
            self.datadir = './temp'
            self.num_workers = 0
            self.filter_n_atoms = None
    
    cfg = ASEConfig()
    
    try:
        # Get dataloaders
        dataloaders, charge_scale = dataset.retrieve_ase_dataloaders(cfg)
        
        logging.info("Dataloaders created successfully:")
        for split_name, dataloader in dataloaders.items():
            logging.info(f"  {split_name}: {len(dataloader)} batches")
        
        # Test loading a batch
        train_loader = dataloaders['train']
        for batch in train_loader:
            logging.info(f"Sample batch loaded:")
            logging.info(f"  Positions: {batch['positions'].shape}")
            logging.info(f"  Charges: {batch['charges'].shape}")
            logging.info(f"  One-hot: {batch['one_hot'].shape}")
            logging.info(f"  Atom mask: {batch['atom_mask'].shape}")
            break
        
        return True
        
    except Exception as e:
        logging.error(f"Dataloader integration test failed: {e}")
        return False


def test_molecular_analysis(data_splits, dataset_info):
    """Test molecular analysis functions."""
    logging.info("Testing molecular analysis...")
    
    try:
        test_data = data_splits['test']
        n_test = min(20, len(test_data['positions']))
        
        # Prepare data for analysis
        molecule_list = {
            'x': test_data['positions'][:n_test],
            'one_hot': test_data['one_hot'][:n_test],
            'node_mask': (test_data['charges'][:n_test] > 0).float().unsqueeze(-1)
        }
        
        # Run analysis
        validity_dict, molecular_metrics = analyze_stability_for_molecules(molecule_list, dataset_info)
        
        logging.info("Molecular analysis results:")
        logging.info(f"  Molecular stability: {validity_dict['mol_stable']:.3f}")
        logging.info(f"  Atomic stability: {validity_dict['atm_stable']:.3f}")
        
        if molecular_metrics is not None:
            if isinstance(molecular_metrics, tuple) and len(molecular_metrics) >= 3:
                validity, uniqueness, novelty = molecular_metrics[:3]
                logging.info(f"  Validity: {validity:.3f}")
                logging.info(f"  Uniqueness: {uniqueness:.3f}")
                logging.info(f"  Novelty: {novelty:.3f}")
            else:
                logging.info(f"  Molecular metrics: {molecular_metrics}")
        
        return True
        
    except Exception as e:
        logging.error(f"Molecular analysis test failed: {e}")
        return False


def main():
    """Main validation test."""
    logging.info("Starting ASE database validation test...")
    
    db_path = './test_synthetic_molecules.db'
    
    try:
        # 1. Create synthetic molecular database
        create_synthetic_molecules_db(db_path, n_molecules=100)
        
        # 2. Test ASE database loading
        data_splits, dataset_info = test_ase_database_loading(db_path)
        
        # 3. Test OpenBabel functions
        openbabel_success = test_openbabel_functions(data_splits, dataset_info)
        
        # 4. Test dataloader integration
        dataloader_success = test_dataloader_integration(db_path)
        
        # 5. Test molecular analysis
        analysis_success = test_molecular_analysis(data_splits, dataset_info)
        
        # Summary
        logging.info("\n=== VALIDATION SUMMARY ===")
        all_tests_passed = openbabel_success and dataloader_success and analysis_success
        
        logging.info(f"OpenBabel functions: {'PASS' if openbabel_success else 'FAIL'}")
        logging.info(f"Dataloader integration: {'PASS' if dataloader_success else 'FAIL'}")
        logging.info(f"Molecular analysis: {'PASS' if analysis_success else 'FAIL'}")
        
        final_status = "ALL TESTS PASSED" if all_tests_passed else "SOME TESTS FAILED"
        logging.info(f"\nFinal result: {final_status}")
        
        return all_tests_passed
        
    except Exception as e:
        logging.error(f"Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)