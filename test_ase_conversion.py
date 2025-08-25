#!/usr/bin/env python3
"""
Test script to convert QM9 dataset to ASE database format and validate equivalence.
This script will:
1. Download/load QM9 dataset
2. Convert it to ASE database format  
3. Load both original and ASE versions
4. Compare molecular properties and ensure identical results
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

from qm9.ase_database import ASEDatabaseReader, create_ase_database_from_qm9
from qm9.data.utils import initialize_datasets
from qm9.data.args import init_argparse
from qm9.analyze import analyze_stability_for_molecules
from configs.datasets_config import get_dataset_info
import qm9.dataset as dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_qm9_original():
    """Setup original QM9 dataset."""
    logging.info("Setting up original QM9 dataset...")
    
    # Create a simple config object
    class QM9Config:
        def __init__(self):
            self.batch_size = 32
            self.num_workers = 0
            self.remove_h = False
            self.filter_n_atoms = None
            self.datadir = 'qm9/temp'
            self.dataset = 'qm9'
            self.include_charges = True
            self.filter_molecule_size = None
            self.sequential = False
            
    cfg = QM9Config()
    
    # Initialize QM9 dataset
    args = init_argparse('qm9')
    args, datasets, num_species, charge_scale = initialize_datasets(
        args, cfg.datadir, cfg.dataset,
        subtract_thermo=args.subtract_thermo,
        force_download=args.force_download,
        remove_h=cfg.remove_h
    )
    
    return datasets, cfg


def convert_qm9_to_ase(datasets, output_dir='./test_ase_conversion'):
    """Convert QM9 datasets to ASE database format."""
    logging.info("Converting QM9 to ASE database format...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    ase_db_paths = {}
    
    for split_name, dataset_obj in datasets.items():
        # Save the split data to npz format first
        npz_path = os.path.join(output_dir, f'{split_name}.npz')
        data_dict = {}
        for key, value in dataset_obj.data.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.numpy()
            else:
                data_dict[key] = value
        np.savez_compressed(npz_path, **data_dict)
        
        # Convert to ASE database
        ase_db_path = os.path.join(output_dir, f'{split_name}.db')
        create_ase_database_from_qm9(npz_path, ase_db_path, split_name)
        ase_db_paths[split_name] = ase_db_path
        
        logging.info(f"Created ASE database for {split_name}: {ase_db_path}")
        
    return ase_db_paths


def load_ase_datasets(ase_db_paths):
    """Load datasets from ASE databases."""
    logging.info("Loading datasets from ASE databases...")
    
    ase_datasets = {}
    dataset_info = None
    
    for split_name, db_path in ase_db_paths.items():
        # Create ASE reader
        ase_reader = ASEDatabaseReader(db_path)
        
        # Get dataset info (use first split for configuration)
        if dataset_info is None:
            dataset_info = ase_reader.get_dataset_info('qm9_from_ase', with_h=True)
            
        # Load data
        ase_data = ase_reader.load_data_split()
        ase_datasets[split_name] = ase_data
        
        logging.info(f"Loaded {len(ase_data['positions'])} molecules from ASE {split_name}")
        
    return ase_datasets, dataset_info


def compare_datasets(original_datasets, ase_datasets, dataset_info, n_samples=100):
    """Compare original and ASE datasets for equivalence."""
    logging.info("Comparing original and ASE datasets...")
    
    comparisons = {}
    
    for split_name in ['train', 'test']:  # Focus on train and test splits
        if split_name not in original_datasets or split_name not in ase_datasets:
            continue
            
        logging.info(f"Comparing {split_name} split...")
        
        orig_data = original_datasets[split_name].data
        ase_data = ase_datasets[split_name]
        
        # Limit comparison to n_samples for efficiency
        n_compare = min(n_samples, len(orig_data['positions']), len(ase_data['positions']))
        
        comparison = {
            'split': split_name,
            'n_samples_compared': n_compare,
            'position_mse': 0.0,
            'charges_match': 0,
            'atom_types_match': 0,
            'num_atoms_match': 0
        }
        
        for i in range(n_compare):
            # Compare positions (after centering)
            orig_pos = orig_data['positions'][i].numpy()
            ase_pos = ase_data['positions'][i].numpy()
            
            orig_n_atoms = int(orig_data['num_atoms'][i])
            ase_n_atoms = int(ase_data['num_atoms'][i])
            
            if orig_n_atoms == ase_n_atoms:
                comparison['num_atoms_match'] += 1
                
                # Center both position sets
                orig_pos_centered = orig_pos[:orig_n_atoms] - np.mean(orig_pos[:orig_n_atoms], axis=0)
                ase_pos_centered = ase_pos[:ase_n_atoms] - np.mean(ase_pos[:ase_n_atoms], axis=0)
                
                # Calculate MSE (positions might be in different order)
                mse = np.mean((orig_pos_centered - ase_pos_centered) ** 2)
                comparison['position_mse'] += mse
                
                # Compare charges
                orig_charges = orig_data['charges'][i][:orig_n_atoms].numpy()
                ase_charges = ase_data['charges'][i][:ase_n_atoms].numpy()
                
                if np.array_equal(orig_charges, ase_charges):
                    comparison['charges_match'] += 1
                    
                # Compare atom types (one-hot vs direct)
                orig_atom_types = orig_data['one_hot'][i][:orig_n_atoms].argmax(1).numpy()
                ase_atom_types = ase_data['one_hot'][i][:ase_n_atoms].argmax(1).numpy()
                
                if np.array_equal(orig_atom_types, ase_atom_types):
                    comparison['atom_types_match'] += 1
        
        # Calculate percentages
        comparison['position_mse'] /= n_compare
        comparison['charges_match_pct'] = comparison['charges_match'] / n_compare * 100
        comparison['atom_types_match_pct'] = comparison['atom_types_match'] / n_compare * 100
        comparison['num_atoms_match_pct'] = comparison['num_atoms_match'] / n_compare * 100
        
        comparisons[split_name] = comparison
        
        logging.info(f"{split_name} comparison results:")
        logging.info(f"  Position MSE: {comparison['position_mse']:.6f}")
        logging.info(f"  Charges match: {comparison['charges_match_pct']:.1f}%")
        logging.info(f"  Atom types match: {comparison['atom_types_match_pct']:.1f}%")
        logging.info(f"  Num atoms match: {comparison['num_atoms_match_pct']:.1f}%")
        
    return comparisons


def test_molecular_analysis(ase_datasets, dataset_info, n_samples=50):
    """Test molecular analysis functions with ASE data."""
    logging.info("Testing molecular analysis with ASE data...")
    
    try:
        # Prepare molecule list for analysis
        test_data = ase_datasets['test']
        n_test = min(n_samples, len(test_data['positions']))
        
        molecule_list = {
            'x': test_data['positions'][:n_test],
            'one_hot': test_data['one_hot'][:n_test],
            'node_mask': (test_data['charges'][:n_test] > 0).float().unsqueeze(-1)
        }
        
        # Run molecular analysis
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
        else:
            logging.info("  No molecular metrics available (requires RDKit or OpenBabel)")
            
        return True
        
    except Exception as e:
        logging.error(f"Molecular analysis failed: {e}")
        return False


def main():
    """Main test function."""
    logging.info("Starting QM9 to ASE conversion and validation test...")
    
    try:
        # 1. Setup original QM9 dataset
        original_datasets, cfg = setup_qm9_original()
        logging.info(f"Loaded original QM9 with {len(original_datasets)} splits")
        
        # 2. Convert to ASE format
        ase_db_paths = convert_qm9_to_ase(original_datasets)
        
        # 3. Load ASE datasets
        ase_datasets, dataset_info = load_ase_datasets(ase_db_paths)
        
        # 4. Compare datasets
        comparisons = compare_datasets(original_datasets, ase_datasets, dataset_info)
        
        # 5. Test molecular analysis
        analysis_success = test_molecular_analysis(ase_datasets, dataset_info)
        
        # 6. Summary
        logging.info("\n=== TEST SUMMARY ===")
        all_tests_passed = True
        
        for split, comp in comparisons.items():
            logging.info(f"{split} split:")
            
            # Check if results are acceptable
            pos_ok = comp['position_mse'] < 1e-10  # Very small difference allowed
            charges_ok = comp['charges_match_pct'] > 95.0
            atoms_ok = comp['atom_types_match_pct'] > 95.0
            nums_ok = comp['num_atoms_match_pct'] > 95.0
            
            split_ok = pos_ok and charges_ok and atoms_ok and nums_ok
            all_tests_passed = all_tests_passed and split_ok
            
            status = "PASS" if split_ok else "FAIL"
            logging.info(f"  Overall: {status}")
            
        analysis_ok = analysis_success
        all_tests_passed = all_tests_passed and analysis_ok
        
        logging.info(f"Molecular analysis: {'PASS' if analysis_ok else 'FAIL'}")
        
        final_status = "ALL TESTS PASSED" if all_tests_passed else "SOME TESTS FAILED"
        logging.info(f"\nFinal result: {final_status}")
        
        return all_tests_passed
        
    except Exception as e:
        logging.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)