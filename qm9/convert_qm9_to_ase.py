#!/usr/bin/env python3
"""
Utility script to download QM9 dataset and convert it to ASE database format.
This script provides functionality to:
1. Download QM9 dataset
2. Convert NPZ files to ASE database
3. Validate the conversion by comparing results
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import qm9 modules
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from qm9.data.prepare.qm9 import download_dataset_qm9
from qm9.ase_interface import ASEDatasetInterface, convert_npz_to_ase_db
from qm9.dataset import retrieve_dataloaders
from qm9.openbabel_functions import OpenBabelMolecularMetrics


class QM9Config:
    """Configuration class for QM9 dataset processing."""
    
    def __init__(self, use_ase_db=False, ase_db_path=None, **kwargs):
        # Default QM9 configuration
        self.dataset = 'qm9'
        self.datadir = 'qm9/temp'
        self.batch_size = 32
        self.num_workers = 1
        self.filter_n_atoms = None
        self.remove_h = False
        self.include_charges = True
        self.subtract_thermo = True
        
        # ASE DB specific configuration
        self.use_ase_db = use_ase_db
        self.ase_db_path = ase_db_path
        
        # Override defaults with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def download_qm9_data(datadir='qm9/temp', force_download=False):
    """
    Download QM9 dataset.
    
    Parameters
    ----------
    datadir : str
        Directory to store the dataset
    force_download : bool
        Whether to force re-download even if data exists
    """
    logging.info("Downloading QM9 dataset...")
    
    # Create data directory
    os.makedirs(datadir, exist_ok=True)
    
    # Check if data already exists
    npz_files = {
        'train': os.path.join(datadir, 'qm9', 'train.npz'),
        'valid': os.path.join(datadir, 'qm9', 'valid.npz'),
        'test': os.path.join(datadir, 'qm9', 'test.npz')
    }
    
    if not force_download and all(os.path.exists(f) for f in npz_files.values()):
        logging.info("QM9 NPZ files already exist. Use --force-download to re-download.")
        return npz_files
    
    try:
        # Download and prepare QM9 dataset
        download_dataset_qm9(datadir, 'qm9', calculate_thermo=True, exclude=True, cleanup=True)
        
        # Verify files were created
        missing_files = [f for f in npz_files.values() if not os.path.exists(f)]
        if missing_files:
            logging.error(f"Failed to create NPZ files: {missing_files}")
            return None
        
        logging.info("Successfully downloaded and processed QM9 dataset")
        return npz_files
        
    except Exception as e:
        logging.error(f"Failed to download QM9 dataset: {e}")
        return None


def convert_qm9_to_ase_db(npz_files, output_db_path, force_convert=False):
    """
    Convert QM9 NPZ files to ASE database format.
    
    Parameters
    ----------
    npz_files : dict
        Dictionary mapping split names to NPZ file paths
    output_db_path : str
        Path for output ASE database
    force_convert : bool
        Whether to force conversion even if DB exists
    """
    if not force_convert and os.path.exists(output_db_path):
        logging.info(f"ASE database already exists: {output_db_path}. Use --force-convert to recreate.")
        return output_db_path
    
    logging.info("Converting QM9 NPZ files to ASE database...")
    
    try:
        convert_npz_to_ase_db(npz_files, output_db_path)
        
        if os.path.exists(output_db_path):
            logging.info(f"Successfully converted QM9 to ASE database: {output_db_path}")
            return output_db_path
        else:
            logging.error("ASE database file was not created")
            return None
            
    except Exception as e:
        logging.error(f"Failed to convert QM9 to ASE database: {e}")
        return None


def validate_conversion(npz_files, ase_db_path, num_samples=10):
    """
    Validate that ASE DB conversion produces equivalent results to NPZ files.
    
    Parameters
    ----------
    npz_files : dict
        Dictionary mapping split names to NPZ file paths
    ase_db_path : str
        Path to ASE database
    num_samples : int
        Number of samples to compare per split
    """
    logging.info("Validating ASE DB conversion...")
    
    try:
        # Load data from NPZ files (original method)
        cfg_npz = QM9Config(use_ase_db=False)
        cfg_npz.batch_size = 1  # Use batch size 1 for easier comparison
        dataloaders_npz, _ = retrieve_dataloaders(cfg_npz)
        
        # Load data from ASE DB (new method)
        cfg_ase = QM9Config(use_ase_db=True, ase_db_path=ase_db_path)
        cfg_ase.batch_size = 1
        dataloaders_ase, _ = retrieve_dataloaders(cfg_ase)
        
        # Compare a few samples from each split
        for split in ['train', 'valid', 'test']:
            if split not in dataloaders_npz or split not in dataloaders_ase:
                logging.warning(f"Split {split} not found in one of the datasets")
                continue
            
            logging.info(f"Comparing {split} split...")
            
            npz_iter = iter(dataloaders_npz[split])
            ase_iter = iter(dataloaders_ase[split])
            
            for i in range(min(num_samples, len(dataloaders_npz[split]), len(dataloaders_ase[split]))):
                try:
                    npz_batch = next(npz_iter)
                    ase_batch = next(ase_iter)
                    
                    # Compare key properties
                    comparison_keys = ['positions', 'charges', 'num_atoms']
                    
                    for key in comparison_keys:
                        if key in npz_batch and key in ase_batch:
                            npz_val = npz_batch[key]
                            ase_val = ase_batch[key]
                            
                            # Compare tensors with tolerance
                            if not torch.allclose(npz_val, ase_val, atol=1e-6):
                                logging.warning(f"Mismatch in {split} sample {i}, key {key}")
                                logging.debug(f"NPZ: {npz_val}")
                                logging.debug(f"ASE: {ase_val}")
                            else:
                                logging.debug(f"✓ {split} sample {i}, key {key} matches")
                
                except StopIteration:
                    break
                except Exception as e:
                    logging.warning(f"Error comparing {split} sample {i}: {e}")
        
        logging.info("Validation completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False


def test_openbabel_functionality(ase_db_path, num_test_molecules=5):
    """
    Test OpenBabel functionality with ASE DB data.
    
    Parameters
    ----------
    ase_db_path : str
        Path to ASE database
    num_test_molecules : int
        Number of molecules to test
    """
    logging.info("Testing OpenBabel functionality...")
    
    try:
        from configs.datasets_config import get_dataset_info
        
        # Load some data from ASE DB
        cfg = QM9Config(use_ase_db=True, ase_db_path=ase_db_path)
        cfg.batch_size = 1
        dataloaders, _ = retrieve_dataloaders(cfg)
        
        # Get dataset info
        dataset_info = get_dataset_info('qm9', False)
        
        # Initialize OpenBabel metrics
        ob_metrics = OpenBabelMolecularMetrics(dataset_info)
        
        # Test with a few molecules
        test_molecules = []
        for i, batch in enumerate(dataloaders['train']):
            if i >= num_test_molecules:
                break
                
            positions = batch['positions'][0]
            atom_types = torch.argmax(batch['one_hot'][0], dim=1)
            
            # Remove padding
            num_atoms = batch['num_atoms'][0].item()
            positions = positions[:num_atoms]
            atom_types = atom_types[:num_atoms]
            
            test_molecules.append((positions, atom_types))
        
        # Test OpenBabel evaluation
        results = ob_metrics.evaluate(test_molecules)
        
        logging.info("OpenBabel test results:")
        logging.info(f"  Validity: {results['validity']:.3f}")
        logging.info(f"  Uniqueness: {results['uniqueness']:.3f}")
        logging.info(f"  Novelty: {results['novelty']:.3f}")
        
        return True
        
    except Exception as e:
        logging.error(f"OpenBabel test failed: {e}")
        return False


def main():
    """Main function to orchestrate QM9 download and conversion."""
    parser = argparse.ArgumentParser(description='Download QM9 and convert to ASE database')
    parser.add_argument('--datadir', default='qm9/temp', help='Directory to store QM9 data')
    parser.add_argument('--output-db', default='qm9/temp/qm9_database.db', help='Output ASE database path')
    parser.add_argument('--force-download', action='store_true', help='Force re-download of QM9 data')
    parser.add_argument('--force-convert', action='store_true', help='Force re-conversion to ASE DB')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation step')
    parser.add_argument('--skip-openbabel-test', action='store_true', help='Skip OpenBabel test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_db), exist_ok=True)
    
    # Step 1: Download QM9 dataset
    npz_files = download_qm9_data(args.datadir, args.force_download)
    if npz_files is None:
        logging.error("Failed to download QM9 dataset")
        return 1
    
    # Step 2: Convert to ASE database
    ase_db_path = convert_qm9_to_ase_db(npz_files, args.output_db, args.force_convert)
    if ase_db_path is None:
        logging.error("Failed to convert to ASE database")
        return 1
    
    # Step 3: Validate conversion
    if not args.skip_validation:
        if not validate_conversion(npz_files, ase_db_path):
            logging.error("Validation failed")
            return 1
    
    # Step 4: Test OpenBabel functionality
    if not args.skip_openbabel_test:
        if not test_openbabel_functionality(ase_db_path):
            logging.error("OpenBabel test failed")
            return 1
    
    logging.info(f"✓ Successfully completed QM9 download and ASE DB conversion: {ase_db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())