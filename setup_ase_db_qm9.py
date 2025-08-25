#!/usr/bin/env python3
"""
Script to download QM9 dataset and convert it to ASE database format.
This provides a test dataset for the ASE DB functionality.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qm9.data.prepare.qm9 import download_dataset_qm9
from qm9.ase_db_dataset import convert_qm9_to_ase_db
from configs.datasets_config import get_dataset_info


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def download_qm9_data(data_dir, force_download=False):
    """Download QM9 dataset if not already present."""
    qm9_dir = os.path.join(data_dir, 'qm9')
    
    # Check if QM9 data already exists
    train_file = os.path.join(qm9_dir, 'train.npz')
    valid_file = os.path.join(qm9_dir, 'valid.npz')
    test_file = os.path.join(qm9_dir, 'test.npz')
    
    if not force_download and all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
        logging.info("QM9 dataset already exists, skipping download")
        return qm9_dir
    
    logging.info("Downloading QM9 dataset...")
    os.makedirs(qm9_dir, exist_ok=True)
    
    try:
        download_dataset_qm9(
            datadir=data_dir,
            dataname='qm9',
            splits=None,
            calculate_thermo=True,
            exclude=True,
            cleanup=True
        )
        logging.info("QM9 dataset downloaded successfully")
        return qm9_dir
    except Exception as e:
        logging.error(f"Failed to download QM9 dataset: {e}")
        raise


def convert_to_ase_db(qm9_dir, output_path, include_properties=True):
    """Convert QM9 dataset to ASE database format."""
    logging.info(f"Converting QM9 to ASE database: {output_path}")
    
    try:
        convert_qm9_to_ase_db(
            qm9_data_dir=qm9_dir,
            output_db_path=output_path,
            include_properties=include_properties
        )
        logging.info(f"Conversion completed successfully: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to convert QM9 to ASE database: {e}")
        raise


def test_ase_db_loading(db_path):
    """Test that the ASE database can be loaded properly."""
    logging.info("Testing ASE database loading...")
    
    try:
        from ase.db import connect
        from qm9.ase_db_dataset import load_ase_db_datasets
        
        # Test basic ASE DB connection
        db = connect(db_path)
        count = db.count()
        logging.info(f"ASE database contains {count} molecules")
        
        # Test dataset loading
        dataset_info = get_dataset_info('ase_db_qm9', remove_h=False)
        datasets = load_ase_db_datasets(
            db_path=db_path,
            dataset_info=dataset_info,
            split_ratio=(0.8, 0.1, 0.1),
            remove_h=False,
            max_atoms=None,
            random_seed=42
        )
        
        logging.info("Dataset split sizes:")
        for split, dataset in datasets.items():
            logging.info(f"  {split}: {len(dataset)} molecules")
        
        # Test a sample
        sample = datasets['train'][0]
        logging.info(f"Sample data keys: {list(sample.keys())}")
        logging.info(f"Sample positions shape: {sample['positions'].shape}")
        logging.info(f"Sample charges shape: {sample['charges'].shape}")
        
        logging.info("ASE database loading test passed!")
        return True
        
    except Exception as e:
        logging.error(f"ASE database loading test failed: {e}")
        return False


def test_openbabel_functions(db_path):
    """Test OpenBabel functions with ASE database."""
    logging.info("Testing OpenBabel functions...")
    
    try:
        from qm9.openbabel_functions import BasicMolecularMetricsOB
        from qm9.ase_db_dataset import load_ase_db_datasets
        
        # Load dataset
        dataset_info = get_dataset_info('ase_db_qm9', remove_h=False)
        datasets = load_ase_db_datasets(
            db_path=db_path,
            dataset_info=dataset_info,
            split_ratio=(0.8, 0.1, 0.1),
            remove_h=False,
            max_atoms=10,  # Small molecules for testing
            random_seed=42
        )
        
        # Get a few samples for testing
        test_molecules = []
        for i in range(min(5, len(datasets['test']))):
            sample = datasets['test'][i]
            positions = sample['positions']
            charges = sample['charges']
            atom_mask = sample['atom_mask']
            
            # Filter to actual atoms
            n_atoms = torch.sum(atom_mask).item()
            positions = positions[:n_atoms]
            
            # Convert charges to atom types
            atom_types = []
            for charge in charges[:n_atoms]:
                charge_int = int(charge.item())
                # Map atomic number to atom type index
                if charge_int == 1:  # H
                    atom_types.append(0)
                elif charge_int == 6:  # C
                    atom_types.append(1)
                elif charge_int == 7:  # N
                    atom_types.append(2)
                elif charge_int == 8:  # O
                    atom_types.append(3)
                elif charge_int == 9:  # F
                    atom_types.append(4)
                else:
                    break
            else:
                atom_types = torch.tensor(atom_types, dtype=torch.long)
                test_molecules.append((positions, atom_types))
        
        if len(test_molecules) == 0:
            logging.warning("No suitable test molecules found")
            return False
        
        # Test molecular metrics
        import torch
        metrics = BasicMolecularMetricsOB(dataset_info)
        results = metrics.evaluate(test_molecules)
        
        logging.info(f"OpenBabel metrics results: {results}")
        logging.info("OpenBabel functions test passed!")
        return True
        
    except Exception as e:
        logging.error(f"OpenBabel functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download QM9 and convert to ASE database")
    parser.add_argument('--data-dir', default='./qm9/temp', 
                       help='Directory to store QM9 data')
    parser.add_argument('--output-db', default='./qm9/temp/qm9_ase.db',
                       help='Output path for ASE database')
    parser.add_argument('--force-download', action='store_true',
                       help='Force redownload of QM9 data')
    parser.add_argument('--skip-properties', action='store_true',
                       help='Skip molecular properties in conversion')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run tests on existing database')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # Create directories
        os.makedirs(os.path.dirname(args.output_db), exist_ok=True)
        
        if not args.test_only:
            # Download QM9 dataset
            qm9_dir = download_qm9_data(args.data_dir, args.force_download)
            
            # Convert to ASE database
            convert_to_ase_db(
                qm9_dir=qm9_dir,
                output_path=args.output_db,
                include_properties=not args.skip_properties
            )
        
        # Test the database
        if os.path.exists(args.output_db):
            test_ase_db_loading(args.output_db)
            test_openbabel_functions(args.output_db)
        else:
            logging.error(f"Database file not found: {args.output_db}")
            return 1
        
        logging.info("All tests completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())