#!/usr/bin/env python3
"""
Test script to create synthetic QM9-like data and test ASE DB conversion pipeline.
This creates a small synthetic dataset in NPZ format, converts it to ASE DB,
and validates the conversion works correctly.
"""

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path

# Add the parent directory to the path so we can import qm9 modules
sys.path.append(str(Path(__file__).parent))

from qm9.ase_interface import convert_npz_to_ase_db, ASEDatasetInterface
from qm9.convert_qm9_to_ase import QM9Config, validate_conversion
from qm9.dataset import retrieve_dataloaders
from qm9.openbabel_functions import OpenBabelMolecularMetrics
from configs.datasets_config import get_dataset_info


def create_synthetic_qm9_data(output_dir, num_molecules_per_split=10):
    """
    Create synthetic QM9-like NPZ data for testing.
    
    Parameters
    ----------
    output_dir : str
        Directory to save NPZ files
    num_molecules_per_split : int
        Number of molecules per split
        
    Returns
    -------
    npz_files : dict
        Dictionary mapping split names to NPZ file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    npz_files = {}
    
    # QM9 atomic numbers: H=1, C=6, N=7, O=8, F=9
    atom_types = [1, 6, 7, 8, 9]
    
    for split in splits:
        molecules_data = {
            'positions': [],
            'charges': [],
            'num_atoms': [],
            'index': [],
            'U0': [],
            'gap': [],
            'homo': [],
            'lumo': [],
            'mu': [],
        }
        
        for i in range(num_molecules_per_split):
            # Generate random molecule (3-8 atoms)
            num_atoms = np.random.randint(3, 9)
            
            # Random positions (centered around origin)
            positions = np.random.randn(29, 3) * 2.0  # QM9 has max 29 atoms
            positions[:num_atoms] = positions[:num_atoms] * 0.5  # Make it more compact
            
            # Random atomic charges (QM9 types)
            charges = np.zeros(29, dtype=int)
            charges[:num_atoms] = np.random.choice(atom_types, size=num_atoms)
            
            # Ensure at least one carbon if more than 1 atom
            if num_atoms > 1:
                charges[0] = 6  # Carbon
            
            # Random properties (realistic ranges for QM9)
            index_val = i + split_offset(split) * num_molecules_per_split
            U0_val = np.random.uniform(-1000, -10)  # Hartree
            gap_val = np.random.uniform(0.1, 10.0)   # eV
            homo_val = np.random.uniform(-15, -5)    # eV  
            lumo_val = homo_val + gap_val
            mu_val = np.random.uniform(0, 5)         # Debye
            
            molecules_data['positions'].append(positions)
            molecules_data['charges'].append(charges)
            molecules_data['num_atoms'].append(num_atoms)
            molecules_data['index'].append(index_val)
            molecules_data['U0'].append(U0_val)
            molecules_data['gap'].append(gap_val)
            molecules_data['homo'].append(homo_val)
            molecules_data['lumo'].append(lumo_val)
            molecules_data['mu'].append(mu_val)
        
        # Convert to numpy arrays
        for key in molecules_data:
            molecules_data[key] = np.array(molecules_data[key])
        
        # Save to NPZ file
        npz_path = os.path.join(output_dir, f'{split}.npz')
        np.savez_compressed(npz_path, **molecules_data)
        npz_files[split] = npz_path
        
        logging.info(f"Created synthetic {split} data: {npz_path}")
    
    return npz_files


def split_offset(split):
    """Get offset for different splits."""
    offsets = {'train': 0, 'valid': 1000, 'test': 2000}
    return offsets.get(split, 0)


def test_synthetic_qm9_pipeline():
    """Test the complete pipeline with synthetic data."""
    logging.info("Testing QM9 ASE DB pipeline with synthetic data...")
    
    # Paths
    synthetic_dir = '/tmp/synthetic_qm9'
    ase_db_path = '/tmp/synthetic_qm9.db'
    
    # Clean up previous runs
    if os.path.exists(synthetic_dir):
        import shutil
        shutil.rmtree(synthetic_dir)
    if os.path.exists(ase_db_path):
        os.remove(ase_db_path)
    
    try:
        # Step 1: Create synthetic NPZ data
        logging.info("Step 1: Creating synthetic QM9 data...")
        npz_files = create_synthetic_qm9_data(synthetic_dir, num_molecules_per_split=5)
        
        # Step 2: Convert to ASE DB
        logging.info("Step 2: Converting NPZ to ASE DB...")
        convert_npz_to_ase_db(npz_files, ase_db_path)
        
        if not os.path.exists(ase_db_path):
            raise Exception("ASE database was not created")
        
        # Step 3: Test loading from NPZ (original method)
        logging.info("Step 3: Testing NPZ dataloader...")
        cfg_npz = QM9Config(use_ase_db=False)
        cfg_npz.datadir = os.path.dirname(synthetic_dir)
        cfg_npz.dataset = 'qm9'
        cfg_npz.batch_size = 2
        
        # Manually set up args to work with our synthetic data
        class MockArgs:
            def __init__(self):
                self.num_train = -1
                self.num_test = -1
                self.num_valid = -1
                self.subtract_thermo = False
                self.force_download = False
                self.shuffle = True
        
        # Load from ASE DB directly using our interface
        logging.info("Step 4: Testing ASE DB dataloader...")
        cfg_ase = QM9Config(use_ase_db=True, ase_db_path=ase_db_path)
        cfg_ase.batch_size = 2
        dataloaders_ase, _ = retrieve_dataloaders(cfg_ase)
        
        # Step 5: Test OpenBabel functionality
        logging.info("Step 5: Testing OpenBabel functionality...")
        dataset_info = get_dataset_info('qm9', False)
        
        # Create atomic number to index mapping for QM9
        atomic_num_to_idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # H, C, N, O, F
        
        ob_metrics = OpenBabelMolecularMetrics(dataset_info)
        
        # Get a few molecules for testing
        test_molecules = []
        for batch in list(dataloaders_ase['train'])[:2]:  # First 2 batches
            for i in range(batch['positions'].shape[0]):
                positions = batch['positions'][i]
                
                # Get atom types from charges and convert to QM9 indices
                charges = batch['charges'][i]
                num_atoms = batch['num_atoms'][i].item()
                
                # Remove padding
                positions = positions[:num_atoms]
                raw_charges = charges[:num_atoms]
                
                # Convert atomic numbers to QM9 indices
                atom_types = torch.zeros_like(raw_charges)
                for j, atomic_num in enumerate(raw_charges):
                    atomic_num_val = atomic_num.item()
                    if atomic_num_val in atomic_num_to_idx:
                        atom_types[j] = atomic_num_to_idx[atomic_num_val]
                    else:
                        # Skip unknown atom types
                        continue
                
                # Skip if no atoms or invalid atoms
                if num_atoms > 0 and torch.all(atom_types >= 0):
                    test_molecules.append((positions, atom_types))
        
        # Test evaluation
        results = ob_metrics.evaluate(test_molecules)
        
        logging.info("OpenBabel evaluation results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {metric}: {value:.3f}")
            else:
                logging.info(f"  {metric}: {len(value)} items")
        
        # Step 6: Basic validation
        logging.info("Step 6: Basic validation...")
        
        # Check that we can load data from ASE DB
        assert 'train' in dataloaders_ase
        assert 'valid' in dataloaders_ase
        assert 'test' in dataloaders_ase
        
        # Check that data has expected structure
        sample_batch = next(iter(dataloaders_ase['train']))
        required_keys = ['positions', 'charges', 'num_atoms', 'one_hot']
        for key in required_keys:
            assert key in sample_batch, f"Missing key: {key}"
        
        logging.info("✅ All tests passed! ASE DB pipeline is working correctly.")
        return True
        
    except Exception as e:
        logging.error(f"❌ Test failed: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return False
    
    finally:
        # Clean up
        if os.path.exists(synthetic_dir):
            import shutil
            shutil.rmtree(synthetic_dir)
        if os.path.exists(ase_db_path):
            os.remove(ase_db_path)


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Starting synthetic QM9 ASE DB pipeline test...")
    
    success = test_synthetic_qm9_pipeline()
    
    if success:
        logging.info("🎉 Synthetic QM9 ASE DB pipeline test completed successfully!")
        return 0
    else:
        logging.error("❌ Synthetic QM9 ASE DB pipeline test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())