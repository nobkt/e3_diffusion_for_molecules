#!/usr/bin/env python3
"""
Integration test for ASE DB dataset with the main framework.
Tests that ASE DB datasets can be used with the existing dataloader infrastructure.
"""

import torch
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_ase_openbabel import create_test_ase_db
from qm9.dataset import retrieve_dataloaders
from qm9.data.collate import PreprocessQM9


class TestConfig:
    """Test configuration for ASE DB dataset."""
    def __init__(self, db_path):
        self.dataset = 'ase_db_qm9'
        self.batch_size = 2
        self.num_workers = 0
        self.remove_h = False
        self.filter_n_atoms = None
        self.datadir = os.path.dirname(db_path)
        self.ase_db_path = db_path
        self.include_charges = True
        self.filter_molecule_size = None
        self.sequential = False


def test_dataloader_integration():
    """Test that ASE DB datasets work with the dataloader system."""
    print("=== Testing Dataloader Integration ===")
    
    # Create test database
    db_path = create_test_ase_db()
    
    try:
        # Create config
        cfg = TestConfig(db_path)
        
        # Test dataloader retrieval
        dataloaders, charge_scale = retrieve_dataloaders(cfg)
        
        print(f"Retrieved dataloaders for dataset: {cfg.dataset}")
        print(f"Charge scale: {charge_scale}")
        print(f"Available splits: {list(dataloaders.keys())}")
        
        # Test each dataloader
        for split_name, dataloader in dataloaders.items():
            print(f"\nTesting {split_name} dataloader:")
            print(f"  Dataset size: {len(dataloader.dataset)}")
            print(f"  Batch size: {dataloader.batch_size}")
            
            if len(dataloader.dataset) > 0:
                # Test batch loading
                for i, batch in enumerate(dataloader):
                    print(f"  Batch {i}:")
                    print(f"    Keys: {list(batch.keys())}")
                    for key, value in batch.items():
                        if torch.is_tensor(value):
                            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"    {key}: {type(value)}")
                    
                    # Only test first batch
                    break
            else:
                print(f"  No data in {split_name}")
        
        return True
        
    except Exception as e:
        print(f"Dataloader integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_preprocessing():
    """Test that preprocessing works with ASE DB datasets."""
    print("\n=== Testing Preprocessing ===")
    
    # Create test database
    db_path = create_test_ase_db()
    
    try:
        from qm9.ase_db_dataset import load_ase_db_datasets
        from configs.datasets_config import get_dataset_info
        
        # Load dataset
        dataset_info = get_dataset_info('ase_db_qm9', remove_h=False)
        datasets = load_ase_db_datasets(
            db_path=db_path,
            dataset_info=dataset_info,
            split_ratio=(0.7, 0.2, 0.1),
            remove_h=False,
            max_atoms=10,
            random_seed=42
        )
        
        # Test preprocessing
        preprocess = PreprocessQM9(load_charges=True)
        
        if len(datasets['train']) > 0:
            # Get raw samples
            samples = [datasets['train'][i] for i in range(min(2, len(datasets['train'])))]
            
            # Test collate function
            batch = preprocess.collate_fn(samples)
            
            print("Preprocessing test successful:")
            print(f"  Input samples: {len(samples)}")
            print(f"  Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            
            return True
        else:
            print("No training data available for preprocessing test")
            return False
            
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_backward_compatibility():
    """Test that existing QM9 functionality still works."""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test QM9 dataset info
        from configs.datasets_config import get_dataset_info
        
        qm9_info = get_dataset_info('qm9', remove_h=False)
        print("QM9 dataset info retrieved successfully")
        print(f"  Name: {qm9_info['name']}")
        print(f"  Use OpenBabel: {qm9_info.get('use_openbabel', False)}")
        print(f"  Atom decoder: {qm9_info['atom_decoder']}")
        
        # Test RDKit functions without trying to load dataset
        from qm9.rdkit_functions import BasicMolecularMetrics
        
        # Create mock SMILES list to avoid network calls
        mock_smiles = ['C', 'O', 'N', 'F']
        
        metrics = BasicMolecularMetrics(qm9_info, dataset_smiles_list=mock_smiles)
        print("RDKit metrics initialized successfully")
        print(f"  Use OpenBabel: {metrics.use_openbabel}")
        
        # Test with simple molecules
        import torch
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        atom_types = torch.tensor([1, 0], dtype=torch.long)  # C-H
        
        # Test validity (this should work without network)
        test_molecules = [(positions, atom_types)]
        valid, validity = metrics.compute_validity(test_molecules)
        print(f"  Validity test: {validity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=== ASE DB Framework Integration Test ===")
    
    success = True
    
    # Test dataloader integration
    if not test_dataloader_integration():
        success = False
    
    # Test preprocessing
    if not test_preprocessing():
        success = False
    
    # Test backward compatibility
    if not test_backward_compatibility():
        success = False
    
    if success:
        print("\n=== All integration tests passed! ===")
    else:
        print("\n=== Some integration tests failed! ===")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)