#!/usr/bin/env python3
"""
Example training script using ASE database dataset.
Demonstrates how to train the E3 diffusion model with ASE DB datasets and OpenBabel.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_ase_openbabel import create_test_ase_db
from qm9.dataset import retrieve_dataloaders
from configs.datasets_config import get_dataset_info


class ASETrainingConfig:
    """Configuration for training with ASE DB dataset."""
    
    def __init__(self, db_path, dataset_name='ase_db_qm9'):
        # Dataset configuration
        self.dataset = dataset_name
        self.ase_db_path = db_path
        self.datadir = os.path.dirname(db_path)
        self.remove_h = False
        self.include_charges = True
        
        # Training configuration
        self.batch_size = 4
        self.num_workers = 0
        self.filter_n_atoms = None
        self.filter_molecule_size = None
        self.sequential = False
        
        # Model configuration (basic example)
        self.n_epochs = 5
        self.lr = 1e-4
        self.nf = 64
        self.n_layers = 3
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_ase_db_training(db_path):
    """Test basic training setup with ASE DB dataset."""
    print("=== Testing ASE DB Training Setup ===")
    
    try:
        # Create configuration
        config = ASETrainingConfig(db_path)
        print(f"Using dataset: {config.dataset}")
        print(f"Database path: {config.ase_db_path}")
        print(f"Device: {config.device}")
        
        # Get dataset info
        dataset_info = get_dataset_info(config.dataset, config.remove_h)
        print(f"Dataset info: {dataset_info['name']}")
        print(f"Use OpenBabel: {dataset_info.get('use_openbabel', False)}")
        print(f"Atom decoder: {dataset_info['atom_decoder']}")
        
        # Load dataloaders
        dataloaders, charge_scale = retrieve_dataloaders(config)
        print(f"Loaded dataloaders with {len(dataloaders)} splits")
        
        # Test data iteration
        for split_name, dataloader in dataloaders.items():
            if len(dataloader.dataset) > 0:
                print(f"\nTesting {split_name} dataloader:")
                for i, batch in enumerate(dataloader):
                    print(f"  Batch {i}: {len(batch)} keys, batch_size={batch['positions'].shape[0]}")
                    
                    # Basic validation that data looks correct
                    positions = batch['positions']
                    charges = batch['charges']
                    atom_mask = batch['atom_mask']
                    
                    print(f"    Positions: {positions.shape}")
                    print(f"    Charges: {charges.shape}")
                    print(f"    Atom mask: {atom_mask.shape}")
                    print(f"    Actual atoms per molecule: {atom_mask.sum(dim=1).tolist()}")
                    
                    # Only test first batch
                    break
        
        print("\n✅ ASE DB training setup test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ ASE DB training setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_molecular_metrics_integration(db_path):
    """Test molecular metrics with ASE DB dataset."""
    print("\n=== Testing Molecular Metrics Integration ===")
    
    try:
        # Load dataset for testing
        config = ASETrainingConfig(db_path)
        dataset_info = get_dataset_info(config.dataset, config.remove_h)
        dataloaders, _ = retrieve_dataloaders(config)
        
        # Get some sample molecules
        test_molecules = []
        if len(dataloaders['train'].dataset) > 0:
            for i, sample in enumerate(dataloaders['train'].dataset):
                if i >= 2:  # Only take 2 samples
                    break
                    
                positions = sample['positions']
                charges = sample['charges']
                atom_mask = sample['atom_mask']
                
                # Extract actual atoms
                n_atoms = int(atom_mask.sum())
                positions = positions[:n_atoms]
                charges = charges[:n_atoms]
                
                # Convert charges to atom types
                atom_types = []
                for charge in charges:
                    charge_int = int(charge.item())
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
            print("No test molecules available")
            return False
        
        print(f"Testing with {len(test_molecules)} molecules")
        
        # Test OpenBabel metrics
        from qm9.rdkit_functions import BasicMolecularMetrics
        
        metrics = BasicMolecularMetrics(dataset_info)
        results = metrics.evaluate(test_molecules)
        
        print(f"Molecular metrics results:")
        print(f"  Validity: {results[0][0]:.3f}")
        print(f"  Uniqueness: {results[0][1]:.3f}")
        print(f"  Novelty: {results[0][2]:.3f}")
        
        print("✅ Molecular metrics integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Molecular metrics integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test ASE DB training integration")
    parser.add_argument('--db-path', type=str, help='Path to ASE database file')
    args = parser.parse_args()
    
    print("=== ASE DB Training Integration Test ===")
    
    # Create test database if none provided
    if args.db_path:
        db_path = args.db_path
        if not os.path.exists(db_path):
            print(f"Database file not found: {db_path}")
            return 1
    else:
        print("Creating test database...")
        db_path = create_test_ase_db()
    
    try:
        success = True
        
        # Test basic training setup
        if not test_ase_db_training(db_path):
            success = False
        
        # Test molecular metrics integration
        if not test_molecular_metrics_integration(db_path):
            success = False
        
        if success:
            print("\n🎉 All ASE DB training integration tests passed!")
            print("\nASE DB datasets are ready for training with the E3 diffusion model!")
            print(f"Example command to use ASE DB dataset:")
            print(f"  python main_qm9.py --dataset ase_db_qm9 --ase_db_path {db_path} --batch_size 32")
        else:
            print("\n💥 Some tests failed!")
        
        return 0 if success else 1
        
    finally:
        # Cleanup if we created the test database
        if not args.db_path and os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == '__main__':
    sys.exit(main())