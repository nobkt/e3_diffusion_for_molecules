#!/usr/bin/env python3

import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qm9 import dataset

def test_original_command_scenario():
    """Test scenario that matches the original command from the problem statement"""
    print("Testing original command scenario...")
    print("python main_qm9.py --dataset ase_db --ase_db_path select.db --n_epochs 200 ...")
    
    # Create a larger test database with 1000 molecules like in the original problem
    from create_test_db import create_test_database
    db_path = create_test_database(db_path='select.db', n_molecules=1000)
    
    # Create args that match the original command
    class Args:
        def __init__(self):
            self.dataset = 'ase_db'
            self.ase_db_path = 'select.db'
            self.batch_size = 32
            self.num_workers = 0
            self.filter_n_atoms = None
            self.include_charges = False  # This was the root cause
            self.remove_h = False
            self.split_ratios = (0.8, 0.1, 0.1)  # 800 train, 100 valid, 100 test
            self.seed = 42
    
    try:
        args = Args()
        print(f"Loading ASE database: {args.ase_db_path}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Include charges: {args.include_charges}")
        print(f"  Remove hydrogen: {args.remove_h}")
        print(f"  Batch size: {args.batch_size}")
        
        # This was the line that failed in the original error
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        print("✅ Successfully loaded ASE database!")
        print(f"Charge scale: {charge_scale}")
        
        # Verify the splits match the original output
        for split_name, dataloader in dataloaders.items():
            dataset_obj = dataloader.dataset
            print(f"Split '{split_name}': {len(dataset_obj)} samples")
            
            # Test that we can get a batch without errors
            if len(dataset_obj) > 0:
                try:
                    batch = next(iter(dataloader))
                    print(f"  ✅ Successfully got batch from {split_name}")
                except Exception as e:
                    print(f"  ❌ Error getting batch from {split_name}: {e}")
                    return False
            else:
                print(f"  ⚠️  {split_name} split is empty")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during ASE database loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_original_command_scenario()
    
    if success:
        print("\n🎉 Original command scenario test PASSED!")
        print("The original error has been fixed!")
    else:
        print("\n💥 Original command scenario test FAILED!")
        sys.exit(1)