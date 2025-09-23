#!/usr/bin/env python3

import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qm9 import dataset

def test_ase_loading_functionality():
    """Test ASE database loading with our fixes"""
    print("Testing ASE database loading functionality...")
    
    # Create a simple class to mimic the args object
    class Args:
        def __init__(self):
            self.dataset = 'ase_db'
            self.ase_db_path = 'test_select.db'
            self.batch_size = 32
            self.num_workers = 0
            self.filter_n_atoms = None
            self.include_charges = False  # This is what causes empty charges tensors
            self.remove_h = False
            self.split_ratios = (0.8, 0.1, 0.1)
            self.seed = 42
    
    try:
        args = Args()
        print(f"Loading ASE database from: {args.ase_db_path}")
        print(f"include_charges: {args.include_charges}")
        print(f"remove_h: {args.remove_h}")
        
        # This should work now with our fixes
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        print("✅ ASE database loaded successfully!")
        print(f"charge_scale: {charge_scale}")
        
        # Check each split
        for split_name, dataloader in dataloaders.items():
            print(f"Split '{split_name}': {len(dataloader.dataset)} samples")
            
            # Try to get one batch if the dataset is not empty
            if len(dataloader.dataset) > 0:
                try:
                    batch = next(iter(dataloader))
                    print(f"  Sample batch from {split_name}: {list(batch.keys())}")
                except Exception as e:
                    print(f"  Error getting batch from {split_name}: {e}")
            else:
                print(f"  {split_name} split is empty")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during ASE database loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First create a test database
    print("Creating test database...")
    os.system("python create_test_db.py")
    
    # Then test loading
    success = test_ase_loading_functionality()
    
    if success:
        print("\n✅ ASE database loading test passed!")
    else:
        print("\n❌ ASE database loading test failed!")
        sys.exit(1)