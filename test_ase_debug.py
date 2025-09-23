#!/usr/bin/env python3

import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qm9 import dataset

def test_ase_loading_with_debug():
    """Test ASE database loading with debug output"""
    print("Testing ASE database loading with debug output...")
    
    # Create a simple class to mimic the args object
    class Args:
        def __init__(self):
            self.dataset = 'ase_db'
            self.ase_db_path = 'test_select.db'
            self.batch_size = 2  # Smaller batch for easier debugging
            self.num_workers = 0
            self.filter_n_atoms = None
            self.include_charges = False
            self.remove_h = False
            self.split_ratios = (0.8, 0.1, 0.1)
            self.seed = 42
    
    try:
        args = Args()
        print(f"Loading ASE database from: {args.ase_db_path}")
        
        # This should work now with our fixes
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        print("✅ ASE database loaded successfully!")
        
        # Focus on train split for debugging
        train_dataloader = dataloaders['train']
        print(f"Train split: {len(train_dataloader.dataset)} samples")
        
        if len(train_dataloader.dataset) > 0:
            print("Getting first sample...")
            first_sample = train_dataloader.dataset[0]
            print("First sample keys:", list(first_sample.keys()))
            for key, val in first_sample.items():
                if hasattr(val, 'shape'):
                    print(f"  {key}: {val.shape}")
                else:
                    print(f"  {key}: {type(val)}")
            
            print("\nTrying to get a batch...")
            try:
                batch = next(iter(train_dataloader))
                print("✅ Successfully got batch!")
                print("Batch keys:", list(batch.keys()))
                for key, val in batch.items():
                    if hasattr(val, 'shape'):
                        print(f"  {key}: {val.shape}")
                    else:
                        print(f"  {key}: {type(val)}")
            except Exception as e:
                print(f"❌ Error getting batch: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during ASE database loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ase_loading_with_debug()
    
    if success:
        print("\n✅ ASE database loading test passed!")
    else:
        print("\n❌ ASE database loading test failed!")
        sys.exit(1)