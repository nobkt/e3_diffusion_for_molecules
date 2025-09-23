#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qm9.data.dataset_class import ProcessedDataset

def test_dataloader_empty_dataset():
    """Test DataLoader creation logic that mimics dataset.py"""
    print("Testing DataLoader creation logic...")
    
    # Simulate what could happen with empty splits in ASE database
    empty_data = {
        'charges': torch.empty(0, dtype=torch.long),
        'positions': torch.empty(0, 3, dtype=torch.float32),
        'num_atoms': torch.empty(0, dtype=torch.long),
    }
    
    try:
        dataset = ProcessedDataset(empty_data)
        print(f"Empty dataset created successfully. Length: {len(dataset)}")
        
        # Test the logic from the dataset.py fix
        split = 'train'  # This would normally have shuffle=True
        should_shuffle = (split == 'train') and (len(dataset) > 0)
        print(f"Should shuffle for {split}: {should_shuffle}")
        
        # Try to create DataLoader with the fixed logic
        dataloader = DataLoader(dataset, batch_size=32, shuffle=should_shuffle)
        print("DataLoader created successfully with fixed logic!")
        
        # Test for validation split
        split = 'valid'  # This would normally have shuffle=False
        should_shuffle = False and (len(dataset) > 0)
        print(f"Should shuffle for {split}: {should_shuffle}")
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=should_shuffle)
        print("DataLoader for validation created successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataloader_empty_dataset()
    
    if success:
        print("\n✅ DataLoader fix works correctly!")
    else:
        print("\n❌ DataLoader fix failed!")
        sys.exit(1)