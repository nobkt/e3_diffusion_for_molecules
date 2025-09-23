#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qm9.data.dataset_class import ProcessedDataset

def test_fixed_empty_dataset():
    """Test that the fix works for empty datasets"""
    print("Testing fixed empty dataset scenario...")
    
    # Simulate what could happen with empty splits in ASE database
    empty_data = {
        'charges': torch.empty(0, dtype=torch.long),
        'positions': torch.empty(0, 3, dtype=torch.float32),
        'num_atoms': torch.empty(0, dtype=torch.long),
    }
    
    try:
        dataset = ProcessedDataset(empty_data)
        print(f"Empty dataset created successfully. Length: {len(dataset)}")
        print(f"num_species: {dataset.num_species}")
        print(f"max_charge: {dataset.max_charge}")
        print(f"one_hot shape: {dataset.data['one_hot'].shape}")
        
        # Try to create DataLoader with shuffle=True (this should work now)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print("DataLoader with shuffle=True created successfully!")
        
        # Try to create DataLoader with shuffle=False
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        print("DataLoader with shuffle=False created successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_non_empty_dataset():
    """Test that the fix doesn't break non-empty datasets"""
    print("\nTesting non-empty dataset still works...")
    
    # Create minimal dataset with one sample
    minimal_data = {
        'charges': torch.tensor([6], dtype=torch.long),  # One carbon atom
        'positions': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        'num_atoms': torch.tensor([1], dtype=torch.long),
    }
    
    try:
        dataset = ProcessedDataset(minimal_data)
        print(f"Non-empty dataset created successfully. Length: {len(dataset)}")
        print(f"num_species: {dataset.num_species}")
        print(f"max_charge: {dataset.max_charge}")
        print(f"one_hot shape: {dataset.data['one_hot'].shape}")
        
        # Try to create DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print("DataLoader created successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_fixed_empty_dataset()
    success2 = test_non_empty_dataset()
    
    if success1 and success2:
        print("\n✅ All tests passed! The fix works correctly.")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)