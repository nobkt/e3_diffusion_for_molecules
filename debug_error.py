#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qm9.data.dataset_class import ProcessedDataset

def test_empty_dataset():
    """Test what happens with an empty dataset"""
    print("Testing empty dataset scenario...")
    
    # Simulate what could happen with empty splits in ASE database
    empty_data = {
        'charges': torch.empty(0, dtype=torch.long),
        'positions': torch.empty(0, 3, dtype=torch.float32),
        'num_atoms': torch.empty(0, dtype=torch.long),
    }
    
    try:
        dataset = ProcessedDataset(empty_data)
        print(f"Empty dataset created successfully. Length: {len(dataset)}")
        
        # Try to create DataLoader with shuffle=True (this should fail)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print("DataLoader with shuffle=True created successfully")
        
    except Exception as e:
        print(f"Error with shuffle=True: {e}")
        
    try:
        # Try to create DataLoader with shuffle=False
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        print("DataLoader with shuffle=False created successfully")
        
    except Exception as e:
        print(f"Error with shuffle=False: {e}")


def test_minimal_dataset():
    """Test with minimal non-empty dataset"""
    print("\nTesting minimal non-empty dataset...")
    
    # Create minimal dataset with one sample
    minimal_data = {
        'charges': torch.tensor([6], dtype=torch.long),  # One carbon atom
        'positions': torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        'num_atoms': torch.tensor([1], dtype=torch.long),
    }
    
    try:
        dataset = ProcessedDataset(minimal_data)
        print(f"Minimal dataset created successfully. Length: {len(dataset)}")
        
        # Try to create DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print("DataLoader created successfully")
        
    except Exception as e:
        print(f"Error creating DataLoader: {e}")


if __name__ == "__main__":
    test_empty_dataset()
    test_minimal_dataset()