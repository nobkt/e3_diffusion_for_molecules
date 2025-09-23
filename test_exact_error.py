#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qm9.data.dataset_class import ProcessedDataset

def test_exact_error():
    """Test the exact error scenario that occurs with empty splits"""
    print("Testing exact error scenario...")
    
    # This reproduces the exact scenario when one of the splits is empty
    empty_data = {
        'charges': torch.empty(0, dtype=torch.long),
        'positions': torch.empty(0, 3, dtype=torch.float32),
        'num_atoms': torch.empty(0, dtype=torch.long),
    }
    
    # This will reproduce the exact error from the traceback
    try:
        dataset = ProcessedDataset(empty_data)
        print(f"Dataset length: {len(dataset)}")
        
        # This reproduces the exact error - RandomSampler with num_samples=0
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,  # This causes RandomSampler to be used
            num_workers=0
        )
        print("DataLoader created successfully - this should NOT be reached")
        
    except ValueError as e:
        print(f"Exact error reproduced: {e}")
        return True
    except Exception as e:
        print(f"Different error: {e}")
        return False

if __name__ == "__main__":
    test_exact_error()