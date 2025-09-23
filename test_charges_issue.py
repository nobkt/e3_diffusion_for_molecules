#!/usr/bin/env python3

import torch

def test_exact_issue():
    """Test the exact issue with empty charges tensor"""
    print("Testing exact issue with empty charges tensor...")
    
    # This reproduces the exact error in ProcessedDataset.__init__
    charges = torch.empty(0, dtype=torch.long)
    print(f"charges shape: {charges.shape}")
    
    # This line succeeds
    unique_charges = torch.unique(charges, sorted=True)
    print(f"unique_charges: {unique_charges}")
    print(f"unique_charges shape: {unique_charges.shape}")
    
    # But this line fails because unique_charges is empty
    try:
        first_element = unique_charges[0]
        print(f"first_element: {first_element}")
    except IndexError as e:
        print(f"Error accessing first element: {e}")
        
    # And then the length check fails too
    print(f"len(unique_charges): {len(unique_charges)}")
    if len(unique_charges) > 0:
        print(f"First element would be: {unique_charges[0]}")
    else:
        print("unique_charges is empty, cannot access first element")

if __name__ == "__main__":
    test_exact_issue()