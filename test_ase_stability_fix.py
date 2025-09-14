#!/usr/bin/env python3
"""
Test script to demonstrate the ASE database training stability fix.

This script shows the difference between the old and new normalization methods
and provides recommendations for training with ASE databases.
"""

import torch
import numpy as np
from qm9.utils import compute_mean_mad_from_dataloader, prepare_context

def create_problematic_ase_data():
    """Create data that simulates the problematic patterns in ASE databases."""
    batch_size = 32
    
    # Very sparse atom types encoding (many zeros, few 1s)
    atom_types = torch.zeros(batch_size, 50)
    for i in range(batch_size):
        # Most molecules have only 2-5 atom types out of 50 possible
        num_types = np.random.randint(2, 6)
        indices = np.random.choice(50, num_types, replace=False)
        atom_types[i, indices] = 1.0
    
    # Very sparse functional groups (even sparser)
    functional_groups = torch.zeros(batch_size, 100)
    for i in range(batch_size):
        # Most molecules have 0-2 functional groups out of 100 possible
        num_groups = np.random.randint(0, 3)
        if num_groups > 0:
            indices = np.random.choice(100, num_groups, replace=False)
            functional_groups[i, indices] = 1.0
    
    # Molecular properties with wide ranges
    molecular_weight = torch.tensor([
        50.0, 100.0, 200.0, 500.0, 800.0, 1200.0, 150.0, 300.0,
        75.0, 125.0, 250.0, 400.0, 600.0, 900.0, 80.0, 350.0,
        90.0, 180.0, 280.0, 450.0, 650.0, 850.0, 110.0, 220.0,
        95.0, 190.0, 290.0, 490.0, 690.0, 890.0, 120.0, 320.0
    ])
    
    pi_conjugation = torch.rand(batch_size)  # 0-1 range
    
    return {
        'molecular_weight': molecular_weight,
        'pi_conjugation_ratio': pi_conjugation,
        'atom_types_encoding': atom_types,
        'functional_groups_encoding': functional_groups
    }

def old_compute_mean_mad(values):
    """Old normalization method (problematic)."""
    if values.dim() > 1:
        mean = torch.mean(values, dim=0)
        ma = torch.abs(values - mean.unsqueeze(0))
        mad = torch.mean(ma, dim=0)
        min_mad = torch.clamp(torch.abs(mean) * 0.01, min=0.1)  # Too small!
        mad = torch.clamp(mad, min=min_mad)
    else:
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        min_mad = max(abs(float(mean)) * 0.01, 0.1)  # Too small!
        mad = torch.max(mad, torch.tensor(min_mad))
    return mean, mad

def old_normalize(properties, mean, mad):
    """Old normalization (problematic)."""
    if mean.dim() == 0:
        normalized = (properties - mean) / mad
    else:
        if properties.dim() == 2 and mean.dim() == 1:
            normalized = (properties - mean.unsqueeze(0)) / mad.unsqueeze(0)
        else:
            normalized = (properties - mean) / mad
    
    normalized = torch.clamp(normalized, min=-50.0, max=50.0)  # Too extreme!
    return normalized

# Mock dataset for testing
class MockDataset:
    def __init__(self, data):
        self.data = data

class MockDataloader:
    def __init__(self, data):
        self.dataset = MockDataset(data)

def compare_normalization_methods():
    """Compare old vs new normalization methods."""
    print("ASE Database Training Stability Fix Demonstration")
    print("=" * 60)
    
    # Create problematic data
    data = create_problematic_ase_data()
    dataloader = MockDataloader(data)
    
    print("\nTesting with problematic ASE database patterns...")
    print("(Very sparse binary features + wide-range continuous features)")
    
    properties = list(data.keys())
    
    # Test new method
    print(f"\n{'='*20} NEW METHOD (FIXED) {'='*20}")
    property_norms_new = compute_mean_mad_from_dataloader(dataloader, properties)
    
    max_normalized_values_new = {}
    for prop in properties:
        mean = property_norms_new[prop]['mean']
        mad = property_norms_new[prop]['mad']
        
        # Apply normalization
        values = data[prop]
        if mean.dim() == 0:
            normalized = (values - mean) / mad
        else:
            normalized = (values - mean.unsqueeze(0)) / mad.unsqueeze(0)
        
        normalized = torch.clamp(normalized, min=-8.0, max=8.0)  # New conservative range
        max_val = torch.max(torch.abs(normalized)).item()
        max_normalized_values_new[prop] = max_val
        
        print(f"{prop}: max_abs_normalized = {max_val:.2f}")
    
    # Test old method for comparison
    print(f"\n{'='*20} OLD METHOD (PROBLEMATIC) {'='*15}")
    
    max_normalized_values_old = {}
    for prop in properties:
        mean, mad = old_compute_mean_mad(data[prop])
        normalized = old_normalize(data[prop], mean, mad)
        max_val = torch.max(torch.abs(normalized)).item()
        max_normalized_values_old[prop] = max_val
        
        print(f"{prop}: max_abs_normalized = {max_val:.2f}")
    
    # Summary
    print(f"\n{'='*25} SUMMARY {'='*25}")
    print("Improvement in max absolute normalized values:")
    for prop in properties:
        old_val = max_normalized_values_old[prop]
        new_val = max_normalized_values_new[prop]
        improvement = (old_val - new_val) / old_val * 100 if old_val > 0 else 0
        print(f"{prop:25s}: {old_val:6.2f} → {new_val:6.2f} ({improvement:+5.1f}%)")
    
    # Gradient explosion risk assessment
    print(f"\n{'='*20} GRADIENT STABILITY {'='*20}")
    for prop in properties:
        old_val = max_normalized_values_old[prop]
        new_val = max_normalized_values_new[prop]
        
        old_risk = "HIGH" if old_val > 10 else "MEDIUM" if old_val > 5 else "LOW"
        new_risk = "HIGH" if new_val > 10 else "MEDIUM" if new_val > 5 else "LOW"
        
        print(f"{prop:25s}: {old_risk:6s} → {new_risk:6s}")

def training_recommendations():
    """Provide training recommendations for ASE databases."""
    print(f"\n{'='*15} TRAINING RECOMMENDATIONS {'='*15}")
    print("""
For training with ASE databases and molecular descriptors:

1. **Use these improved settings:**
   - Batch size: 16 (start conservative)
   - Learning rate: 1e-5 to 5e-5 (lower than QM9)
   - Gradient clipping: automatically adjusted (starts at 1.0)

2. **Monitor these metrics:**
   - Gradient norm should stay below 100
   - Loss should decrease gradually without spikes
   - Watch for "Large normalized values" warnings

3. **If training is still unstable:**
   - Reduce learning rate to 1e-6
   - Use smaller batch size (8 or 4)
   - Consider filtering very large molecules (>100 atoms)
   - Remove rare atom types/functional groups

4. **Example command with improved stability:**
   python main_qm9.py \\
     --dataset ase_db \\
     --ase_db_path your_database.db \\
     --conditioning molecular_weight pi_conjugation_ratio atom_types_encoding \\
     --batch_size 16 \\
     --lr 2e-5 \\
     --n_epochs 200 \\
     --nf 256 \\
     --n_layers 8 \\
     --diffusion_steps 500
""")

if __name__ == "__main__":
    compare_normalization_methods()
    training_recommendations()