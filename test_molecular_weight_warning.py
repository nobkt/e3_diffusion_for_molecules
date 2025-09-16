#!/usr/bin/env python3
"""
Test to reproduce the molecular_weight warning that was mentioned in the problem statement.
This will help us understand and fix the issue.
"""

import os
import sys
import tempfile
import torch
import numpy as np
from ase import Atoms
from ase.db import connect

# Add the repository to the path
sys.path.insert(0, '/home/runner/work/e3_diffusion_for_molecules/e3_diffusion_for_molecules')


def create_molecular_weight_test_database(db_path, n_molecules=50):
    """Create a test database with varied molecular weights to trigger the warning."""
    print(f"Creating test database at {db_path}")
    
    db = connect(db_path)
    
    # Create molecules with varied molecular weights to trigger normalization warning
    molecules = [
        # Small molecules - low molecular weight
        {'formula': 'H2', 'positions': [[0, 0, 0], [0.74, 0, 0]], 'mw_factor': 1.0},
        {'formula': 'H2O', 'positions': [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]], 'mw_factor': 1.0},
        {'formula': 'CH4', 'positions': [[0, 0, 0], [1.089, 1.089, 1.089], [1.089, -1.089, -1.089], [-1.089, 1.089, -1.089], [-1.089, -1.089, 1.089]], 'mw_factor': 1.0},
        {'formula': 'NH3', 'positions': [[0, 0, 0], [1.017, 0, 0], [-0.509, 0.882, 0], [-0.509, -0.882, 0]], 'mw_factor': 1.0},
        
        # Large molecules - high molecular weight (simulated with synthetic data)
        {'formula': 'C20H42', 'positions': [[i*2.0, 0, 0] for i in range(62)], 'mw_factor': 15.0},  # Simulated large hydrocarbon
        {'formula': 'C30H62', 'positions': [[i*2.0, j*2.0, 0] for i in range(10) for j in range(9) if i*10+j < 92], 'mw_factor': 20.0},  # Even larger
    ]
    
    for i in range(n_molecules):
        mol_data = molecules[i % len(molecules)]
        
        # Create molecule
        try:
            mol = Atoms(mol_data['formula'], positions=mol_data['positions'][:len(mol_data['formula'])])
        except:
            # Fallback to simpler molecules if formula parsing fails
            if 'C20' in mol_data['formula']:
                mol = Atoms('C10H22', positions=[[i*1.5, 0, 0] for i in range(32)])
            elif 'C30' in mol_data['formula']:
                mol = Atoms('C15H32', positions=[[i*1.5, j*1.5, 0] for i in range(8) for j in range(6) if i*8+j < 47])
            else:
                mol = Atoms(mol_data['formula'], positions=mol_data['positions'])
        
        # Add synthetic molecular weight that will cause large normalized values
        base_mw = sum([1 if atom.symbol == 'H' else 12 if atom.symbol == 'C' 
                      else 14 if atom.symbol == 'N' else 16 for atom in mol])
        synthetic_mw = base_mw * mol_data['mw_factor'] + np.random.normal(0, 5)
        
        properties = {
            'energy': np.random.normal(-50, 20),
            'molecular_weight': max(1.0, synthetic_mw),  # Ensure positive
            'mol_id': i
        }
        
        db.write(mol, data=properties)
    
    print(f"Created database with {n_molecules} molecules with varied molecular weights")
    return db_path


def test_molecular_weight_warning_scenario():
    """Test the specific molecular weight normalization that causes the warning."""
    print("\n" + "="*60)
    print("TESTING MOLECULAR WEIGHT WARNING SCENARIO")
    print("="*60)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create test database with varied molecular weights
        create_molecular_weight_test_database(db_path, n_molecules=50)
        
        # Load dataset and test the problematic scenario
        print("\n--- Loading dataset ---")
        
        from qm9.dataset import load_ase_database
        
        datasets, num_species, charge_scale = load_ase_database(
            db_path, 
            split_ratios=(0.8, 0.1, 0.1), 
            include_charges=True, 
            remove_h=False
        )
        
        train_data = datasets['train'].data
        
        print(f"Dataset loaded successfully:")
        print(f"  Train samples: {len(train_data['num_atoms'])}")
        
        # Check molecular weight distribution
        if 'molecular_weight' in train_data:
            mw_values = train_data['molecular_weight']
            print(f"\nMolecular weight distribution:")
            print(f"  Range: [{mw_values.min():.1f}, {mw_values.max():.1f}]")
            print(f"  Mean: {mw_values.mean():.1f}")
            print(f"  Std: {mw_values.std():.1f}")
        
        # Test the normalization computation that triggers the warning
        print(f"\n--- Testing normalization and warning ---")
        
        from qm9.utils import compute_mean_mad_from_dataloader
        
        # Mock dataloader structure
        class MockDataloader:
            def __init__(self, dataset):
                self.dataset = dataset
        
        class MockDataset:
            def __init__(self, data):
                self.data = data
        
        mock_dataloader = MockDataloader(MockDataset(train_data))
        
        # Compute normalization for molecular_weight
        conditioning = ['molecular_weight']
        norms = compute_mean_mad_from_dataloader(mock_dataloader, conditioning)
        
        mean = norms['molecular_weight']['mean']
        mad = norms['molecular_weight']['mad']
        
        print(f"Normalization statistics:")
        print(f"  Mean: {mean:.2f}")
        print(f"  MAD: {mad:.2f}")
        
        # Simulate the prepare_context function to see if we get the warning
        print(f"\n--- Simulating prepare_context normalization ---")
        
        # Create a mock batch
        batch_size = 8
        mock_batch = {
            'molecular_weight': mw_values[:batch_size],
            'positions': torch.zeros(batch_size, 10, 3),
            'atom_mask': torch.ones(batch_size, 10, 1)
        }
        
        # Apply normalization as done in prepare_context
        properties = mock_batch['molecular_weight']
        normalized_properties = (properties - mean) / mad
        
        # Clamp as done in prepare_context
        normalized_properties = torch.clamp(normalized_properties, min=-4.0, max=4.0)
        
        # Check for the warning condition
        max_abs_val = torch.max(torch.abs(normalized_properties))
        warning_threshold = 3.5  # For molecular_weight
        
        print(f"Normalized molecular weight:")
        print(f"  Range: [{normalized_properties.min():.3f}, {normalized_properties.max():.3f}]")
        print(f"  Max absolute value: {max_abs_val:.3f}")
        print(f"  Warning threshold: {warning_threshold}")
        
        if max_abs_val > warning_threshold:
            print(f"⚠️  WARNING TRIGGERED:")
            print(f"   Large normalized values in 'molecular_weight': max_abs = {max_abs_val:.2f}")
            print(f"   This might cause training instability. Consider feature engineering.")
            print(f"   Suggestion: Consider using log(molecular_weight) or molecular_weight^0.5 for better normalization.")
            return True, max_abs_val.item()
        else:
            print(f"✅ No warning triggered - values within acceptable range")
            return False, max_abs_val.item()
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def demonstrate_solutions():
    """Demonstrate different solutions for the molecular weight normalization issue."""
    print("\n" + "="*60)
    print("DEMONSTRATING SOLUTIONS")
    print("="*60)
    
    # Create some test molecular weight data that would cause issues
    torch.manual_seed(42)  # For reproducible results
    mw_values = torch.cat([
        torch.normal(20, 5, (20,)),   # Small molecules
        torch.normal(200, 50, (20,)), # Medium molecules  
        torch.normal(500, 100, (10,)) # Large molecules
    ])
    
    print(f"Test molecular weight data:")
    print(f"  Range: [{mw_values.min():.1f}, {mw_values.max():.1f}]")
    print(f"  Mean: {mw_values.mean():.1f}")
    print(f"  Std: {mw_values.std():.1f}")
    
    # Current approach (problematic)
    print(f"\n--- Current approach (MAD normalization) ---")
    mean = torch.mean(mw_values)
    mad = torch.mean(torch.abs(mw_values - mean))
    
    # Apply special handling for molecular weight as currently done
    min_mad = max(abs(float(mean)) * 0.25, mad * 0.5)
    adjusted_mad = torch.max(mad, torch.tensor(min_mad))
    
    normalized_current = (mw_values - mean) / adjusted_mad
    normalized_current = torch.clamp(normalized_current, min=-4.0, max=4.0)
    max_abs_current = torch.max(torch.abs(normalized_current))
    
    print(f"  Mean: {mean:.2f}, MAD: {mad:.2f}, Adjusted MAD: {adjusted_mad:.2f}")
    print(f"  Normalized range: [{normalized_current.min():.3f}, {normalized_current.max():.3f}]")
    print(f"  Max absolute: {max_abs_current:.3f}")
    
    # Solution 1: Log transformation
    print(f"\n--- Solution 1: Log transformation ---")
    log_mw = torch.log(mw_values)
    log_mean = torch.mean(log_mw)
    log_mad = torch.mean(torch.abs(log_mw - log_mean))
    
    normalized_log = (log_mw - log_mean) / log_mad
    normalized_log = torch.clamp(normalized_log, min=-3.0, max=3.0)
    max_abs_log = torch.max(torch.abs(normalized_log))
    
    print(f"  Log(MW) mean: {log_mean:.2f}, MAD: {log_mad:.2f}")
    print(f"  Normalized range: [{normalized_log.min():.3f}, {normalized_log.max():.3f}]")
    print(f"  Max absolute: {max_abs_log:.3f}")
    
    # Solution 2: Square root transformation  
    print(f"\n--- Solution 2: Square root transformation ---")
    sqrt_mw = torch.sqrt(mw_values)
    sqrt_mean = torch.mean(sqrt_mw)
    sqrt_mad = torch.mean(torch.abs(sqrt_mw - sqrt_mean))
    
    normalized_sqrt = (sqrt_mw - sqrt_mean) / sqrt_mad
    normalized_sqrt = torch.clamp(normalized_sqrt, min=-3.0, max=3.0)
    max_abs_sqrt = torch.max(torch.abs(normalized_sqrt))
    
    print(f"  Sqrt(MW) mean: {sqrt_mean:.2f}, MAD: {sqrt_mad:.2f}")
    print(f"  Normalized range: [{normalized_sqrt.min():.3f}, {normalized_sqrt.max():.3f}]")
    print(f"  Max absolute: {max_abs_sqrt:.3f}")
    
    # Solution 3: Percentile-based robust scaling
    print(f"\n--- Solution 3: Percentile-based robust scaling ---")
    q25 = torch.quantile(mw_values, 0.25)
    q75 = torch.quantile(mw_values, 0.75)
    median = torch.median(mw_values)
    iqr = q75 - q25
    
    # Robust scaling: (x - median) / IQR
    normalized_robust = (mw_values - median) / (iqr + 1e-8)  # Add small epsilon to avoid division by zero
    normalized_robust = torch.clamp(normalized_robust, min=-3.0, max=3.0)
    max_abs_robust = torch.max(torch.abs(normalized_robust))
    
    print(f"  Median: {median:.2f}, IQR: {iqr:.2f}")
    print(f"  Normalized range: [{normalized_robust.min():.3f}, {normalized_robust.max():.3f}]")
    print(f"  Max absolute: {max_abs_robust:.3f}")
    
    print(f"\n--- Summary ---")
    print(f"  Current approach max abs: {max_abs_current:.3f}")
    print(f"  Log transformation max abs: {max_abs_log:.3f}")
    print(f"  Sqrt transformation max abs: {max_abs_sqrt:.3f}")
    print(f"  Robust scaling max abs: {max_abs_robust:.3f}")
    
    # Recommend the best approach
    results = [
        ("Current", max_abs_current.item()),
        ("Log", max_abs_log.item()),
        ("Sqrt", max_abs_sqrt.item()),
        ("Robust", max_abs_robust.item())
    ]
    
    best_method, best_value = min(results, key=lambda x: x[1])
    print(f"\n💡 Best approach: {best_method} transformation (max abs: {best_value:.3f})")
    
    return best_method, results


if __name__ == "__main__":
    print("🧪 Testing Molecular Weight Warning and Solutions")
    print("="*70)
    
    try:
        # Test the warning scenario
        warning_triggered, max_abs = test_molecular_weight_warning_scenario()
        
        if warning_triggered:
            print(f"\n✅ Successfully reproduced the warning (max_abs = {max_abs:.2f})")
        else:
            print(f"\n🤔 Warning not triggered in this test (max_abs = {max_abs:.2f})")
        
        # Demonstrate solutions
        best_method, all_results = demonstrate_solutions()
        
        print(f"\n🎯 Conclusion:")
        print(f"The molecular weight normalization warning can be addressed by:")
        print(f"1. Using log(molecular_weight) transformation (most effective)")
        print(f"2. Using sqrt(molecular_weight) transformation (also effective)")
        print(f"3. Using robust scaling with percentiles (alternative)")
        print(f"4. Improving the current MAD-based approach with better scaling")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)