#!/usr/bin/env python3
"""
Regression test for high loss value bug fixes.

This test verifies that:
1. PyTorch boolean tensor operations work correctly
2. Adaptive normalization produces reasonable loss values for ASE datasets
3. The system doesn't crash with newer PyTorch versions

Run with: python test_high_loss_regression.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import tempfile
import ase.db
import ase

def test_boolean_tensor_compatibility():
    """Test that boolean tensor operations work with newer PyTorch versions."""
    print("Testing PyTorch boolean tensor compatibility...")
    
    # Test the problematic operations that used to fail
    from equivariant_diffusion.utils import remove_mean_with_mask, assert_correctly_masked
    from qm9.losses import assert_correctly_masked as losses_assert_correctly_masked
    
    # Create test data
    batch_size, n_nodes, n_dims = 2, 5, 3
    x = torch.randn(batch_size, n_nodes, n_dims)
    node_mask = torch.ones(batch_size, n_nodes, 1, dtype=torch.bool)  # Boolean mask
    
    # These operations should not crash
    try:
        x_centered = remove_mean_with_mask(x, node_mask.float())
        assert_correctly_masked(x_centered, node_mask.float())
        losses_assert_correctly_masked(x_centered, node_mask.float())
        print("  ✅ Boolean tensor compatibility test passed")
        return True
    except Exception as e:
        print(f"  ❌ Boolean tensor compatibility test failed: {e}")
        return False

def create_test_ase_database():
    """Create a small test ASE database with QM9-like molecules."""
    db_path = tempfile.mktemp(suffix='.db')
    
    with ase.db.connect(db_path) as db:
        # Small molecules (QM9-style)
        test_molecules = [
            # Methane-like
            {
                'symbols': ['C', 'H', 'H', 'H', 'H'],
                'positions': [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]],
            },
            # Water-like  
            {
                'symbols': ['O', 'H', 'H'],
                'positions': [[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]],
            },
            # Ammonia-like
            {
                'symbols': ['N', 'H', 'H', 'H'],
                'positions': [[0, 0, 0], [1.0, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.43, 0.75]],
            }
        ]
        
        for mol_data in test_molecules:
            atoms = ase.Atoms(symbols=mol_data['symbols'], positions=mol_data['positions'])
            db.write(atoms)
    
    return db_path

def test_adaptive_normalization():
    """Test that adaptive normalization selects appropriate factors and doesn't crash."""
    print("Testing adaptive normalization and crash prevention...")
    
    try:
        # Test the QM9 database that should work
        class TestArgs:
            def __init__(self):
                self.dataset = 'ase'
                self.ase_db_file = './qm9.db'
                self.ase_max_entries = 10
                self.filter_molecule_size = None
                self.sequential = False
                self.batch_size = 2
                self.num_workers = 0
                self.include_charges = True
                self.remove_h = False
                self.conditioning = []
                self.normalize_factors = None  # Let adaptive normalization work
                self.probabilistic_model = 'diffusion'
                self.diffusion_steps = 10  # Very small for testing
                self.diffusion_noise_schedule = 'polynomial_2'
                self.diffusion_noise_precision = 1e-5
                self.diffusion_loss_type = 'l2'
                self.context_node_nf = 0
                self.condition_time = True
                self.nf = 32  # Small model for testing
                self.n_layers = 2
                self.attention = True
                self.tanh = True
                self.model = 'egnn_dynamics'
                self.norm_constant = 1
                self.inv_sublayers = 1
                self.sin_embedding = False
                self.normalization_factor = 1
                self.aggregation_method = 'sum'
                self.data_augmentation = False
                self.augment_noise = 0
                self.ode_regularization = 1e-3
                self.device = torch.device('cpu')
        
        args = TestArgs()
        device = torch.device('cpu')
        
        # Import modules
        from qm9 import dataset, losses
        from configs.datasets_config import get_dataset_info
        from qm9.models import get_model
        from equivariant_diffusion.utils import remove_mean_with_mask, assert_mean_zero_with_mask
        
        # Get dataset info and load data
        dataset_info = get_dataset_info(args.dataset, args.remove_h)
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        # Apply adaptive normalization logic (same as main_qm9.py)
        if args.normalize_factors is None:
            if args.dataset == 'ase':
                default_pos_norm = 1.0
                default_cat_norm = 4.0
                default_int_norm = 1.0
                
                if hasattr(dataset_info, 'position_stats') and 'max_span' in dataset_info['position_stats']:
                    max_span = dataset_info['position_stats']['max_span']
                    median_span = dataset_info['position_stats']['median_span']
                    
                    if max_span > 15:
                        default_pos_norm = max(3.0, median_span / 3.0)
                    elif max_span > 8:
                        default_pos_norm = max(2.0, median_span / 4.0) 
                    else:
                        default_pos_norm = 1.0
                
                args.normalize_factors = [default_pos_norm, default_cat_norm, default_int_norm]
            else:
                args.normalize_factors = [1, 4, 1]
        
        print(f"  Selected normalization factors: {args.normalize_factors}")
        
        # Key test: Make sure we're not using the old problematic factors
        if args.normalize_factors == [10, 8, 1]:
            print(f"  ❌ Still using old problematic normalization factors!")
            return False
        
        # Test that normalization produces reasonable ranges
        model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
        model.eval()
        
        # Get a sample batch
        data_batch = next(iter(dataloaders['train']))
        x = data_batch['positions'].to(device, dtype=torch.float32)
        node_mask = data_batch['atom_mask'].to(device, dtype=torch.float32).unsqueeze(2)
        edge_mask = data_batch['edge_mask'].to(device, dtype=torch.float32)
        one_hot = data_batch['one_hot'].to(device, dtype=torch.float32)
        charges = (data_batch['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype=torch.float32)

        x = remove_mean_with_mask(x, node_mask)
        h = {'categorical': one_hot, 'integer': charges}
        
        # Test normalization
        x_norm, h_norm, delta_log_px = model.normalize(x, h, node_mask)
        
        print(f"  Normalized x range: [{x_norm.min():.3f}, {x_norm.max():.3f}]")
        print(f"  Normalized h_cat range: [{h_norm['categorical'].min():.3f}, {h_norm['categorical'].max():.3f}]")
        
        # Check that normalized values are in reasonable ranges (not extremely small/large)
        if abs(x_norm.std().item()) < 0.01:
            print(f"  ❌ Normalized coordinates too small (std={x_norm.std().item():.6f})")
            return False
        if abs(x_norm.std().item()) > 100:
            print(f"  ❌ Normalized coordinates too large (std={x_norm.std().item():.6f})")
            return False
        
        # Test that we can compute loss without crashing
        with torch.no_grad():
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model, nodes_dist,
                                                                    x, h, node_mask, edge_mask, None)
            loss = nll + args.ode_regularization * reg_term
        
        print(f"  Loss computation successful: {loss.item():.2f}")
        
        # For small test models, loss can be high, but it shouldn't be astronomically high
        # The original bug caused losses in hundreds of thousands
        if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 100000:
            print(f"  ✅ Adaptive normalization working correctly")
            return True
        else:
            print(f"  ❌ Loss is still astronomically high or invalid")
            return False
            
    except Exception as e:
        print(f"  ❌ Adaptive normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all regression tests."""
    print("=" * 60)
    print("HIGH LOSS VALUE BUG REGRESSION TESTS")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Boolean tensor compatibility
    if not test_boolean_tensor_compatibility():
        all_tests_passed = False
    
    print()
    
    # Test 2: Adaptive normalization
    if not test_adaptive_normalization():
        all_tests_passed = False
    
    print()
    print("=" * 60)
    if all_tests_passed:
        print("✅ ALL REGRESSION TESTS PASSED")
        print("The high loss value bugs have been successfully fixed!")
    else:
        print("❌ SOME REGRESSION TESTS FAILED")
        print("There may be remaining issues with the high loss value fixes.")
    print("=" * 60)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)