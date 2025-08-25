#!/usr/bin/env python3
"""
Debug test to reproduce the high loss issue with ASE datasets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

def test_with_custom_db():
    """Test using test_qm9.db to reproduce the issue."""
    print("Testing with test_qm9.db to reproduce high loss issue...")
    
    try:
        class TestArgs:
            def __init__(self):
                self.dataset = 'ase'
                self.ase_db_file = './test_qm9.db'  # Use the smaller test database
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
        
        # Test that we can compute loss without crashing
        with torch.no_grad():
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model, nodes_dist,
                                                                    x, h, node_mask, edge_mask, None)
            loss = nll + args.ode_regularization * reg_term
        
        print(f"  Loss computation successful: {loss.item():.2f}")
        
        # Check for high loss
        if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 100000:
            print(f"  ✅ Loss is reasonable")
            return True
        else:
            print(f"  ❌ Loss is astronomically high or invalid")
            return False
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unit_conversion_multiple_times():
    """Test if unit conversion is applied multiple times causing the issue."""
    print("\nTesting if unit conversion is applied multiple times...")
    
    try:
        import build_ase_dataset
        
        # Create a simple test dataset
        test_data = [
            {
                'geometry': np.array([[6, 0, 0, 0], [1, 1, 0, 0]]),  # C-H
                'properties': {'U0_Ha': -1.0, 'HOMO_Ha': -0.5}
            }
        ]
        
        dataset = build_ase_dataset.ASEDataset(test_data)
        
        # Check initial values
        print(f"  Initial U0_Ha: {dataset.properties[0]['U0_Ha']}")
        print(f"  Initial HOMO_Ha: {dataset.properties[0]['HOMO_Ha']}")
        
        # Apply conversion once
        qm9_to_eV = {'U0': 27.2114, 'homo': 27.2114}
        dataset.convert_units(qm9_to_eV)
        
        print(f"  After 1st conversion - U0_Ha: {dataset.properties[0]['U0_Ha']}")
        print(f"  After 1st conversion - HOMO_Ha: {dataset.properties[0]['HOMO_Ha']}")
        
        # Apply conversion again (this should NOT happen in normal use)
        dataset.convert_units(qm9_to_eV)
        
        print(f"  After 2nd conversion - U0_Ha: {dataset.properties[0]['U0_Ha']}")
        print(f"  After 2nd conversion - HOMO_Ha: {dataset.properties[0]['HOMO_Ha']}")
        
        # Check if values have grown exponentially
        if abs(dataset.properties[0]['U0_Ha']) > 1000:
            print(f"  ❌ Unit conversion applied multiple times - values are too large!")
            return False
        else:
            print(f"  ✅ Unit conversion appears to work correctly when applied once")
            return True
            
    except Exception as e:
        print(f"  ❌ Unit conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG TEST FOR HIGH LOSS ISSUE")
    print("=" * 60)
    
    test1_passed = test_with_custom_db()
    test2_passed = test_unit_conversion_multiple_times()
    
    print()
    print("=" * 60)
    if test1_passed and test2_passed:
        print("✅ ALL DEBUG TESTS PASSED")
    else:
        print("❌ SOME DEBUG TESTS FAILED") 
        if not test1_passed:
            print("  - High loss issue reproduced")
        if not test2_passed:
            print("  - Unit conversion issue detected")
    print("=" * 60)