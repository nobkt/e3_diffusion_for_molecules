#!/usr/bin/env python3
"""
Integration test to verify the KeyError fix works in the actual training scenario.
This simulates the exact error from the problem statement where training with
conditional generation on ASE database fails at epoch 10 during sampling.
"""

import torch
import sys
import os

def test_integration():
    """Test that the fix works in an integration scenario similar to the actual error"""
    print("=" * 70)
    print("Integration Test: Simulating conditional training with ASE database")
    print("=" * 70)
    print()
    
    # Add current directory to path
    sys.path.insert(0, '.')
    
    # Import necessary modules
    from qm9.dataset import retrieve_dataloaders
    from qm9.models import get_model, DistributionProperty
    from qm9 import dataset
    from configs.datasets_config import get_dataset_info
    from qm9.utils import compute_mean_mad, prepare_context
    
    # Create a minimal args object similar to the command in the problem statement
    class Args:
        def __init__(self):
            self.dataset = 'ase_db'
            self.ase_db_path = 'select.db'
            self.batch_size = 4
            self.num_workers = 0
            self.filter_n_atoms = None
            self.remove_h = False
            self.include_charges = False
            
            # Conditioning settings from the error command
            self.conditioning = ['molecular_weight', 'pi_conjugation_ratio', 
                               'atom_types_encoding', 'functional_groups_encoding']
            
            # Model settings
            self.model = 'egnn_dynamics'
            self.nf = 64
            self.n_layers = 3
            self.attention = False
            self.tanh = False
            self.norm_constant = 0
            self.inv_sublayers = 2
            self.sin_embedding = False
            self.normalization_factor = 1
            self.aggregation_method = 'sum'
            
            # Diffusion settings
            self.probabilistic_model = 'diffusion'
            self.diffusion_steps = 100
            self.diffusion_noise_schedule = 'polynomial_2'
            self.diffusion_noise_precision = 1e-5
            self.diffusion_loss_type = 'l2'
            self.normalize_factors = [1, 4, 1]
            self.condition_time = True
            
            # Other
            self.context_node_nf = 0
            self.dequantization = 'deterministic'
            self.n_report_steps = 1
    
    args = Args()
    device = torch.device('cpu')
    
    print("Step 1: Loading ASE database...")
    try:
        # Load the dataset
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        print(f"✅ Loaded dataloaders successfully")
        print(f"   Training samples: {len(dataloaders['train'].dataset)}")
        
        # Check node distribution
        node_counts = {}
        for data in dataloaders['train'].dataset.data['num_atoms']:
            n = int(data.item())
            node_counts[n] = node_counts.get(n, 0) + 1
        print(f"   Node distribution: {dict(sorted(node_counts.items()))}")
        print(f"   Contains 19 nodes: {19 in node_counts}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    print("Step 2: Computing property normalizers...")
    try:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        print(f"✅ Computed normalizers for {len(property_norms)} properties")
        
        # Prepare context to get context_node_nf
        data_dummy = next(iter(dataloaders['train']))
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
        args.context_node_nf = context_node_nf
        print(f"   Context node features: {context_node_nf}")
        print()
        
    except Exception as e:
        print(f"❌ Failed to compute normalizers: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Step 3: Creating model and property distribution...")
    try:
        dataset_info = get_dataset_info(args.dataset, args.remove_h)
        model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
        
        if prop_dist is not None:
            prop_dist.set_normalizer(property_norms)
            print(f"✅ Created property distribution")
            print(f"   Properties: {prop_dist.properties}")
            for prop in prop_dist.properties:
                available_nodes = sorted(prop_dist.distributions[prop].keys())
                print(f"   {prop}: available node counts = {available_nodes}")
        else:
            print(f"✅ No scalar property distribution (only multi-dimensional properties)")
        print()
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Step 4: Testing sample_sweep_conditional (THE CRITICAL TEST)...")
    print("   This is where the original error occurred at epoch 10...")
    try:
        from qm9.sampling import sample_sweep_conditional
        
        # This is the call that was failing with KeyError
        # We use n_nodes=4 which is within max_n_nodes but not in the distribution
        # (select.db only has 3, 5, 8 nodes, so 4 is missing)
        max_n_nodes_actual = max(node_counts.keys())
        test_n_nodes = 4  # A node count that doesn't exist in the actual data
        print(f"   Dataset max_n_nodes: {max_n_nodes_actual}")
        print(f"   Available node counts in data: {sorted(node_counts.keys())}")
        print(f"   Testing with n_nodes={test_n_nodes} (not in distribution)...")
        one_hot, charges, x, node_mask = sample_sweep_conditional(
            args, device, model, dataset_info, prop_dist, n_nodes=test_n_nodes, n_frames=5
        )
        
        print(f"✅ Successfully generated molecules!")
        print(f"   Shape one_hot: {one_hot.shape}")
        print(f"   Shape charges: {charges.shape}")
        print(f"   Shape x: {x.shape}")
        print(f"   Shape node_mask: {node_mask.shape}")
        print()
        
    except KeyError as e:
        print(f"❌ FAILED with KeyError: {e}")
        print("   This is the exact error from the problem statement!")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 70)
    print("✅ INTEGRATION TEST PASSED!")
    print("=" * 70)
    print()
    print("The fix successfully resolves the KeyError that occurred during")
    print("conditional generation at epoch 10 when using ASE database with")
    print("molecules that don't contain the default node count (19).")
    print()
    
    return True


if __name__ == "__main__":
    # Check if select.db exists
    if not os.path.exists('select.db'):
        print("❌ Error: select.db not found!")
        print("   This test requires the select.db database file.")
        sys.exit(1)
    
    success = test_integration()
    sys.exit(0 if success else 1)
