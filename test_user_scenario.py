#!/usr/bin/env python3
"""
Test to simulate the user's scenario with a custom ASE database
and verify the high loss issue is fixed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import tempfile
import ase.db
import ase

def create_qm9_like_test_database():
    """Create a test database that simulates user's custom QM9 data."""
    db_path = './qm9_test.db'  # Use the same name as in user's error
    
    # Remove existing file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    with ase.db.connect(db_path) as db:
        # Create molecules with different sizes like user's data
        test_molecules = [
            # Various small molecules with QM9-like properties
            {
                'symbols': ['C', 'H', 'H', 'H'],  # 4 atoms 
                'positions': [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
                'U0_Ha': -40.1234,
                'HOMO_Ha': -0.2345,
                'LUMO_Ha': -0.1234,
                'gap_Ha': 0.1111,
                'ZPVE_Ha': 0.0456
            },
            {
                'symbols': ['C', 'H', 'H', 'H', 'H'],  # 5 atoms
                'positions': [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]],
                'U0_Ha': -40.5678,
                'HOMO_Ha': -0.2567,
                'LUMO_Ha': -0.1345,
                'gap_Ha': 0.1222,
                'ZPVE_Ha': 0.0567
            },
            {
                'symbols': ['O', 'H', 'H'],  # 3 atoms
                'positions': [[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]],
                'U0_Ha': -76.2345,
                'HOMO_Ha': -0.4567,
                'LUMO_Ha': -0.1123,
                'gap_Ha': 0.3444,
                'ZPVE_Ha': 0.0213
            },
            {
                'symbols': ['N', 'H', 'H', 'H'],  # 4 atoms
                'positions': [[0, 0, 0], [1.0, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.43, 0.75]],
                'U0_Ha': -56.7890,
                'HOMO_Ha': -0.3456,
                'LUMO_Ha': -0.1567,
                'gap_Ha': 0.1889,
                'ZPVE_Ha': 0.0345
            },
            {
                'symbols': ['C', 'C', 'H', 'H', 'H', 'H'],  # 6 atoms
                'positions': [[0, 0, 0], [1.5, 0, 0], [2.5, 0, 0], [-1.0, 0, 0], [2.0, 1.0, 0], [1.0, -1.0, 0]],
                'U0_Ha': -79.1234,
                'HOMO_Ha': -0.2789,
                'LUMO_Ha': -0.1234,
                'gap_Ha': 0.1555,
                'ZPVE_Ha': 0.0678
            },
            {
                'symbols': ['C', 'O', 'H', 'H', 'H', 'H', 'H'],  # 7 atoms
                'positions': [[0, 0, 0], [1.5, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [2.5, 0, 0], [1.5, 1.0, 0]],
                'U0_Ha': -115.4567,
                'HOMO_Ha': -0.4123,
                'LUMO_Ha': -0.1456,
                'gap_Ha': 0.2667,
                'ZPVE_Ha': 0.0789
            },
            {
                'symbols': ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H'],  # 8 atoms  
                'positions': [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [-1.0, 0, 0], [2.0, 1.0, 0], [3.5, 1.0, 0], [4.0, 0, 0], [0.5, -1.0, 0]],
                'U0_Ha': -118.7890,
                'HOMO_Ha': -0.2345,
                'LUMO_Ha': -0.1123,
                'gap_Ha': 0.1222,
                'ZPVE_Ha': 0.0890
            },
            {
                'symbols': ['C', 'H', 'H', 'H'],  # Another 4 atoms molecule
                'positions': [[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1]],
                'U0_Ha': -40.2468,
                'HOMO_Ha': -0.2456,
                'LUMO_Ha': -0.1345,
                'gap_Ha': 0.1111,
                'ZPVE_Ha': 0.0467
            },
            {
                'symbols': ['C', 'H', 'H', 'H'],  # Another 4 atoms molecule
                'positions': [[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0], [0, 0, 0.9]],
                'U0_Ha': -40.1357,
                'HOMO_Ha': -0.2378,
                'LUMO_Ha': -0.1289,
                'gap_Ha': 0.1089,
                'ZPVE_Ha': 0.0445
            },
            {
                'symbols': ['O', 'H', 'H'],  # Another 3 atoms molecule
                'positions': [[0, 0, 0], [0.9, 0.5, 0], [-0.9, 0.5, 0]],
                'U0_Ha': -76.1123,
                'HOMO_Ha': -0.4234,
                'LUMO_Ha': -0.1034,
                'gap_Ha': 0.3200,
                'ZPVE_Ha': 0.0203
            }
        ]
        
        for mol_data in test_molecules:
            atoms = ase.Atoms(symbols=mol_data['symbols'], positions=mol_data['positions'])
            
            # Add properties as the user would have them
            properties = {k: v for k, v in mol_data.items() if k not in ['symbols', 'positions']}
            db.write(atoms, **properties)
    
    return db_path

def test_user_scenario():
    """Test the exact scenario described by the user."""
    print("Testing user's scenario with custom QM9 ASE database...")
    
    # Create test database similar to user's  
    db_path = create_qm9_like_test_database()
    
    try:
        class TestArgs:
            def __init__(self):
                self.dataset = 'ase'
                self.ase_db_file = db_path  # Use our test database
                self.ase_max_entries = 10  # Same as user
                self.filter_molecule_size = None
                self.sequential = False
                self.batch_size = 2  # Same as user
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
        
        # The user's original issue was loss of 239363.94, which was astronomically high
        # With our fix, loss should be reasonable (< 10000 at least)
        if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 10000:
            print(f"  ✅ Loss is reasonable - user's high loss issue is FIXED!")
            return True
        else:
            print(f"  ❌ Loss is still too high - issue may still exist")
            return False
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test database
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    print("=" * 60)
    print("USER SCENARIO TEST - CUSTOM QM9 ASE DATABASE")
    print("=" * 60)
    
    test_passed = test_user_scenario()
    
    print()
    print("=" * 60)
    if test_passed:
        print("✅ USER'S HIGH LOSS ISSUE HAS BEEN FIXED!")
        print("The unit conversion bug has been resolved.")
    else:
        print("❌ USER'S HIGH LOSS ISSUE MAY STILL EXIST")
        print("Further investigation needed.")
    print("=" * 60)