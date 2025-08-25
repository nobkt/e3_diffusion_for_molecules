#!/usr/bin/env python3
"""
Debug the actual property values and see what's causing high loss.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import tempfile
import ase.db
import ase

def debug_property_values():
    """Debug what property values look like after conversion."""
    
    # Create a simple test molecule with realistic QM9-like values
    db_path = tempfile.mktemp(suffix='.db')
    
    with ase.db.connect(db_path) as db:
        # Use more realistic QM9 property values (from actual QM9 dataset)
        atoms = ase.Atoms(symbols=['C', 'H', 'H', 'H', 'H'], 
                         positions=[[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]])
        
        # These are typical QM9 values in Hartree units (should be converted to eV)
        db.write(atoms,
                 U0_Ha=-40.5,      # Total energy at 0K
                 HOMO_Ha=-0.25,    # HOMO energy 
                 LUMO_Ha=-0.1,     # LUMO energy
                 gap_Ha=0.15,      # HOMO-LUMO gap
                 ZPVE_Ha=0.05)     # Zero point vibrational energy
    
    try:
        import build_ase_dataset
        from configs.datasets_config import get_dataset_info
        
        # Load data and create dataset
        data_list = build_ase_dataset.load_ase_data(db_path, max_entries=1)
        dataset_info = get_dataset_info('ase', False)
        
        transform = build_ase_dataset.ASETransform(
            dataset_info,
            include_charges=True,
            device=torch.device('cpu'),
            sequential=False
        )
        
        dataset = build_ase_dataset.ASEDataset(data_list, transform=transform)
        
        # Check values before conversion
        print("Property values before unit conversion:")
        props = dataset.properties[0]
        for key, value in props.items():
            print(f"  {key}: {value}")
        
        # Apply unit conversion
        qm9_to_eV = {'U0': 27.2114, 'homo': 27.2114, 'lumo': 27.2114, 'gap': 27.2114, 'zpve': 27211.4}
        dataset.convert_units(qm9_to_eV)
        
        # Check values after conversion  
        print("\nProperty values after unit conversion:")
        props = dataset.properties[0]
        for key, value in props.items():
            print(f"  {key}: {value}")
        
        # Check what the transform produces
        data_item = dataset[0]
        print("\nTransformed data item properties:")
        for key, value in data_item.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                print(f"  {key}: {value.item()}")
            elif key in ['positions', 'one_hot', 'charges', 'atom_mask']:
                print(f"  {key}: shape {value.shape}")
        
        # Compare with typical QM9 ranges
        print("\nTypical QM9 property ranges (in eV):")
        print("  U0: -2000 to 0 eV")
        print("  HOMO: -15 to -5 eV")  
        print("  LUMO: -5 to 5 eV")
        print("  gap: 0 to 15 eV")
        print("  zpve: 0 to 3000 cm^-1 (converted)")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_with_realistic_qm9_values():
    """Test with more realistic QM9 property value ranges."""
    print("\n" + "="*60)
    print("Testing with realistic QM9 property values...")
    
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        with ase.db.connect(db_path) as db:
            # Use typical QM9 molecular property values from the actual dataset
            molecules = [
                {
                    'symbols': ['C', 'H', 'H', 'H', 'H'],
                    'positions': [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]],
                    'U0_Ha': -40.502155,     # Typical methane-like value
                    'HOMO_Ha': -0.379072,    # Typical HOMO
                    'LUMO_Ha': 0.082394,     # Typical LUMO
                    'gap_Ha': 0.461466,      # HOMO-LUMO gap
                    'ZPVE_Ha': 0.044752      # Zero-point energy
                },
                {
                    'symbols': ['O', 'H', 'H'],
                    'positions': [[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]],
                    'U0_Ha': -76.404702,     # Water-like
                    'HOMO_Ha': -0.484602,
                    'LUMO_Ha': 0.184402,
                    'gap_Ha': 0.669004,
                    'ZPVE_Ha': 0.021072
                }
            ]
            
            for mol_data in molecules:
                atoms = ase.Atoms(symbols=mol_data['symbols'], positions=mol_data['positions'])
                properties = {k: v for k, v in mol_data.items() if k not in ['symbols', 'positions']}
                db.write(atoms, **properties)
        
        # Now test with this more realistic data
        class TestArgs:
            def __init__(self):
                self.dataset = 'ase'
                self.ase_db_file = db_path
                self.ase_max_entries = 2
                self.filter_molecule_size = None
                self.sequential = False
                self.batch_size = 2
                self.num_workers = 0
                self.include_charges = True
                self.remove_h = False
                self.conditioning = []
                self.normalize_factors = None
                self.probabilistic_model = 'diffusion'
                self.diffusion_steps = 10
                self.diffusion_noise_schedule = 'polynomial_2'
                self.diffusion_noise_precision = 1e-5
                self.diffusion_loss_type = 'l2'
                self.context_node_nf = 0
                self.condition_time = True
                self.nf = 32
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
        
        from qm9 import dataset, losses
        from configs.datasets_config import get_dataset_info
        from qm9.models import get_model
        from equivariant_diffusion.utils import remove_mean_with_mask
        
        dataset_info = get_dataset_info(args.dataset, args.remove_h)
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        if args.normalize_factors is None:
            args.normalize_factors = [1.0, 4.0, 1.0]  # Use standard factors for this test
        
        print(f"  Selected normalization factors: {args.normalize_factors}")
        
        model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
        model.eval()
        
        data_batch = next(iter(dataloaders['train']))
        x = data_batch['positions'].to(device, dtype=torch.float32)
        node_mask = data_batch['atom_mask'].to(device, dtype=torch.float32).unsqueeze(2)
        edge_mask = data_batch['edge_mask'].to(device, dtype=torch.float32)
        one_hot = data_batch['one_hot'].to(device, dtype=torch.float32)
        charges = (data_batch['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype=torch.float32)

        x = remove_mean_with_mask(x, node_mask)
        h = {'categorical': one_hot, 'integer': charges}
        
        # Check property values in the batch
        print("  Property values in batch:")
        for key, value in data_batch.items():
            if isinstance(value, torch.Tensor) and value.numel() <= 10 and 'mask' not in key:
                print(f"    {key}: {value.flatten()}")
        
        x_norm, h_norm, delta_log_px = model.normalize(x, h, node_mask)
        
        print(f"  Normalized x range: [{x_norm.min():.3f}, {x_norm.max():.3f}]")
        
        with torch.no_grad():
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model, nodes_dist,
                                                                    x, h, node_mask, edge_mask, None)
            loss = nll + args.ode_regularization * reg_term
        
        print(f"  Loss with realistic QM9 values: {loss.item():.2f}")
        
        if loss.item() < 1000:
            print(f"  ✅ Loss is much more reasonable with realistic QM9 values!")
            return True
        else:
            print(f"  ⚠️  Loss still high, may need further investigation")
            return False
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG: PROPERTY VALUES AND LOSS ANALYSIS")
    print("=" * 60)
    
    debug_property_values()
    test_passed = test_with_realistic_qm9_values()
    
    print()
    print("=" * 60)
    if test_passed:
        print("✅ REALISTIC QM9 VALUES PRODUCE REASONABLE LOSS")
        print("The issue may be with unrealistic property values in user's dataset.")
    else:
        print("❌ STILL INVESTIGATING HIGH LOSS ISSUE")
    print("=" * 60)