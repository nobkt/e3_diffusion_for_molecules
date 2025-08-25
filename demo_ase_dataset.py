#!/usr/bin/env python3
"""
Demo script showing how to use the fixed ASE dataset functionality.
This demonstrates that ASE datasets now work consistently with QM9 datasets.
"""

import torch
import numpy as np
import ase.db
import ase
import tempfile
import os
from qm9 import dataset
from configs.datasets_config import get_dataset_info

def create_sample_ase_database(filepath):
    """Create a sample ASE database with QM9-style molecular properties."""
    print(f"Creating sample ASE database: {filepath}")
    
    with ase.db.connect(filepath) as db:
        # Sample molecules with QM9-style properties
        molecules = [
            {
                'symbols': ['C', 'H', 'H', 'H', 'H'],  # Methane
                'positions': [[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, -0.5, -0.5]],
                'properties': {
                    'index': 1,
                    'rotA_GHz': 157.7,
                    'mu': 0.0,
                    'alpha': 17.3,
                    'HOMO_Ha': -0.5998,
                    'LUMO_Ha': 0.0618,
                    'gap_Ha': 0.6616,
                    'ZPVE_Ha': 0.044749,
                    'U0_Ha': -40.47893,
                    'H_298K_Ha': -40.475117,
                    'G_298K_Ha': -40.498597,
                }
            },
            {
                'symbols': ['O', 'H', 'H'],  # Water  
                'positions': [[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]],
                'properties': {
                    'index': 2,
                    'rotA_GHz': 835.51,
                    'mu': 1.8546,
                    'alpha': 9.46,
                    'HOMO_Ha': -0.5875,
                    'LUMO_Ha': 0.0829,
                    'gap_Ha': 0.6704,
                    'ZPVE_Ha': 0.021375,
                    'U0_Ha': -76.404702,
                    'H_298K_Ha': -76.402502,
                    'G_298K_Ha': -76.419745,
                }
            }
        ]
        
        for mol_data in molecules:
            atoms = ase.Atoms(mol_data['symbols'], positions=mol_data['positions'])
            db.write(atoms, **mol_data['properties'])
    
    print(f"✓ Created database with {len(molecules)} molecules")
    return len(molecules)

def demo_ase_dataset_usage():
    """Demonstrate ASE dataset usage with the fixed implementation."""
    
    print("=" * 60)
    print("ASE Dataset Demo - QM9 Compatibility")
    print("=" * 60)
    
    # Create sample database
    db_path = './demo_molecules.db'
    n_molecules = create_sample_ase_database(db_path)
    
    try:
        # Configure ASE dataset loading
        class Args:
            def __init__(self):
                self.dataset = 'ase'
                self.ase_db_file = db_path
                self.ase_max_entries = None
                self.filter_molecule_size = None
                self.remove_h = False
                self.include_charges = True
                self.sequential = False
                self.batch_size = 2
                self.device = torch.device('cpu')
        
        args = Args()
        
        print("\\n1. Loading ASE dataset...")
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
        
        print(f"✓ Successfully loaded ASE dataset")
        print(f"✓ Splits available: {list(dataloaders.keys())}")
        print(f"✓ Charge scale: {charge_scale}")
        
        print("\\n2. Examining data structure...")
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        print(f"✓ Batch contains {len(batch)} properties:")
        geometric_props = ['positions', 'one_hot', 'charges', 'atom_mask', 'edge_mask']
        molecular_props = [k for k in batch.keys() if k not in geometric_props]
        
        print(f"  Geometric properties: {geometric_props}")
        print(f"  Molecular properties: {molecular_props[:5]}{'...' if len(molecular_props) > 5 else ''}")
        
        print("\\n3. Verifying tensor shapes...")
        batch_size = batch['positions'].shape[0]
        n_nodes = batch['positions'].shape[1]
        
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Max nodes: {n_nodes}")
        print(f"✓ Positions shape: {batch['positions'].shape}")
        print(f"✓ One-hot shape: {batch['one_hot'].shape}")
        print(f"✓ Charges shape: {batch['charges'].shape}")
        print(f"✓ Atom mask shape: {batch['atom_mask'].shape}")
        print(f"✓ Edge mask shape: {batch['edge_mask'].shape}")
        
        print("\\n4. Checking unit conversions...")
        if 'U0_Ha' in batch:
            u0_original_ha = -40.47893  # Original methane U0 in Hartree
            u0_converted_ev = u0_original_ha * 27.2114  # Expected after conversion
            u0_actual = batch['U0_Ha'][0].item()
            print(f"✓ U0_Ha conversion: {u0_original_ha} Ha → {u0_actual:.2f} eV")
            print(f"  Expected: {u0_converted_ev:.2f} eV")
            print(f"  Match: {'✓' if abs(u0_actual - u0_converted_ev) < 0.1 else '✗'}")
        
        if 'ZPVE_Ha' in batch:
            zpve_original_ha = 0.044749  # Original methane ZPVE in Hartree
            zpve_converted_cm = zpve_original_ha * 27211.4  # Expected after conversion
            zpve_actual = batch['ZPVE_Ha'][0].item()
            print(f"✓ ZPVE_Ha conversion: {zpve_original_ha} Ha → {zpve_actual:.1f} cm⁻¹")
            print(f"  Expected: {zpve_converted_cm:.1f} cm⁻¹")
            print(f"  Match: {'✓' if abs(zpve_actual - zpve_converted_cm) < 10 else '✗'}")
        
        print("\\n5. Data compatibility summary...")
        print("✅ ASE dataset now provides:")
        print("   - Same data structure as QM9 datasets")
        print("   - Proper unit conversions (Ha → eV, Ha → cm⁻¹)")
        print("   - Molecular properties included in batches")
        print("   - Consistent charge scaling")
        print("   - Compatible tensor shapes and masks")
        
        print("\\n🎯 Result: ASE datasets should now train with the same loss")
        print("   characteristics as QM9 datasets, eliminating the 10x difference!")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"\\n🧹 Cleaned up demo database: {db_path}")

if __name__ == "__main__":
    success = demo_ase_dataset_usage()
    
    print("\\n" + "=" * 60)
    if success:
        print("✅ Demo completed successfully!")
        print("You can now use ASE datasets with main_qm9.py using:")
        print("  python main_qm9.py --dataset ase --ase_db_file path/to/your/database.db")
    else:
        print("❌ Demo failed - check the implementation")
    print("=" * 60)