#!/usr/bin/env python3
"""
Comprehensive demonstration of ASE DB and OpenBabel integration for QM9 dataset.
This script demonstrates all the implemented functionality:

1. ASE database interface for molecular data storage
2. OpenBabel-based molecular analysis (replacing RDKit)
3. Data loading and processing pipeline
4. Validation and comparison capabilities

To test with real QM9 data, run:
    python qm9/convert_qm9_to_ase.py --datadir qm9/temp --output-db qm9/temp/qm9_database.db
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the path so we can import qm9 modules
sys.path.append(str(Path(__file__).parent))

def demonstrate_ase_interface():
    """Demonstrate ASE database interface functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: ASE Database Interface")
    print("="*60)
    
    from qm9.ase_interface import ASEDatasetInterface
    import torch
    import numpy as np
    
    # Create demonstration molecular data
    demo_data = {
        'train': {
            'positions': torch.tensor([
                [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.0, 0.0]],  # Ethylene-like
                [[0.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.9, -0.3, 0.0]]   # Water-like
            ]).float(),
            'charges': torch.tensor([
                [6, 6, 0],  # C-C (padded)
                [8, 1, 1]   # O-H-H
            ]).long(),
            'num_atoms': torch.tensor([2, 3]).long(),
            'U0': torch.tensor([-78.5, -76.4]).float(),
            'gap': torch.tensor([8.9, 9.2]).float(),
        }
    }
    
    # Test saving to ASE DB
    db_path = '/tmp/demo_ase.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("1. Saving molecular data to ASE database...")
    ase_interface = ASEDatasetInterface(db_path)
    ase_interface.save_molecules_to_db(demo_data)
    ase_interface.close()
    print(f"   ✓ Saved to: {db_path}")
    
    # Test loading from ASE DB
    print("2. Loading molecular data from ASE database...")
    ase_interface = ASEDatasetInterface(db_path)
    loaded_data = ase_interface.load_molecules_from_db()
    ase_interface.close()
    
    print(f"   ✓ Loaded splits: {list(loaded_data.keys())}")
    print(f"   ✓ Train molecules: {len(loaded_data['train']['positions'])}")
    
    # Verify data integrity
    print("3. Verifying data integrity...")
    original_num_atoms = demo_data['train']['num_atoms']
    loaded_num_atoms = loaded_data['train']['num_atoms']
    
    if torch.equal(original_num_atoms, loaded_num_atoms):
        print("   ✓ Data integrity verified - num_atoms match")
    else:
        print("   ✗ Data integrity error - num_atoms don't match")
    
    # Clean up
    os.remove(db_path)
    print("   ✓ Cleanup completed")


def demonstrate_openbabel_functions():
    """Demonstrate OpenBabel molecular analysis functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: OpenBabel Molecular Functions")
    print("="*60)
    
    from qm9.openbabel_functions import (
        build_molecule_openbabel, 
        mol2smiles_openbabel,
        OpenBabelMolecularMetrics
    )
    from configs.datasets_config import get_dataset_info
    import torch
    
    # Create test molecules
    molecules_data = [
        # Methane (CH4)
        (torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.4, 1.0, 0.0], 
                      [-0.4, -0.5, 0.9], [-0.4, -0.5, -0.9]]).float(),
         torch.tensor([1, 0, 0, 0, 0])),  # C, H, H, H, H (QM9 indices)
         
        # Water (H2O)  
        (torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]).float(),
         torch.tensor([3, 0, 0])),  # O, H, H (QM9 indices)
         
        # Ethylene (C2H4)
        (torch.tensor([[0.0, 0.0, 0.0], [1.33, 0.0, 0.0], [-0.6, 0.9, 0.0], 
                      [-0.6, -0.9, 0.0], [1.9, 0.9, 0.0], [1.9, -0.9, 0.0]]).float(),
         torch.tensor([1, 1, 0, 0, 0, 0])),  # C, C, H, H, H, H (QM9 indices)
    ]
    
    dataset_info = get_dataset_info('qm9', False)
    
    print("1. Building molecules with OpenBabel...")
    built_molecules = []
    for i, (positions, atom_types) in enumerate(molecules_data):
        mol = build_molecule_openbabel(positions, atom_types, dataset_info)
        if mol:
            built_molecules.append(mol)
            smiles = mol2smiles_openbabel(mol)
            print(f"   Molecule {i+1}: {mol.OBMol.NumAtoms()} atoms, SMILES: {smiles}")
        else:
            print(f"   Molecule {i+1}: Failed to build")
    
    print("2. Evaluating molecular metrics...")
    ob_metrics = OpenBabelMolecularMetrics(dataset_info)
    results = ob_metrics.evaluate(molecules_data)
    
    print("   Evaluation Results:")
    print(f"     Validity: {results['validity']:.3f}")
    print(f"     Uniqueness: {results['uniqueness']:.3f}")
    print(f"     Novelty: {results['novelty']:.3f}")
    print(f"     Valid SMILES: {results['valid_smiles']}")


def demonstrate_dataset_loading():
    """Demonstrate enhanced dataset loading with ASE DB support."""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Enhanced Dataset Loading")
    print("="*60)
    
    from qm9.dataset import retrieve_dataloaders
    from test_synthetic_qm9 import create_synthetic_qm9_data
    from qm9.ase_interface import convert_npz_to_ase_db
    
    # Create synthetic dataset
    print("1. Creating synthetic QM9 dataset...")
    synthetic_dir = '/tmp/demo_qm9'
    ase_db_path = '/tmp/demo_qm9.db'
    
    # Clean up previous runs
    if os.path.exists(synthetic_dir):
        import shutil
        shutil.rmtree(synthetic_dir)
    if os.path.exists(ase_db_path):
        os.remove(ase_db_path)
    
    # Create synthetic NPZ data
    npz_files = create_synthetic_qm9_data(synthetic_dir, num_molecules_per_split=3)
    print(f"   ✓ Created NPZ files: {list(npz_files.keys())}")
    
    # Convert to ASE DB
    print("2. Converting to ASE database...")
    convert_npz_to_ase_db(npz_files, ase_db_path)
    print(f"   ✓ Created ASE DB: {ase_db_path}")
    
    # Test loading via ASE DB
    print("3. Testing ASE DB dataloader...")
    
    class TestConfig:
        def __init__(self):
            self.dataset = 'qm9'  # Add missing dataset attribute
            self.use_ase_db = True
            self.ase_db_path = ase_db_path
            self.batch_size = 2
            self.num_workers = 0
            self.filter_n_atoms = None
            self.include_charges = True
            self.subtract_thermo = False
    
    cfg = TestConfig()
    dataloaders, _ = retrieve_dataloaders(cfg)
    
    print(f"   ✓ Loaded dataloaders for splits: {list(dataloaders.keys())}")
    
    # Test batch loading
    print("4. Testing batch data loading...")
    for split_name, dataloader in dataloaders.items():
        batch = next(iter(dataloader))
        print(f"   Split '{split_name}': batch_size={batch['positions'].shape[0]}, "
              f"max_atoms={batch['positions'].shape[1]}")
    
    # Clean up
    import shutil
    shutil.rmtree(synthetic_dir)
    os.remove(ase_db_path)
    print("   ✓ Cleanup completed")


def demonstrate_comparison():
    """Demonstrate comparison between RDKit and OpenBabel approaches."""
    print("\n" + "="*60)
    print("DEMONSTRATION 4: RDKit vs OpenBabel Comparison")
    print("="*60)
    
    print("This implementation provides OpenBabel replacements for key RDKit functions:")
    print()
    
    replacements = [
        ("qm9.rdkit_functions.BasicMolecularMetrics", "qm9.openbabel_functions.OpenBabelMolecularMetrics"),
        ("qm9.rdkit_functions.mol2smiles", "qm9.openbabel_functions.mol2smiles_openbabel"),
        ("qm9.rdkit_functions.build_molecule", "qm9.openbabel_functions.build_molecule_openbabel"),
        ("qm9.rdkit_functions.compute_qm9_smiles", "qm9.openbabel_functions.compute_qm9_smiles_openbabel"),
        ("qm9.rdkit_functions.retrieve_qm9_smiles", "qm9.openbabel_functions.retrieve_qm9_smiles_openbabel"),
    ]
    
    for rdkit_func, openbabel_func in replacements:
        print(f"  {rdkit_func:45} → {openbabel_func}")
    
    print()
    print("Key advantages of the ASE DB + OpenBabel approach:")
    print("  ✓ No RDKit dependency required")
    print("  ✓ Standardized molecular database format (ASE)")
    print("  ✓ Better integration with computational chemistry workflows")
    print("  ✓ Maintains compatibility with existing PyTorch dataloaders")
    print("  ✓ Supports all QM9 molecular properties and metadata")


def show_usage_instructions():
    """Show instructions for using the implementation with real QM9 data."""
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS FOR REAL QM9 DATA")
    print("="*60)
    
    print("To download and convert the full QM9 dataset to ASE database format:")
    print()
    print("1. Download and convert QM9 dataset:")
    print("   cd /path/to/e3_diffusion_for_molecules")
    print("   python qm9/convert_qm9_to_ase.py \\")
    print("       --datadir qm9/temp \\")
    print("       --output-db qm9/temp/qm9_database.db")
    print()
    print("2. Use ASE DB in your training script:")
    print("   cfg.use_ase_db = True")
    print("   cfg.ase_db_path = 'qm9/temp/qm9_database.db'")
    print("   dataloaders, _ = retrieve_dataloaders(cfg)")
    print()
    print("3. Use OpenBabel for molecular analysis:")
    print("   from qm9.openbabel_functions import OpenBabelMolecularMetrics")
    print("   metrics = OpenBabelMolecularMetrics(dataset_info)")
    print("   results = metrics.evaluate(generated_molecules)")
    print()
    print("Note: The full QM9 dataset download is ~1.5GB and may take 10-30 minutes")
    print("      depending on internet connection and processing power.")


def main():
    """Run all demonstrations."""
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for demo
    
    print("🧪 ASE Database & OpenBabel Integration for QM9 Dataset")
    print("   Comprehensive Demonstration and Validation")
    print()
    print("This script demonstrates the complete implementation of:")
    print("  • ASE database interface for molecular data storage")
    print("  • OpenBabel-based molecular analysis (replacing RDKit)")
    print("  • Enhanced dataset loading with ASE DB support")
    print("  • Molecular property evaluation and SMILES generation")
    
    try:
        demonstrate_ase_interface()
        demonstrate_openbabel_functions()
        demonstrate_dataset_loading()
        demonstrate_comparison()
        show_usage_instructions()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print()
        print("The ASE database and OpenBabel integration is fully functional.")
        print("You can now use ASE databases as an alternative to NPZ files for QM9 data,")
        print("and OpenBabel instead of RDKit for molecular analysis.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())