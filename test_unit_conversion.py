#!/usr/bin/env python3
"""
Test to check if ASE dataset unit conversion is causing the high loss issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import tempfile
import ase.db
import ase

def create_test_database():
    """Create a test database with known property values."""
    db_path = tempfile.mktemp(suffix='.db')
    
    with ase.db.connect(db_path) as db:
        # Create a simple molecule with known properties
        atoms = ase.Atoms(symbols=['C', 'H'], positions=[[0, 0, 0], [1.0, 0, 0]])
        
        # Add properties in Hartree units (should be converted to eV)
        db.write(atoms, 
                 U0_Ha=-1.0,       # Should become -27.2114 eV 
                 HOMO_Ha=-0.5,     # Should become -13.6057 eV
                 LUMO_Ha=-0.3,     # Should become -8.16342 eV
                 gap_Ha=0.2,       # Should become 5.44228 eV
                 ZPVE_Ha=0.001)    # Should become 27.2114 eV
    
    return db_path

def test_ase_dataset_unit_conversion():
    """Test ASE dataset unit conversion behavior."""
    print("Testing ASE dataset unit conversion...")
    
    # Create test database
    db_path = create_test_database()
    
    try:
        import build_ase_dataset
        
        # Load data
        data_list = build_ase_dataset.load_ase_data(db_path, max_entries=1)
        print(f"  Loaded {len(data_list)} molecules")
        
        # Check initial values
        props = data_list[0]['properties']
        print(f"  Initial U0_Ha: {props.get('U0_Ha', 'N/A')}")
        print(f"  Initial HOMO_Ha: {props.get('HOMO_Ha', 'N/A')}")
        print(f"  Initial ZPVE_Ha: {props.get('ZPVE_Ha', 'N/A')}")
        
        # Create ASE dataset
        dataset = build_ase_dataset.ASEDataset(data_list)
        
        # Check properties in dataset
        print(f"  Dataset U0_Ha: {dataset.properties[0].get('U0_Ha', 'N/A')}")
        print(f"  Dataset HOMO_Ha: {dataset.properties[0].get('HOMO_Ha', 'N/A')}")
        print(f"  Dataset ZPVE_Ha: {dataset.properties[0].get('ZPVE_Ha', 'N/A')}")
        
        # Apply unit conversion
        qm9_to_eV = {'U0': 27.2114, 'homo': 27.2114, 'lumo': 27.2114, 'gap': 27.2114, 'zpve': 27211.4}
        print("  Applying unit conversion...")
        dataset.convert_units(qm9_to_eV)
        
        # Check converted values
        print(f"  After conversion U0_Ha: {dataset.properties[0].get('U0_Ha', 'N/A')}")
        print(f"  After conversion HOMO_Ha: {dataset.properties[0].get('HOMO_Ha', 'N/A')}")
        print(f"  After conversion ZPVE_Ha: {dataset.properties[0].get('ZPVE_Ha', 'N/A')}")
        
        # Apply again to test if it causes exponential growth
        print("  Applying unit conversion again...")
        dataset.convert_units(qm9_to_eV)
        
        print(f"  After 2nd conversion U0_Ha: {dataset.properties[0].get('U0_Ha', 'N/A')}")
        print(f"  After 2nd conversion HOMO_Ha: {dataset.properties[0].get('HOMO_Ha', 'N/A')}")
        print(f"  After 2nd conversion ZPVE_Ha: {dataset.properties[0].get('ZPVE_Ha', 'N/A')}")
        
        # Check if values grew exponentially
        u0_val = dataset.properties[0].get('U0_Ha', 0)
        if abs(u0_val) > 1000:  # Much larger than expected ~27 eV
            print(f"  ❌ Unit conversion applied multiple times - values too large!")
            return False
        else:
            print(f"  ✅ Unit conversion values seem reasonable")
            return True
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.remove(db_path)

def test_full_pipeline_conversion():
    """Test the full pipeline to see where repeated conversions might happen."""
    print("\nTesting full ASE pipeline...")
    
    db_path = create_test_database()
    
    try:
        import build_ase_dataset
        from configs.datasets_config import get_dataset_info
        
        # Simulate what happens in qm9/dataset.py for ASE datasets
        dataset_info = get_dataset_info('ase', False)
        
        # Load and split data
        split_data = build_ase_dataset.load_split_data(
            db_path,
            val_proportion=0.1,
            test_proportion=0.1,
            filter_size=None,
            max_entries=1
        )
        
        # Create transform
        transform = build_ase_dataset.ASETransform(
            dataset_info,
            include_charges=True,
            device=torch.device('cpu'),
            sequential=False
        )
        
        # Create datasets
        datasets = {}
        for key, data_list in zip(['train', 'valid', 'test'], split_data):
            if len(data_list) > 0:  # Only create if we have data
                datasets[key] = build_ase_dataset.ASEDataset(data_list, transform=transform)
        
        # Check properties before unit conversion
        if 'train' in datasets and len(datasets['train']) > 0:
            train_props = datasets['train'].properties[0]
            print(f"  Before conversion - U0_Ha: {train_props.get('U0_Ha', 'N/A')}")
            print(f"  Before conversion - ZPVE_Ha: {train_props.get('ZPVE_Ha', 'N/A')}")
        
        # Apply unit conversion (like in qm9/dataset.py)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 
                      'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        
        for split_name, dataset in datasets.items():
            if hasattr(dataset, 'convert_units'):
                print(f"  Converting units for {split_name} split")
                dataset.convert_units(qm9_to_eV)
        
        # Check properties after unit conversion
        if 'train' in datasets and len(datasets['train']) > 0:
            train_props = datasets['train'].properties[0]
            print(f"  After conversion - U0_Ha: {train_props.get('U0_Ha', 'N/A')}")
            print(f"  After conversion - ZPVE_Ha: {train_props.get('ZPVE_Ha', 'N/A')}")
        
        # Now test what happens when we get data from the dataset
        if 'train' in datasets and len(datasets['train']) > 0:
            data_item = datasets['train'][0]  # This calls the transform
            print(f"  Data item U0: {data_item.get('U0', 'N/A')}")
            print(f"  Data item zpve: {data_item.get('zpve', 'N/A')}")
            
            # Check if values are reasonable for loss computation
            u0_val = data_item.get('U0', torch.tensor(0.0)).item()
            zpve_val = data_item.get('zpve', torch.tensor(0.0)).item()
            
            print(f"  Final U0 value: {u0_val:.6f}")
            print(f"  Final zpve value: {zpve_val:.6f}")
            
            # These should be in reasonable eV ranges
            if abs(u0_val) > 1000 or abs(zpve_val) > 1000:
                print(f"  ❌ Property values too large for reasonable loss computation!")
                return False
            else:
                print(f"  ✅ Property values seem reasonable")
                return True
        else:
            print(f"  ❌ No training data to test")
            return False
            
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    print("=" * 60)
    print("ASE DATASET UNIT CONVERSION TESTS")
    print("=" * 60)
    
    test1_passed = test_ase_dataset_unit_conversion()
    test2_passed = test_full_pipeline_conversion()
    
    print()
    print("=" * 60)
    if test1_passed and test2_passed:
        print("✅ ALL UNIT CONVERSION TESTS PASSED")
    else:
        print("❌ SOME UNIT CONVERSION TESTS FAILED")
        if not test1_passed:
            print("  - Basic unit conversion issue detected")
        if not test2_passed:
            print("  - Full pipeline conversion issue detected")
    print("=" * 60)