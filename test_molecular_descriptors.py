#!/usr/bin/env python3
"""
Test script for the new molecular descriptor extraction functionality
"""

import sys
import numpy as np
import torch

def test_molecular_descriptors():
    """Test the molecular descriptor extraction functions"""
    
    try:
        from ase import Atoms
        from qm9.openbabel_functions import (
            extract_atom_types_from_ase,
            extract_molecular_weight_from_ase,
            extract_molecular_descriptors_ase_openbabel,
            is_openbabel_available
        )
        
        print("Testing molecular descriptor extraction...")
        print(f"OpenBabel available: {is_openbabel_available()}")
        
        # Create test molecules
        
        # 1. Water molecule (H2O)
        water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
        print("\n--- Water molecule (H2O) ---")
        print(f"Atom types: {extract_atom_types_from_ase(water)}")
        print(f"Molecular weight: {extract_molecular_weight_from_ase(water):.3f} u")
        
        descriptors = extract_molecular_descriptors_ase_openbabel(water)
        print(f"Full descriptors: {descriptors}")
        
        # 2. Methane molecule (CH4)
        methane = Atoms('CH4', positions=[
            [0, 0, 0],          # C
            [1.089, 1.089, 1.089],   # H
            [1.089, -1.089, -1.089], # H
            [-1.089, 1.089, -1.089], # H
            [-1.089, -1.089, 1.089]  # H
        ])
        print("\n--- Methane molecule (CH4) ---")
        print(f"Atom types: {extract_atom_types_from_ase(methane)}")
        print(f"Molecular weight: {extract_molecular_weight_from_ase(methane):.3f} u")
        
        descriptors = extract_molecular_descriptors_ase_openbabel(methane)
        print(f"Full descriptors: {descriptors}")
        
        # 3. Benzene molecule (C6H6) - simple approximation
        benzene_positions = []
        # Carbon ring
        for i in range(6):
            angle = i * np.pi / 3
            x = 1.4 * np.cos(angle)
            y = 1.4 * np.sin(angle)
            benzene_positions.append([x, y, 0])
        
        # Hydrogen atoms
        for i in range(6):
            angle = i * np.pi / 3
            x = 2.4 * np.cos(angle)
            y = 2.4 * np.sin(angle)
            benzene_positions.append([x, y, 0])
        
        benzene = Atoms('C6H6', positions=benzene_positions)
        print("\n--- Benzene molecule (C6H6) ---")
        print(f"Atom types: {extract_atom_types_from_ase(benzene)}")
        print(f"Molecular weight: {extract_molecular_weight_from_ase(benzene):.3f} u")
        
        descriptors = extract_molecular_descriptors_ase_openbabel(benzene)
        print(f"Full descriptors: {descriptors}")
        
        print("\n✅ Molecular descriptor extraction test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure ASE and OpenBabel are installed:")
        print("pip install ase openbabel-wheel")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ase_dataset_conversion():
    """Test the ASE dataset conversion functionality"""
    
    try:
        from ase import Atoms
        from qm9.dataset import convert_ase_to_dataset_format
        
        print("\n" + "="*50)
        print("Testing ASE dataset conversion...")
        
        # Create test molecules
        water = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
        methane = Atoms('CH4', positions=[
            [0, 0, 0],          # C
            [1.089, 1.089, 1.089],   # H
            [1.089, -1.089, -1.089], # H
            [-1.089, 1.089, -1.089], # H
            [-1.089, -1.089, 1.089]  # H
        ])
        
        atoms_list = [water, methane]
        properties_list = [
            {'energy': -76.4, 'homo': -12.6},
            {'energy': -40.5, 'homo': -14.4}
        ]
        
        # Test conversion
        dataset_data = convert_ase_to_dataset_format(atoms_list, properties_list)
        
        print(f"Dataset keys: {list(dataset_data.keys())}")
        print(f"Positions shape: {dataset_data['positions'].shape}")
        print(f"Charges shape: {dataset_data['charges'].shape}")
        print(f"Number of atoms: {dataset_data['num_atoms']}")
        
        if 'molecular_weight' in dataset_data:
            print(f"Molecular weights: {dataset_data['molecular_weight']}")
        if 'pi_conjugation_ratio' in dataset_data:
            print(f"π conjugation ratios: {dataset_data['pi_conjugation_ratio']}")
        if 'atom_types_encoding' in dataset_data:
            print(f"Atom types encoding shape: {dataset_data['atom_types_encoding'].shape}")
        if 'functional_groups_encoding' in dataset_data:
            print(f"Functional groups encoding shape: {dataset_data['functional_groups_encoding'].shape}")
        if '_atom_types_mapping' in dataset_data:
            print(f"Atom types mapping: {dataset_data['_atom_types_mapping']}")
        if '_functional_groups_mapping' in dataset_data:
            print(f"Functional groups mapping: {dataset_data['_functional_groups_mapping']}")
        
        print("\n✅ ASE dataset conversion test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during dataset conversion testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing molecular descriptor functionality for ASE database conditioning")
    
    success1 = test_molecular_descriptors()
    success2 = test_ase_dataset_conversion()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)