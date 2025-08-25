#!/usr/bin/env python3
"""
Simple test script to verify ASE DB and OpenBabel functionality.
Creates a small test ASE database and tests basic operations.
"""

import torch
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ase import Atoms
from ase.db import connect
from qm9.ase_db_dataset import load_ase_db_datasets
from qm9.openbabel_functions import BasicMolecularMetricsOB, build_molecule_ob, mol2smiles_ob
from configs.datasets_config import get_dataset_info


def create_test_ase_db():
    """Create a small test ASE database with simple molecules."""
    # Create temporary database
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    db = connect(db_path)
    
    # Add some simple molecules
    molecules = [
        # Methane (CH4)
        {'symbols': ['C', 'H', 'H', 'H', 'H'],
         'positions': np.array([[0.0, 0.0, 0.0],
                               [0.629, 0.629, 0.629],
                               [-0.629, -0.629, 0.629],
                               [-0.629, 0.629, -0.629],
                               [0.629, -0.629, -0.629]]),
         'properties': {'gap': 10.0, 'homo': -8.0}},
        
        # Water (H2O)  
        {'symbols': ['O', 'H', 'H'],
         'positions': np.array([[0.0, 0.0, 0.0],
                               [0.757, 0.586, 0.0],
                               [-0.757, 0.586, 0.0]]),
         'properties': {'gap': 12.0, 'homo': -12.0}},
        
        # Ammonia (NH3)
        {'symbols': ['N', 'H', 'H', 'H'],
         'positions': np.array([[0.0, 0.0, 0.0],
                               [0.941, 0.0, 0.0],
                               [-0.471, 0.816, 0.0],
                               [-0.471, -0.816, 0.0]]),
         'properties': {'gap': 11.0, 'homo': -10.0}},
    ]
    
    for i, mol_data in enumerate(molecules):
        atoms = Atoms(symbols=mol_data['symbols'], 
                     positions=mol_data['positions'])
        properties = mol_data['properties']
        properties['split'] = 'train' if i < 2 else 'test'
        db.write(atoms, key_value_pairs=properties)
    
    print(f"Created test database with {len(molecules)} molecules: {db_path}")
    return db_path


def test_ase_db_loading(db_path):
    """Test ASE database loading."""
    print("\n=== Testing ASE DB Loading ===")
    
    # Test dataset info
    dataset_info = get_dataset_info('ase_db_qm9', remove_h=False)
    print(f"Dataset info: {dataset_info['name']}")
    print(f"Atom decoder: {dataset_info['atom_decoder']}")
    print(f"Use OpenBabel: {dataset_info.get('use_openbabel', False)}")
    
    # Load datasets
    datasets = load_ase_db_datasets(
        db_path=db_path,
        dataset_info=dataset_info,
        split_ratio=(0.7, 0.2, 0.1),
        remove_h=False,
        max_atoms=10,
        random_seed=42
    )
    
    print(f"Loaded datasets with splits:")
    for split, dataset in datasets.items():
        print(f"  {split}: {len(dataset)} molecules")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"    Sample keys: {list(sample.keys())}")
            print(f"    Positions shape: {sample['positions'].shape}")
            print(f"    Charges shape: {sample['charges'].shape}")
            print(f"    Atom mask sum: {sample['atom_mask'].sum()}")
    
    return datasets


def test_openbabel_functions(datasets):
    """Test OpenBabel molecular functions."""
    print("\n=== Testing OpenBabel Functions ===")
    
    dataset_info = get_dataset_info('ase_db_qm9', remove_h=False)
    
    # Extract some test molecules
    test_molecules = []
    for dataset in datasets.values():
        if len(dataset) > 0:
            sample = dataset[0]
            positions = sample['positions']
            charges = sample['charges']
            atom_mask = sample['atom_mask']
            
            # Get actual atoms
            n_atoms = int(atom_mask.sum())
            positions = positions[:n_atoms]
            charges = charges[:n_atoms]
            
            # Convert atomic numbers to atom types
            atom_types = []
            for charge in charges:
                charge_int = int(charge.item())
                if charge_int == 1:  # H
                    atom_types.append(0)
                elif charge_int == 6:  # C  
                    atom_types.append(1)
                elif charge_int == 7:  # N
                    atom_types.append(2)
                elif charge_int == 8:  # O
                    atom_types.append(3)
                elif charge_int == 9:  # F
                    atom_types.append(4)
                else:
                    print(f"Warning: Unknown charge {charge_int}")
                    continue
            
            if len(atom_types) == n_atoms:
                atom_types = torch.tensor(atom_types, dtype=torch.long)
                test_molecules.append((positions, atom_types))
                print(f"Added test molecule with {n_atoms} atoms")
    
    if len(test_molecules) == 0:
        print("No test molecules could be created")
        return False
    
    # Test individual functions
    print("\nTesting individual molecule building...")
    for i, (positions, atom_types) in enumerate(test_molecules):
        try:
            # Test build_molecule_ob
            mol = build_molecule_ob(positions, atom_types, dataset_info)
            print(f"Molecule {i}: Built successfully")
            
            # Test mol2smiles_ob  
            smiles = mol2smiles_ob(mol)
            print(f"Molecule {i}: SMILES = {smiles}")
            
        except Exception as e:
            print(f"Molecule {i}: Error = {e}")
    
    # Test metrics
    print("\nTesting molecular metrics...")
    try:
        metrics = BasicMolecularMetricsOB(dataset_info)
        results = metrics.evaluate(test_molecules)
        print(f"Metrics results: validity={results[0][0]:.3f}, uniqueness={results[0][1]:.3f}, novelty={results[0][2]:.3f}")
        return True
    except Exception as e:
        print(f"Metrics error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_compatibility():
    """Test that ASE DB datasets work with existing infrastructure."""
    print("\n=== Testing Dataset Compatibility ===")
    
    try:
        # Test dataset configuration
        from configs.datasets_config import get_dataset_info
        
        # Test all ASE DB configurations  
        configs = [
            ('ase_db_qm9', False),
            ('ase_db_qm9', True),
            ('ase_db_generic', False)
        ]
        
        for dataset_name, remove_h in configs:
            try:
                info = get_dataset_info(dataset_name, remove_h)
                print(f"Config {dataset_name} (remove_h={remove_h}): OK")
                print(f"  - Use OpenBabel: {info.get('use_openbabel', False)}")
                print(f"  - Atom decoder: {info['atom_decoder']}")
            except Exception as e:
                print(f"Config {dataset_name} (remove_h={remove_h}): Error = {e}")
        
        return True
        
    except Exception as e:
        print(f"Dataset compatibility test failed: {e}")
        return False


def main():
    print("=== ASE DB and OpenBabel Functionality Test ===")
    
    try:
        # Create test database
        db_path = create_test_ase_db()
        
        # Test dataset loading
        datasets = test_ase_db_loading(db_path)
        
        # Test OpenBabel functions
        test_openbabel_functions(datasets)
        
        # Test dataset compatibility
        test_dataset_compatibility()
        
        # Cleanup
        os.unlink(db_path)
        
        print("\n=== All tests completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n=== Test failed: {e} ===")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)