#!/usr/bin/env python3
"""
Example script demonstrating ASE database support with OpenBabel.

This script shows how to:
1. Create an ASE database from molecular structures
2. Load the database with the E3 diffusion framework
3. Train or analyze molecules using OpenBabel instead of RDKit
"""

import sys
import os
import numpy as np
import torch
from ase import Atoms
from ase.db import connect

# Add current directory to path
sys.path.insert(0, '/home/runner/work/e3_diffusion_for_molecules/e3_diffusion_for_molecules')

from qm9.ase_dataset import create_sample_ase_db, load_ase_dataset, get_ase_dataset_info
from qm9.dataset import retrieve_dataloaders
from qm9.openbabel_functions import BasicMolecularMetricsOB, build_molecule_ob
from qm9.analyze import analyze_stability_for_molecules


def create_custom_ase_database(db_path: str):
    """
    Create a custom ASE database with various molecules.
    This example shows how to populate a database with molecular structures.
    """
    print(f"Creating custom ASE database at: {db_path}")
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = connect(db_path)
    
    # Define various molecular structures
    molecules = [
        # Water
        {
            'symbols': ['O', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]],
            'name': 'water'
        },
        # Methane
        {
            'symbols': ['C', 'H', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [-0.36, 1.03, 0.0], 
                         [-0.36, -0.52, 0.89], [-0.36, -0.52, -0.89]],
            'name': 'methane'
        },
        # Ammonia
        {
            'symbols': ['N', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [1.01, 0.0, 0.0], [-0.34, 0.94, 0.0], [-0.34, -0.47, 0.82]],
            'name': 'ammonia'
        },
        # Carbon dioxide
        {
            'symbols': ['C', 'O', 'O'],
            'positions': [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]],
            'name': 'CO2'
        },
        # Ethane
        {
            'symbols': ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], 
                         [-0.51, 1.03, 0.0], [-0.51, -0.51, 0.89], [-0.51, -0.51, -0.89],
                         [2.05, 1.03, 0.0], [2.05, -0.51, 0.89], [2.05, -0.51, -0.89]],
            'name': 'ethane'
        }
    ]
    
    # Add molecules to database
    for mol_data in molecules:
        atoms = Atoms(symbols=mol_data['symbols'], positions=mol_data['positions'])
        
        # You can add additional properties to the database
        db.write(atoms, mol_name=mol_data['name'], 
                sample_energy=np.random.uniform(-10, 0),  # Example energy
                sample_forces=str(np.random.normal(0, 0.1, (len(atoms), 3)).tolist()))  # Example forces
    
    print(f"Created database with {len(molecules)} molecules")
    return len(molecules)


def demonstrate_ase_dataset_loading():
    """
    Demonstrate how to load and use ASE datasets with the framework.
    """
    print("\n" + "="*60)
    print("Demonstrating ASE Dataset Loading")
    print("="*60)
    
    # Create database
    db_path = "/tmp/example_molecules.db"
    n_molecules = create_custom_ase_database(db_path)
    
    # Method 1: Direct ASE dataset loading
    print("\n1. Direct ASE dataset loading:")
    
    class Config:
        batch_size = 2
        num_workers = 0
        max_atoms = 20
        include_charges = True
        dataset = 'ase_example'
        ase_db_path = db_path
    
    config = Config()
    dataloaders, charge_scale = load_ase_dataset(db_path, config)
    
    print(f"✓ Loaded {len(dataloaders)} data splits")
    for split, loader in dataloaders.items():
        print(f"  {split}: {len(loader.dataset)} molecules")
    
    # Method 2: Using the main dataset retrieval function
    print("\n2. Using main dataset retrieval:")
    
    config.dataset = 'ase_general'  # Any name containing 'ase'
    dataloaders2, charge_scale2 = retrieve_dataloaders(config)
    
    print(f"✓ Retrieved {len(dataloaders2)} data splits via main loader")
    
    # Cleanup
    os.unlink(db_path)
    
    return dataloaders


def demonstrate_openbabel_analysis():
    """
    Demonstrate molecular analysis using OpenBabel instead of RDKit.
    """
    print("\n" + "="*60)
    print("Demonstrating OpenBabel Molecular Analysis")
    print("="*60)
    
    # Create a sample dataset
    db_path = "/tmp/analysis_molecules.db"
    create_sample_ase_db(db_path, 20)
    
    # Load dataset
    class Config:
        batch_size = 4
        num_workers = 0
        max_atoms = 20
        include_charges = True
        dataset = 'ase_analysis'
        ase_db_path = db_path
    
    config = Config()
    dataloaders, _ = retrieve_dataloaders(config)
    dataset_info = dataloaders['train'].dataset_info
    
    print(f"Dataset info:")
    print(f"  Name: {dataset_info['name']}")
    print(f"  Atom types: {dataset_info['atom_decoder']}")
    print(f"  Max atoms: {dataset_info['max_n_nodes']}")
    
    # Get sample molecules
    train_loader = dataloaders['train']
    for batch in train_loader:
        # Convert to molecule list format
        batch_size = batch['positions'].shape[0]
        molecules = {
            'one_hot': [batch['one_hot'][i] for i in range(batch_size)],
            'x': [batch['positions'][i] for i in range(batch_size)],
            'node_mask': [batch['atom_mask'][i] for i in range(batch_size)]
        }
        break
    
    print(f"\n✓ Analyzing {batch_size} molecules...")
    
    # Analyze using stability check and OpenBabel metrics
    validity_dict, metrics = analyze_stability_for_molecules(molecules, dataset_info)
    
    print(f"Stability Results:")
    print(f"  Molecular stability: {validity_dict['mol_stable']:.2%}")
    print(f"  Atomic stability: {validity_dict['atm_stable']:.2%}")
    
    if metrics is not None:
        print(f"OpenBabel Metrics:")
        print(f"  Validity: {metrics[0][0]:.2%}")
        print(f"  Uniqueness: {metrics[0][1]:.2%}")
        print(f"  Novelty: {metrics[0][2]:.2%}")
    
    # Cleanup
    os.unlink(db_path)


def demonstrate_molecule_building():
    """
    Demonstrate building molecules with OpenBabel.
    """
    print("\n" + "="*60)
    print("Demonstrating OpenBabel Molecule Building")
    print("="*60)
    
    # Example: Build a water molecule
    from configs.datasets_config import qm9_with_h
    
    # Water molecule: O-H-H
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]])
    atom_types = torch.tensor([3, 0, 0])  # O, H, H (using QM9 encoding)
    
    print("Building water molecule...")
    mol = build_molecule_ob(positions, atom_types, qm9_with_h)
    
    if mol is not None:
        print(f"✓ Successfully built molecule with {mol.NumAtoms()} atoms")
        print(f"  Bonds: {mol.NumBonds()}")
        
        # Try to get molecular formula
        formula = mol.GetFormula()
        print(f"  Formula: {formula}")
        
        # Test molecular metrics
        metrics = BasicMolecularMetricsOB(qm9_with_h)
        generated = [(positions, atom_types)]
        result = metrics.evaluate(generated)
        
        print(f"  Validity: {result[0][0]:.2%}")
    else:
        print("✗ Failed to build molecule")


def main():
    """
    Main demonstration function.
    """
    print("ASE Database and OpenBabel Integration Demo")
    print("="*60)
    
    try:
        # Demonstrate different aspects
        demonstrate_ase_dataset_loading()
        demonstrate_openbabel_analysis()
        demonstrate_molecule_building()
        
        print("\n" + "="*60)
        print("✓ All demonstrations completed successfully!")
        print("\nKey features demonstrated:")
        print("- ASE database loading with automatic dataset info generation")
        print("- OpenBabel-based molecular analysis without SMILES dependency")
        print("- Integration with existing E3 diffusion framework")
        print("- Support for general molecules beyond QM9/GEOM")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)