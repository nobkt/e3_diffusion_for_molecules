"""
Create a simple test dataset for molecule-conditioned crystal generation.

This script creates paired molecule-crystal databases with simple examples
for testing the conditional generation system.
"""

import numpy as np
from ase import Atoms
from ase.db import connect
from pathlib import Path
import argparse


def create_simple_molecule(name='benzene'):
    """Create a simple molecular structure."""
    if name == 'benzene':
        # Benzene ring
        symbols = ['C'] * 6 + ['H'] * 6
        positions = [
            [0.000, 1.400, 0.000],   # C1
            [1.212, 0.700, 0.000],   # C2
            [1.212, -0.700, 0.000],  # C3
            [0.000, -1.400, 0.000],  # C4
            [-1.212, -0.700, 0.000], # C5
            [-1.212, 0.700, 0.000],  # C6
            [0.000, 2.480, 0.000],   # H1
            [2.148, 1.240, 0.000],   # H2
            [2.148, -1.240, 0.000],  # H3
            [0.000, -2.480, 0.000],  # H4
            [-2.148, -1.240, 0.000], # H5
            [-2.148, 1.240, 0.000],  # H6
        ]
    elif name == 'ethane':
        # Ethane
        symbols = ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
        positions = [
            [0.000, 0.000, 0.765],   # C1
            [0.000, 0.000, -0.765],  # C2
            [1.019, 0.000, 1.160],   # H1
            [-0.510, 0.882, 1.160],  # H2
            [-0.510, -0.882, 1.160], # H3
            [1.019, 0.000, -1.160],  # H4
            [-0.510, 0.882, -1.160], # H5
            [-0.510, -0.882, -1.160],# H6
        ]
    elif name == 'methane':
        # Methane
        symbols = ['C', 'H', 'H', 'H', 'H']
        positions = [
            [0.000, 0.000, 0.000],   # C
            [0.631, 0.631, 0.631],   # H1
            [-0.631, -0.631, 0.631], # H2
            [-0.631, 0.631, -0.631], # H3
            [0.631, -0.631, -0.631], # H4
        ]
    else:
        raise ValueError(f"Unknown molecule: {name}")
    
    mol = Atoms(symbols=symbols, positions=positions)
    return mol


def create_simple_crystal(molecule, Z=4, cell_size=10.0, random_rotation=True):
    """
    Create a simple crystal by replicating a molecule.
    
    Args:
        molecule: ASE Atoms object
        Z: Number of molecules per unit cell
        cell_size: Cubic cell size in Angstroms
        random_rotation: Whether to apply random rotations
    """
    # Get molecule center
    mol_positions = molecule.positions
    mol_center = mol_positions.mean(axis=0)
    mol_positions_centered = mol_positions - mol_center
    
    # Create crystal
    crystal_symbols = []
    crystal_positions = []
    
    for i in range(Z):
        # Random rotation matrix
        if random_rotation:
            # Random rotation using Euler angles
            alpha = np.random.uniform(0, 2*np.pi)
            beta = np.random.uniform(0, 2*np.pi)
            gamma = np.random.uniform(0, 2*np.pi)
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]
            ])
            Ry = np.array([
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ])
            Rz = np.array([
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]
            ])
            R = Rz @ Ry @ Rx
            
            rotated_positions = (R @ mol_positions_centered.T).T
        else:
            rotated_positions = mol_positions_centered
        
        # Random translation within unit cell
        translation = np.random.uniform(0, cell_size, 3)
        
        translated_positions = rotated_positions + translation
        
        crystal_symbols.extend(molecule.symbols)
        crystal_positions.extend(translated_positions)
    
    # Create crystal Atoms object
    crystal = Atoms(
        symbols=crystal_symbols,
        positions=crystal_positions,
        cell=[cell_size, cell_size, cell_size],
        pbc=[True, True, True]
    )
    
    return crystal


def main():
    parser = argparse.ArgumentParser(
        description='Create test dataset for molecule-conditioned crystal generation'
    )
    parser.add_argument('--output_dir', type=str, default='data/test_paired',
                        help='Output directory for databases')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples per molecule type')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create databases
    mol_db_path = output_dir / 'molecules.db'
    crystal_db_path = output_dir / 'crystals.db'
    
    # Remove existing databases
    if mol_db_path.exists():
        mol_db_path.unlink()
    if crystal_db_path.exists():
        crystal_db_path.unlink()
    
    mol_db = connect(str(mol_db_path))
    crystal_db = connect(str(crystal_db_path))
    
    print(f"Creating test dataset in {output_dir}")
    print(f"Samples per molecule: {args.n_samples}")
    
    # Molecule types to generate
    molecule_types = ['methane', 'ethane', 'benzene']
    
    for mol_type in molecule_types:
        print(f"\nGenerating {mol_type} samples...")
        
        # Create molecule
        molecule = create_simple_molecule(mol_type)
        
        # Add molecule to database
        mol_id = mol_db.write(
            molecule,
            name=mol_type,
        )
        
        print(f"  Molecule ID: {mol_id}")
        print(f"  Formula: {molecule.get_chemical_formula()}")
        print(f"  Atoms: {len(molecule)}")
        
        # Generate multiple crystal samples
        for sample_idx in range(args.n_samples):
            # Vary Z and cell size
            Z = np.random.choice([2, 4, 6, 8])
            cell_size = np.random.uniform(8.0, 15.0)
            
            # Create crystal
            crystal = create_simple_crystal(
                molecule,
                Z=Z,
                cell_size=cell_size,
                random_rotation=True,
            )
            
            # Calculate density
            mass = crystal.get_masses().sum()  # amu
            volume = crystal.get_volume()  # Angstrom^3
            density = mass / volume * 1.660539  # g/cm^3
            
            # Add crystal to database
            crystal_db.write(
                crystal,
                molecule_id=mol_id,
                Z=Z,
                density=density,
                cell_size=cell_size,
                sample_idx=sample_idx,
            )
        
        print(f"  Generated {args.n_samples} crystal samples")
    
    # Summary
    print(f"\nDataset created successfully!")
    print(f"Molecules: {len(mol_db)} ({mol_db_path})")
    print(f"Crystals: {len(crystal_db)} ({crystal_db_path})")
    print(f"\nTo train a model:")
    print(f"  python main_crystal_conditional.py \\")
    print(f"    --molecule_db_path {mol_db_path} \\")
    print(f"    --crystal_db_path {crystal_db_path} \\")
    print(f"    --epochs 10 --batch_size 2")


if __name__ == '__main__':
    main()
