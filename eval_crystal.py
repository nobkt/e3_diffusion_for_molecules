"""
Evaluation script for crystal generation model.

This script evaluates generated crystal structures against reference data.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from ase.io import read
from ase.db import connect

from crystal.data.periodic_utils import cell_vectors_to_params


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate generated crystals')
    
    parser.add_argument('--generated_dir', type=str, required=True,
                        help='Directory containing generated crystal structures')
    parser.add_argument('--reference_db', type=str, default=None,
                        help='ASE database with reference structures')
    parser.add_argument('--format', type=str, default='cif',
                        choices=['cif', 'xyz'],
                        help='Format of generated structures')
    
    return parser.parse_args()


def compute_lattice_statistics(structures):
    """Compute statistics of lattice parameters."""
    lengths_list = []
    angles_list = []
    volumes = []
    
    for atoms in structures:
        cell = atoms.cell
        lengths = cell.lengths()
        angles = cell.angles()
        volume = cell.volume
        
        lengths_list.append(lengths)
        angles_list.append(angles)
        volumes.append(volume)
    
    lengths_array = np.array(lengths_list)
    angles_array = np.array(angles_list)
    volumes_array = np.array(volumes)
    
    stats = {
        'lengths_mean': lengths_array.mean(axis=0),
        'lengths_std': lengths_array.std(axis=0),
        'angles_mean': angles_array.mean(axis=0),
        'angles_std': angles_array.std(axis=0),
        'volume_mean': volumes_array.mean(),
        'volume_std': volumes_array.std(),
        'n_structures': len(structures)
    }
    
    return stats


def compute_coordination_statistics(structures):
    """Compute coordination number statistics."""
    coord_numbers = []
    
    for atoms in structures:
        # Get distances between all pairs
        positions = atoms.positions
        cell = atoms.cell.array
        
        n_atoms = len(atoms)
        coords = np.zeros(n_atoms)
        
        # Simple distance-based coordination (cutoff 3 Angstrom)
        cutoff = 3.0
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    # Compute distance with minimum image convention
                    diff = positions[i] - positions[j]
                    # Simple periodic boundary (not exact)
                    diff = diff - np.round(diff / cell.diagonal()) * cell.diagonal()
                    dist = np.linalg.norm(diff)
                    if dist < cutoff:
                        coords[i] += 1
        
        coord_numbers.extend(coords)
    
    coord_array = np.array(coord_numbers)
    
    stats = {
        'coord_mean': coord_array.mean(),
        'coord_std': coord_array.std(),
        'coord_min': coord_array.min(),
        'coord_max': coord_array.max()
    }
    
    return stats


def evaluate_crystals(generated_dir, reference_db=None, file_format='cif'):
    """Evaluate generated crystal structures."""
    generated_dir = Path(generated_dir)
    
    # Load generated structures
    print(f"Loading generated structures from {generated_dir}")
    pattern = f'*.{file_format}'
    generated_files = list(generated_dir.glob(pattern))
    
    if len(generated_files) == 0:
        print(f"No {file_format} files found in {generated_dir}")
        return
    
    print(f"Found {len(generated_files)} generated structures")
    
    generated_structures = []
    for file_path in generated_files:
        try:
            atoms = read(file_path)
            generated_structures.append(atoms)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Successfully loaded {len(generated_structures)} structures")
    
    # Compute statistics for generated structures
    print("\n=== Generated Structures Statistics ===")
    lattice_stats = compute_lattice_statistics(generated_structures)
    
    print(f"Number of structures: {lattice_stats['n_structures']}")
    print(f"\nLattice lengths (a, b, c):")
    print(f"  Mean: {lattice_stats['lengths_mean']}")
    print(f"  Std:  {lattice_stats['lengths_std']}")
    print(f"\nLattice angles (α, β, γ):")
    print(f"  Mean: {lattice_stats['angles_mean']}")
    print(f"  Std:  {lattice_stats['angles_std']}")
    print(f"\nCell volume:")
    print(f"  Mean: {lattice_stats['volume_mean']:.2f} Ų")
    print(f"  Std:  {lattice_stats['volume_std']:.2f} Ų")
    
    # Compute coordination statistics
    print("\n=== Coordination Statistics ===")
    coord_stats = compute_coordination_statistics(generated_structures)
    print(f"Coordination number:")
    print(f"  Mean: {coord_stats['coord_mean']:.2f}")
    print(f"  Std:  {coord_stats['coord_std']:.2f}")
    print(f"  Range: [{coord_stats['coord_min']:.0f}, {coord_stats['coord_max']:.0f}]")
    
    # If reference database provided, compare
    if reference_db:
        print("\n=== Reference Structures Statistics ===")
        try:
            db = connect(reference_db)
            reference_structures = []
            for row in db.select():
                reference_structures.append(row.toatoms())
            
            ref_lattice_stats = compute_lattice_statistics(reference_structures)
            
            print(f"Number of structures: {ref_lattice_stats['n_structures']}")
            print(f"\nLattice lengths (a, b, c):")
            print(f"  Mean: {ref_lattice_stats['lengths_mean']}")
            print(f"  Std:  {ref_lattice_stats['lengths_std']}")
            print(f"\nLattice angles (α, β, γ):")
            print(f"  Mean: {ref_lattice_stats['angles_mean']}")
            print(f"  Std:  {ref_lattice_stats['angles_std']}")
            print(f"\nCell volume:")
            print(f"  Mean: {ref_lattice_stats['volume_mean']:.2f} Ų")
            print(f"  Std:  {ref_lattice_stats['volume_std']:.2f} Ų")
            
        except Exception as e:
            print(f"Error loading reference database: {e}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    evaluate_crystals(
        generated_dir=args.generated_dir,
        reference_db=args.reference_db,
        file_format=args.format
    )


if __name__ == '__main__':
    main()
