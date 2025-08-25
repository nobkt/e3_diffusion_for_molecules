#!/usr/bin/env python3
"""Create a test database with molecules of varying sizes to test the fix."""

from ase import Atoms
from ase.db import connect
import numpy as np
import os

def create_test_database(db_path: str):
    """Create test database with molecules of varying sizes."""
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database
    db = connect(db_path)
    
    # Create molecules with different numbers of atoms
    molecules = []
    
    # Small molecules (5-10 atoms)
    for i in range(20):
        # Water-like molecules
        symbols = ['O', 'H', 'H']
        positions = np.random.rand(3, 3) * 2 - 1
        molecules.append((symbols, positions))
    
    # Medium molecules (11-15 atoms) 
    for i in range(30):
        # Methane + extra atoms
        symbols = ['C'] + ['H'] * (10 + i % 5)
        n_atoms = len(symbols)
        positions = np.random.rand(n_atoms, 3) * 3 - 1.5
        molecules.append((symbols, positions))
    
    # Large molecules (16-20 atoms)
    for i in range(25):
        symbols = ['C'] * (6 + i % 4) + ['H'] * (10 + i % 4)
        n_atoms = len(symbols)
        positions = np.random.rand(n_atoms, 3) * 4 - 2
        molecules.append((symbols, positions))
    
    # Extra large molecules (21-25 atoms) - these should trigger the KeyError before fix
    for i in range(15):
        symbols = ['C'] * (8 + i % 3) + ['H'] * (13 + i % 5) + ['O', 'N']
        n_atoms = len(symbols)
        positions = np.random.rand(n_atoms, 3) * 5 - 2.5
        molecules.append((symbols, positions))
    
    # Write to database
    molecules_written = 0
    for symbols, positions in molecules:
        try:
            atoms = Atoms(symbols=symbols, positions=positions)
            db.write(atoms)
            molecules_written += 1
        except Exception as e:
            print(f"Warning: Failed to write molecule: {e}")
    
    print(f"Created test database with {molecules_written} molecules at {db_path}")
    
    # Verify molecule size distribution
    db_verify = connect(db_path)
    sizes = []
    for row in db_verify.select():
        atoms = row.toatoms()
        sizes.append(len(atoms))
    
    from collections import Counter
    size_counts = Counter(sizes)
    print(f"Molecule size distribution: {dict(sorted(size_counts.items()))}")
    print(f"Sizes range from {min(sizes)} to {max(sizes)} atoms")
    
    return molecules_written

if __name__ == "__main__":
    create_test_database("qm9_test.db")