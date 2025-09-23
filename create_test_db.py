#!/usr/bin/env python3

import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a simple test database with a few molecules to test ASE loading
from ase import Atoms
from ase.db import connect
import numpy as np

def create_test_database(db_path='test_select.db', n_molecules=10):
    """Create a small test database similar to the one mentioned in the problem"""
    print(f"Creating test database with {n_molecules} molecules...")
    
    # Create or connect to database
    db = connect(db_path)
    
    # Clear existing entries
    try:
        db.delete([row.id for row in db.select()])
    except:
        pass
    
    # Add some simple molecules
    molecules = []
    
    # Add some methane molecules (C + 4H)
    for i in range(n_molecules // 3):
        atoms = Atoms(
            symbols=['C', 'H', 'H', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0]
            ]
        )
        molecules.append(atoms)
    
    # Add some water molecules (O + 2H)
    for i in range(n_molecules // 3):
        atoms = Atoms(
            symbols=['O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],
                [0.757, 0.587, 0.0],
                [-0.757, 0.587, 0.0]
            ]
        )
        molecules.append(atoms)
    
    # Add some more complex molecules
    for i in range(n_molecules - len(molecules)):
        # Simple ethane molecule (2C + 6H)
        atoms = Atoms(
            symbols=['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [-0.5, 1.0, 0.0],
                [-0.5, -0.5, 0.9],
                [-0.5, -0.5, -0.9],
                [2.0, 1.0, 0.0],
                [2.0, -0.5, 0.9],
                [2.0, -0.5, -0.9]
            ]
        )
        molecules.append(atoms)
    
    # Add molecules to database with some properties
    for i, atoms in enumerate(molecules):
        db.write(
            atoms, 
            cid=i+1000,  # Some arbitrary compound ID
            conformer_index=0,
            record_type='molecule',
            source='test_generation'
        )
    
    print(f"Created database with {len(molecules)} molecules")
    print(f"Database saved to: {db_path}")
    return db_path

if __name__ == "__main__":
    create_test_database()