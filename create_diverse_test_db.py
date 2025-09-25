#!/usr/bin/env python3

import sys
import os
import numpy as np

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a more diverse test database to better test the duplicate detection
from ase import Atoms
from ase.db import connect

def create_diverse_test_database(db_path='diverse_test.db', n_molecules=1000):
    """Create a diverse test database with various molecular structures"""
    print(f"Creating diverse test database with {n_molecules} molecules...")
    
    # Create or connect to database
    db = connect(db_path)
    
    # Clear existing entries
    try:
        db.delete([row.id for row in db.select()])
    except:
        pass
    
    molecules = []
    
    # Create a variety of different molecules
    molecule_templates = [
        # Methane
        {
            'symbols': ['C', 'H', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
        },
        # Water
        {
            'symbols': ['O', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [0.757, 0.587, 0.0], [-0.757, 0.587, 0.0]]
        },
        # Ammonia
        {
            'symbols': ['N', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.866, -0.5, 0.0], [-0.866, -0.5, 0.0]]
        },
        # Ethane
        {
            'symbols': ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            'positions': [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-0.5, 1.0, 0.0], [-0.5, -0.5, 0.9], [-0.5, -0.5, -0.9], [2.0, 1.0, 0.0], [2.0, -0.5, 0.9], [2.0, -0.5, -0.9]]
        },
        # Carbon dioxide
        {
            'symbols': ['C', 'O', 'O'],
            'positions': [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]
        },
        # Hydrogen fluoride
        {
            'symbols': ['H', 'F'],
            'positions': [[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]
        },
        # Benzene (simplified)
        {
            'symbols': ['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            'positions': [[1.4, 0.0, 0.0], [0.7, 1.2, 0.0], [-0.7, 1.2, 0.0], [-1.4, 0.0, 0.0], [-0.7, -1.2, 0.0], [0.7, -1.2, 0.0], [2.5, 0.0, 0.0], [1.2, 2.2, 0.0], [-1.2, 2.2, 0.0], [-2.5, 0.0, 0.0], [-1.2, -2.2, 0.0], [1.2, -2.2, 0.0]]
        }
    ]
    
    # Add molecules with some variations and duplicates
    for i in range(n_molecules):
        template_idx = i % len(molecule_templates)
        template = molecule_templates[template_idx]
        
        # Add some random variation to positions (small perturbations)
        if i % 10 == 0:  # Every 10th molecule is exact duplicate
            positions = template['positions']
        else:
            # Add small random perturbations
            noise_scale = 0.1 * (i % 5) / 5.0  # Variable noise
            positions = []
            for pos in template['positions']:
                new_pos = [
                    pos[0] + np.random.normal(0, noise_scale),
                    pos[1] + np.random.normal(0, noise_scale),
                    pos[2] + np.random.normal(0, noise_scale)
                ]
                positions.append(new_pos)
        
        atoms = Atoms(
            symbols=template['symbols'],
            positions=positions
        )
        molecules.append(atoms)
    
    # Add molecules to database with some properties
    for i, atoms in enumerate(molecules):
        db.write(
            atoms, 
            data={
                'energy': np.random.uniform(-100, -50),  # Random energy values
                'dipole_moment': np.random.uniform(0, 5)  # Random dipole moments
            },
            cid=i+1000,  # Some arbitrary compound ID
            conformer_index=0,
            record_type='molecule',
            source='diverse_test_generation'
        )
    
    print(f"Created diverse database with {len(molecules)} molecules")
    print(f"Database saved to: {db_path}")
    return db_path

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    create_diverse_test_database()