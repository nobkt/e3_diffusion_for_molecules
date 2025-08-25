"""
ASE Database interface for molecular data storage and retrieval.
This module provides functionality to convert molecular data to/from ASE database format.
"""

import os
import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from typing import Dict, List, Optional, Union, Tuple
import logging


class ASEDatasetInterface:
    """Interface for loading/saving molecular datasets using ASE database format."""
    
    def __init__(self, db_path: str):
        """
        Initialize ASE database interface.
        
        Parameters
        ----------
        db_path : str
            Path to ASE database file
        """
        self.db_path = db_path
        self.db = None
        
    def connect(self):
        """Connect to ASE database."""
        if self.db is None:
            self.db = connect(self.db_path)
        return self.db
    
    def close(self):
        """Close database connection."""
        if self.db is not None:
            self.db = None
    
    def save_molecules_to_db(self, molecules_data: Dict[str, Dict], split_info: Optional[Dict] = None):
        """
        Save molecular data to ASE database.
        
        Parameters
        ----------
        molecules_data : dict
            Dictionary with splits as keys and molecular data as values
        split_info : dict, optional
            Information about data splits
        """
        db = self.connect()
        
        for split_name, split_data in molecules_data.items():
            num_molecules = len(split_data['charges'])
            
            for i in range(num_molecules):
                # Extract molecular data
                positions = split_data['positions'][i].numpy() if isinstance(split_data['positions'][i], torch.Tensor) else split_data['positions'][i]
                charges = split_data['charges'][i].numpy() if isinstance(split_data['charges'][i], torch.Tensor) else split_data['charges'][i]
                num_atoms = split_data['num_atoms'][i].item() if isinstance(split_data['num_atoms'][i], torch.Tensor) else split_data['num_atoms'][i]
                
                # Handle padding - only take actual atoms
                actual_positions = positions[:num_atoms]
                actual_charges = charges[:num_atoms]
                
                # Convert atomic numbers to symbols
                symbols = [self._atomic_number_to_symbol(int(charge)) for charge in actual_charges]
                
                # Create ASE Atoms object
                atoms = Atoms(symbols=symbols, positions=actual_positions)
                
                # Prepare additional properties
                properties = {'split': split_name, 'molecule_idx': i}
                
                # Add QM9 specific properties if available
                qm9_properties = ['index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 
                                'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'omega1']
                
                for prop in qm9_properties:
                    if prop in split_data:
                        value = split_data[prop][i]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        properties[prop] = float(value)
                
                # Add thermochemical properties if available
                for key in split_data.keys():
                    if key.endswith('_thermo'):
                        value = split_data[key][i]
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        properties[key] = float(value)
                
                # Write to database
                db.write(atoms, **properties)
                
                if (i + 1) % 1000 == 0:
                    logging.info(f"Saved {i + 1}/{num_molecules} molecules from {split_name} split")
        
        logging.info(f"Successfully saved molecular data to {self.db_path}")
    
    def load_molecules_from_db(self, splits: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Load molecular data from ASE database.
        
        Parameters
        ----------
        splits : list of str, optional
            Which data splits to load. If None, loads all available splits.
            
        Returns
        -------
        molecules_data : dict
            Dictionary with splits as keys and molecular data as values
        """
        db = self.connect()
        
        # Get all available splits if not specified
        if splits is None:
            all_rows = list(db.select())
            splits = list(set(row.get('split', 'unknown') for row in all_rows))
        
        molecules_data = {}
        
        for split in splits:
            split_rows = list(db.select(split=split))
            if not split_rows:
                logging.warning(f"No data found for split: {split}")
                continue
                
            # Initialize data containers
            split_data = {
                'positions': [],
                'charges': [],
                'num_atoms': [],
            }
            
            # Get property names from first row
            first_row = split_rows[0]
            qm9_properties = ['index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 
                            'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'omega1']
            
            for prop in qm9_properties:
                if prop in first_row.data:
                    split_data[prop] = []
            
            # Check for thermochemical properties
            for key in first_row.data.keys():
                if key.endswith('_thermo'):
                    split_data[key] = []
            
            # Process each molecule
            max_atoms = 0
            for row in split_rows:
                atoms = row.toatoms()
                num_atoms = len(atoms)
                max_atoms = max(max_atoms, num_atoms)
                
                # Get positions and atomic numbers
                positions = atoms.positions
                atomic_numbers = atoms.numbers
                
                split_data['positions'].append(positions)
                split_data['charges'].append(atomic_numbers)
                split_data['num_atoms'].append(num_atoms)
                
                # Add properties
                for prop in qm9_properties:
                    if prop in split_data and prop in row.data:
                        split_data[prop].append(row.data[prop])
                
                for key in row.data.keys():
                    if key.endswith('_thermo') and key in split_data:
                        split_data[key].append(row.data[key])
            
            # Pad and convert to tensors
            split_data = self._pad_and_tensorize(split_data, max_atoms)
            molecules_data[split] = split_data
            
            logging.info(f"Loaded {len(split_rows)} molecules from {split} split")
        
        return molecules_data
    
    def _pad_and_tensorize(self, data: Dict, max_atoms: int) -> Dict:
        """Pad molecular data to consistent size and convert to tensors."""
        tensorized_data = {}
        
        for key, values in data.items():
            if key in ['positions', 'charges']:
                # Pad positions and charges
                if key == 'positions':
                    padded_values = np.zeros((len(values), max_atoms, 3))
                    for i, pos in enumerate(values):
                        padded_values[i, :len(pos)] = pos
                else:  # charges
                    padded_values = np.zeros((len(values), max_atoms))
                    for i, charges in enumerate(values):
                        padded_values[i, :len(charges)] = charges
                
                tensorized_data[key] = torch.from_numpy(padded_values).float()
            else:
                # Convert scalar properties to tensors
                tensorized_data[key] = torch.tensor(values)
        
        return tensorized_data
    
    def _atomic_number_to_symbol(self, atomic_number: int) -> str:
        """Convert atomic number to element symbol."""
        element_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
        return element_map.get(atomic_number, f'X{atomic_number}')


def convert_npz_to_ase_db(npz_files: Dict[str, str], output_db_path: str):
    """
    Convert QM9 NPZ files to ASE database format.
    
    Parameters
    ----------
    npz_files : dict
        Dictionary mapping split names to NPZ file paths
    output_db_path : str
        Path for output ASE database
    """
    # Load data from NPZ files
    molecules_data = {}
    
    for split_name, npz_path in npz_files.items():
        if os.path.exists(npz_path):
            with np.load(npz_path) as f:
                split_data = {key: torch.from_numpy(val) for key, val in f.items()}
                molecules_data[split_name] = split_data
                logging.info(f"Loaded {split_name} data from {npz_path}")
        else:
            logging.warning(f"NPZ file not found: {npz_path}")
    
    # Save to ASE database
    if molecules_data:
        # Remove existing database if it exists
        if os.path.exists(output_db_path):
            os.remove(output_db_path)
        
        ase_interface = ASEDatasetInterface(output_db_path)
        ase_interface.save_molecules_to_db(molecules_data)
        ase_interface.close()
        
        logging.info(f"Successfully converted NPZ files to ASE database: {output_db_path}")
    else:
        logging.error("No valid NPZ files found to convert")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test conversion from NPZ to ASE DB (when NPZ files are available)
    npz_files = {
        'train': 'qm9/temp/qm9/train.npz',
        'valid': 'qm9/temp/qm9/valid.npz', 
        'test': 'qm9/temp/qm9/test.npz'
    }
    
    output_db = 'qm9/temp/qm9_database.db'
    convert_npz_to_ase_db(npz_files, output_db)