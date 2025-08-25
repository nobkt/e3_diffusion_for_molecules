"""
ASE database dataset loader for general molecules.
This module provides dataset loading functionality for ASE database files.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ase.db import connect
from ase import Atoms
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
from configs.datasets_config import get_dataset_info


def get_ase_dataset_info(db_path: str, max_atoms: int = 100, name: str = "ase_general") -> Dict[str, Any]:
    """
    Create dataset info for ASE database by analyzing the database contents.
    
    Args:
        db_path: Path to ASE database file
        max_atoms: Maximum number of atoms to consider
        name: Name for the dataset
        
    Returns:
        Dataset info dictionary compatible with existing codebase
    """
    # Connect to database
    db = connect(db_path)
    
    # Analyze database to get atom types and statistics
    atom_types = set()
    num_atoms_list = []
    
    print(f"Analyzing ASE database: {db_path}")
    
    for i, row in enumerate(db.select()):
        atoms = row.toatoms()
        symbols = atoms.get_chemical_symbols()
        
        # Skip molecules that are too large
        if len(symbols) > max_atoms:
            continue
            
        atom_types.update(symbols)
        num_atoms_list.append(len(symbols))
        
        # Limit analysis to first 1000 entries for speed
        if i >= 1000:
            break
    
    # Sort atom types for consistent encoding
    atom_types = sorted(list(atom_types))
    
    # Create atom encoder/decoder
    atom_encoder = {symbol: i for i, symbol in enumerate(atom_types)}
    atom_decoder = atom_types
    
    # Create node distribution
    from collections import Counter
    node_counts = Counter(num_atoms_list)
    max_n_nodes = max(num_atoms_list) if num_atoms_list else max_atoms
    
    # Create colors and radii for visualization
    # Use a set of default colors and radii
    default_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    default_radius = 0.77
    
    colors_dic = []
    radius_dic = []
    for i in range(len(atom_types)):
        colors_dic.append(default_colors[i % len(default_colors)])
        radius_dic.append(default_radius)
    
    dataset_info = {
        'name': name,
        'atom_encoder': atom_encoder,
        'atom_decoder': atom_decoder,
        'max_n_nodes': max_n_nodes,
        'n_nodes': dict(node_counts),
        'atom_types': {i: 0 for i in range(len(atom_types))},  # Will be computed properly if needed
        'colors_dic': colors_dic,
        'radius_dic': radius_dic,
        'with_h': 'H' in atom_types,
        'db_path': db_path
    }
    
    print(f"Found {len(atom_types)} atom types: {atom_types}")
    print(f"Max atoms: {max_n_nodes}")
    print(f"Node distribution: {dict(list(node_counts.most_common(10)))}")
    
    return dataset_info


class ASEDataset(Dataset):
    """
    Dataset class for ASE database files.
    """
    
    def __init__(self, db_path: str, dataset_info: Dict[str, Any], 
                 max_atoms: int = 100, include_charges: bool = True,
                 filter_func: Optional[callable] = None):
        """
        Initialize ASE dataset.
        
        Args:
            db_path: Path to ASE database file
            dataset_info: Dataset configuration dictionary
            max_atoms: Maximum number of atoms per molecule
            include_charges: Whether to include atomic charges
            filter_func: Optional function to filter molecules
        """
        self.db_path = db_path
        self.dataset_info = dataset_info
        self.max_atoms = max_atoms
        self.include_charges = include_charges
        self.filter_func = filter_func
        
        # Load data from database
        self.data = []
        self._load_data()
        
    def _load_data(self):
        """Load data from ASE database."""
        db = connect(self.db_path)
        
        atom_encoder = self.dataset_info['atom_encoder']
        
        for row in db.select():
            atoms = row.toatoms()
            
            # Skip if too many atoms
            if len(atoms) > self.max_atoms:
                continue
                
            # Apply filter if provided
            if self.filter_func and not self.filter_func(atoms):
                continue
            
            # Extract positions and atom types
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            
            # Skip if any atom type is not in encoder
            if any(symbol not in atom_encoder for symbol in symbols):
                continue
            
            # Convert to indices
            atom_types = [atom_encoder[symbol] for symbol in symbols]
            
            # Get charges if available
            if self.include_charges:
                try:
                    charges = atoms.get_initial_charges()
                    if charges is None or len(charges) == 0:
                        charges = np.zeros(len(atoms))
                except:
                    charges = np.zeros(len(atoms))
            else:
                charges = np.zeros(len(atoms))
            
            # Store data
            self.data.append({
                'positions': positions,
                'atom_types': atom_types,
                'charges': charges,
                'num_atoms': len(atoms),
                'symbols': symbols
            })
        
        print(f"Loaded {len(self.data)} molecules from {self.db_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        item = self.data[idx]
        
        positions = torch.tensor(item['positions'], dtype=torch.float32)
        atom_types = torch.tensor(item['atom_types'], dtype=torch.long)
        charges = torch.tensor(item['charges'], dtype=torch.float32)
        num_atoms = item['num_atoms']
        
        # Create one-hot encoding
        n_atom_types = len(self.dataset_info['atom_decoder'])
        one_hot = torch.zeros(len(atom_types), n_atom_types)
        one_hot[torch.arange(len(atom_types)), atom_types] = 1
        
        # Create masks
        atom_mask = torch.ones(len(atom_types), dtype=torch.bool)
        
        # Pad to max_atoms if needed
        if len(atom_types) < self.max_atoms:
            pad_size = self.max_atoms - len(atom_types)
            
            # Pad positions
            positions = torch.cat([positions, torch.zeros(pad_size, 3)], dim=0)
            
            # Pad one_hot
            one_hot = torch.cat([one_hot, torch.zeros(pad_size, n_atom_types)], dim=0)
            
            # Pad charges
            charges = torch.cat([charges, torch.zeros(pad_size)], dim=0)
            
            # Extend mask
            atom_mask = torch.cat([atom_mask, torch.zeros(pad_size, dtype=torch.bool)], dim=0)
        
        # Create edge mask (all pairs of real atoms)
        edge_mask = atom_mask.unsqueeze(0) * atom_mask.unsqueeze(1)
        
        return {
            'positions': positions,
            'one_hot': one_hot,
            'atom_mask': atom_mask,
            'edge_mask': edge_mask,
            'charges': charges,
            'num_atoms': torch.tensor(num_atoms, dtype=torch.long)
        }


class ASEDataLoader:
    """
    DataLoader wrapper for ASE datasets.
    """
    
    def __init__(self, dataset: ASEDataset, batch_size: int = 32, 
                 shuffle: bool = True, num_workers: int = 0):
        """
        Initialize ASE DataLoader.
        
        Args:
            dataset: ASE dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
        """
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Stack all tensors
        positions = torch.stack([item['positions'] for item in batch])
        one_hot = torch.stack([item['one_hot'] for item in batch])
        atom_mask = torch.stack([item['atom_mask'] for item in batch])
        edge_mask = torch.stack([item['edge_mask'] for item in batch])
        charges = torch.stack([item['charges'] for item in batch])
        num_atoms = torch.stack([item['num_atoms'] for item in batch])
        
        # Add extra dimension to charges to match the 3D node_mask format expected by training code
        # This is consistent with QM9 dataset collate function
        charges = charges.unsqueeze(2)
        
        return {
            'positions': positions,
            'one_hot': one_hot,
            'atom_mask': atom_mask,
            'edge_mask': edge_mask,
            'charges': charges,
            'num_atoms': num_atoms
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def load_ase_dataset(db_path: str, config: Any) -> Tuple[Dict[str, ASEDataLoader], None]:
    """
    Load ASE dataset and create data loaders.
    
    Args:
        db_path: Path to ASE database file
        config: Configuration object with dataset parameters
        
    Returns:
        Tuple of (dataloaders_dict, charge_scale)
    """
    # Create dataset info
    dataset_info = get_ase_dataset_info(
        db_path, 
        max_atoms=getattr(config, 'max_atoms', 100),
        name=getattr(config, 'dataset', 'ase_general')
    )
    
    # Load full dataset
    full_dataset = ASEDataset(
        db_path=db_path,
        dataset_info=dataset_info,
        max_atoms=getattr(config, 'max_atoms', 100),
        include_charges=getattr(config, 'include_charges', True)
    )
    
    # Split dataset into train/val/test
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = getattr(config, 'batch_size', 32)
    num_workers = getattr(config, 'num_workers', 0)
    
    # Create wrapper datasets to maintain ASEDataset interface
    class SubsetDataset(Dataset):
        def __init__(self, subset, parent_dataset):
            self.subset = subset
            self.parent_dataset = parent_dataset
            self.dataset_info = parent_dataset.dataset_info
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            return self.subset[idx]
    
    train_wrapper = SubsetDataset(train_dataset, full_dataset)
    val_wrapper = SubsetDataset(val_dataset, full_dataset)
    test_wrapper = SubsetDataset(test_dataset, full_dataset)
    
    dataloaders = {
        'train': ASEDataLoader(train_wrapper, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'valid': ASEDataLoader(val_wrapper, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': ASEDataLoader(test_wrapper, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    # Store dataset info for later use
    for loader in dataloaders.values():
        loader.dataset_info = dataset_info
    
    return dataloaders, None  # No charge scaling for general molecules


def create_sample_ase_db(db_path: str, num_molecules: int = 100):
    """
    Create a sample ASE database for testing.
    
    Args:
        db_path: Path where to create the database
        num_molecules: Number of sample molecules to create
    """
    from ase import Atoms
    from ase.db import connect
    import random
    import numpy as np
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database
    db = connect(db_path)
    
    # Sample molecule templates
    templates = [
        # Water
        {'symbols': ['O', 'H', 'H'], 
         'positions': [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]},
        # Methane
        {'symbols': ['C', 'H', 'H', 'H', 'H'],
         'positions': [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-0.37, 1.04, 0.0], 
                      [-0.37, -0.52, 0.9], [-0.37, -0.52, -0.9]]},
        # Ammonia
        {'symbols': ['N', 'H', 'H', 'H'],
         'positions': [[0.0, 0.0, 0.0], [1.01, 0.0, 0.0], [-0.33, 0.94, 0.0], [-0.33, -0.47, 0.82]]},
        # CO2
        {'symbols': ['C', 'O', 'O'],
         'positions': [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]},
    ]
    
    molecules_written = 0
    for i in range(num_molecules):
        try:
            # Select random template
            template = random.choice(templates)
            
            # Add some random noise to positions
            positions = np.array(template['positions'])
            noise = np.random.normal(0, 0.05, positions.shape)  # Smaller noise
            positions += noise
            
            # Create atoms object
            atoms = Atoms(symbols=template['symbols'], positions=positions)
            
            # Add to database
            row_id = db.write(atoms)
            molecules_written += 1
            
        except Exception as e:
            print(f"Warning: Failed to write molecule {i}: {e}")
    
    print(f"Created sample ASE database with {molecules_written} molecules at {db_path}")
    
    # Verify the database
    db_verify = connect(db_path)
    actual_count = len(db_verify)
    if actual_count != molecules_written:
        print(f"Warning: Expected {molecules_written} molecules but database has {actual_count}")
    
    return actual_count


if __name__ == "__main__":
    # Test the ASE dataset loader
    test_db_path = "/tmp/test_molecules.db"
    
    # Create sample database
    create_sample_ase_db(test_db_path, 10)
    
    # Test loading
    class TestConfig:
        batch_size = 2
        num_workers = 0
        max_atoms = 20
        include_charges = True
        dataset = 'ase_test'
    
    config = TestConfig()
    dataloaders, _ = load_ase_dataset(test_db_path, config)
    
    # Test iteration
    for split, loader in dataloaders.items():
        print(f"\n{split} set:")
        for i, batch in enumerate(loader):
            print(f"  Batch {i}: positions {batch['positions'].shape}, one_hot {batch['one_hot'].shape}")
            if i >= 1:  # Only show first 2 batches
                break
    
    print("\nASE dataset loader test completed successfully!")