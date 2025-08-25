"""
ASE Database Reader for molecular datasets.
Provides functionality to read molecular data from ASE databases and convert to 
the format expected by the diffusion model.
"""

import numpy as np
import torch
from ase.db import connect
from ase import Atoms
from collections import defaultdict
import logging


class ASEDatabaseReader:
    """Reader for ASE database files."""
    
    def __init__(self, db_path, atom_types=None):
        """
        Initialize ASE database reader.
        
        Args:
            db_path: Path to ASE database file
            atom_types: List of allowed atom types (e.g., ['H', 'C', 'N', 'O', 'F'])
                       If None, will be determined from dataset
        """
        self.db_path = db_path
        self.db = connect(db_path)
        self.atom_types = atom_types
        self._atom_encoder = None
        self._atom_decoder = None
        self._setup_atom_encoding()
        
    def _setup_atom_encoding(self):
        """Setup atom encoding/decoding based on database content or provided atom_types."""
        if self.atom_types is None:
            # Discover atom types from database
            all_symbols = set()
            for row in self.db.select():
                atoms = row.toatoms()
                all_symbols.update(atoms.get_chemical_symbols())
            self.atom_types = sorted(list(all_symbols))
            
        self._atom_encoder = {symbol: i for i, symbol in enumerate(self.atom_types)}
        self._atom_decoder = self.atom_types.copy()
        
    @property
    def atom_encoder(self):
        return self._atom_encoder
        
    @property 
    def atom_decoder(self):
        return self._atom_decoder
        
    def get_dataset_info(self, dataset_name="ase_molecules", with_h=True):
        """
        Generate dataset_info dictionary compatible with existing codebase.
        
        Args:
            dataset_name: Name for the dataset
            with_h: Whether hydrogen atoms are included
            
        Returns:
            Dictionary with dataset configuration
        """
        # Count atoms and analyze structure
        atom_counts = defaultdict(int)
        node_counts = defaultdict(int)
        max_nodes = 0
        
        for row in self.db.select():
            atoms = row.toatoms()
            n_atoms = len(atoms)
            max_nodes = max(max_nodes, n_atoms)
            node_counts[n_atoms] += 1
            
            for symbol in atoms.get_chemical_symbols():
                if symbol in self.atom_encoder:
                    atom_counts[self.atom_encoder[symbol]] += 1
                    
        # Generate colors and radius for visualization (using default scheme)
        colors = ['#FFFFFF99'] + [f'C{i}' for i in range(len(self.atom_types)-1)]
        radius = [0.46 if self.atom_types[0] == 'H' else 0.77] + [0.77] * (len(self.atom_types)-1)
        
        dataset_info = {
            'name': dataset_name,
            'atom_encoder': self.atom_encoder,
            'atom_decoder': self.atom_decoder,
            'max_n_nodes': max_nodes,
            'n_nodes': dict(node_counts),
            'atom_types': dict(atom_counts),
            'colors_dic': colors[:len(self.atom_types)],
            'radius_dic': radius[:len(self.atom_types)],
            'with_h': with_h
        }
        
        return dataset_info
        
    def load_data_split(self, indices=None, max_molecules=None):
        """
        Load molecular data from ASE database.
        
        Args:
            indices: List of database row indices to load. If None, load all.
            max_molecules: Maximum number of molecules to load
            
        Returns:
            Dictionary with molecular data in format expected by model
        """
        if indices is None:
            if max_molecules is None:
                rows = list(self.db.select())
            else:
                rows = list(self.db.select(limit=max_molecules))
        else:
            # ASE database uses 1-based indexing
            rows = []
            for idx in indices:
                try:
                    row = self.db.get(id=int(idx+1))  # Convert to int and add 1 for 1-based indexing
                    rows.append(row)
                except Exception:
                    continue  # Skip missing entries
            
        if max_molecules is not None:
            rows = rows[:max_molecules]
            
        # Determine maximum number of atoms for padding
        max_atoms = max(len(row.toatoms()) for row in rows) if rows else 0
        
        n_molecules = len(rows)
        
        # Initialize arrays
        positions = np.zeros((n_molecules, max_atoms, 3))
        charges = np.zeros((n_molecules, max_atoms), dtype=np.int64)
        one_hot = np.zeros((n_molecules, max_atoms, len(self.atom_types)))
        num_atoms = np.zeros(n_molecules, dtype=np.int64)
        
        # Additional properties if available
        energies = []
        forces = []
        has_energy = False
        has_forces = False
        
        for i, row in enumerate(rows):
            atoms = row.toatoms()
            n_atoms = len(atoms)
            num_atoms[i] = n_atoms
            
            # Positions (centered)
            pos = atoms.get_positions()
            pos = pos - np.mean(pos, axis=0)  # Center molecule
            positions[i, :n_atoms] = pos
            
            # Atomic numbers and one-hot encoding
            symbols = atoms.get_chemical_symbols()
            for j, symbol in enumerate(symbols):
                if symbol in self.atom_encoder:
                    atom_idx = self.atom_encoder[symbol]
                    charges[i, j] = atoms.get_atomic_numbers()[j]
                    one_hot[i, j, atom_idx] = 1
                    
            # Additional properties
            if hasattr(row, 'energy') and row.energy is not None:
                energies.append(row.energy)
                has_energy = True
            elif 'energy' in row.data:
                energies.append(row.data['energy'])
                has_energy = True
            else:
                energies.append(0.0)
                
            if hasattr(row, 'forces') and row.forces is not None:
                forces.append(row.forces)
                has_forces = True
            elif 'forces' in row.data:
                forces.append(row.data['forces'])
                has_forces = True
                
        # Convert to tensors
        data = {
            'positions': torch.from_numpy(positions).float(),
            'charges': torch.from_numpy(charges),
            'one_hot': torch.from_numpy(one_hot).float(),
            'num_atoms': torch.from_numpy(num_atoms),
        }
        
        if has_energy:
            data['energy'] = torch.tensor(energies).float()
            
        if has_forces:
            # Pad forces to match position dimensions
            forces_padded = np.zeros((n_molecules, max_atoms, 3))
            for i, force in enumerate(forces):
                if force is not None and len(force) > 0:
                    forces_padded[i, :len(force)] = force
            data['forces'] = torch.from_numpy(forces_padded).float()
            
        return data
        
    def create_splits(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=42):
        """
        Create train/validation/test splits.
        
        Args:
            train_ratio: Fraction for training set
            valid_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'valid', 'test' keys containing data splits
        """
        total_molecules = len(self.db)
        np.random.seed(random_seed)
        indices = np.random.permutation(total_molecules)
        
        train_end = int(train_ratio * total_molecules)
        valid_end = train_end + int(valid_ratio * total_molecules)
        
        splits = {
            'train': indices[:train_end],
            'valid': indices[train_end:valid_end], 
            'test': indices[valid_end:]
        }
        
        # Load data for each split
        data_splits = {}
        for split_name, split_indices in splits.items():
            data_splits[split_name] = self.load_data_split(split_indices)
            logging.info(f"Loaded {len(split_indices)} molecules for {split_name} split")
            
        return data_splits


def create_ase_database_from_qm9(qm9_data_path, ase_db_path, split='train'):
    """
    Convert QM9 dataset to ASE database format for testing.
    
    Args:
        qm9_data_path: Path to QM9 .npz file 
        ase_db_path: Output path for ASE database
        split: Which split to convert ('train', 'valid', 'test')
    """
    # Load QM9 data
    with np.load(qm9_data_path) as data:
        positions = data['positions']
        charges = data['charges'] 
        num_atoms = data['num_atoms']
        
    # Create ASE database
    db = connect(ase_db_path)
    
    # QM9 atomic number to symbol mapping
    atomic_num_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    
    for i in range(len(positions)):
        n_atoms = int(num_atoms[i])
        pos = positions[i][:n_atoms]
        atom_charges = charges[i][:n_atoms]
        
        # Create symbols from atomic numbers
        symbols = [atomic_num_to_symbol[int(charge)] for charge in atom_charges if int(charge) > 0]
        valid_pos = pos[:len(symbols)]
        
        # Create ASE Atoms object
        atoms = Atoms(symbols=symbols, positions=valid_pos)
        
        # Add to database
        kvp = {}
        if 'energy' in data:
            kvp['energy'] = float(data['energy'][i])
        
        db.write(atoms, **kvp)
        
    logging.info(f"Created ASE database with {len(db)} molecules at {ase_db_path}")