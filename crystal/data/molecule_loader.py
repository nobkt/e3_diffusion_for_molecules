"""
Single molecule dataset loader

Loads single molecule structures from ASE database for EGNN feature extraction.
This is used to extract molecular features that are then used as conditioning
for crystal generation.
"""

import torch
import numpy as np
from ase.db import connect
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class MoleculeDataset(Dataset):
    """
    Single molecule dataset loader
    
    Loads individual molecules from ASE database (no periodic boundary conditions).
    Used to extract EGNN features for molecular conditioning in crystal generation.
    
    Args:
        db_path: Path to molecules ASE database
        indices: List of database indices to use (None = all)
        remove_h: Whether to remove hydrogen atoms
    """
    
    def __init__(
        self,
        db_path: str,
        indices: Optional[List[int]] = None,
        remove_h: bool = False,
    ):
        self.db_path = db_path
        self.remove_h = remove_h
        
        # Connect to database
        self.db = connect(db_path)
        
        # Set indices (ASE DB is 1-indexed)
        if indices is None:
            self.indices = list(range(1, len(self.db) + 1))
        else:
            self.indices = [i + 1 for i in indices]
        
        # Build atom encoder from dataset
        self._build_atom_encoder()
    
    def _build_atom_encoder(self):
        """Scan dataset to build atom type encoder"""
        all_atomic_numbers = set()
        
        for idx in self.indices:
            row = self.db.get(idx)
            atoms = row.toatoms()
            all_atomic_numbers.update(atoms.numbers)
        
        # Create encoder/decoder
        sorted_atomic_numbers = sorted(all_atomic_numbers)
        self.atom_encoder = {num: i for i, num in enumerate(sorted_atomic_numbers)}
        self.atom_decoder = sorted_atomic_numbers
        self.num_atom_types = len(self.atom_decoder)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single molecule
        
        Returns:
            data: Dictionary containing:
                - positions: [n_atoms, 3] atomic coordinates
                - atom_types: [n_atoms] atom type indices
                - one_hot: [n_atoms, num_atom_types] one-hot encoding
                - molecule_id: molecule identifier (str or int)
                - num_atoms: [1] number of atoms
        """
        # Get from database
        db_idx = self.indices[idx]
        row = self.db.get(db_idx)
        atoms = row.toatoms()
        
        # Extract molecule_id (REQUIRED - no fallback!)
        molecule_id = self._get_molecule_id(row, db_idx)
        
        # Remove hydrogen if requested
        if self.remove_h:
            mask = atoms.numbers != 1
            atoms = atoms[mask]
        
        # Extract data
        positions = torch.tensor(atoms.positions, dtype=torch.float32)
        atomic_numbers = atoms.numbers
        atom_types = torch.tensor(
            [self.atom_encoder[num] for num in atomic_numbers],
            dtype=torch.long
        )
        
        # Create one-hot encoding
        one_hot = torch.zeros(len(atoms), self.num_atom_types, dtype=torch.float32)
        one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
        
        data = {
            'positions': positions,
            'atom_types': atom_types,
            'one_hot': one_hot,
            'molecule_id': molecule_id,
            'num_atoms': torch.tensor([len(atoms)], dtype=torch.long),
        }
        
        return data
    
    def _get_molecule_id(self, row, db_idx: int) -> str:
        """
        Extract molecule_id from database row
        
        NO FALLBACK! molecule_id must be explicitly set.
        """
        # Try different possible locations for molecule_id
        if hasattr(row, 'molecule_id'):
            return str(row.molecule_id)
        
        if hasattr(row, 'data') and 'molecule_id' in row.data:
            return str(row.data['molecule_id'])
        
        if hasattr(row, 'key_value_pairs') and 'molecule_id' in row.key_value_pairs:
            return str(row.key_value_pairs['molecule_id'])
        
        # NO FALLBACK - raise clear error
        raise ValueError(
            f"molecule_id not found in database entry {db_idx}. "
            f"All molecules must have an explicit molecule_id field. "
            f"Available fields: {dir(row)}"
        )
    
    def get_molecule_by_id(self, molecule_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get molecule by its molecule_id"""
        for idx in range(len(self)):
            data = self[idx]
            if data['molecule_id'] == molecule_id:
                return data
        return None
