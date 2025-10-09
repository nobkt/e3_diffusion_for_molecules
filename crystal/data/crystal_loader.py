"""
Crystal structure data loader from ASE database.

This module provides dataset class and collation functions for loading
molecular crystal structures from ASE databases.
"""

import torch
import numpy as np
from ase.db import connect
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset

from .periodic_utils import (
    cartesian_to_fractional,
    fractional_to_cartesian,
    cell_params_to_vectors,
    cell_vectors_to_params,
    wrap_to_unit_cell,
)


class CrystalDataset(Dataset):
    """
    Dataset class for molecular crystal structures from ASE database.
    
    Args:
        db_path: Path to ASE database file
        indices: Optional list of specific indices to load
        use_fractional_coords: If True, store positions in fractional coordinates
        max_atoms: Maximum number of atoms per structure (for padding)
        
    """
    
    def __init__(
        self,
        db_path: str,
        indices: Optional[List[int]] = None,
        use_fractional_coords: bool = True,
        max_atoms: Optional[int] = None,
    ):
        self.db_path = db_path
        self.use_fractional_coords = use_fractional_coords
        
        # Connect to database
        self.db = connect(db_path)
        
        # Get indices
        if indices is None:
            self.indices = list(range(1, len(self.db) + 1))
        else:
            self.indices = indices
        
        # Determine max_atoms if not provided
        if max_atoms is None:
            self.max_atoms = self._compute_max_atoms()
        else:
            self.max_atoms = max_atoms
    
    def _compute_max_atoms(self) -> int:
        """Compute maximum number of atoms across all structures."""
        max_n = 0
        for idx in self.indices:
            row = self.db.get(idx)
            atoms = row.toatoms()
            max_n = max(max_n, len(atoms))
        return max_n
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single crystal structure.
        
        Returns:
            Dict with keys:
                - positions: (max_atoms, 3) - atomic positions
                - atom_types: (max_atoms,) - atomic numbers
                - cell_vectors: (3, 3) - unit cell vectors
                - cell_params: (6,) - [a, b, c, alpha, beta, gamma]
                - atom_mask: (max_atoms,) - mask for real atoms
                - pbc: (3,) - periodic boundary conditions
                - n_atoms: int - actual number of atoms
        """
        db_idx = self.indices[idx]
        row = self.db.get(db_idx)
        atoms = row.toatoms()
        
        # Get atomic positions and numbers
        positions = torch.tensor(atoms.positions, dtype=torch.float32)
        atom_types = torch.tensor(atoms.numbers, dtype=torch.long)
        n_atoms = len(atoms)
        
        # Get cell information
        cell_array = atoms.cell.array
        cell_vectors = torch.tensor(cell_array, dtype=torch.float32)
        
        # Get cell parameters
        cell_lengths = atoms.cell.lengths()
        cell_angles = atoms.cell.angles()
        cell_params = torch.tensor(
            np.concatenate([cell_lengths, cell_angles]),
            dtype=torch.float32
        )
        
        # Get periodic boundary conditions
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool)
        
        # Convert to fractional coordinates if requested
        if self.use_fractional_coords:
            positions = cartesian_to_fractional(positions, cell_vectors)
        
        # Create padding mask
        atom_mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        atom_mask[:n_atoms] = True
        
        # Pad arrays
        positions_padded = torch.zeros(self.max_atoms, 3, dtype=torch.float32)
        positions_padded[:n_atoms] = positions
        
        atom_types_padded = torch.zeros(self.max_atoms, dtype=torch.long)
        atom_types_padded[:n_atoms] = atom_types
        
        return {
            'positions': positions_padded,
            'atom_types': atom_types_padded,
            'cell_vectors': cell_vectors,
            'cell_params': cell_params,
            'atom_mask': atom_mask,
            'pbc': pbc,
            'n_atoms': n_atoms,
        }


def collate_crystal_batch(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching crystal structures.
    
    Args:
        batch: List of dictionaries from CrystalDataset.__getitem__
        
    Returns:
        Batched dictionary with shape (batch_size, ...)
    """
    # Stack all tensors
    batched = {}
    
    for key in batch[0].keys():
        if key == 'n_atoms':
            # Keep as list for n_atoms
            batched[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        else:
            # Stack other tensors
            batched[key] = torch.stack([item[key] for item in batch], dim=0)
    
    return batched
