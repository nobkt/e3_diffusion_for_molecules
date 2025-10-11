"""
Molecular Crystal Dataset Loader with Molecule Conditioning

This module provides a dataset loader that handles paired molecule-crystal data
for conditional crystal generation.
"""

import torch
import numpy as np
from ase.db import connect
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class MolecularCrystalDataset(Dataset):
    """
    Dataset for paired molecular and crystal structures.
    
    Each data point consists of:
    - A single molecule structure (reference)
    - A corresponding molecular crystal structure
    
    The crystal is composed of Z copies of the reference molecule.
    """
    
    def __init__(
        self,
        molecule_db_path: str,
        crystal_db_path: str,
        indices: List[int],
        remove_h: bool = False,
        use_fractional_coords: bool = True,
        cutoff_radius: float = 10.0,
        max_atoms: int = 500,
        include_charges: bool = False,
    ):
        """
        Args:
            molecule_db_path: Path to ASE database containing single molecules
            crystal_db_path: Path to ASE database containing crystals
            indices: List of crystal indices to use
            remove_h: Whether to remove hydrogen atoms
            use_fractional_coords: Whether to use fractional coordinates for crystals
            cutoff_radius: Cutoff radius for neighbor calculations (Å)
            max_atoms: Maximum number of atoms per unit cell
            include_charges: Whether to include atomic charges
        """
        self.molecule_db_path = molecule_db_path
        self.crystal_db_path = crystal_db_path
        self.indices = indices
        self.remove_h = remove_h
        self.use_fractional_coords = use_fractional_coords
        self.cutoff_radius = cutoff_radius
        self.max_atoms = max_atoms
        self.include_charges = include_charges
        
        # Connect to databases
        self.molecule_db = connect(molecule_db_path)
        self.crystal_db = connect(crystal_db_path)
        
        # Build atom encoder/decoder
        self._build_atom_encoder()
        
        logger.info(f"Loaded MolecularCrystalDataset with {len(self)} samples")
        logger.info(f"Number of atom types: {self.num_atom_types}")
    
    def _build_atom_encoder(self):
        """Build atom encoder/decoder from all data in both databases."""
        all_atomic_numbers = set()
        
        # Collect from molecules
        for idx in range(len(self.molecule_db)):
            row = self.molecule_db.get(idx + 1)
            atoms = row.toatoms()
            if not self.remove_h:
                all_atomic_numbers.update(atoms.numbers)
            else:
                all_atomic_numbers.update([n for n in atoms.numbers if n != 1])
        
        # Collect from crystals
        for idx in self.indices:
            row = self.crystal_db.get(idx + 1)
            atoms = row.toatoms()
            if not self.remove_h:
                all_atomic_numbers.update(atoms.numbers)
            else:
                all_atomic_numbers.update([n for n in atoms.numbers if n != 1])
        
        # Sort and create encoder/decoder
        sorted_atomic_numbers = sorted(all_atomic_numbers)
        
        self.atom_encoder = {num: i for i, num in enumerate(sorted_atomic_numbers)}
        self.atom_decoder = sorted_atomic_numbers
        self.num_atom_types = len(self.atom_decoder)
        
        logger.info(f"Atom types: {self.atom_decoder}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a paired molecule-crystal sample.
        
        Returns:
            Dictionary containing:
                - 'molecule': Single molecule data
                - 'crystal': Crystal structure data
                - 'metadata': Additional information
        """
        # Get crystal data
        crystal_idx = self.indices[idx]
        crystal_row = self.crystal_db.get(crystal_idx + 1)
        crystal_atoms = crystal_row.toatoms()
        
        # Get corresponding molecule ID
        if hasattr(crystal_row, 'molecule_id'):
            molecule_id = crystal_row.molecule_id
        elif hasattr(crystal_row, 'data') and 'molecule_id' in crystal_row.data:
            molecule_id = crystal_row.data['molecule_id']
        else:
            raise ValueError(f"Crystal {crystal_idx} missing molecule_id")
        
        # Get molecule data
        molecule_row = self.molecule_db.get(molecule_id)
        molecule_atoms = molecule_row.toatoms()
        
        # Remove hydrogen if requested
        if self.remove_h:
            mol_mask = molecule_atoms.numbers != 1
            molecule_atoms = molecule_atoms[mol_mask]
            
            cryst_mask = crystal_atoms.numbers != 1
            crystal_atoms = crystal_atoms[cryst_mask]
        
        # Check size constraints
        if len(molecule_atoms) > self.max_atoms:
            raise ValueError(f"Molecule has {len(molecule_atoms)} atoms, exceeding max {self.max_atoms}")
        if len(crystal_atoms) > self.max_atoms:
            raise ValueError(f"Crystal has {len(crystal_atoms)} atoms, exceeding max {self.max_atoms}")
        
        # Process molecule
        molecule_data = self._process_molecule(molecule_atoms)
        
        # Process crystal
        crystal_data = self._process_crystal(crystal_atoms, crystal_row)
        
        # Metadata
        metadata = {
            'molecule_id': molecule_id,
            'crystal_id': crystal_idx,
        }
        
        # Optional metadata
        if hasattr(crystal_row, 'space_group'):
            metadata['space_group'] = torch.tensor([crystal_row.space_group], dtype=torch.long)
        elif hasattr(crystal_row, 'data') and 'space_group' in crystal_row.data:
            metadata['space_group'] = torch.tensor([crystal_row.data['space_group']], dtype=torch.long)
        
        if hasattr(crystal_row, 'density'):
            metadata['density'] = torch.tensor([crystal_row.density], dtype=torch.float32)
        elif hasattr(crystal_row, 'data') and 'density' in crystal_row.data:
            metadata['density'] = torch.tensor([crystal_row.data['density']], dtype=torch.float32)
        
        if hasattr(crystal_row, 'Z'):
            metadata['Z'] = torch.tensor([crystal_row.Z], dtype=torch.long)
        elif hasattr(crystal_row, 'data') and 'Z' in crystal_row.data:
            metadata['Z'] = torch.tensor([crystal_row.data['Z']], dtype=torch.long)
        else:
            # Estimate Z from atom counts
            Z_estimate = len(crystal_atoms) // len(molecule_atoms)
            metadata['Z'] = torch.tensor([Z_estimate], dtype=torch.long)
        
        return {
            'molecule': molecule_data,
            'crystal': crystal_data,
            'metadata': metadata,
        }
    
    def _process_molecule(self, atoms) -> Dict[str, torch.Tensor]:
        """Process a single molecule into tensor format."""
        n_atoms = len(atoms)
        
        # Positions (Cartesian)
        positions = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # Center molecule at origin
        positions = positions - positions.mean(dim=0, keepdim=True)
        
        # Atom types
        atomic_numbers = atoms.numbers
        atom_types = torch.tensor(
            [self.atom_encoder[num] for num in atomic_numbers],
            dtype=torch.long
        )
        
        # One-hot encoding
        one_hot = torch.zeros(n_atoms, self.num_atom_types, dtype=torch.float32)
        one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
        
        data = {
            'positions': positions,
            'atom_types': atom_types,
            'one_hot': one_hot,
            'num_atoms': torch.tensor([n_atoms], dtype=torch.long),
        }
        
        # Charges
        if self.include_charges:
            data['charges'] = torch.tensor(atomic_numbers, dtype=torch.float32)
        
        return data
    
    def _process_crystal(self, atoms, row) -> Dict[str, torch.Tensor]:
        """Process a crystal structure into tensor format."""
        n_atoms = len(atoms)
        
        # Positions (Cartesian)
        positions_cart = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # Cell information
        cell_vectors = torch.tensor(atoms.cell.array, dtype=torch.float32)  # [3, 3]
        cell_params = torch.tensor(atoms.cell.cellpar(), dtype=torch.float32)  # [6] (a,b,c,α,β,γ)
        cell_volume = torch.tensor([atoms.cell.volume], dtype=torch.float32)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool)
        
        # Convert to fractional coordinates if requested
        if self.use_fractional_coords:
            positions = self._cartesian_to_fractional(positions_cart, cell_vectors)
        else:
            positions = positions_cart
        
        # Atom types
        atomic_numbers = atoms.numbers
        atom_types = torch.tensor(
            [self.atom_encoder[num] for num in atomic_numbers],
            dtype=torch.long
        )
        
        # One-hot encoding
        one_hot = torch.zeros(n_atoms, self.num_atom_types, dtype=torch.float32)
        one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
        
        data = {
            'positions': positions,
            'positions_cart': positions_cart,
            'atom_types': atom_types,
            'one_hot': one_hot,
            'cell': cell_vectors,
            'cell_params': cell_params,
            'cell_volume': cell_volume,
            'pbc': pbc,
            'num_atoms': torch.tensor([n_atoms], dtype=torch.long),
        }
        
        # Charges
        if self.include_charges:
            data['charges'] = torch.tensor(atomic_numbers, dtype=torch.float32)
        
        return data
    
    @staticmethod
    def _cartesian_to_fractional(
        positions_cart: torch.Tensor,
        cell_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Convert Cartesian coordinates to fractional coordinates."""
        # Solve: cell_vectors^T @ positions_frac^T = positions_cart^T
        positions_frac = torch.linalg.solve(cell_vectors.T, positions_cart.T).T
        return positions_frac


def collate_molecular_crystal_batch(batch: List[Dict]) -> Dict[str, Dict]:
    """
    Collate function for batching molecular-crystal pairs.
    
    Handles variable-sized molecules and crystals with padding and masking.
    
    Args:
        batch: List of samples from MolecularCrystalDataset
        
    Returns:
        Batched data with separate 'molecule' and 'crystal' dictionaries
    """
    batch_size = len(batch)
    
    # Find maximum sizes
    max_mol_atoms = max(item['molecule']['num_atoms'].item() for item in batch)
    max_crystal_atoms = max(item['crystal']['num_atoms'].item() for item in batch)
    num_atom_types = batch[0]['molecule']['one_hot'].shape[1]
    
    # Initialize molecule batch
    mol_positions = torch.zeros(batch_size, max_mol_atoms, 3)
    mol_atom_types = torch.zeros(batch_size, max_mol_atoms, dtype=torch.long)
    mol_one_hot = torch.zeros(batch_size, max_mol_atoms, num_atom_types)
    mol_mask = torch.zeros(batch_size, max_mol_atoms, dtype=torch.bool)
    mol_num_atoms = torch.zeros(batch_size, dtype=torch.long)
    
    # Initialize crystal batch
    cryst_positions = torch.zeros(batch_size, max_crystal_atoms, 3)
    cryst_positions_cart = torch.zeros(batch_size, max_crystal_atoms, 3)
    cryst_atom_types = torch.zeros(batch_size, max_crystal_atoms, dtype=torch.long)
    cryst_one_hot = torch.zeros(batch_size, max_crystal_atoms, num_atom_types)
    cryst_mask = torch.zeros(batch_size, max_crystal_atoms, dtype=torch.bool)
    cryst_num_atoms = torch.zeros(batch_size, dtype=torch.long)
    
    cryst_cell = torch.zeros(batch_size, 3, 3)
    cryst_cell_params = torch.zeros(batch_size, 6)
    cryst_cell_volume = torch.zeros(batch_size, 1)
    cryst_pbc = torch.zeros(batch_size, 3, dtype=torch.bool)
    
    # Fill batches
    for i, item in enumerate(batch):
        # Molecule
        n_mol = item['molecule']['num_atoms'].item()
        mol_positions[i, :n_mol] = item['molecule']['positions']
        mol_atom_types[i, :n_mol] = item['molecule']['atom_types']
        mol_one_hot[i, :n_mol] = item['molecule']['one_hot']
        mol_mask[i, :n_mol] = True
        mol_num_atoms[i] = n_mol
        
        # Crystal
        n_cryst = item['crystal']['num_atoms'].item()
        cryst_positions[i, :n_cryst] = item['crystal']['positions']
        cryst_positions_cart[i, :n_cryst] = item['crystal']['positions_cart']
        cryst_atom_types[i, :n_cryst] = item['crystal']['atom_types']
        cryst_one_hot[i, :n_cryst] = item['crystal']['one_hot']
        cryst_mask[i, :n_cryst] = True
        cryst_num_atoms[i] = n_cryst
        
        cryst_cell[i] = item['crystal']['cell']
        cryst_cell_params[i] = item['crystal']['cell_params']
        cryst_cell_volume[i] = item['crystal']['cell_volume']
        cryst_pbc[i] = item['crystal']['pbc']
    
    batched_data = {
        'molecule': {
            'positions': mol_positions,
            'atom_types': mol_atom_types,
            'one_hot': mol_one_hot,
            'node_mask': mol_mask,
            'num_atoms': mol_num_atoms,
        },
        'crystal': {
            'positions': cryst_positions,
            'positions_cart': cryst_positions_cart,
            'atom_types': cryst_atom_types,
            'one_hot': cryst_one_hot,
            'node_mask': cryst_mask,
            'num_atoms': cryst_num_atoms,
            'cell': cryst_cell,
            'cell_params': cryst_cell_params,
            'cell_volume': cryst_cell_volume,
            'pbc': cryst_pbc,
        },
        'metadata': {},
    }
    
    # Optional fields
    if 'charges' in batch[0]['molecule']:
        mol_charges = torch.zeros(batch_size, max_mol_atoms)
        cryst_charges = torch.zeros(batch_size, max_crystal_atoms)
        for i, item in enumerate(batch):
            n_mol = item['molecule']['num_atoms'].item()
            n_cryst = item['crystal']['num_atoms'].item()
            mol_charges[i, :n_mol] = item['molecule']['charges']
            cryst_charges[i, :n_cryst] = item['crystal']['charges']
        batched_data['molecule']['charges'] = mol_charges
        batched_data['crystal']['charges'] = cryst_charges
    
    # Metadata
    for key in batch[0]['metadata']:
        if key in ['molecule_id', 'crystal_id']:
            batched_data['metadata'][key] = [item['metadata'][key] for item in batch]
        else:
            batched_data['metadata'][key] = torch.stack([item['metadata'][key] for item in batch])
    
    return batched_data


def load_paired_datasets(
    molecule_db_path: str,
    crystal_db_path: str,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    **dataset_kwargs
) -> Tuple[Dict[str, MolecularCrystalDataset], Dict]:
    """
    Load and split paired molecule-crystal datasets.
    
    Args:
        molecule_db_path: Path to molecule database
        crystal_db_path: Path to crystal database
        split_ratios: (train, val, test) ratios
        seed: Random seed for splitting
        **dataset_kwargs: Additional arguments for MolecularCrystalDataset
        
    Returns:
        datasets: Dictionary with 'train', 'valid', 'test' splits
        dataset_info: Dataset metadata
    """
    from ase.db import connect
    
    # Connect to crystal database to get total size
    crystal_db = connect(crystal_db_path)
    n_total = len(crystal_db)
    
    # Split indices
    indices = np.random.RandomState(seed).permutation(n_total)
    
    n_train = int(split_ratios[0] * n_total)
    n_valid = int(split_ratios[1] * n_total)
    
    train_indices = indices[:n_train].tolist()
    valid_indices = indices[n_train:n_train + n_valid].tolist()
    test_indices = indices[n_train + n_valid:].tolist()
    
    # Create datasets
    train_dataset = MolecularCrystalDataset(
        molecule_db_path=molecule_db_path,
        crystal_db_path=crystal_db_path,
        indices=train_indices,
        **dataset_kwargs
    )
    
    valid_dataset = MolecularCrystalDataset(
        molecule_db_path=molecule_db_path,
        crystal_db_path=crystal_db_path,
        indices=valid_indices,
        **dataset_kwargs
    )
    
    test_dataset = MolecularCrystalDataset(
        molecule_db_path=molecule_db_path,
        crystal_db_path=crystal_db_path,
        indices=test_indices,
        **dataset_kwargs
    )
    
    datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset,
    }
    
    # Dataset info
    dataset_info = {
        'atom_encoder': train_dataset.atom_encoder,
        'atom_decoder': train_dataset.atom_decoder,
        'num_atom_types': train_dataset.num_atom_types,
        'n_train': len(train_dataset),
        'n_valid': len(valid_dataset),
        'n_test': len(test_dataset),
    }
    
    logger.info(f"Dataset splits - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    return datasets, dataset_info
