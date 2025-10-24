"""
Crystal structure dataset loader

Loads molecular crystal structures from ASE database with periodic boundary conditions.
Links to molecule dataset for feature extraction.
"""

import torch
import numpy as np
from ase.db import connect
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset

from crystal.data.periodic_utils import cartesian_to_fractional


class CrystalDataset(Dataset):
    """
    Molecular crystal dataset loader
    
    Loads crystal structures with periodic boundary conditions.
    Links to molecule dataset via molecule_id for feature extraction.
    
    Args:
        db_path: Path to crystals ASE database
        indices: List of database indices to use
        molecule_dataset: MoleculeDataset instance (optional)
        molecule_crystal_mapper: MoleculeCrystalMapper instance (optional)
        remove_h: Whether to remove hydrogen atoms
        use_fractional_coords: Use fractional coordinates (default: True)
        cutoff_radius: Cutoff radius for neighbor lists (Angstroms)
        max_atoms: Maximum atoms per unit cell
        include_charges: Include atomic charges
    """
    
    def __init__(
        self,
        db_path: str,
        indices: List[int],
        molecule_dataset: Optional['MoleculeDataset'] = None,
        molecule_crystal_mapper: Optional['MoleculeCrystalMapper'] = None,
        remove_h: bool = False,
        use_fractional_coords: bool = True,
        cutoff_radius: float = 10.0,
        max_atoms: int = 500,
        include_charges: bool = False,
    ):
        self.db_path = db_path
        # Store 0-based indices for consistency with PyTorch datasets
        self.indices = indices
        self.molecule_dataset = molecule_dataset
        self.molecule_crystal_mapper = molecule_crystal_mapper
        self.remove_h = remove_h
        self.use_fractional_coords = use_fractional_coords
        self.cutoff_radius = cutoff_radius
        self.max_atoms = max_atoms
        self.include_charges = include_charges
        
        # Connect to database
        self.db = connect(db_path)
        
        # Build atom encoder
        self._build_atom_encoder()
        
        # Cache for molecule data
        self.molecule_cache: Dict[str, Dict] = {}
    
    def _build_atom_encoder(self):
        """Scan dataset to build atom type encoder"""
        all_atomic_numbers = set()
        
        # self.indices contains 0-based indices (list indices), convert to 1-based for ASE DB
        for idx in self.indices:
            row = self.db.get(idx + 1)  # Convert to 1-based for ASE DB
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
        Get single crystal structure
        
        Returns:
            data: Dictionary containing:
                - positions: [n_atoms, 3] atomic positions (fractional or Cartesian)
                - positions_cart: [n_atoms, 3] Cartesian positions (always included)
                - atom_types: [n_atoms] atom type indices
                - one_hot: [n_atoms, num_atom_types] one-hot encoding
                - cell: [3, 3] unit cell vectors
                - cell_params: [6] (a, b, c, α, β, γ)
                - cell_volume: [1] unit cell volume
                - pbc: [3] periodic boundary flags
                - num_atoms: [1] number of atoms
                - molecule_id: str - ID of constituent molecule
                - crystal_id: str - ID of this crystal
                - space_group: [1] space group number (if available)
                - density: [1] crystal density (if available)
        """
        # Get from database (self.indices contains 0-based indices, convert to 1-based for ASE DB)
        db_idx = self.indices[idx]
        row = self.db.get(db_idx + 1)  # Convert to 1-based for ASE DB
        atoms = row.toatoms()
        
        # Extract IDs
        crystal_id = self._get_crystal_id(row, db_idx)
        molecule_id = self._get_molecule_id(row, crystal_id)
        
        # Remove hydrogen if requested
        if self.remove_h:
            mask = atoms.numbers != 1
            atoms = atoms[mask]
        
        # Check atom count
        if len(atoms) > self.max_atoms:
            raise ValueError(
                f"Crystal {crystal_id} has {len(atoms)} atoms, "
                f"exceeding maximum {self.max_atoms}"
            )
        
        # Extract positions (Cartesian)
        positions_cart = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # Extract cell information
        cell_vectors = torch.tensor(atoms.cell.array, dtype=torch.float32)  # [3, 3]
        cell_params = torch.tensor(atoms.cell.cellpar(), dtype=torch.float32)  # [6]
        cell_volume = torch.tensor([atoms.cell.volume], dtype=torch.float32)
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool)
        
        # Convert to fractional if requested
        if self.use_fractional_coords:
            positions = self._cartesian_to_fractional(positions_cart, cell_vectors)
        else:
            positions = positions_cart
        
        # Extract atom types
        atomic_numbers = atoms.numbers
        atom_types = torch.tensor(
            [self.atom_encoder[num] for num in atomic_numbers],
            dtype=torch.long
        )
        
        # Create one-hot encoding
        one_hot = torch.zeros(len(atoms), self.num_atom_types, dtype=torch.float32)
        one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
        
        # Build data dictionary
        data = {
            'positions': positions,
            'positions_cart': positions_cart,
            'atom_types': atom_types,
            'one_hot': one_hot,
            'cell': cell_vectors,
            'cell_params': cell_params,
            'cell_volume': cell_volume,
            'pbc': pbc,
            'num_atoms': torch.tensor([len(atoms)], dtype=torch.long),
            'molecule_id': molecule_id,
            'crystal_id': crystal_id,
        }
        
        # Add charges if requested
        if self.include_charges:
            data['charges'] = torch.tensor(atomic_numbers, dtype=torch.float32)
        
        # Add metadata if available
        if hasattr(row, 'space_group'):
            data['space_group'] = torch.tensor([row.space_group], dtype=torch.long)
        elif hasattr(row, 'data') and 'space_group' in row.data:
            data['space_group'] = torch.tensor([row.data['space_group']], dtype=torch.long)
        
        if hasattr(row, 'density'):
            data['density'] = torch.tensor([row.density], dtype=torch.float32)
        elif hasattr(row, 'data') and 'density' in row.data:
            data['density'] = torch.tensor([row.data['density']], dtype=torch.float32)
        
        return data
    
    def _get_crystal_id(self, row, db_idx: int) -> str:
        """Extract crystal_id from row"""
        if hasattr(row, 'crystal_id'):
            return str(row.crystal_id)
        if hasattr(row, 'data') and 'crystal_id' in row.data:
            return str(row.data['crystal_id'])
        if hasattr(row, 'key_value_pairs') and 'crystal_id' in row.key_value_pairs:
            return str(row.key_value_pairs['crystal_id'])
        # Use database id as fallback for crystal_id only
        return str(row.id)
    
    def _get_molecule_id(self, row, crystal_id: str) -> str:
        """
        Extract molecule_id from crystal row
        
        NO FALLBACK! All crystals must have molecule_id.
        """
        if hasattr(row, 'molecule_id'):
            return str(row.molecule_id)
        
        if hasattr(row, 'data') and 'molecule_id' in row.data:
            return str(row.data['molecule_id'])
        
        if hasattr(row, 'key_value_pairs') and 'molecule_id' in row.key_value_pairs:
            return str(row.key_value_pairs['molecule_id'])
        
        # NO FALLBACK - raise clear error
        raise ValueError(
            f"molecule_id not found in crystal entry {crystal_id}. "
            f"All crystals must have an explicit molecule_id field. "
            f"This is REQUIRED for homocrystal generation."
        )
    
    @staticmethod
    def _cartesian_to_fractional(
        positions_cart: torch.Tensor,
        cell_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Convert Cartesian to fractional coordinates"""
        return cartesian_to_fractional(positions_cart.unsqueeze(0), cell_vectors.unsqueeze(0)).squeeze(0)


def collate_crystal_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching crystal structures
    
    Pads structures to same size and creates masks.
    
    Args:
        batch: List of data dictionaries from CrystalDataset
        
    Returns:
        batched_data: Dictionary with batched tensors
    """
    batch_size = len(batch)
    max_atoms = max(item['num_atoms'].item() for item in batch)
    num_atom_types = batch[0]['one_hot'].shape[1]
    
    # Initialize batch tensors
    positions = torch.zeros(batch_size, max_atoms, 3)
    positions_cart = torch.zeros(batch_size, max_atoms, 3)
    atom_types = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    one_hot = torch.zeros(batch_size, max_atoms, num_atom_types)
    mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    
    cell = torch.zeros(batch_size, 3, 3)
    cell_params = torch.zeros(batch_size, 6)
    cell_volume = torch.zeros(batch_size, 1)
    pbc = torch.zeros(batch_size, 3, dtype=torch.bool)
    num_atoms = torch.zeros(batch_size, dtype=torch.long)
    
    # Lists for non-tensor data
    molecule_ids = []
    crystal_ids = []
    
    # Fill batch tensors
    for i, item in enumerate(batch):
        n_atoms = item['num_atoms'].item()
        
        positions[i, :n_atoms] = item['positions']
        positions_cart[i, :n_atoms] = item['positions_cart']
        atom_types[i, :n_atoms] = item['atom_types']
        one_hot[i, :n_atoms] = item['one_hot']
        mask[i, :n_atoms] = True
        
        cell[i] = item['cell']
        cell_params[i] = item['cell_params']
        cell_volume[i] = item['cell_volume']
        pbc[i] = item['pbc']
        num_atoms[i] = n_atoms
        
        molecule_ids.append(item['molecule_id'])
        crystal_ids.append(item['crystal_id'])
    
    batched_data = {
        'positions': positions,
        'positions_cart': positions_cart,
        'atom_types': atom_types,
        'one_hot': one_hot,
        'mask': mask,
        'cell': cell,
        'cell_params': cell_params,
        'cell_volume': cell_volume,
        'pbc': pbc,
        'num_atoms': num_atoms,
        'molecule_ids': molecule_ids,
        'crystal_ids': crystal_ids,
    }
    
    # Add optional fields if present in all items
    if 'charges' in batch[0]:
        charges = torch.zeros(batch_size, max_atoms)
        for i, item in enumerate(batch):
            n_atoms = item['num_atoms'].item()
            charges[i, :n_atoms] = item['charges']
        batched_data['charges'] = charges
    
    if 'space_group' in batch[0]:
        space_group = torch.stack([item['space_group'] for item in batch])
        batched_data['space_group'] = space_group
    
    if 'density' in batch[0]:
        density = torch.stack([item['density'] for item in batch])
        batched_data['density'] = density
    
    # Add properties if present
    if 'properties' in batch[0]:
        properties = torch.stack([item['properties'] for item in batch])
        batched_data['properties'] = properties
    
    return batched_data


class CrystalDatasetWithProperties(CrystalDataset):
    """
    Extended crystal dataset that includes physical property values.
    
    This class extends CrystalDataset to load and provide physical properties
    (e.g., bandgap, melting point) along with crystal structures.
    
    Properties are extracted from the ASE database 'data' field and statistics
    (mean, std) are computed for normalization during training.
    
    Args:
        db_path: Path to crystals ASE database
        indices: List of database indices to use
        property_names: List of property names to extract
        molecule_dataset: MoleculeDataset instance (optional)
        molecule_crystal_mapper: MoleculeCrystalMapper instance (optional)
        **kwargs: Additional arguments passed to CrystalDataset
    
    Example:
        >>> dataset = CrystalDatasetWithProperties(
        ...     db_path='crystals.db',
        ...     indices=list(range(1000)),
        ...     property_names=['bandgap', 'melting_point']
        ... )
        >>> data = dataset[0]
        >>> print(data['properties'])  # [bandgap, melting_point]
        >>> print(dataset.property_mean)  # Mean of each property
        >>> print(dataset.property_std)  # Std of each property
    """
    
    def __init__(
        self,
        db_path: str,
        indices: List[int],
        property_names: Optional[List[str]] = None,
        molecule_dataset: Optional['MoleculeDataset'] = None,
        molecule_crystal_mapper: Optional['MoleculeCrystalMapper'] = None,
        **kwargs
    ):
        # Initialize parent class
        super().__init__(
            db_path=db_path,
            indices=indices,
            molecule_dataset=molecule_dataset,
            molecule_crystal_mapper=molecule_crystal_mapper,
            **kwargs
        )
        
        self.property_names = property_names or []
        
        # Compute property statistics if properties are requested
        if self.property_names:
            self._compute_property_statistics()
        else:
            self.property_mean = None
            self.property_std = None
    
    def _compute_property_statistics(self):
        """
        Compute mean and standard deviation of property values.
        
        Statistics are computed across all samples in the dataset and
        used for property normalization during training.
        
        Raises:
            ValueError: If a property is missing in the database
            ValueError: If all values of a property are the same (std=0)
        """
        property_values = []
        missing_properties = set()
        
        # Collect all property values
        for idx in self.indices:
            row = self.db.get(idx + 1)  # Convert to 1-based for ASE DB
            values = []
            
            for prop_name in self.property_names:
                # Try different locations for property values
                value = None
                
                # Check in data field (most common)
                if hasattr(row, 'data') and prop_name in row.data:
                    value = row.data[prop_name]
                # Check as direct attribute
                elif hasattr(row, prop_name):
                    value = getattr(row, prop_name)
                # Check in key_value_pairs
                elif hasattr(row, 'key_value_pairs') and prop_name in row.key_value_pairs:
                    value = row.key_value_pairs[prop_name]
                
                if value is None:
                    missing_properties.add(prop_name)
                    # Use NaN for missing values (will be caught later)
                    values.append(float('nan'))
                else:
                    values.append(float(value))
            
            property_values.append(values)
        
        # Raise error if any properties are missing
        if missing_properties:
            raise ValueError(
                f"The following properties are missing in some database entries: "
                f"{missing_properties}. All properties must be present for all crystals."
            )
        
        # Convert to tensor
        property_values = torch.tensor(property_values, dtype=torch.float32)
        
        # Check for NaN values
        if torch.any(torch.isnan(property_values)):
            raise ValueError(
                "NaN values found in property values. "
                "Ensure all properties are present and have valid numeric values."
            )
        
        # Compute statistics
        self.property_mean = property_values.mean(dim=0)
        self.property_std = property_values.std(dim=0)
        
        # Check for zero std (all values the same)
        zero_std_mask = self.property_std < 1e-8
        if torch.any(zero_std_mask):
            zero_std_props = [
                self.property_names[i] 
                for i in range(len(self.property_names)) 
                if zero_std_mask[i]
            ]
            raise ValueError(
                f"The following properties have zero standard deviation "
                f"(all values are the same): {zero_std_props}. "
                f"Remove these properties or add more diverse data."
            )
        
        # Log statistics
        print(f"\nProperty statistics for {len(self.indices)} crystals:")
        for i, name in enumerate(self.property_names):
            print(
                f"  {name:30s}: "
                f"mean={self.property_mean[i]:8.3f}, "
                f"std={self.property_std[i]:8.3f}"
            )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single crystal structure with properties.
        
        Returns:
            data: Dictionary containing all CrystalDataset fields plus:
                - properties: [property_dim] property values
        """
        # Get base data from parent class
        data = super().__getitem__(idx)
        
        # Add property values if requested
        if self.property_names:
            db_idx = self.indices[idx]
            row = self.db.get(db_idx + 1)  # Convert to 1-based for ASE DB
            
            properties = []
            for prop_name in self.property_names:
                # Try different locations (same as in _compute_property_statistics)
                value = None
                
                if hasattr(row, 'data') and prop_name in row.data:
                    value = row.data[prop_name]
                elif hasattr(row, prop_name):
                    value = getattr(row, prop_name)
                elif hasattr(row, 'key_value_pairs') and prop_name in row.key_value_pairs:
                    value = row.key_value_pairs[prop_name]
                
                if value is None:
                    # This should not happen if _compute_property_statistics succeeded
                    raise ValueError(
                        f"Property '{prop_name}' not found for crystal {data['crystal_id']}"
                    )
                
                properties.append(float(value))
            
            data['properties'] = torch.tensor(properties, dtype=torch.float32)
        
        return data
