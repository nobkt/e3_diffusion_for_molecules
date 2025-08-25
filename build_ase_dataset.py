import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
from qm9.data import collate as qm9_collate
import ase.db


def load_ase_data(db_file, max_entries=None, exclude_keys=None):
    """
    Load data from ASE database file with molecular properties.
    
    Parameters
    ----------
    db_file : str
        Path to ASE database file (.db)
    max_entries : int, optional
        Maximum number of entries to load
    exclude_keys : list, optional
        Keys to exclude from the database query
        
    Returns
    -------
    data_list : list
        List of molecular data dictionaries, each containing:
        - 'geometry': array of shape (n_atoms, 4) with [atomic_number, x, y, z]
        - 'properties': dict of molecular properties from the database
    """
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"ASE database file not found: {db_file}")
    
    # Connect to ASE database
    db = ase.db.connect(db_file)
    
    data_list = []
    count = 0
    
    # Iterate through database entries
    for row in db.select():
        if max_entries is not None and count >= max_entries:
            break
            
        atoms = row.toatoms()
        
        # Get atomic numbers and positions
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        
        # Combine into single array: [atomic_number, x, y, z]
        geometry = np.column_stack([atomic_numbers, positions])
        
        # Extract molecular properties from key-value pairs
        properties = dict(row.key_value_pairs)
        
        # Also check for properties in the data field
        if hasattr(row, 'data') and row.data:
            properties.update(row.data)
        
        mol_data = {
            'geometry': geometry,
            'properties': properties
        }
        data_list.append(mol_data)
        
        count += 1
    
    print(f"Loaded {count} molecules from ASE database: {db_file}")
    return data_list


def load_split_data(db_file, val_proportion=0.1, test_proportion=0.1, 
                   filter_size=None, max_entries=None):
    """
    Load and split ASE database data into train/validation/test sets.
    
    Parameters
    ----------
    db_file : str
        Path to ASE database file
    val_proportion : float
        Proportion of data for validation set
    test_proportion : float  
        Proportion of data for test set
    filter_size : int, optional
        If specified, only keep molecules with this many atoms
    max_entries : int, optional
        Maximum number of entries to load from database
        
    Returns
    -------
    tuple
        (train_data, val_data, test_data) where each is a list of molecular arrays
    """
    # Load data from ASE database
    data_list = load_ase_data(db_file, max_entries=max_entries)
    
    # Filter by size if requested
    if filter_size is not None:
        data_list = [mol for mol in data_list if mol['geometry'].shape[0] == filter_size]
        print(f"Filtered to {len(data_list)} molecules with {filter_size} atoms")
    
    if len(data_list) == 0:
        raise ValueError("No data available after filtering")
    
    # Calculate split indices
    n_total = len(data_list)
    n_test = int(n_total * test_proportion)
    n_val = int(n_total * val_proportion)
    n_train = n_total - n_test - n_val
    
    print(f"Splitting data: {n_train} train, {n_val} val, {n_test} test")
    
    # Random shuffle and split
    np.random.seed(42)  # For reproducible splits
    indices = np.random.permutation(n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    
    return train_data, val_data, test_data


class ASEDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        ASE dataset class.
        
        Parameters
        ----------
        data_list : list
            List of molecular data dictionaries with 'geometry' and 'properties' keys
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # Sort the data list by size
        lengths = [mol['geometry'].shape[0] for mol in data_list]
        argsort = np.argsort(lengths)
        self.data_list = [data_list[i] for i in argsort]
        
        # Store indices where the size changes
        self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]
        
        # Store all molecular properties for unit conversion
        self.properties = [mol['properties'] for mol in self.data_list]

    def convert_units(self, units_dict):
        """
        Convert units of molecular properties to match QM9 dataset behavior.
        
        Parameters
        ----------
        units_dict : dict
            Dictionary mapping property names to conversion factors
        """
        for mol_props in self.properties:
            for prop_name, conversion_factor in units_dict.items():
                # Handle different possible property key formats
                property_keys = []
                
                # Check for Hartree keys (from ASE conversion script)
                if prop_name in ['U0', 'U', 'G', 'H', 'zpve', 'gap', 'homo', 'lumo']:
                    ha_key = f"{prop_name.upper()}_Ha" if prop_name != 'zpve' else "ZPVE_Ha"
                    if prop_name == 'homo':
                        ha_key = "HOMO_Ha"
                    elif prop_name == 'lumo':
                        ha_key = "LUMO_Ha"
                    elif prop_name == 'gap':
                        ha_key = "gap_Ha"
                    property_keys.append(ha_key)
                
                # Also check direct property name
                property_keys.append(prop_name)
                
                # Apply conversion to any matching keys
                for key in property_keys:
                    if key in mol_props and isinstance(mol_props[key], (int, float)):
                        mol_props[key] = float(mol_props[key]) * conversion_factor
                        print(f"Converted {key}: original * {conversion_factor}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class CustomBatchSampler(BatchSampler):
    """Creates batches where all molecules have the same size."""
    
    def __init__(self, sampler, batch_size, drop_last, split_indices):
        super().__init__(sampler, batch_size, drop_last)
        self.split_indices = split_indices

    def __iter__(self):
        batch = []
        split_idx = 0
        for idx in self.sampler:
            if split_idx < len(self.split_indices) and idx >= self.split_indices[split_idx]:
                # Yield current batch if we're moving to molecules of different size
                if len(batch) > 0:
                    yield batch
                    batch = []
                split_idx += 1
            
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch


def collate_fn(batch):
    """Collate function for ASE datasets that matches QM9 collate behavior."""
    # Get all property keys from the batch
    all_keys = set()
    for mol in batch:
        all_keys.update(mol.keys())
    
    # Separate geometric properties from molecular properties
    geometric_keys = {'positions', 'one_hot', 'charges', 'atom_mask', 'edge_mask'}
    molecular_prop_keys = all_keys - geometric_keys
    
    # Stack geometric properties
    batch_dict = {}
    for prop in geometric_keys:
        if prop in batch[0]:
            batch_dict[prop] = qm9_collate.batch_stack([mol[prop] for mol in batch])
    
    # Handle molecular properties (scalar values)
    for prop in molecular_prop_keys:
        prop_values = []
        for mol in batch:
            if prop in mol:
                if torch.is_tensor(mol[prop]):
                    prop_values.append(mol[prop])
                else:
                    # Try to convert to float, skip if not possible
                    try:
                        prop_values.append(torch.tensor(float(mol[prop])))
                    except (ValueError, TypeError):
                        # Skip non-numeric properties
                        continue
            else:
                prop_values.append(torch.tensor(0.0))  # Default value for missing properties
        
        # Only add to batch if we have numeric values
        if prop_values:
            batch_dict[prop] = torch.stack(prop_values)

    # Ensure atom_mask is correctly formed
    if 'charges' in batch_dict:
        # Handle the case where charges contain atomic numbers
        atom_mask = batch_dict['charges'] > 0
        # Flatten charges dimension if it's 2D
        if batch_dict['charges'].dim() > 1:
            atom_mask = atom_mask.squeeze(-1)
        batch_dict['atom_mask'] = atom_mask
    elif 'atom_mask' not in batch_dict:
        # Fallback to positions-based mask
        batch_dict['atom_mask'] = torch.ones(batch_dict['positions'].shape[:2], dtype=torch.bool)

    # Obtain edges
    batch_size, n_nodes = batch_dict['atom_mask'].size()
    edge_mask = batch_dict['atom_mask'].unsqueeze(1) * batch_dict['atom_mask'].unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool,
                           device=edge_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    batch_dict['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
    
    # Ensure charges have the right format for compatibility with QM9
    if 'charges' in batch_dict:
        if batch_dict['charges'].dim() == 2:
            batch_dict['charges'] = batch_dict['charges'].unsqueeze(2)

    return batch_dict


class ASEDataLoader(DataLoader):
    def __init__(self, sequential, dataset, batch_size, shuffle, drop_last=False):
        if sequential:
            # Sequential processing for memory efficiency
            assert not shuffle
            sampler = SequentialSampler(dataset)
            batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last,
                                               dataset.split_indices)
            super().__init__(dataset, batch_sampler=batch_sampler)
        else:
            # Random processing with padding
            super().__init__(dataset, batch_size, shuffle=shuffle,
                             collate_fn=collate_fn, drop_last=drop_last)


class ASETransform(object):
    def __init__(self, dataset_info, include_charges, device, sequential):
        """
        Transform for ASE dataset.
        
        Parameters
        ----------
        dataset_info : dict
            Dataset configuration containing atomic_nb list
        include_charges : bool
            Whether to include charges
        device : torch.device
            Device to place tensors on
        sequential : bool
            Whether using sequential processing
        """
        self.atomic_number_list = torch.Tensor(dataset_info['atomic_nb'])[None, :]
        self.device = device
        self.include_charges = include_charges
        self.sequential = sequential

    def __call__(self, data):
        """
        Transform molecular data.
        
        Parameters
        ----------
        data : dict
            Molecular data with 'geometry' and 'properties' keys
            geometry: np.ndarray of shape (n_atoms, 4) with columns [atomic_number, x, y, z]
            properties: dict of molecular properties
            
        Returns
        -------
        dict
            Dictionary with keys: positions, one_hot, charges, atom_mask, edge_mask (if sequential)
            and molecular properties
        """
        geometry = data['geometry']
        properties = data['properties']
        n = geometry.shape[0]
        new_data = {}
        
        # Extract positions (last 3 columns)
        new_data['positions'] = torch.from_numpy(geometry[:, -3:]).float()
        
        # Extract atomic numbers and create one-hot encoding
        atom_types = torch.from_numpy(geometry[:, 0].astype(int)[:, None])
        one_hot = (atom_types == self.atomic_number_list.long()).float()
        new_data['one_hot'] = one_hot
        
        # Handle charges - for consistency with QM9, create charges based on atomic numbers
        if self.include_charges:
            # Use atomic numbers as charges for compatibility
            new_data['charges'] = torch.from_numpy(geometry[:, 0].astype(float)[:, None])
        else:
            new_data['charges'] = torch.zeros(0, device=self.device)
        
        # Atom mask
        new_data['atom_mask'] = torch.ones(n, device=self.device)

        # Edge mask for sequential processing
        if self.sequential:
            edge_mask = torch.ones((n, n), device=self.device)
            edge_mask[~torch.eye(edge_mask.shape[0], dtype=torch.bool)] = 0
            new_data['edge_mask'] = edge_mask.flatten()
        
        # Add molecular properties for compatibility with QM9 dataset
        for prop_name, prop_value in properties.items():
            try:
                # Try to convert to numeric tensor
                if isinstance(prop_value, (int, float)):
                    new_data[prop_name] = torch.tensor(float(prop_value))
                elif isinstance(prop_value, str):
                    # Try to parse string as number
                    new_data[prop_name] = torch.tensor(float(prop_value))
                else:
                    # Skip non-numeric properties to avoid collate issues
                    continue
            except (ValueError, TypeError):
                # Skip properties that cannot be converted to numeric values
                continue
            
        return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_file", type=str, required=True,
                        help="Path to ASE database file (.db)")
    parser.add_argument("--max_entries", type=int, default=None,
                        help="Maximum number of entries to load")
    parser.add_argument("--filter_size", type=int, default=None,
                        help="Filter molecules by number of atoms")
    parser.add_argument("--val_proportion", type=float, default=0.1,
                        help="Proportion of data for validation")
    parser.add_argument("--test_proportion", type=float, default=0.1,
                        help="Proportion of data for test")
    
    args = parser.parse_args()
    
    # Load and split data
    train_data, val_data, test_data = load_split_data(
        args.db_file,
        val_proportion=args.val_proportion,
        test_proportion=args.test_proportion,
        filter_size=args.filter_size,
        max_entries=args.max_entries
    )
    
    print(f"Data split completed:")
    print(f"  Train: {len(train_data)} molecules")
    print(f"  Val: {len(val_data)} molecules") 
    print(f"  Test: {len(test_data)} molecules")