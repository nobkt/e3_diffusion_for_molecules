import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
from qm9.data import collate as qm9_collate
import ase.db


def validate_ase_database(db_file, max_entries=None, verbose=True):
    """
    Validate an ASE database file to check if it's suitable for training.
    
    This function checks for common issues that can cause high loss values
    and fragmented molecules when training with ASE datasets.
    
    Parameters
    ----------
    db_file : str
        Path to ASE database file (.db)
    max_entries : int, optional
        Maximum number of entries to examine (None = all)
    verbose : bool
        Whether to print detailed output
        
    Returns
    -------
    bool
        True if database passes validation, False otherwise
    dict
        Dictionary with validation results and statistics
    """
    if verbose:
        print("=" * 70)
        print(f"Validating ASE Database: {db_file}")
        print("=" * 70)
    
    validation_results = {
        'file_exists': False,
        'can_connect': False,
        'num_molecules': 0,
        'valid_molecules': 0,
        'issues': [],
        'warnings': [],
        'statistics': {},
        'passed': False
    }
    
    # Check if file exists
    if not os.path.exists(db_file):
        validation_results['issues'].append(f"Database file not found: {db_file}")
        if verbose:
            print(f"❌ Database file not found: {db_file}")
        return False, validation_results
    
    validation_results['file_exists'] = True
    if verbose:
        print(f"✓ Database file exists: {db_file}")
    
    try:
        # Connect to database
        db = ase.db.connect(db_file)
        validation_results['can_connect'] = True
        if verbose:
            print(f"✓ Successfully connected to database")
        
        # Count total entries
        total_entries = len(db)
        validation_results['num_molecules'] = total_entries
        
        if total_entries == 0:
            validation_results['issues'].append("Database is empty")
            if verbose:
                print(f"❌ Database is empty")
            return False, validation_results
        
        if verbose:
            print(f"✓ Database contains {total_entries} molecules")
        
        # Examine entries
        molecules_examined = 0
        valid_molecules = 0
        atomic_numbers = []
        bond_lengths = []
        property_keys = set()
        molecule_sizes = []
        energy_values = []
        
        examine_limit = min(max_entries or total_entries, total_entries)
        
        for i, row in enumerate(db.select()):
            if i >= examine_limit:
                break
                
            molecules_examined += 1
            atoms = row.toatoms()
            
            # Check molecular structure
            atomic_nums = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
            molecule_sizes.append(len(atomic_nums))
            atomic_numbers.extend(atomic_nums)
            
            # Check for reasonable atomic numbers (common elements)
            valid_elements = {1, 6, 7, 8, 9, 15, 16, 17}  # H, C, N, O, F, P, S, Cl
            invalid_elements = set(atomic_nums) - valid_elements
            if invalid_elements:
                validation_results['warnings'].append(
                    f"Molecule {i+1}: Contains uncommon elements {invalid_elements}"
                )
            
            # Check bond lengths for unreasonable values
            if len(positions) > 1:
                distances = []
                for a in range(len(positions)):
                    for b in range(a+1, len(positions)):
                        dist = np.linalg.norm(positions[a] - positions[b])
                        distances.append(dist)
                
                bond_lengths.extend(distances)
                
                # Check for extremely short or long bonds
                min_dist = min(distances)
                max_dist = max(distances)
                
                if min_dist < 0.5:  # Atoms too close (< 0.5 Å)
                    validation_results['issues'].append(
                        f"Molecule {i+1}: Unreasonably short bond distance {min_dist:.2f} Å"
                    )
                    continue
                
                if max_dist > 20.0:  # Atoms too far apart for molecular structure
                    validation_results['warnings'].append(
                        f"Molecule {i+1}: Very long distance {max_dist:.2f} Å (possible fragment)"
                    )
            
            # Check properties
            properties = dict(row.key_value_pairs)
            if hasattr(row, 'data') and row.data:
                properties.update(row.data)
            
            property_keys.update(properties.keys())
            
            # Check for energy values in reasonable ranges
            for energy_key in ['U0_Ha', 'U0', 'energy', 'total_energy']:
                if energy_key in properties:
                    try:
                        energy_val = float(properties[energy_key])
                        energy_values.append(energy_val)
                        
                        # Check for reasonable energy ranges
                        if energy_key.endswith('_Ha'):  # Hartree units
                            if abs(energy_val) > 10000:  # Very large energy
                                validation_results['warnings'].append(
                                    f"Molecule {i+1}: Unusually large energy {energy_val} Ha"
                                )
                        elif abs(energy_val) > 100000:  # eV or other units
                            validation_results['warnings'].append(
                                f"Molecule {i+1}: Unusually large energy {energy_val}"
                            )
                    except (ValueError, TypeError):
                        validation_results['warnings'].append(
                            f"Molecule {i+1}: Non-numeric energy value in {energy_key}"
                        )
            
            valid_molecules += 1
        
        validation_results['valid_molecules'] = valid_molecules
        
        # Compute statistics
        if atomic_numbers:
            unique_elements = set(atomic_numbers)
            validation_results['statistics']['unique_elements'] = sorted(unique_elements)
            validation_results['statistics']['element_counts'] = {
                elem: atomic_numbers.count(elem) for elem in unique_elements
            }
        
        if molecule_sizes:
            validation_results['statistics']['mol_size_range'] = (min(molecule_sizes), max(molecule_sizes))
            validation_results['statistics']['avg_mol_size'] = np.mean(molecule_sizes)
        
        if bond_lengths:
            validation_results['statistics']['bond_length_range'] = (min(bond_lengths), max(bond_lengths))
            validation_results['statistics']['avg_bond_length'] = np.mean(bond_lengths)
        
        if energy_values:
            validation_results['statistics']['energy_range'] = (min(energy_values), max(energy_values))
            validation_results['statistics']['avg_energy'] = np.mean(energy_values)
        
        validation_results['statistics']['property_keys'] = sorted(property_keys)
        
        # Summary validation
        if verbose:
            print(f"\n📊 Database Statistics:")
            print(f"   Examined: {molecules_examined} molecules")
            print(f"   Valid structures: {valid_molecules}")
            
            if 'unique_elements' in validation_results['statistics']:
                elements = validation_results['statistics']['unique_elements']
                print(f"   Elements found: {elements}")
            
            if 'mol_size_range' in validation_results['statistics']:
                size_range = validation_results['statistics']['mol_size_range']
                avg_size = validation_results['statistics']['avg_mol_size']
                print(f"   Molecule sizes: {size_range[0]}-{size_range[1]} atoms (avg: {avg_size:.1f})")
            
            if 'property_keys' in validation_results['statistics']:
                props = validation_results['statistics']['property_keys'][:5]
                print(f"   Properties: {props}{'...' if len(validation_results['statistics']['property_keys']) > 5 else ''}")
        
        # Check for critical issues
        critical_issues = 0
        for issue in validation_results['issues']:
            if any(word in issue.lower() for word in ['short bond', 'empty', 'not found']):
                critical_issues += 1
        
        # Determine if validation passed
        passed = (
            validation_results['valid_molecules'] > 0 and
            critical_issues == 0 and
            len(validation_results['issues']) < validation_results['valid_molecules'] * 0.1  # < 10% issues
        )
        
        validation_results['passed'] = passed
        
        # Print warnings and issues
        if validation_results['warnings'] and verbose:
            print(f"\n⚠️  Warnings ({len(validation_results['warnings'])}):")
            for warning in validation_results['warnings'][:5]:  # Show first 5
                print(f"   {warning}")
            if len(validation_results['warnings']) > 5:
                print(f"   ... and {len(validation_results['warnings']) - 5} more warnings")
        
        if validation_results['issues'] and verbose:
            print(f"\n❌ Issues ({len(validation_results['issues'])}):")
            for issue in validation_results['issues'][:5]:  # Show first 5
                print(f"   {issue}")
            if len(validation_results['issues']) > 5:
                print(f"   ... and {len(validation_results['issues']) - 5} more issues")
        
        # Final assessment
        if verbose:
            print(f"\n" + "=" * 70)
            if passed:
                print("✅ VALIDATION PASSED")
                print("Database appears suitable for training.")
                if validation_results['warnings']:
                    print(f"Note: {len(validation_results['warnings'])} warnings detected (check output above)")
            else:
                print("❌ VALIDATION FAILED")
                print("Database has issues that may cause training problems:")
                if critical_issues > 0:
                    print(f"  • {critical_issues} critical structural issues")
                if len(validation_results['issues']) >= validation_results['valid_molecules'] * 0.1:
                    print(f"  • High issue rate: {len(validation_results['issues'])} issues out of {valid_molecules} molecules")
                
                print("\n🔧 Recommendations:")
                print("  • Check molecular structures for reasonable bond lengths")
                print("  • Verify energy values are in expected ranges")
                print("  • Consider filtering or cleaning the database")
            print("=" * 70)
        
        return passed, validation_results
        
    except Exception as e:
        validation_results['issues'].append(f"Database validation error: {str(e)}")
        if verbose:
            print(f"❌ Error validating database: {e}")
        return False, validation_results


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
        
        # Track whether unit conversion has been applied to prevent multiple applications
        self._units_converted = False
        self._converted_properties = set()

    def convert_units(self, units_dict):
        """
        Convert units of molecular properties to match QM9 dataset behavior.
        
        Parameters
        ----------
        units_dict : dict
            Dictionary mapping property names to conversion factors
        """
        # Prevent multiple applications of unit conversion
        if self._units_converted:
            print(f"Warning: Unit conversion already applied to this dataset. Skipping to prevent double conversion.")
            return
        
        # Track which properties we actually convert
        converted_count = 0
        
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
                        converted_count += 1
                        self._converted_properties.add(key)
        
        # Mark conversion as complete and log summary
        self._units_converted = True
        if converted_count > 0:
            print(f"Converted {converted_count} property values using conversion factors: {list(self._converted_properties)}")

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
    # Filter out None values (molecules that were filtered out by transform)
    valid_batch = [mol for mol in batch if mol is not None]
    
    if len(valid_batch) == 0:
        # If all molecules in the batch were filtered out, return None
        # This should be handled by the DataLoader
        raise RuntimeError("All molecules in batch were filtered out. Consider using a larger batch size or checking your dataset.")
    
    # Get all property keys from the valid batch
    all_keys = set()
    for mol in valid_batch:
        all_keys.update(mol.keys())
    
    # Separate geometric properties from molecular properties
    geometric_keys = {'positions', 'one_hot', 'charges', 'atom_mask', 'edge_mask'}
    molecular_prop_keys = all_keys - geometric_keys
    
    # Stack geometric properties
    batch_dict = {}
    for prop in geometric_keys:
        if prop in valid_batch[0]:
            batch_dict[prop] = qm9_collate.batch_stack([mol[prop] for mol in valid_batch])
    
    # Handle molecular properties (scalar values)
    for prop in molecular_prop_keys:
        prop_values = []
        for mol in valid_batch:
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
        
        # Extract positions (last 3 columns) - ensure float32 for compatibility
        new_data['positions'] = torch.from_numpy(geometry[:, -3:]).float()
        
        # Extract atomic numbers and create one-hot encoding
        atom_types = torch.from_numpy(geometry[:, 0].astype(int)[:, None])
        
        # Safety check: molecules should already be filtered at dataset level
        # but this provides a fallback in case filtering was missed
        qm9_atomic_numbers = set(self.atomic_number_list.squeeze().tolist())
        
        for atomic_num in geometry[:, 0].astype(int):
            if atomic_num not in qm9_atomic_numbers:
                # This should not happen if dataset-level filtering worked correctly
                return None  # Skip this molecule as fallback
        
        one_hot = (atom_types == self.atomic_number_list.long()).float()
        new_data['one_hot'] = one_hot
        
        # Handle charges - for consistency with QM9, create charges based on atomic numbers
        # Ensure we use float32 for compatibility
        if self.include_charges:
            # Use atomic numbers as charges for compatibility
            new_data['charges'] = torch.from_numpy(geometry[:, 0].astype(np.float32)[:, None])
        else:
            new_data['charges'] = torch.zeros(0, device=self.device)
        
        # Atom mask - ensure float32 tensor on correct device
        new_data['atom_mask'] = torch.ones(n, dtype=torch.bool, device=self.device)

        # Edge mask for sequential processing
        if self.sequential:
            edge_mask = torch.ones((n, n), device=self.device)
            edge_mask[~torch.eye(edge_mask.shape[0], dtype=torch.bool)] = 0
            new_data['edge_mask'] = edge_mask.flatten()
        
        # Add molecular properties for compatibility with QM9 dataset
        # Property key mapping from ASE format to QM9 format
        property_mapping = {
            'U0_Ha': 'U0',
            'HOMO_Ha': 'HOMO', 
            'LUMO_Ha': 'LUMO',
            'gap_Ha': 'gap',
            'ZPVE_Ha': 'zpve',
            # alpha stays the same
        }
        
        for prop_name, prop_value in properties.items():
            try:
                # Map property names to QM9 format
                qm9_prop_name = property_mapping.get(prop_name, prop_name)
                
                # Try to convert to numeric tensor with float32 dtype
                if isinstance(prop_value, (int, float)):
                    new_data[qm9_prop_name] = torch.tensor(float(prop_value), dtype=torch.float32)
                elif isinstance(prop_value, str):
                    # Try to parse string as number
                    new_data[qm9_prop_name] = torch.tensor(float(prop_value), dtype=torch.float32)
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
                        help="Maximum number of entries to examine")
    parser.add_argument("--filter_size", type=int, default=None,
                        help="Filter molecules by number of atoms")
    parser.add_argument("--val_proportion", type=float, default=0.1,
                        help="Proportion of data for validation")
    parser.add_argument("--test_proportion", type=float, default=0.1,
                        help="Proportion of data for test")
    parser.add_argument("--validate", action='store_true',
                        help="Run database validation only")
    
    args = parser.parse_args()
    
    if args.validate:
        # Run database validation
        print("Running ASE database validation...")
        passed, results = validate_ase_database(args.db_file, args.max_entries, verbose=True)
        exit(0 if passed else 1)
    else:
        # Load and split data (original functionality)
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