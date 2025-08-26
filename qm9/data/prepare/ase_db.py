"""
ASE Database dataset preparation module for E3 Diffusion.

This module provides functionality to load molecular datasets from ASE database format
and convert them to the internal format used by the E3 Diffusion framework.
"""

import logging
import os
import torch
import numpy as np
from ase.db import connect
from ase.data import atomic_numbers, chemical_symbols
from torch.nn.utils.rnn import pad_sequence


def process_ase_database(db_path, max_molecules=None, include_properties=None):
    """
    Process ASE database and convert to internal format.
    
    Parameters
    ----------
    db_path : str
        Path to the ASE database file
    max_molecules : int, optional
        Maximum number of molecules to load. If None, loads all molecules.
    include_properties : list, optional
        List of properties to include from the database. If None, includes all available.
        
    Returns
    -------
    molecules : dict
        Dictionary containing molecular data in the format expected by the framework:
        - 'charges': atomic numbers
        - 'positions': atomic coordinates
        - 'num_atoms': number of atoms per molecule
        - Additional properties from the database
    """
    logging.info(f'Loading ASE database from: {db_path}')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ASE database not found: {db_path}")
    
    # Connect to ASE database
    db = connect(db_path)
    
    molecules = []
    count = 0
    
    for row in db.select():
        if max_molecules is not None and count >= max_molecules:
            break
            
        atoms = row.toatoms()
        
        # Extract basic atomic information
        charges = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        num_atoms = torch.tensor(len(atoms), dtype=torch.long)
        
        # Create molecule dictionary
        molecule = {
            'charges': charges,
            'positions': positions,
            'num_atoms': num_atoms
        }
        
        # Add properties from database if available
        if include_properties is None:
            # Include all available properties
            for key, value in row.data.items():
                if isinstance(value, (int, float)):
                    molecule[key] = torch.tensor(float(value), dtype=torch.float32)
        else:
            # Include only specified properties
            for prop in include_properties:
                if prop in row.data:
                    value = row.data[prop]
                    if isinstance(value, (int, float)):
                        molecule[prop] = torch.tensor(float(value), dtype=torch.float32)
        
        molecules.append(molecule)
        count += 1
    
    logging.info(f'Loaded {len(molecules)} molecules from ASE database')
    
    if len(molecules) == 0:
        raise ValueError("No molecules found in the database")
    
    # Check that all molecules have the same set of properties
    props = molecules[0].keys()
    if not all(props == mol.keys() for mol in molecules):
        logging.warning('Not all molecules have the same set of properties!')
    
    # Convert list-of-dicts to dict-of-lists
    molecules_dict = {prop: [mol[prop] for mol in molecules] for prop in props}
    
    # Pad and stack tensors
    padded_molecules = {}
    for key, val_list in molecules_dict.items():
        if val_list[0].dim() > 0:
            # Pad sequences for variable-length data
            padded_molecules[key] = pad_sequence(val_list, batch_first=True)
        else:
            # Stack scalar values
            padded_molecules[key] = torch.stack(val_list)
    
    return padded_molecules


def create_ase_splits(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Create train/validation/test splits for ASE dataset.
    Ensures all splits contain all atomic species present in the dataset.
    
    Parameters
    ----------
    data : dict
        Molecular data dictionary
    train_ratio : float
        Fraction of data for training
    valid_ratio : float
        Fraction of data for validation
    test_ratio : float
        Fraction of data for testing
    random_seed : int
        Random seed for reproducible splits
        
    Returns
    -------
    splits : dict
        Dictionary with 'train', 'valid', 'test' keys containing data splits
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    num_molecules = len(data['charges'])
    logging.info(f'Creating splits for {num_molecules} molecules')
    
    if num_molecules < 3:
        logging.warning(f'Very small dataset ({num_molecules} molecules). Using single split for all.')
        splits = {
            'train': data,
            'valid': data,
            'test': data
        }
        return splits
    
    # Get unique atomic species in the dataset
    all_charges = data['charges']
    unique_species = torch.unique(all_charges[all_charges > 0]).tolist()
    logging.info(f'Dataset contains atomic species: {unique_species}')
    
    # Find molecules that contain each species
    species_to_molecules = {}
    for species in unique_species:
        molecules_with_species = []
        for mol_idx in range(num_molecules):
            mol_charges = all_charges[mol_idx]
            # Check if this species is present in this molecule
            if torch.any(mol_charges == species):
                molecules_with_species.append(mol_idx)
        species_to_molecules[species] = molecules_with_species
        logging.info(f'Species {species}: found in {len(molecules_with_species)} molecules')
    
    # Generate random permutation for splitting
    np.random.seed(random_seed)
    indices = np.random.permutation(num_molecules)
    
    # Calculate split sizes with minimum 1 molecule per split
    train_size = max(1, int(train_ratio * num_molecules))
    valid_size = max(1, int(valid_ratio * num_molecules))
    test_size = max(1, num_molecules - train_size - valid_size)
    
    # Adjust if total exceeds molecules available
    if train_size + valid_size + test_size > num_molecules:
        if num_molecules >= 3:
            train_size = num_molecules - 2
            valid_size = 1
            test_size = 1
        else:
            # Fall back to simple split
            train_size = num_molecules
            valid_size = 0
            test_size = 0
    
    # Strategy: Ensure each split gets at least one molecule with each species
    # Then distribute the remaining molecules randomly
    
    # Start with empty splits
    train_indices = []
    valid_indices = []
    test_indices = []
    assigned_molecules = set()
    
    # First, assign one molecule with each species to each split (if possible)
    for species in unique_species:
        candidates = [mol for mol in species_to_molecules[species] if mol not in assigned_molecules]
        if len(candidates) >= 3:
            # Enough molecules to assign one to each split
            np.random.shuffle(candidates)
            train_indices.append(candidates[0])
            valid_indices.append(candidates[1])
            test_indices.append(candidates[2])
            assigned_molecules.update(candidates[:3])
        elif len(candidates) >= 2:
            # Assign to two splits, duplicate one to the third split
            np.random.shuffle(candidates)
            train_indices.append(candidates[0])
            valid_indices.append(candidates[1])
            test_indices.append(candidates[0])  # Duplicate to ensure all splits have this species
            assigned_molecules.update(candidates[:2])
        elif len(candidates) >= 1:
            # Only one molecule with this species, add to all splits
            mol = candidates[0]
            train_indices.append(mol)
            valid_indices.append(mol)
            test_indices.append(mol)
            assigned_molecules.add(mol)
        else:
            logging.warning(f'No molecules found with species {species}')
    
    # Remove duplicates while preserving order
    train_indices = list(dict.fromkeys(train_indices))
    valid_indices = list(dict.fromkeys(valid_indices))
    test_indices = list(dict.fromkeys(test_indices))
    
    # Now distribute remaining molecules randomly according to ratios
    remaining_molecules = [mol for mol in indices if mol not in assigned_molecules]
    
    # Calculate how many more molecules each split needs
    remaining_train = max(0, train_size - len(train_indices))
    remaining_valid = max(0, valid_size - len(valid_indices))
    remaining_test = max(0, test_size - len(test_indices))
    
    # Distribute remaining molecules
    np.random.shuffle(remaining_molecules)
    idx = 0
    
    # Add to train split
    train_indices.extend(remaining_molecules[idx:idx + remaining_train])
    idx += remaining_train
    
    # Add to valid split
    valid_indices.extend(remaining_molecules[idx:idx + remaining_valid])
    idx += remaining_valid
    
    # Add remaining to test split
    test_indices.extend(remaining_molecules[idx:])
    
    # Convert to numpy arrays and create splits
    train_indices = np.array(train_indices)
    valid_indices = np.array(valid_indices)
    test_indices = np.array(test_indices)
    
    splits = {
        'train': {key: val[train_indices] for key, val in data.items()},
        'valid': {key: val[valid_indices] for key, val in data.items()},
        'test': {key: val[test_indices] for key, val in data.items()}
    }
    
    logging.info(f'Split sizes - Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}')
    
    # Verify that all splits contain all species
    for split_name, split_data in splits.items():
        split_species = torch.unique(split_data['charges'][split_data['charges'] > 0]).tolist()
        logging.info(f'{split_name} contains species: {split_species}')
        if set(split_species) != set(unique_species):
            logging.warning(f'{split_name} missing species: {set(unique_species) - set(split_species)}')
    
    return splits


def get_ase_dataset_info(data, with_h=True):
    """
    Extract dataset information from ASE data to create dataset config.
    
    Parameters
    ----------
    data : dict
        Molecular data dictionary
    with_h : bool
        Whether to include hydrogen atoms
        
    Returns
    -------
    dataset_info : dict
        Dataset configuration dictionary
    """
    # Get unique atomic numbers
    all_charges = data['charges']
    unique_charges = torch.unique(all_charges[all_charges > 0]).tolist()
    
    if not with_h and 1 in unique_charges:
        unique_charges.remove(1)
    
    # Create atom encoder/decoder
    unique_elements = [chemical_symbols[z] for z in sorted(unique_charges)]
    atom_encoder = {elem: i for i, elem in enumerate(unique_elements)}
    atom_decoder = unique_elements
    
    # Calculate statistics
    num_atoms_per_mol = data['num_atoms']
    max_n_nodes = int(torch.max(num_atoms_per_mol).item())
    
    # Count number of molecules by atom count
    n_nodes = {}
    for n in range(1, max_n_nodes + 1):
        count = int(torch.sum(num_atoms_per_mol == n).item())
        if count > 0:
            n_nodes[n] = count
    
    # Count atom types
    atom_types = {}
    for i, elem in enumerate(unique_elements):
        z = atomic_numbers[elem]
        count = int(torch.sum(all_charges == z).item())
        atom_types[i] = count
    
    dataset_info = {
        'name': 'ase',
        'atom_encoder': atom_encoder,
        'atom_decoder': atom_decoder,
        'n_nodes': n_nodes,
        'max_n_nodes': max_n_nodes,
        'atom_types': atom_types,
        'with_h': with_h,
        'colors_dic': [f'C{i}' for i in range(len(unique_elements))],
        'radius_dic': [0.77] * len(unique_elements)  # Default radius
    }
    
    return dataset_info


def prepare_ase_dataset(db_path, output_dir, max_molecules=None, include_properties=None,
                       train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Prepare ASE dataset for training by loading, splitting, and saving data.
    
    Parameters
    ----------
    db_path : str
        Path to ASE database file
    output_dir : str
        Directory to save processed data
    max_molecules : int, optional
        Maximum number of molecules to process
    include_properties : list, optional
        List of properties to include
    train_ratio, valid_ratio, test_ratio : float
        Data split ratios
    random_seed : int
        Random seed for reproducible results
        
    Returns
    -------
    datafiles : dict
        Dictionary with paths to saved data files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process ASE database
    data = process_ase_database(db_path, max_molecules, include_properties)
    
    # Create splits
    splits = create_ase_splits(data, train_ratio, valid_ratio, test_ratio, random_seed)
    
    # Save splits to files
    datafiles = {}
    for split_name, split_data in splits.items():
        filename = os.path.join(output_dir, f'ase_{split_name}.npz')
        
        # Convert tensors to numpy for saving
        numpy_data = {key: val.numpy() for key, val in split_data.items()}
        np.savez(filename, **numpy_data)
        
        datafiles[split_name] = filename
        logging.info(f'Saved {split_name} split to {filename}')
    
    return datafiles